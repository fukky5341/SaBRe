## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 7200 seconds
Split limit: 100
Threshold: 11.5728484671


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2692566, 25.2692566)
1: (-1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7759514, 19.7759552)
2: (-1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2921982, 17.2921982)
3: (-9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0309601, 22.0309525)
4: (-3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7619209, 21.7619209)
5: (-7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7809639, 23.7809677)
6: (-28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2222900, 23.2222900)
7: (-7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6115227, 23.6115303)
8: (-14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5333672, 26.5333710)
9: (-5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2980270, 24.2980309)
10: (-17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3112411, 31.3112411)
11: (-26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8951187, 27.8951187)
12: (-34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2311554, 27.2311478)
13: (-26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9192047, 33.9192047)
14: (-55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8087692, 37.8087769)
15: (-14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9327469, 27.9327469)
16: (-14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0715408, 31.0715408)
17: (-57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6438599, 41.6438522)
18: (-21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6204224, 29.6204224)
19: (-22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7970428, 22.7970467)
20: (-23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2281265, 19.2281265)
21: (-26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5346451, 25.5346451)
22: (-28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7372169, 24.7372131)
23: (-22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0434837, 22.0434837)
24: (-18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8599319, 22.8599319)
25: (-23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4248238, 24.4248238)
26: (-41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6490631, 30.6490631)
27: (-21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4651260, 26.4651260)
28: (-24.1100121, 6.1282749, -24.1100121, 6.1282749, -22.0117798, 22.0117836)
29: (-27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9739227, 23.9739189)
30: (-28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1395111, 26.1395111)
31: (-22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0944595, 25.0944595)
32: (-23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3960495, 21.3960419)
33: (-36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4106445, 33.4106522)
34: (-37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7963409, 27.7963333)
35: (-32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2126007, 28.2126007)
36: (-36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0765915, 29.0765915)
37: (-44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8305664, 38.8305664)
38: (-43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7540741, 40.7540741)
39: (-43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4122162, 41.4122162)
40: (-32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0533371, 31.0533447)
41: (-20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5302048, 26.5302048)
42: (-22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4770126, 18.4770164)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.14 + 44.45 = 47.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -11.5844329, upper bound: 11.5844329

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1748

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5536042, upper bound: 11.5825229
time: 32.54 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5826125, upper bound: 11.5826126
time: 30.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 63.31 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 63.31
Output dim: 2, lower bound: -11.5536042, upper bound: 11.5825229
IS_A2, status: Status.UNKNOWN, split count: 1, time: 63.31
Output dim: 2, lower bound: -11.5826125, upper bound: 11.5826126

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.3402271, 19.0136986, -9.3820267, 19.0150719, -25.2074013, 25.2494659
1: -1.2079921, 22.8010368, -1.2401023, 22.8025932, -19.7342529, 19.7611237
2: -1.6164184, 20.9130745, -1.6449940, 20.9144363, -17.2541008, 17.2783279
3: -9.3487911, 16.4751968, -9.3734426, 16.4796143, -21.9942474, 22.0138092
4: -3.1367774, 22.2231922, -3.1714931, 22.2244415, -21.7158318, 21.7446747
5: -7.8344159, 20.6021023, -7.8635774, 20.6059074, -23.7415657, 23.7626114
6: -28.8246193, -1.3898387, -28.8285427, -1.3782101, -23.2090759, 23.1952667
7: -7.6924896, 21.6246910, -7.7231965, 21.6270351, -23.5712204, 23.5956688
8: -14.7838860, 14.7501955, -14.8250723, 14.7544832, -26.4793472, 26.5116272
9: -5.1931233, 21.2772388, -5.2120018, 21.2798748, -24.2576370, 24.2810478
10: -17.8781204, 17.5540752, -17.8995266, 17.5590191, -31.2759552, 31.2938995
11: -26.7491341, 3.5663528, -26.7529373, 3.5897217, -27.8803864, 27.8616028
12: -34.8722839, -2.3496175, -34.8889008, -2.3401413, -27.2040024, 27.2126465
13: -26.2508450, 15.6957302, -26.2898197, 15.7014952, -33.8611908, 33.8951569
14: -55.8753738, -17.5577660, -55.9186783, -17.5511360, -37.7431030, 37.7820587
15: -14.3656607, 15.5094194, -14.3840332, 15.5146704, -27.9026413, 27.9157257
16: -14.0406456, 20.8055916, -14.0670547, 20.8072872, -31.0344238, 31.0581284
17: -57.8179779, -14.4294653, -57.8514748, -14.4202518, -41.5949707, 41.6209869
18: -21.5687027, 12.1755362, -21.5768890, 12.1999798, -29.5979767, 29.5794678
19: -22.2677612, 3.5670469, -22.2738056, 3.6027186, -22.7739258, 22.7439194
20: -23.2789745, 1.3705788, -23.2815857, 1.4100804, -19.2075233, 19.1719666
21: -26.7845955, 2.4011731, -26.7901764, 2.4399669, -25.5101852, 25.4773140
22: -28.4867973, 3.3260508, -28.4911537, 3.3654177, -24.7137985, 24.6791229
23: -22.2772369, 5.7060184, -22.2802200, 5.7455530, -22.0225830, 21.9870834
24: -18.2940655, 9.4450579, -18.2979336, 9.4797916, -22.8404007, 22.8094711
25: -23.8123245, 5.3899593, -23.8158646, 5.4247737, -24.4035835, 24.3736916
26: -41.0209885, -0.4649701, -41.0246162, -0.4123173, -30.6214523, 30.5743408
27: -21.5626640, 8.5774841, -21.5674667, 8.6166916, -26.4428329, 26.4069824
28: -24.1066551, 6.0673714, -24.1090069, 6.1103735, -21.9895897, 21.9499321
29: -27.8341999, -0.2175801, -27.8381290, -0.1889012, -23.9555511, 23.9314766
30: -28.1074810, 3.7526748, -28.1098289, 3.7909448, -26.1199379, 26.0877342
31: -22.6584358, 5.0586271, -22.6648331, 5.0880771, -25.0724945, 25.0480423
32: -23.9455566, 2.3184059, -23.9497414, 2.3274124, -21.3861008, 21.3805275
33: -36.4322052, 3.6537809, -36.4394760, 3.6693735, -33.3912201, 33.3849640
34: -37.8594933, -4.7548389, -37.8624878, -4.7277489, -27.7810669, 27.7584229
35: -32.9239845, 0.3041096, -32.9291763, 0.3201632, -28.1985779, 28.1877289
36: -36.8419952, -0.6616116, -36.8468132, -0.6336160, -29.0590439, 29.0335770
37: -44.5641823, -1.7148595, -44.5752258, -1.6957555, -38.8039856, 38.7970352
38: -43.9597816, 2.8804269, -43.9671593, 2.9056392, -40.7319641, 40.7048111
39: -43.6051140, 3.0179591, -43.6184120, 3.0235562, -41.3903503, 41.3956604
40: -32.7429276, -0.0085402, -32.7558784, -0.0057979, -31.0293503, 31.0400925
41: -20.7328033, 7.2734623, -20.7396145, 7.2923942, -26.5086212, 26.4890747
42: -22.9918900, -0.2118957, -22.9939461, -0.2013192, -18.4683304, 18.4606094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=173, inp2_unstable=174, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1731

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5519104, upper bound: 11.5444519
time: 31.51 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5519104, upper bound: 11.5806461
time: 33.66 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.4091663, 19.0551910, -9.3963757, 19.0155640, -25.2699852, 25.3078079
1: -1.2578282, 22.8439827, -1.2514038, 22.8030567, -19.7732086, 19.8167114
2: -1.6577034, 20.9477348, -1.6550651, 20.9148884, -17.2845230, 17.3262329
3: -9.3796663, 16.5268326, -9.3813286, 16.4810715, -22.0228157, 22.0690460
4: -3.1888533, 22.2469902, -3.1836224, 22.2248058, -21.7585754, 21.7722168
5: -7.8753591, 20.6544342, -7.8738647, 20.6072083, -23.7737503, 23.8231430
6: -28.8371277, -1.3678312, -28.8298340, -1.3760614, -23.2362213, 23.2224197
7: -7.7411175, 21.6644115, -7.7342048, 21.6277599, -23.6113548, 23.6462631
8: -14.8397980, 14.7924204, -14.8394861, 14.7556562, -26.5231552, 26.5571136
9: -5.2219214, 21.3037910, -5.2182140, 21.2806969, -24.2874222, 24.3398514
10: -17.9116669, 17.5932255, -17.9059410, 17.5607548, -31.3102341, 31.3437958
11: -26.8141994, 3.6001067, -26.7539883, 3.5980287, -27.9559021, 27.8910217
12: -34.8990479, -2.3011327, -34.8949890, -2.3371925, -27.2318726, 27.2678986
13: -26.3050804, 15.7871304, -26.3036652, 15.7033710, -33.9097137, 34.0025024
14: -55.9415817, -17.4802513, -55.9342613, -17.5489922, -37.8035583, 37.8758850
15: -14.3927393, 15.5231752, -14.3880301, 15.5161266, -27.9336090, 27.9348907
16: -14.0905409, 20.8620377, -14.0763674, 20.8079033, -31.0763168, 31.1239243
17: -57.8773804, -14.3331242, -57.8622589, -14.4171162, -41.6565399, 41.7327652
18: -21.6201019, 12.2086935, -21.5794868, 12.2089148, -29.6615219, 29.6139526
19: -22.3521633, 3.6151781, -22.2757568, 3.6157532, -22.8720856, 22.7857285
20: -23.3597584, 1.4225903, -23.2824860, 1.4240274, -19.3069038, 19.2086334
21: -26.8754959, 2.4517946, -26.7919655, 2.4540417, -25.6166382, 25.5167847
22: -28.5934696, 3.3765032, -28.4925442, 3.3791952, -24.8397713, 24.7173843
23: -22.3550415, 5.7595682, -22.2810078, 5.7595491, -22.1176453, 22.0296402
24: -18.3952065, 9.4904957, -18.2991886, 9.4922256, -22.9560318, 22.8466148
25: -23.8802376, 5.4389763, -23.8170509, 5.4372082, -24.4878540, 24.4128418
26: -41.1240845, -0.3976660, -41.0258026, -0.3934507, -30.7472229, 30.6226883
27: -21.6614342, 8.6291981, -21.5691452, 8.6307335, -26.5578232, 26.4525528
28: -24.1907864, 6.1234884, -24.1097717, 6.1255383, -22.0929298, 21.9909706
29: -27.9373589, -0.1793631, -27.8394051, -0.1786439, -24.0727234, 23.9611282
30: -28.1860466, 3.8051488, -28.1105595, 3.8043532, -26.2145157, 26.1261826
31: -22.7232533, 5.0986791, -22.6669140, 5.0986991, -25.1508560, 25.0862389
32: -23.9672699, 2.3342619, -23.9511757, 2.3284750, -21.4176483, 21.3985481
33: -36.4767075, 3.6735821, -36.4418640, 3.6714272, -33.4361420, 33.4133224
34: -37.9017639, -4.7102485, -37.8633995, -4.7189951, -27.8376160, 27.7987671
35: -32.9659729, 0.3341117, -32.9309502, 0.3246851, -28.2465439, 28.2192764
36: -36.8959427, -0.6156659, -36.8484879, -0.6234851, -29.1251984, 29.0785828
37: -44.6311798, -1.6883178, -44.5787239, -1.6913133, -38.8771820, 38.8295670
38: -44.0143738, 2.9291816, -43.9696655, 2.9136596, -40.8125305, 40.7622833
39: -43.6460953, 3.0354609, -43.6227875, 3.0233293, -41.4458008, 41.4181061
40: -32.7804260, 0.0114820, -32.7600250, -0.0050654, -31.0707321, 31.0647354
41: -20.7735176, 7.2998304, -20.7417870, 7.2977362, -26.5736771, 26.5224457
42: -23.0002289, -0.1971669, -22.9944534, -0.1995764, -18.4794998, 18.4771805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=173, inp2_unstable=174, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1731

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5807192, upper bound: 11.5445066
time: 30.20 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5807192, upper bound: 11.5807191
time: 36.04 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 68.86 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 68.86
Output dim: 2, lower bound: -11.5519104, upper bound: 11.5444519
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 68.86
Output dim: 2, lower bound: -11.5519104, upper bound: 11.5806461
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 68.86
Output dim: 2, lower bound: -11.5807192, upper bound: 11.5445066
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 68.86
Output dim: 2, lower bound: -11.5807192, upper bound: 11.5807191

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.3378229, 19.0136490, -9.3986082, 19.0579834, -25.2490921, 25.2644310
1: -1.2055645, 22.8009415, -1.2461214, 22.8397675, -19.7695847, 19.7703743
2: -1.6144819, 20.9129238, -1.6486776, 20.9668713, -17.3035011, 17.2813911
3: -9.3471317, 16.4747066, -9.3715706, 16.5474911, -22.0575485, 22.0114174
4: -3.1344414, 22.2230492, -3.1764193, 22.2553024, -21.7406464, 21.7509575
5: -7.8322563, 20.6016827, -7.8630905, 20.6676388, -23.7852173, 23.7684555
6: -28.8239594, -1.3904982, -28.8341713, -1.3559504, -23.2101822, 23.2240868
7: -7.6902409, 21.6245193, -7.7292418, 21.6671791, -23.6057587, 23.6060028
8: -14.7813816, 14.7497654, -14.8259983, 14.8055496, -26.5279312, 26.5123672
9: -5.1900244, 21.2770519, -5.2150249, 21.3068562, -24.3076630, 24.2793884
10: -17.8753281, 17.5537567, -17.9073257, 17.5797729, -31.2878876, 31.3070984
11: -26.7485600, 3.5649533, -26.8303604, 3.5927782, -27.8827667, 27.9419556
12: -34.8714981, -2.3502717, -34.8947830, -2.3014646, -27.2430344, 27.2185173
13: -26.2481499, 15.6952295, -26.2911224, 15.8356590, -33.9974213, 33.8947449
14: -55.8748093, -17.5592842, -55.9314232, -17.5415955, -37.7864685, 37.7739105
15: -14.3635530, 15.5090199, -14.3895912, 15.5369024, -27.9241943, 27.9220200
16: -14.0385151, 20.8054581, -14.0894251, 20.8343067, -31.0632782, 31.0872879
17: -57.8172722, -14.4301147, -57.8719215, -14.3774910, -41.6644821, 41.6328125
18: -21.5680351, 12.1731224, -21.6883011, 12.1980534, -29.5940781, 29.6912689
19: -22.2671604, 3.5653291, -22.3638363, 3.6015439, -22.7722855, 22.8314705
20: -23.2787285, 1.3683209, -23.3765640, 1.4098103, -19.2053223, 19.2681923
21: -26.7841148, 2.3987834, -26.9004059, 2.4377267, -25.5062485, 25.5841637
22: -28.4863777, 3.3235908, -28.6169510, 3.3654346, -24.7121010, 24.8040237
23: -22.2767944, 5.7040157, -22.3727093, 5.7447939, -22.0222015, 22.0747681
24: -18.2935810, 9.4430294, -18.4300404, 9.4775858, -22.8363342, 22.9406586
25: -23.8118858, 5.3873096, -23.9282417, 5.4255562, -24.4028625, 24.4843750
26: -41.0206299, -0.4679470, -41.1326714, -0.4173136, -30.6162872, 30.6777344
27: -21.5623341, 8.5750732, -21.6880341, 8.6145725, -26.4392014, 26.5287857
28: -24.1063843, 6.0650539, -24.2115440, 6.1062837, -21.9842758, 22.0468750
29: -27.8336811, -0.2192569, -27.9508629, -0.1868072, -23.9562683, 24.0462036
30: -28.1071110, 3.7500017, -28.2311783, 3.7906144, -26.1194954, 26.2077942
31: -22.6579285, 5.0565510, -22.7609787, 5.0884113, -25.0711441, 25.1420212
32: -23.9435196, 2.3176839, -23.9628029, 2.3618608, -21.4172363, 21.3954086
33: -36.4310760, 3.6533442, -36.4565353, 3.6879463, -33.4136429, 33.3951874
34: -37.8592682, -4.7573676, -37.8918762, -4.7196717, -27.7994308, 27.7833481
35: -32.9236717, 0.3035932, -32.9438477, 0.3387933, -28.2199326, 28.2004623
36: -36.8416977, -0.6635904, -36.8651085, -0.6195173, -29.0762405, 29.0509720
37: -44.5631714, -1.7151618, -44.6125488, -1.6814203, -38.8217316, 38.8312073
38: -43.9593353, 2.8771725, -43.9964256, 2.9283543, -40.7545471, 40.7462387
39: -43.6017113, 3.0176167, -43.6377258, 3.0727434, -41.4352112, 41.4237137
40: -32.7415619, -0.0088396, -32.7841263, 0.0554767, -31.0875473, 31.0670853
41: -20.7304420, 7.2730823, -20.7466106, 7.3104420, -26.5199890, 26.5043945
42: -22.9904289, -0.2124083, -22.9946728, -0.1796470, -18.4912758, 18.4625664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=173, inp2_unstable=173, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1745

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5292498, upper bound: 11.5791595
time: 32.07 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5509860, upper bound: 11.5797208
time: 28.70 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.4049225, 19.0550728, -9.3628283, 19.0146866, -25.2647705, 25.2728271
1: -1.2549448, 22.8438416, -1.2288017, 22.8018913, -19.7692986, 19.7968178
2: -1.6540458, 20.9475937, -1.6260252, 20.9136963, -17.2798309, 17.2990952
3: -9.3762226, 16.5263119, -9.3540754, 16.4768906, -22.0147362, 22.0413551
4: -3.1850457, 22.2468090, -3.1536984, 22.2232609, -21.7532196, 21.7440414
5: -7.8725467, 20.6539822, -7.8520026, 20.6033497, -23.7659683, 23.8054047
6: -28.8367348, -1.3687162, -28.8264809, -1.3831568, -23.2182846, 23.2169533
7: -7.7379713, 21.6642494, -7.7094879, 21.6265488, -23.6070404, 23.6248779
8: -14.8350220, 14.7920761, -14.8030310, 14.7530880, -26.5159950, 26.5210876
9: -5.2200522, 21.3035660, -5.2035484, 21.2788258, -24.2824173, 24.3167267
10: -17.9101353, 17.5928459, -17.8939362, 17.5577221, -31.3026199, 31.3302917
11: -26.8137531, 3.5976701, -26.7505989, 3.5786185, -27.9362106, 27.8849716
12: -34.8982506, -2.3022265, -34.8886871, -2.3458242, -27.2231674, 27.2608757
13: -26.3001804, 15.7866211, -26.2648048, 15.6993599, -33.9010620, 33.9640350
14: -55.9411011, -17.4811440, -55.9303665, -17.5561714, -37.7953033, 37.8608170
15: -14.3901939, 15.5225925, -14.3678417, 15.5114250, -27.9259720, 27.9131622
16: -14.0889349, 20.8618660, -14.0637474, 20.8066769, -31.0736237, 31.1118469
17: -57.8770065, -14.3340120, -57.8589783, -14.4240742, -41.6491089, 41.7210236
18: -21.6196213, 12.2038412, -21.5754013, 12.1703568, -29.6242218, 29.6050835
19: -22.3516083, 3.6118515, -22.2711296, 3.5893285, -22.8450394, 22.7782173
20: -23.3594398, 1.4185028, -23.2804871, 1.3914757, -19.2763901, 19.2025986
21: -26.8749847, 2.4472549, -26.7878780, 2.4182198, -25.5801697, 25.5079918
22: -28.5931168, 3.3720620, -28.4898567, 3.3440156, -24.8057861, 24.7102814
23: -22.3546505, 5.7558928, -22.2779083, 5.7305822, -22.0926476, 22.0230713
24: -18.3948498, 9.4865150, -18.2961922, 9.4608564, -22.9273224, 22.8400383
25: -23.8798866, 5.4342375, -23.8140602, 5.4002705, -24.4523315, 24.4047852
26: -41.1239014, -0.4033813, -41.0237579, -0.4384727, -30.7057953, 30.6147690
27: -21.6610699, 8.6245956, -21.5660458, 8.5940685, -26.5213623, 26.4452362
28: -24.1905174, 6.1188765, -24.1078510, 6.0891986, -22.0598869, 21.9840889
29: -27.9370155, -0.1824368, -27.8365479, -0.2031064, -24.0487137, 23.9548531
30: -28.1858025, 3.8000948, -28.1083527, 3.7641497, -26.1752777, 26.1186066
31: -22.7226639, 5.0949240, -22.6622448, 5.0688767, -25.1200867, 25.0780754
32: -23.9663525, 2.3335409, -23.9438858, 2.3229604, -21.4105301, 21.3896446
33: -36.4757538, 3.6731544, -36.4338799, 3.6680822, -33.4301147, 33.4020081
34: -37.9015198, -4.7120976, -37.8612747, -4.7336016, -27.8242264, 27.7947540
35: -32.9654846, 0.3334413, -32.9274254, 0.3197064, -28.2410049, 28.2151108
36: -36.8955421, -0.6168728, -36.8457108, -0.6327519, -29.1149063, 29.0749817
37: -44.6298523, -1.6886530, -44.5684738, -1.6936407, -38.8706665, 38.8163376
38: -44.0138245, 2.9276066, -43.9652405, 2.9010420, -40.7974548, 40.7568359
39: -43.6436996, 3.0352397, -43.6036072, 3.0214810, -41.4414825, 41.4008179
40: -32.7779694, 0.0111177, -32.7407303, -0.0079274, -31.0653610, 31.0460129
41: -20.7723312, 7.2994003, -20.7323608, 7.2942576, -26.5666580, 26.5111923
42: -22.9993591, -0.1976445, -22.9876213, -0.2034159, -18.4745636, 18.4702339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=173, inp2_unstable=173, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1731

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5445065, upper bound: 11.5445066
time: 32.85 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5445065, upper bound: 11.5445066
time: 36.54 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.4066973, 19.0551167, -9.4128971, 19.0584488, -25.3116341, 25.3226967
1: -1.2553806, 22.8438835, -1.2573714, 22.8402004, -19.8085327, 19.8259277
2: -1.6557536, 20.9475975, -1.6587379, 20.9672985, -17.3338814, 17.3292618
3: -9.3779335, 16.5263367, -9.3794279, 16.5489616, -22.0860596, 22.0666504
4: -3.1864715, 22.2468548, -3.1885400, 22.2556801, -21.7833862, 21.7782784
5: -7.8731303, 20.6540203, -7.8733158, 20.6689262, -23.8173332, 23.8292465
6: -28.8364601, -1.3685284, -28.8353996, -1.3538728, -23.2372971, 23.2512169
7: -7.7388506, 21.6642170, -7.7402248, 21.6678886, -23.6458588, 23.6565933
8: -14.8372841, 14.7919483, -14.8404579, 14.8066750, -26.5716400, 26.5578461
9: -5.2188172, 21.3035545, -5.2211971, 21.3076820, -24.3374023, 24.3381920
10: -17.9087868, 17.5928745, -17.9134712, 17.5814857, -31.3220062, 31.3568878
11: -26.8135643, 3.5986891, -26.8314362, 3.6010141, -27.9582062, 27.9713287
12: -34.8981857, -2.3018227, -34.9006424, -2.2985311, -27.2708130, 27.2736626
13: -26.3024330, 15.7866163, -26.3048973, 15.8375177, -34.0459290, 34.0020523
14: -55.9410095, -17.4817829, -55.9470253, -17.5394440, -37.8468781, 37.8676834
15: -14.3905888, 15.5227652, -14.3935032, 15.5383625, -27.9550552, 27.9410706
16: -14.0883818, 20.8619194, -14.0986004, 20.8349380, -31.1050949, 31.1530380
17: -57.8766327, -14.3337784, -57.8827019, -14.3743744, -41.7255249, 41.7444153
18: -21.6194153, 12.2062550, -21.6907806, 12.2069302, -29.6575928, 29.7256165
19: -22.3515720, 3.6134694, -22.3657932, 3.6146147, -22.8704224, 22.8732605
20: -23.3595161, 1.4203386, -23.3774414, 1.4237285, -19.3046341, 19.3048553
21: -26.8749866, 2.4493334, -26.9022560, 2.4517355, -25.6127396, 25.6235733
22: -28.5929871, 3.3741102, -28.6183205, 3.3792479, -24.8380203, 24.8419113
23: -22.3546047, 5.7575212, -22.3735180, 5.7587814, -22.1174278, 22.1172943
24: -18.3947487, 9.4884090, -18.4313660, 9.4900112, -22.9518738, 22.9777374
25: -23.8798122, 5.4363050, -23.9294376, 5.4379539, -24.4871559, 24.5232658
26: -41.1237335, -0.4006677, -41.1338539, -0.3985267, -30.7420654, 30.7254257
27: -21.6611404, 8.6267834, -21.6896935, 8.6286583, -26.5541916, 26.5743561
28: -24.1905308, 6.1211171, -24.2123413, 6.1214466, -22.0876007, 22.0878525
29: -27.9368973, -0.1810619, -27.9520435, -0.1766080, -24.0734520, 24.0758400
30: -28.1856689, 3.8024111, -28.2319450, 3.8039536, -26.2138824, 26.2462006
31: -22.7227402, 5.0966368, -22.7630138, 5.0989866, -25.1494217, 25.1801376
32: -23.9652481, 2.3334746, -23.9642448, 2.3629444, -21.4486885, 21.4133873
33: -36.4755821, 3.6731381, -36.4589272, 3.6899595, -33.4585724, 33.4235229
34: -37.9015808, -4.7128282, -37.8928108, -4.7109404, -27.8558655, 27.8235245
35: -32.9655800, 0.3335171, -32.9456062, 0.3432274, -28.2677307, 28.2319412
36: -36.8956490, -0.6176906, -36.8667641, -0.6094718, -29.1422424, 29.0959320
37: -44.6301041, -1.6886401, -44.6160965, -1.6771040, -38.8945465, 38.8636703
38: -44.0138969, 2.9259229, -43.9989090, 2.9362111, -40.8350525, 40.8035583
39: -43.6424942, 3.0351267, -43.6421165, 3.0725350, -41.4906311, 41.4461365
40: -32.7790413, 0.0111985, -32.7881889, 0.0562038, -31.1290436, 31.0916367
41: -20.7710991, 7.2994719, -20.7487946, 7.3157992, -26.5850067, 26.5376968
42: -22.9987907, -0.1976471, -22.9951916, -0.1779449, -18.5023079, 18.4791183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=173, inp2_unstable=173, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1745

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5581328, upper bound: 11.5792189
time: 35.29 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5797930, upper bound: 11.5797931
time: 31.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 69.46 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 69.46
Output dim: 2, lower bound: -11.5292498, upper bound: 11.5791595
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 69.46
Output dim: 2, lower bound: -11.5509860, upper bound: 11.5797208
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 69.46
Output dim: 2, lower bound: -11.5445065, upper bound: 11.5445066
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 69.46
Output dim: 2, lower bound: -11.5445065, upper bound: 11.5445066
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 69.46
Output dim: 2, lower bound: -11.5581328, upper bound: 11.5792189
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 69.46
Output dim: 2, lower bound: -11.5797930, upper bound: 11.5797931

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.1992970, 18.9779339, -9.3538008, 19.0564995, -25.1092072, 25.1796036
1: -1.1126804, 22.7674866, -1.2150273, 22.8386574, -19.6753311, 19.7021408
2: -1.5504367, 20.8949776, -1.6275353, 20.9658566, -17.2386398, 17.2416267
3: -9.2860432, 16.4444351, -9.3506508, 16.5442429, -21.9917603, 21.9538727
4: -3.0445514, 22.2133083, -3.1472964, 22.2540283, -21.6488342, 21.7113647
5: -7.7411098, 20.5643311, -7.8324971, 20.6653156, -23.6919708, 23.7005959
6: -28.7849922, -1.4729576, -28.8303890, -1.3831682, -23.1389771, 23.1332588
7: -7.6011229, 21.5974693, -7.6998763, 21.6659241, -23.5133820, 23.5484390
8: -14.6457882, 14.6947460, -14.7801771, 14.8026676, -26.3898010, 26.4118805
9: -5.1094160, 21.2495613, -5.1881657, 21.3044796, -24.2266388, 24.2256775
10: -17.7702751, 17.5214195, -17.8724823, 17.5757351, -31.1795197, 31.2384796
11: -26.7228642, 3.5170321, -26.8242302, 3.5769014, -27.8421631, 27.8852463
12: -34.8538818, -2.4142871, -34.8894653, -2.3210597, -27.2021255, 27.1466179
13: -26.2299118, 15.6388474, -26.2863789, 15.8204584, -33.9536819, 33.8262863
14: -55.7759399, -17.6055508, -55.8998795, -17.5470428, -37.6713562, 37.6363144
15: -14.2518473, 15.4622507, -14.3532963, 15.5342245, -27.8083725, 27.8382568
16: -13.9739828, 20.7867985, -14.0681200, 20.8335876, -31.0003815, 31.0479279
17: -57.7404175, -14.4785767, -57.8467255, -14.3852606, -41.5621033, 41.5111847
18: -21.5156021, 12.1181822, -21.6819916, 12.1796560, -29.5165405, 29.6227264
19: -22.2160072, 3.4997017, -22.3582096, 3.5792813, -22.6981430, 22.7600517
20: -23.2249260, 1.2970872, -23.3740234, 1.3858485, -19.1253319, 19.1946526
21: -26.7293339, 2.3275814, -26.8941536, 2.4137232, -25.4277878, 25.5025063
22: -28.4397659, 3.2686157, -28.6134720, 3.3475745, -24.6465988, 24.7454948
23: -22.2445107, 5.6527243, -22.3696976, 5.7280703, -21.9741898, 22.0202599
24: -18.2401772, 9.3778086, -18.4274979, 9.4552402, -22.7607956, 22.8738480
25: -23.7814960, 5.3154240, -23.9252701, 5.4027090, -24.3539734, 24.4095650
26: -40.9775887, -0.5426726, -41.1298561, -0.4428339, -30.5438309, 30.6009064
27: -21.5291271, 8.5288315, -21.6838608, 8.5988903, -26.3847809, 26.4779587
28: -24.0672951, 5.9878254, -24.2091675, 6.0809078, -21.9193153, 21.9675598
29: -27.7970695, -0.2401938, -27.9445419, -0.1926367, -23.9068680, 24.0123329
30: -28.0693741, 3.6799889, -28.2288742, 3.7680631, -26.0696983, 26.1412354
31: -22.6023636, 4.9874854, -22.7551270, 5.0651312, -24.9908676, 25.0654144
32: -23.9023094, 2.2494400, -23.9595871, 2.3393908, -21.3487778, 21.3220215
33: -36.3379593, 3.4987226, -36.4518661, 3.6348104, -33.2673798, 33.2359543
34: -37.7939644, -4.8796577, -37.8896408, -4.7610373, -27.6893234, 27.6551819
35: -32.8502884, 0.1607823, -32.9404526, 0.2896662, -28.0984192, 28.0542831
36: -36.7753716, -0.7966213, -36.8618317, -0.6647449, -28.9657135, 28.9146881
37: -44.4866066, -1.8078928, -44.6052475, -1.7130671, -38.7113495, 38.7304001
38: -43.8632393, 2.7048259, -43.9900894, 2.8694353, -40.6026611, 40.5685196
39: -43.4912071, 2.8602185, -43.6302605, 3.0185165, -41.2725067, 41.2586136
40: -32.6818695, -0.0598006, -32.7769814, 0.0384924, -31.0055466, 31.0049057
41: -20.6879692, 7.2042351, -20.7418575, 7.2876663, -26.4343414, 26.4261322
42: -22.9714909, -0.2393467, -22.9890003, -0.1873631, -18.4614487, 18.4261551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=172, inp2_unstable=173, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1741

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5274331, upper bound: 11.5553538
time: 35.74 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5286038, upper bound: 11.5785170
time: 47.57 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.3297052, 19.0133190, -9.3958664, 19.0578766, -25.2059174, 25.2625122
1: -1.1997104, 22.8005848, -1.2441669, 22.8396568, -19.7317162, 19.7689323
2: -1.6108792, 20.9125500, -1.6473739, 20.9667225, -17.2894669, 17.2798729
3: -9.3432474, 16.4737053, -9.3703308, 16.5471287, -22.0520935, 22.0132828
4: -3.1298590, 22.2225285, -3.1747780, 22.2551327, -21.7313576, 21.7463646
5: -7.8265605, 20.6009216, -7.8612132, 20.6673698, -23.7654419, 23.7658081
6: -28.8228264, -1.3939285, -28.8337498, -1.3570809, -23.2073212, 23.2097435
7: -7.6847663, 21.6241837, -7.7273455, 21.6670723, -23.5882187, 23.6036606
8: -14.7733612, 14.7486897, -14.8233757, 14.8051834, -26.4992447, 26.5086441
9: -5.1846857, 21.2765522, -5.2131681, 21.3066978, -24.2874374, 24.2774048
10: -17.8678036, 17.5527611, -17.9047260, 17.5794315, -31.2796326, 31.3048019
11: -26.7465229, 3.5609245, -26.8295994, 3.5912170, -27.8789444, 27.9342651
12: -34.8697815, -2.3540382, -34.8941383, -2.3026552, -27.2377090, 27.2185211
13: -26.2465706, 15.6913462, -26.2905426, 15.8344507, -33.9860764, 33.8941879
14: -55.8681450, -17.5606136, -55.9288025, -17.5420570, -37.7676163, 37.8175583
15: -14.3586502, 15.5081854, -14.3877048, 15.5366278, -27.9028015, 27.9189148
16: -14.0315113, 20.8052635, -14.0868721, 20.8342514, -31.0473633, 31.0744629
17: -57.8110962, -14.4321041, -57.8693848, -14.3782043, -41.6473694, 41.6636276
18: -21.5665913, 12.1690674, -21.6878357, 12.1967020, -29.5931244, 29.6856003
19: -22.2655773, 3.5617335, -22.3632832, 3.6002645, -22.7695923, 22.8214035
20: -23.2781067, 1.3646441, -23.3763428, 1.4083412, -19.2036209, 19.2292900
21: -26.7823925, 2.3952122, -26.8998566, 2.4362960, -25.5031662, 25.5664368
22: -28.4853249, 3.3199310, -28.6166382, 3.3641977, -24.7100868, 24.7742691
23: -22.2758522, 5.7007108, -22.3723545, 5.7437229, -22.0200882, 22.0561066
24: -18.2928810, 9.4386902, -18.4298325, 9.4761791, -22.8344650, 22.9102707
25: -23.8109589, 5.3835073, -23.9279385, 5.4241657, -24.4005585, 24.4647522
26: -41.0199699, -0.4737649, -41.1324654, -0.4192386, -30.6138229, 30.6424332
27: -21.5614510, 8.5724421, -21.6877689, 8.6136627, -26.4414291, 26.5238953
28: -24.1056442, 6.0603280, -24.2112980, 6.1047268, -21.9819374, 22.0289536
29: -27.8297920, -0.2205749, -27.9495087, -0.1873012, -23.9525299, 24.0448608
30: -28.1062889, 3.7455871, -28.2308884, 3.7889075, -26.1159401, 26.1849518
31: -22.6563225, 5.0537744, -22.7604370, 5.0873861, -25.0686798, 25.1211662
32: -23.9425735, 2.3134608, -23.9624557, 2.3605015, -21.4147949, 21.3678932
33: -36.4297180, 3.6434731, -36.4560814, 3.6848083, -33.4092331, 33.3240738
34: -37.8585739, -4.7657466, -37.8916626, -4.7224088, -27.7958450, 27.7250519
35: -32.9225082, 0.2941685, -32.9434471, 0.3357067, -28.2158051, 28.1575775
36: -36.8408127, -0.6724095, -36.8648071, -0.6224098, -29.0725021, 29.0181961
37: -44.5610733, -1.7211599, -44.6119194, -1.6834393, -38.8204956, 38.8235016
38: -43.9579620, 2.8653126, -43.9959641, 2.9244761, -40.7496338, 40.6961670
39: -43.5992699, 3.0073113, -43.6369057, 3.0694017, -41.4298096, 41.3533096
40: -32.7396851, -0.0122933, -32.7835083, 0.0543532, -31.0842743, 31.0615730
41: -20.7291069, 7.2687788, -20.7461586, 7.3090391, -26.5342407, 26.4949722
42: -22.9883690, -0.2161031, -22.9939613, -0.1808617, -18.4879723, 18.4623451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=172, inp2_unstable=173, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1741

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5492303, upper bound: 11.5559611
time: 31.50 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5503477, upper bound: 11.5790874
time: 35.06 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.2682285, 19.0194340, -9.3681068, 19.0570030, -25.1717834, 25.2379074
1: -1.1625504, 22.8104057, -1.2262707, 22.8390980, -19.7143135, 19.7576904
2: -1.5917361, 20.9296379, -1.6375890, 20.9662914, -17.2690506, 17.2895164
3: -9.3168612, 16.4960442, -9.3585320, 16.5457516, -22.0202560, 22.0091324
4: -3.0965986, 22.2371082, -3.1594024, 22.2543945, -21.6915627, 21.7387009
5: -7.7819943, 20.6167545, -7.8427601, 20.6665878, -23.7241364, 23.7614174
6: -28.7974720, -1.4509439, -28.8316174, -1.3810835, -23.1660538, 23.1603737
7: -7.6497202, 21.6371155, -7.7108564, 21.6666222, -23.5534935, 23.5990143
8: -14.7017021, 14.7369261, -14.7946262, 14.8037891, -26.4335480, 26.4573441
9: -5.1381888, 21.2760696, -5.1943407, 21.3053246, -24.2564011, 24.2844963
10: -17.8036175, 17.5605316, -17.8786583, 17.5774593, -31.2136078, 31.2882996
11: -26.7879066, 3.5507069, -26.8252983, 3.5851631, -27.9176025, 27.9145737
12: -34.8805618, -2.3658028, -34.8953094, -2.3181343, -27.2299194, 27.2018547
13: -26.2841072, 15.7302876, -26.3001518, 15.8223000, -34.0022125, 33.9336700
14: -55.8421326, -17.5279484, -55.9155273, -17.5448742, -37.7318115, 37.7301636
15: -14.2788658, 15.4759598, -14.3572693, 15.5357122, -27.8391953, 27.8572922
16: -14.0238056, 20.8432369, -14.0772877, 20.8341522, -31.0422440, 31.1136398
17: -57.7998962, -14.3822269, -57.8574753, -14.3821144, -41.6232071, 41.6228943
18: -21.5669880, 12.1512966, -21.6844482, 12.1885176, -29.5800247, 29.6571159
19: -22.3004208, 3.5477974, -22.3601513, 3.5923495, -22.7962952, 22.8018036
20: -23.3056793, 1.3490639, -23.3748856, 1.3997569, -19.2246399, 19.2312965
21: -26.8202019, 2.3781781, -26.8959656, 2.4277620, -25.5342407, 25.5419273
22: -28.5463867, 3.3190439, -28.6148071, 3.3613532, -24.7724609, 24.7834015
23: -22.3223705, 5.7062273, -22.3705063, 5.7420788, -22.0694199, 22.0628090
24: -18.3413467, 9.4231796, -18.4287834, 9.4675951, -22.8763199, 22.9109840
25: -23.8494148, 5.3643541, -23.9264870, 5.4150963, -24.4382515, 24.4484329
26: -41.0806541, -0.4754615, -41.1310387, -0.4240003, -30.6696091, 30.6486053
27: -21.6278553, 8.5805864, -21.6854534, 8.6129532, -26.4997253, 26.5235214
28: -24.1514206, 6.0438623, -24.2099609, 6.0960860, -22.0226402, 22.0086021
29: -27.9003010, -0.2019836, -27.9457436, -0.1824173, -24.0239983, 24.0419769
30: -28.1478729, 3.7323964, -28.2296352, 3.7814136, -26.1640129, 26.1796303
31: -22.6671772, 5.0275297, -22.7571850, 5.0757008, -25.0691452, 25.1035423
32: -23.9240131, 2.2652504, -23.9610157, 2.3404312, -21.3802185, 21.3400230
33: -36.3824768, 3.5186005, -36.4542542, 3.6368079, -33.3123322, 33.2642975
34: -37.8362846, -4.8350668, -37.8905563, -4.7523212, -27.7457733, 27.6953735
35: -32.8922729, 0.1907353, -32.9422112, 0.2941279, -28.1462402, 28.0857925
36: -36.8292542, -0.7506814, -36.8635101, -0.6547294, -29.0317383, 28.9597015
37: -44.5535164, -1.7813778, -44.6087799, -1.7087355, -38.7842102, 38.7629166
38: -43.9178391, 2.7536564, -43.9925804, 2.8773370, -40.6832123, 40.6259308
39: -43.5320816, 2.8776550, -43.6346588, 3.0182610, -41.3278961, 41.2811279
40: -32.7193146, -0.0398109, -32.7810669, 0.0392480, -31.0470123, 31.0294266
41: -20.7286606, 7.2305837, -20.7440186, 7.2929673, -26.4993744, 26.4594040
42: -22.9798374, -0.2246087, -22.9895382, -0.1856294, -18.4724770, 18.4426651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=172, inp2_unstable=173, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1741

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5563451, upper bound: 11.5554252
time: 37.27 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5574892, upper bound: 11.5785817
time: 42.10 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.3985643, 19.0548172, -9.4102154, 19.0583572, -25.2684326, 25.3207855
1: -1.2495403, 22.8435135, -1.2554498, 22.8400612, -19.7706451, 19.8245239
2: -1.6521277, 20.9472313, -1.6574421, 20.9671516, -17.3198738, 17.3277512
3: -9.3741016, 16.5253162, -9.3781633, 16.5486183, -22.0806274, 22.0685425
4: -3.1819162, 22.2463303, -3.1868801, 22.2555008, -21.7740593, 21.7737236
5: -7.8674302, 20.6532593, -7.8714848, 20.6686668, -23.7975731, 23.8266144
6: -28.8352795, -1.3719273, -28.8349876, -1.3549948, -23.2344284, 23.2368660
7: -7.7333899, 21.6638927, -7.7383194, 21.6678009, -23.6283035, 23.6542549
8: -14.8292618, 14.7908592, -14.8378410, 14.8062859, -26.5430107, 26.5541153
9: -5.2134595, 21.3030624, -5.2193947, 21.3075466, -24.3171539, 24.3362350
10: -17.9012375, 17.5918827, -17.9109097, 17.5811501, -31.3137512, 31.3546371
11: -26.8115673, 3.5946383, -26.8307095, 3.5994644, -27.9543686, 27.9636002
12: -34.8964539, -2.3055520, -34.9000053, -2.2997074, -27.2655029, 27.2737083
13: -26.3007927, 15.7827415, -26.3043060, 15.8363113, -34.0346222, 34.0015182
14: -55.9343185, -17.4830894, -55.9444199, -17.5399208, -37.8280029, 37.9113846
15: -14.3856659, 15.5219126, -14.3916855, 15.5380850, -27.9336700, 27.9379807
16: -14.0814056, 20.8616924, -14.0960236, 20.8348427, -31.0892029, 31.1402130
17: -57.8704376, -14.3357716, -57.8801422, -14.3750486, -41.7084885, 41.7752838
18: -21.6179848, 12.2022390, -21.6903076, 12.2056179, -29.6566620, 29.7199707
19: -22.3499851, 3.6098735, -22.3652344, 3.6133032, -22.8677139, 22.8631363
20: -23.3588352, 1.4166789, -23.3772240, 1.4222672, -19.3029099, 19.2659264
21: -26.8733101, 2.4457624, -26.9016418, 2.4502847, -25.6096573, 25.6058769
22: -28.5920181, 3.3704038, -28.6179543, 3.3779986, -24.8360291, 24.8122101
23: -22.3536701, 5.7541947, -22.3731651, 5.7576890, -22.1153336, 22.0986557
24: -18.3940506, 9.4840889, -18.4311504, 9.4886055, -22.9499893, 22.9474220
25: -23.8788662, 5.4324522, -23.9291458, 5.4365492, -24.4848061, 24.5036697
26: -41.1230545, -0.4064698, -41.1336136, -0.4004283, -30.7396164, 30.6901550
27: -21.6602249, 8.6241856, -21.6894131, 8.6276932, -26.5564346, 26.5694580
28: -24.1897488, 6.1164217, -24.2121010, 6.1199074, -22.0852737, 22.0699768
29: -27.9329872, -0.1823647, -27.9507027, -0.1770855, -24.0697174, 24.0744858
30: -28.1848507, 3.7980232, -28.2316647, 3.8022661, -26.2103500, 26.2233467
31: -22.7211304, 5.0937967, -22.7624626, 5.0979605, -25.1469955, 25.1593170
32: -23.9643059, 2.3292954, -23.9638901, 2.3615963, -21.4462357, 21.3858719
33: -36.4742546, 3.6632547, -36.4584579, 3.6867628, -33.4541245, 33.3524246
34: -37.9008865, -4.7211404, -37.8925934, -4.7136459, -27.8522568, 27.7652435
35: -32.9644585, 0.3241286, -32.9452591, 0.3401704, -28.2636032, 28.1890564
36: -36.8947258, -0.6265039, -36.8664818, -0.6123333, -29.1385269, 29.0631409
37: -44.6280365, -1.6946058, -44.6153641, -1.6791058, -38.8933411, 38.8560104
38: -44.0125885, 2.9140706, -43.9984665, 2.9323645, -40.8301544, 40.7534637
39: -43.6401443, 3.0247517, -43.6413040, 3.0692234, -41.4852600, 41.3757858
40: -32.7771530, 0.0077512, -32.7875786, 0.0551105, -31.1257935, 31.0860977
41: -20.7697792, 7.2951670, -20.7483273, 7.3143792, -26.5993118, 26.5281982
42: -22.9967422, -0.2013397, -22.9945011, -0.1791646, -18.4990082, 18.4788857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=172, inp2_unstable=173, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1741

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5780694, upper bound: 11.5560426
time: 36.40 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5791628, upper bound: 11.5791627
time: 32.98 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 72.04 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 72.04
Output dim: 2, lower bound: -11.5274331, upper bound: 11.5553538
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 72.04
Output dim: 2, lower bound: -11.5286038, upper bound: 11.5785170
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 72.04
Output dim: 2, lower bound: -11.5492303, upper bound: 11.5559611
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 72.04
Output dim: 2, lower bound: -11.5503477, upper bound: 11.5790874
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 72.04
Output dim: 2, lower bound: -11.5563451, upper bound: 11.5554252
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 72.04
Output dim: 2, lower bound: -11.5574892, upper bound: 11.5785817
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 72.04
Output dim: 2, lower bound: -11.5780694, upper bound: 11.5560426
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 72.04
Output dim: 2, lower bound: -11.5791628, upper bound: 11.5791627

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.1978807, 18.9779072, -9.3591661, 19.0661449, -25.1202736, 25.1832047
1: -1.1117635, 22.7674465, -1.2154999, 22.8607960, -19.6958733, 19.6966324
2: -1.5496733, 20.8949127, -1.6279488, 20.9849339, -17.2589531, 17.2342758
3: -9.2851601, 16.4442902, -9.3499365, 16.5755959, -22.0116119, 21.9427490
4: -3.0433393, 22.2132607, -3.1498203, 22.2728081, -21.6631165, 21.7056732
5: -7.7402759, 20.5642071, -7.8330374, 20.6984024, -23.7166977, 23.6914406
6: -28.7842712, -1.4733682, -28.8295135, -1.3643460, -23.1457062, 23.1220589
7: -7.6000428, 21.5973740, -7.6987243, 21.7034435, -23.5494423, 23.5364876
8: -14.6445312, 14.6946259, -14.7814989, 14.8248386, -26.3986130, 26.4052277
9: -5.1074185, 21.2495308, -5.1907110, 21.3259583, -24.2526550, 24.2256775
10: -17.7677002, 17.5212955, -17.8720818, 17.5885944, -31.1959381, 31.2342529
11: -26.7226295, 3.5156193, -26.8289146, 3.5774221, -27.8427887, 27.8921356
12: -34.8535347, -2.4155574, -34.8903770, -2.3174686, -27.2131195, 27.1378937
13: -26.2283363, 15.6388016, -26.2864552, 15.8661098, -34.0013123, 33.8182678
14: -55.7756119, -17.6063385, -55.9084549, -17.5455933, -37.6875916, 37.6166382
15: -14.2513428, 15.4615374, -14.3646889, 15.5336447, -27.8043365, 27.8487396
16: -13.9726496, 20.7867241, -14.0706787, 20.8722725, -31.0380249, 31.0420532
17: -57.7401543, -14.4808464, -57.8470154, -14.3826427, -41.5987549, 41.4777985
18: -21.5154400, 12.1168938, -21.7313004, 12.1795139, -29.5057449, 29.6744080
19: -22.2158260, 3.4990048, -22.3911495, 3.5789461, -22.6883087, 22.7920685
20: -23.2248363, 1.2965436, -23.4044361, 1.3869262, -19.1184311, 19.2270660
21: -26.7291985, 2.3269196, -26.9227028, 2.4145505, -25.4203186, 25.5263519
22: -28.4396667, 3.2677743, -28.6677628, 3.3467834, -24.6304359, 24.7992783
23: -22.2443619, 5.6518502, -22.4117432, 5.7290897, -21.9616051, 22.0633163
24: -18.2400513, 9.3771124, -18.4737015, 9.4551315, -22.7492599, 22.9210663
25: -23.7813206, 5.3145380, -23.9696350, 5.4033298, -24.3420334, 24.4540405
26: -40.9774094, -0.5437207, -41.1918945, -0.4442158, -30.5210953, 30.6667633
27: -21.5290203, 8.5280209, -21.7239857, 8.6003628, -26.3786697, 26.5187531
28: -24.0671844, 5.9868498, -24.2581234, 6.0802784, -21.9019928, 22.0174561
29: -27.7968807, -0.2412293, -27.9923592, -0.1930974, -23.8939667, 24.0584755
30: -28.0692558, 3.6792614, -28.2549553, 3.7715518, -26.0666847, 26.1647186
31: -22.6021919, 4.9866910, -22.7887383, 5.0662508, -24.9869461, 25.1000595
32: -23.9016151, 2.2490759, -23.9620323, 2.3567479, -21.3624191, 21.3172531
33: -36.3376541, 3.4986196, -36.4600639, 3.6394176, -33.2731476, 33.2393951
34: -37.7938995, -4.8819947, -37.8967285, -4.7646060, -27.6850662, 27.6625519
35: -32.8501358, 0.1582346, -32.9486237, 0.2848148, -28.0929794, 28.0613174
36: -36.7752800, -0.7991815, -36.8679237, -0.6699743, -28.9605560, 28.9231720
37: -44.4860992, -1.8101392, -44.6150131, -1.7174249, -38.7152100, 38.7278519
38: -43.8630600, 2.7018843, -44.0028687, 2.8669353, -40.6033630, 40.5915527
39: -43.4894333, 2.8600969, -43.6360779, 3.0368891, -41.2859802, 41.2616577
40: -32.6806717, -0.0600309, -32.7868233, 0.0641646, -31.0302353, 31.0106201
41: -20.6871033, 7.2039957, -20.7421074, 7.2927761, -26.4349060, 26.4245453
42: -22.9709320, -0.2395318, -22.9882545, -0.1798780, -18.4691124, 18.4215164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=172, inp2_unstable=172, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5648322
time: 29.28 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5760283
time: 34.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.3283262, 19.0133018, -9.4012671, 19.0675087, -25.2169533, 25.2661247
1: -1.1987643, 22.8005371, -1.2446752, 22.8617859, -19.7522278, 19.7634621
2: -1.6100950, 20.9124851, -1.6478031, 20.9858360, -17.3097801, 17.2725143
3: -9.3423882, 16.4735775, -9.3696280, 16.5785065, -22.0719604, 22.0021706
4: -3.1286573, 22.2224808, -3.1773248, 22.2739105, -21.7456131, 21.7406921
5: -7.8257279, 20.6007690, -7.8617477, 20.7004852, -23.7901421, 23.7566452
6: -28.8221359, -1.3943300, -28.8328629, -1.3382945, -23.2140045, 23.1985283
7: -7.6836534, 21.6240997, -7.7261848, 21.7046108, -23.6242905, 23.5917435
8: -14.7721071, 14.7485895, -14.8246498, 14.8273602, -26.5080338, 26.5020370
9: -5.1826663, 21.2765369, -5.2157850, 21.3281708, -24.3134308, 24.2774162
10: -17.8652649, 17.5526600, -17.9043293, 17.5922642, -31.2960129, 31.3006210
11: -26.7463379, 3.5595121, -26.8343544, 3.5917668, -27.8795624, 27.9411926
12: -34.8694496, -2.3552904, -34.8950462, -2.2990999, -27.2486343, 27.2097778
13: -26.2450123, 15.6912556, -26.2906456, 15.8800611, -34.0336914, 33.8861923
14: -55.8678055, -17.5614281, -55.9373245, -17.5405483, -37.7838745, 37.7978973
15: -14.3581314, 15.5074902, -14.3990822, 15.5360365, -27.8987808, 27.9293823
16: -14.0301666, 20.8052406, -14.0894146, 20.8729744, -31.0849609, 31.0685959
17: -57.8107986, -14.4343472, -57.8696938, -14.3755016, -41.6840820, 41.6301804
18: -21.5664139, 12.1677599, -21.7371483, 12.1965723, -29.5823746, 29.7372894
19: -22.2654133, 3.5610704, -22.3962345, 3.5998676, -22.7597580, 22.8534317
20: -23.2779961, 1.3641305, -23.4067535, 1.4094746, -19.1967468, 19.2616882
21: -26.7822685, 2.3945522, -26.9283943, 2.4370852, -25.4956894, 25.5902672
22: -28.4852352, 3.3191075, -28.6708927, 3.3633945, -24.6939812, 24.8280220
23: -22.2757034, 5.6998305, -22.4144096, 5.7447224, -22.0075111, 22.0991478
24: -18.2927475, 9.4380083, -18.4760818, 9.4761276, -22.8229294, 22.9575195
25: -23.8108025, 5.3826051, -23.9722786, 5.4247880, -24.3886185, 24.5092316
26: -41.0198517, -0.4747481, -41.1944733, -0.4206314, -30.5911102, 30.7082825
27: -21.5613518, 8.5716257, -21.7279396, 8.6151695, -26.4353333, 26.5646515
28: -24.1055679, 6.0593834, -24.2602558, 6.1041579, -21.9645996, 22.0788612
29: -27.8296547, -0.2216334, -27.9973412, -0.1877502, -23.9396286, 24.0909996
30: -28.1061764, 3.7448707, -28.2569466, 3.7924213, -26.1129150, 26.2084351
31: -22.6561203, 5.0529361, -22.7940063, 5.0884762, -25.0647812, 25.1557999
32: -23.9418926, 2.3131332, -23.9649296, 2.3778615, -21.4284172, 21.3631325
33: -36.4294167, 3.6432929, -36.4643097, 3.6893973, -33.4149780, 33.3275375
34: -37.8584976, -4.7681413, -37.8987465, -4.7259560, -27.7915802, 27.7324600
35: -32.9223785, 0.2916517, -32.9516602, 0.3308454, -28.2103577, 28.1646118
36: -36.8406906, -0.6749988, -36.8708954, -0.6275887, -29.0673370, 29.0266418
37: -44.5606194, -1.7234073, -44.6216316, -1.6877494, -38.8243103, 38.8209381
38: -43.9577789, 2.8623867, -44.0087624, 2.9219241, -40.7503510, 40.7191467
39: -43.5975266, 3.0072193, -43.6427460, 3.0878406, -41.4433746, 41.3563385
40: -32.7385178, -0.0125399, -32.7933464, 0.0800102, -31.1089630, 31.0672836
41: -20.7282581, 7.2685628, -20.7464161, 7.3141928, -26.5348053, 26.4933395
42: -22.9878197, -0.2163038, -22.9931927, -0.1734016, -18.4956245, 18.4577103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=172, inp2_unstable=172, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5653754
time: 37.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5766147
time: 34.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2668533, 19.0194149, -9.3735256, 19.0666313, -25.1828613, 25.2415199
1: -1.1615920, 22.8103580, -1.2267718, 22.8612213, -19.7348175, 19.7521744
2: -1.5909455, 20.9295635, -1.6380053, 20.9853992, -17.2893448, 17.2821503
3: -9.3159828, 16.4959221, -9.3578415, 16.5770779, -22.0401154, 21.9980011
4: -3.0953984, 22.2370777, -3.1619663, 22.2731781, -21.7058372, 21.7330132
5: -7.7811184, 20.6165981, -7.8433180, 20.6996994, -23.7488327, 23.7522583
6: -28.7967567, -1.4513168, -28.8307457, -1.3622937, -23.1727982, 23.1492004
7: -7.6486497, 21.6370811, -7.7097154, 21.7041664, -23.5895348, 23.5870895
8: -14.7004738, 14.7368107, -14.7959538, 14.8259602, -26.4423141, 26.4506836
9: -5.1361976, 21.2760773, -5.1969166, 21.3268318, -24.2823563, 24.2844925
10: -17.8010864, 17.5604534, -17.8782597, 17.5902672, -31.2299957, 31.2840805
11: -26.7877216, 3.5492992, -26.8299770, 3.5856843, -27.9182129, 27.9215088
12: -34.8802414, -2.3670168, -34.8962555, -2.3145022, -27.2409286, 27.1931534
13: -26.2825279, 15.7301941, -26.3002434, 15.8679762, -34.0498352, 33.9256134
14: -55.8417664, -17.5287647, -55.9240341, -17.5434017, -37.7480469, 37.7104492
15: -14.2783766, 15.4752512, -14.3686743, 15.5351048, -27.8351746, 27.8677521
16: -14.0224199, 20.8432102, -14.0798721, 20.8728523, -31.0798492, 31.1077881
17: -57.7995377, -14.3844681, -57.8577423, -14.3795128, -41.6598969, 41.5894623
18: -21.5668106, 12.1500196, -21.7337646, 12.1883831, -29.5692673, 29.7088242
19: -22.3002434, 3.5471530, -22.3930740, 3.5919416, -22.7864685, 22.8338699
20: -23.3056412, 1.3485270, -23.4053097, 1.4008524, -19.2177505, 19.2637024
21: -26.8200741, 2.3774920, -26.9245148, 2.4285812, -25.5267334, 25.5657654
22: -28.5462418, 3.3182635, -28.6690884, 3.3605645, -24.7563324, 24.8371506
23: -22.3222046, 5.7054029, -22.4125443, 5.7430530, -22.0568352, 22.1058617
24: -18.3411751, 9.4224796, -18.4750252, 9.4675550, -22.8647995, 22.9582214
25: -23.8492107, 5.3634806, -23.9708061, 5.4157181, -24.4263153, 24.4929352
26: -41.0805664, -0.4765277, -41.1930923, -0.4253840, -30.6468582, 30.7144394
27: -21.6277771, 8.5797348, -21.7256355, 8.6144638, -26.4936523, 26.5643158
28: -24.1513081, 6.0429168, -24.2588959, 6.0954704, -22.0053406, 22.0584679
29: -27.9001083, -0.2030059, -27.9936218, -0.1829084, -24.0110970, 24.0880814
30: -28.1477394, 3.7316890, -28.2556763, 3.7849121, -26.1609840, 26.2031097
31: -22.6669636, 5.0267248, -22.7907143, 5.0768228, -25.0652466, 25.1381798
32: -23.9233055, 2.2649095, -23.9634762, 2.3577902, -21.3938637, 21.3352661
33: -36.3822136, 3.5183892, -36.4624863, 3.6413708, -33.3180847, 33.2677612
34: -37.8362274, -4.8374071, -37.8976440, -4.7558312, -27.7415466, 27.7027512
35: -32.8921280, 0.1882172, -32.9503632, 0.2892623, -28.1407928, 28.0928268
36: -36.8292236, -0.7532730, -36.8696136, -0.6599154, -29.0265732, 28.9682007
37: -44.5530853, -1.7836413, -44.6184998, -1.7130795, -38.7880402, 38.7603531
38: -43.9176865, 2.7506776, -44.0053787, 2.8748059, -40.6838989, 40.6489868
39: -43.5302582, 2.8774838, -43.6405563, 3.0366507, -41.3414307, 41.2841339
40: -32.7181168, -0.0400264, -32.7908936, 0.0649593, -31.0717087, 31.0351524
41: -20.7277927, 7.2303810, -20.7443275, 7.2981234, -26.4999619, 26.4578552
42: -22.9792557, -0.2248251, -22.9887581, -0.1781831, -18.4801254, 18.4380188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=172, inp2_unstable=172, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5549674, upper bound: 11.5648912
time: 32.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5549674, upper bound: 11.5760878
time: 39.14 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.3810139, 19.0544052, -9.3546524, 19.0569534, -25.2477798, 25.2589149
1: -1.2358856, 22.8432484, -1.2127271, 22.8392315, -19.7557869, 19.7808914
2: -1.6407223, 20.9469070, -1.6216216, 20.9661942, -17.3071671, 17.2902641
3: -9.3596983, 16.5243301, -9.3323002, 16.5455246, -22.0628357, 22.0226746
4: -3.1633410, 22.2462254, -3.1288691, 22.2550755, -21.7551498, 21.7172089
5: -7.8538589, 20.6519585, -7.8288178, 20.6645927, -23.7787132, 23.7850990
6: -28.8286896, -1.3761306, -28.8145905, -1.3684630, -23.2059784, 23.2087517
7: -7.7156954, 21.6633091, -7.6835761, 21.6659088, -23.6090698, 23.5999451
8: -14.8104362, 14.7902346, -14.7786541, 14.8042221, -26.5219803, 26.4947662
9: -5.2010875, 21.3027210, -5.1800671, 21.3064346, -24.3020782, 24.2916183
10: -17.8918495, 17.5908070, -17.8817768, 17.5777359, -31.3004150, 31.3233643
11: -26.8110886, 3.5892000, -26.8292809, 3.5823436, -27.9330978, 27.9549332
12: -34.8934174, -2.3116531, -34.8902512, -2.3191352, -27.2428589, 27.2506981
13: -26.2850456, 15.7815304, -26.2540855, 15.8324165, -34.0148010, 33.9483490
14: -55.9315033, -17.4868317, -55.9352760, -17.5512486, -37.8029327, 37.8737717
15: -14.3801155, 15.5181160, -14.3738461, 15.5259171, -27.9157028, 27.9158020
16: -14.0706396, 20.8614311, -14.0622396, 20.8340244, -31.0771103, 31.1056976
17: -57.8692017, -14.3407202, -57.8760529, -14.3899832, -41.6888580, 41.7395325
18: -21.6166878, 12.1891260, -21.6861858, 12.1645575, -29.6113892, 29.7022095
19: -22.3488998, 3.5976641, -22.3618393, 3.5749514, -22.8289413, 22.8474388
20: -23.3583298, 1.4073064, -23.3755684, 1.3931189, -19.2701035, 19.2540855
21: -26.8725281, 2.4346271, -26.8992710, 2.4152942, -25.5731277, 25.5913620
22: -28.5912666, 3.3553627, -28.6155910, 3.3312042, -24.7876587, 24.7941589
23: -22.3530540, 5.7393904, -22.3714409, 5.7119617, -22.0678558, 22.0820694
24: -18.3928223, 9.4708176, -18.4272842, 9.4471407, -22.9047394, 22.9295654
25: -23.8777466, 5.4169178, -23.9254398, 5.3874884, -24.4333000, 24.4833450
26: -41.1220207, -0.4248724, -41.1302338, -0.4584041, -30.6780167, 30.6672745
27: -21.6592579, 8.6103945, -21.6863708, 8.5843019, -26.5107574, 26.5531769
28: -24.1891518, 6.0987144, -24.2101307, 6.0644450, -22.0287552, 22.0504150
29: -27.9324780, -0.1964471, -27.9491329, -0.2219617, -24.0239487, 24.0582008
30: -28.1843796, 3.7878175, -28.2301579, 3.7696424, -26.1789474, 26.2105560
31: -22.7196255, 5.0804090, -22.7576714, 5.0558891, -25.1026077, 25.1410561
32: -23.9574203, 2.3259339, -23.9419594, 2.3508739, -21.4266129, 21.3593369
33: -36.4711456, 3.6619287, -36.4486580, 3.6825457, -33.4442139, 33.3366470
34: -37.9001541, -4.7242713, -37.8903427, -4.7234578, -27.8412323, 27.7594833
35: -32.9628220, 0.3200274, -32.9400482, 0.3270922, -28.2477570, 28.1789932
36: -36.8934937, -0.6314082, -36.8625717, -0.6270409, -29.1183395, 29.0541534
37: -44.6243515, -1.6966500, -44.6038589, -1.6855946, -38.8776550, 38.8354797
38: -44.0102539, 2.9070263, -43.9910164, 2.9113636, -40.8005066, 40.7388687
39: -43.6302490, 3.0244446, -43.6099205, 3.0683289, -41.4748535, 41.3456039
40: -32.7686005, 0.0061011, -32.7606506, 0.0498941, -31.1126251, 31.0588074
41: -20.7663383, 7.2929745, -20.7374516, 7.3074560, -26.5858765, 26.5143127
42: -22.9937725, -0.2037096, -22.9852409, -0.1866682, -18.4882202, 18.4661026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=172, inp2_unstable=172, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5755830, upper bound: 11.5423393
time: 36.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5755830, upper bound: 11.5535544
time: 30.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.3972120, 19.0547905, -9.4155807, 19.0680084, -25.2795105, 25.3244171
1: -1.2486072, 22.8434582, -1.2559319, 22.8622246, -19.7911758, 19.8190079
2: -1.6513956, 20.9471664, -1.6578414, 20.9862595, -17.3401794, 17.3203926
3: -9.3732166, 16.5251846, -9.3774710, 16.5799732, -22.1004868, 22.0574112
4: -3.1807532, 22.2462788, -3.1894326, 22.2742577, -21.7883415, 21.7680321
5: -7.8665652, 20.6531506, -7.8720427, 20.7017765, -23.8222923, 23.8174591
6: -28.8345909, -1.3723097, -28.8340893, -1.3362350, -23.2410736, 23.2256508
7: -7.7322893, 21.6638222, -7.7371798, 21.7053242, -23.6643639, 23.6423264
8: -14.8279991, 14.7907534, -14.8391142, 14.8284578, -26.5517578, 26.5475006
9: -5.2114735, 21.3030396, -5.2219515, 21.3290176, -24.3431854, 24.3362274
10: -17.8987083, 17.5917912, -17.9104862, 17.5939598, -31.3301086, 31.3504486
11: -26.8113995, 3.5932713, -26.8353977, 3.5999928, -27.9549866, 27.9705887
12: -34.8961220, -2.3067436, -34.9009476, -2.2961798, -27.2764435, 27.2650032
13: -26.2992477, 15.7826195, -26.3044357, 15.8819132, -34.0822449, 33.9935226
14: -55.9340363, -17.4838600, -55.9528961, -17.5383797, -37.8442078, 37.8917084
15: -14.3851881, 15.5212145, -14.4030304, 15.5374928, -27.9296112, 27.9484406
16: -14.0800505, 20.8616657, -14.0985804, 20.8735046, -31.1267853, 31.1343765
17: -57.8701859, -14.3380356, -57.8804474, -14.3724031, -41.7451401, 41.7418518
18: -21.6177921, 12.2009029, -21.7396545, 12.2054405, -29.6458817, 29.7717056
19: -22.3497944, 3.6092153, -22.3981533, 3.6129010, -22.8578644, 22.8951645
20: -23.3587666, 1.4161115, -23.4076424, 1.4233699, -19.2960587, 19.2983284
21: -26.8731422, 2.4451008, -26.9301872, 2.4511187, -25.6021652, 25.6296463
22: -28.5919228, 3.3696084, -28.6722221, 3.3771923, -24.8199081, 24.8659286
23: -22.3535347, 5.7533288, -22.4152355, 5.7586856, -22.1027222, 22.1417007
24: -18.3938828, 9.4833441, -18.4773674, 9.4885283, -22.9384842, 22.9946556
25: -23.8787155, 5.4315672, -23.9734631, 5.4371557, -24.4729118, 24.5481300
26: -41.1229172, -0.4074745, -41.1956635, -0.4017711, -30.7168579, 30.7559967
27: -21.6601257, 8.6233206, -21.7295990, 8.6292171, -26.5503235, 26.6101990
28: -24.1896591, 6.1154461, -24.2610626, 6.1193061, -22.0679169, 22.1198654
29: -27.9328079, -0.1834160, -27.9985390, -0.1775206, -24.0567970, 24.1206207
30: -28.1847038, 3.7973192, -28.2577057, 3.8057637, -26.2072830, 26.2468414
31: -22.7209244, 5.0929871, -22.7960262, 5.0990524, -25.1430817, 25.1938934
32: -23.9636116, 2.3289452, -23.9663696, 2.3789315, -21.4598541, 21.3810883
33: -36.4739685, 3.6630106, -36.4667320, 3.6913795, -33.4598694, 33.3558578
34: -37.9007950, -4.7235422, -37.8996429, -4.7172217, -27.8480072, 27.7726364
35: -32.9643135, 0.3216243, -32.9533615, 0.3352866, -28.2581482, 28.1961288
36: -36.8946190, -0.6290956, -36.8725319, -0.6175356, -29.1333694, 29.0716324
37: -44.6276093, -1.6968708, -44.6251526, -1.6834040, -38.8971405, 38.8534317
38: -44.0123901, 2.9111223, -44.0112648, 2.9298301, -40.8308411, 40.7765350
39: -43.6383400, 3.0246243, -43.6471405, 3.0875826, -41.4987946, 41.3787460
40: -32.7759552, 0.0075023, -32.7973938, 0.0807858, -31.1504974, 31.0918236
41: -20.7689133, 7.2949491, -20.7486115, 7.3195152, -26.5998459, 26.5266724
42: -22.9961891, -0.2015789, -22.9937248, -0.1717143, -18.5066681, 18.4742203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=172, inp2_unstable=172, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5766894, upper bound: 11.5654349
time: 41.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5766894, upper bound: 11.5766893
time: 33.20 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 77.47 seconds
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 77.47
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5648322
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 77.47
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5760283
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 77.47
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5653754
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 77.47
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5766147
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 77.47
Output dim: 2, lower bound: -11.5549674, upper bound: 11.5648912
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 77.47
Output dim: 2, lower bound: -11.5549674, upper bound: 11.5760878
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 77.47
Output dim: 2, lower bound: -11.5755830, upper bound: 11.5423393
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 77.47
Output dim: 2, lower bound: -11.5755830, upper bound: 11.5535544
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 77.47
Output dim: 2, lower bound: -11.5766894, upper bound: 11.5654349
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 77.47
Output dim: 2, lower bound: -11.5766894, upper bound: 11.5766893

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.2260561, 18.9990883, -9.3554668, 19.0660210, -25.1529350, 25.2072372
1: -1.1262407, 22.7948112, -1.2109780, 22.8603268, -19.7118988, 19.7321472
2: -1.5923331, 20.8938980, -1.6274536, 20.9833431, -17.2591972, 17.2738342
3: -9.3031006, 16.4439812, -9.3495064, 16.5718517, -22.0116043, 21.9563217
4: -3.0777869, 22.2148514, -3.1490002, 22.2719212, -21.6845016, 21.7445679
5: -7.7765255, 20.5641212, -7.8324656, 20.6954765, -23.7292976, 23.6957703
6: -28.7875404, -1.4028072, -28.8273354, -1.3648558, -23.1441956, 23.1956635
7: -7.6324840, 21.6043167, -7.6981063, 21.7028866, -23.5698929, 23.5550880
8: -14.7174110, 14.7034101, -14.7812490, 14.8230886, -26.4525833, 26.4497147
9: -5.1807899, 21.2560844, -5.1887579, 21.3234482, -24.3254395, 24.2264786
10: -17.9748249, 17.5249653, -17.8702869, 17.5830021, -31.3973694, 31.2179642
11: -26.7895412, 3.5160389, -26.8279057, 3.5724258, -27.9138794, 27.8884964
12: -34.8724442, -2.3969622, -34.8897133, -2.3185735, -27.2547531, 27.1349106
13: -26.2437897, 15.6611462, -26.2846565, 15.8656950, -34.0224838, 33.8378983
14: -55.9260750, -17.6052170, -55.9073715, -17.5502815, -37.8360596, 37.5928040
15: -14.2955666, 15.4658079, -14.3637486, 15.5326109, -27.8628464, 27.8481293
16: -14.0490608, 20.7888927, -14.0679474, 20.8691998, -31.1055832, 31.0364532
17: -57.8383904, -14.4704418, -57.8463516, -14.3860149, -41.7071228, 41.4845200
18: -21.5498810, 12.1413784, -21.7305527, 12.1774426, -29.5239868, 29.7358017
19: -22.2400856, 3.5027642, -22.3898926, 3.5772245, -22.7209702, 22.7905731
20: -23.2412415, 1.2934313, -23.4038811, 1.3828053, -19.1365585, 19.2209778
21: -26.7704391, 2.3305264, -26.9214649, 2.4113145, -25.4714737, 25.5278168
22: -28.4486160, 3.2960870, -28.6662750, 3.3456402, -24.6955795, 24.7773972
23: -22.2691956, 5.6495318, -22.4109631, 5.7239323, -21.9821358, 22.0660629
24: -18.2476082, 9.3827438, -18.4722996, 9.4545507, -22.7529373, 22.9265060
25: -23.7969589, 5.3299580, -23.9685783, 5.4015799, -24.3895721, 24.4504318
26: -41.0156212, -0.5456729, -41.1914024, -0.4491916, -30.5590210, 30.6658173
27: -21.5366917, 8.5941696, -21.7220039, 8.5990114, -26.3748550, 26.5882874
28: -24.0711727, 6.0198078, -24.2561111, 6.0797434, -21.9096565, 22.0339890
29: -27.8134232, -0.2106369, -27.9909058, -0.1936182, -23.9746704, 24.0184822
30: -28.0827217, 3.6927047, -28.2539711, 3.7687900, -26.0972023, 26.1715889
31: -22.6476803, 4.9872942, -22.7874527, 5.0624475, -25.0346146, 25.1003456
32: -23.9032879, 2.2914412, -23.9601860, 2.3562808, -21.3559456, 21.3598671
33: -36.3435516, 3.5965905, -36.4566002, 3.6389332, -33.2700653, 33.3454895
34: -37.7936745, -4.7911506, -37.8942642, -4.7661505, -27.6754074, 27.7624893
35: -32.8527222, 0.2740159, -32.9456863, 0.2841034, -28.0839844, 28.1767654
36: -36.7765656, -0.6728992, -36.8645325, -0.6706476, -28.9436798, 29.0472031
37: -44.4931145, -1.6936440, -44.6096725, -1.7183070, -38.7064972, 38.8527222
38: -43.8757820, 2.8574018, -43.9989471, 2.8652434, -40.6016541, 40.7434845
39: -43.4938164, 2.9237580, -43.6301575, 3.0363088, -41.2936096, 41.3327026
40: -32.6908035, 0.0251305, -32.7837448, 0.0637674, -31.0415268, 31.0972557
41: -20.6879711, 7.2944126, -20.7388439, 7.2924261, -26.4239273, 26.5143967
42: -22.9695625, -0.2215343, -22.9874916, -0.1805618, -18.4906425, 18.4392052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=172, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5600329
time: 39.21 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5760283
time: 33.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.3564415, 19.0344620, -9.3975725, 19.0673923, -25.2496605, 25.2901611
1: -1.2132435, 22.8279037, -1.2401390, 22.8613167, -19.7682648, 19.7989731
2: -1.6527531, 20.9115105, -1.6473210, 20.9842052, -17.3100777, 17.3120842
3: -9.3603306, 16.4732513, -9.3691206, 16.5747356, -22.0719528, 22.0157585
4: -3.1631694, 22.2240887, -3.1764808, 22.2730255, -21.7670555, 21.7795639
5: -7.8619909, 20.6006393, -7.8611956, 20.6975536, -23.8027649, 23.7609634
6: -28.8253880, -1.3237319, -28.8307076, -1.3387823, -23.2124939, 23.2721672
7: -7.7161026, 21.6310215, -7.7255626, 21.7040672, -23.6447487, 23.6103058
8: -14.8449726, 14.7573795, -14.8244057, 14.8255854, -26.5620575, 26.5464935
9: -5.2560577, 21.2829933, -5.2138009, 21.3256397, -24.3861847, 24.2781525
10: -18.0723991, 17.5562782, -17.9025097, 17.5866928, -31.4974136, 31.2842636
11: -26.8132019, 3.5599532, -26.8333473, 3.5867372, -27.9506073, 27.9375381
12: -34.8883324, -2.3366141, -34.8943558, -2.3002110, -27.2902832, 27.2068520
13: -26.2604485, 15.7136898, -26.2888470, 15.8796577, -34.0548706, 33.9058304
14: -56.0183067, -17.5603008, -55.9362602, -17.5452747, -37.9321823, 37.7740097
15: -14.4023991, 15.5117626, -14.3981438, 15.5350323, -27.9573975, 27.9287567
16: -14.1065836, 20.8073311, -14.0866232, 20.8698196, -31.1524658, 31.0629807
17: -57.9090805, -14.4239922, -57.8690186, -14.3789635, -41.7924347, 41.6369019
18: -21.6008530, 12.1922693, -21.7364292, 12.1945333, -29.6006165, 29.7986908
19: -22.2896404, 3.5648327, -22.3949642, 3.5981739, -22.7924194, 22.8519135
20: -23.2944183, 1.3610020, -23.4061623, 1.4053218, -19.2148438, 19.2556076
21: -26.8234062, 2.3981364, -26.9271278, 2.4338639, -25.5467529, 25.5917244
22: -28.4941998, 3.3474307, -28.6694260, 3.3623233, -24.7589874, 24.8061752
23: -22.3005390, 5.6974974, -22.4136410, 5.7396169, -22.0280457, 22.1019058
24: -18.3002968, 9.4436569, -18.4746513, 9.4755297, -22.8265839, 22.9629478
25: -23.8263969, 5.3981185, -23.9712334, 5.4230461, -24.4361038, 24.5057220
26: -41.0580597, -0.4766784, -41.1940079, -0.4256454, -30.6289978, 30.7073441
27: -21.5689907, 8.6377802, -21.7259178, 8.6137867, -26.4313812, 26.6342392
28: -24.1095276, 6.0923781, -24.2582302, 6.1036072, -21.9722595, 22.0953903
29: -27.8461609, -0.1909823, -27.9958611, -0.1882414, -24.0203781, 24.0510178
30: -28.1196060, 3.7583337, -28.2559929, 3.7896564, -26.1433945, 26.2152863
31: -22.7016506, 5.0535851, -22.7927246, 5.0846720, -25.1124496, 25.1561127
32: -23.9435616, 2.3555171, -23.9630890, 2.3774018, -21.4218559, 21.4057426
33: -36.4353447, 3.7412400, -36.4607849, 3.6889091, -33.4118347, 33.4336548
34: -37.8582916, -4.6773047, -37.8963242, -4.7275085, -27.7818680, 27.8323669
35: -32.9248810, 0.4074488, -32.9487305, 0.3301444, -28.2013550, 28.2800751
36: -36.8418961, -0.5486655, -36.8674850, -0.6282825, -29.0504303, 29.1506729
37: -44.5675659, -1.6068416, -44.6163063, -1.6886783, -38.8155670, 38.9458160
38: -43.9704590, 3.0179424, -44.0048599, 2.9202685, -40.7486115, 40.8711166
39: -43.6018753, 3.0709152, -43.6368179, 3.0872397, -41.4509277, 41.4273911
40: -32.7486267, 0.0726926, -32.7902451, 0.0795968, -31.1202850, 31.1539688
41: -20.7291069, 7.3589773, -20.7431183, 7.3138161, -26.5238190, 26.5832062
42: -22.9865131, -0.1983089, -22.9924431, -0.1740987, -18.5171394, 18.4754028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=172, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5607146
time: 34.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5766147
time: 28.15 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.2950144, 19.0405731, -9.3697853, 19.0664997, -25.2155342, 25.2655449
1: -1.1760631, 22.8376980, -1.2222047, 22.8607445, -19.7508278, 19.7876625
2: -1.6336143, 20.9285736, -1.6375070, 20.9837456, -17.2896080, 17.3216972
3: -9.3339386, 16.4955883, -9.3573494, 16.5733147, -22.0401154, 22.0115471
4: -3.1298881, 22.2386856, -3.1611004, 22.2723007, -21.7272339, 21.7718582
5: -7.8174028, 20.6164761, -7.8427382, 20.6967525, -23.7614288, 23.7565842
6: -28.8000698, -1.3807755, -28.8285599, -1.3627777, -23.1712646, 23.2227898
7: -7.6810789, 21.6439724, -7.7090349, 21.7036095, -23.6099701, 23.6056061
8: -14.7733173, 14.7455788, -14.7957134, 14.8241997, -26.4963112, 26.4951553
9: -5.2095766, 21.2825584, -5.1949587, 21.3243179, -24.3551483, 24.2852554
10: -18.0081902, 17.5641041, -17.8764076, 17.5847454, -31.4314575, 31.2677383
11: -26.8546143, 3.5497241, -26.8289261, 3.5806599, -27.9892807, 27.9178238
12: -34.8991318, -2.3484559, -34.8955498, -2.3156271, -27.2825165, 27.1901016
13: -26.2979813, 15.7525368, -26.2984390, 15.8675594, -34.0710220, 33.9451904
14: -55.9923019, -17.5276661, -55.9229813, -17.5480804, -37.8963776, 37.6865311
15: -14.3226643, 15.4795732, -14.3677082, 15.5340767, -27.8937683, 27.8671799
16: -14.0988483, 20.8453217, -14.0771236, 20.8697605, -31.1473770, 31.1021957
17: -57.8978386, -14.3741388, -57.8571091, -14.3828831, -41.7681046, 41.5961609
18: -21.6012611, 12.1744900, -21.7330284, 12.1863346, -29.5874634, 29.7701912
19: -22.3244514, 3.5509074, -22.3918190, 3.5902705, -22.8191376, 22.8323708
20: -23.3219719, 1.3454041, -23.4047508, 1.3967288, -19.2358322, 19.2576256
21: -26.8612595, 2.3810697, -26.9232521, 2.4253423, -25.5779037, 25.5672607
22: -28.5552711, 3.3464978, -28.6676064, 3.3594618, -24.8213882, 24.8152924
23: -22.3470154, 5.7030597, -22.4117966, 5.7379308, -22.0773926, 22.1086197
24: -18.3486900, 9.4281406, -18.4735928, 9.4669571, -22.8684616, 22.9636459
25: -23.8648300, 5.3789406, -23.9697895, 5.4140077, -24.4737930, 24.4893341
26: -41.1186981, -0.4784527, -41.1926727, -0.4303794, -30.6847687, 30.7134933
27: -21.6353951, 8.6458750, -21.7236214, 8.6130466, -26.4897766, 26.6338654
28: -24.1552734, 6.0758963, -24.2568741, 6.0949330, -22.0129776, 22.0750198
29: -27.9166222, -0.1723756, -27.9920883, -0.1834092, -24.0918159, 24.0480881
30: -28.1611824, 3.7451742, -28.2547073, 3.7821467, -26.1914749, 26.2099838
31: -22.7124901, 5.0273066, -22.7894688, 5.0730019, -25.1128616, 25.1384888
32: -23.9250069, 2.3072858, -23.9616222, 2.3573439, -21.3873138, 21.3778076
33: -36.3881264, 3.6163244, -36.4589539, 3.6409287, -33.3150330, 33.3738251
34: -37.8360138, -4.7465940, -37.8952026, -4.7574286, -27.7318420, 27.8026352
35: -32.8946762, 0.3039727, -32.9474869, 0.2885914, -28.1318207, 28.2082596
36: -36.8304520, -0.6269298, -36.8662491, -0.6605620, -29.0096588, 29.0921860
37: -44.5600052, -1.6671286, -44.6131744, -1.7140069, -38.7792969, 38.8852005
38: -43.9303360, 2.9061680, -44.0014191, 2.8731790, -40.6821442, 40.8008881
39: -43.5347023, 2.9412179, -43.6345596, 3.0360832, -41.3489838, 41.3551636
40: -32.7283173, 0.0451865, -32.7877769, 0.0645745, -31.0830460, 31.1217842
41: -20.7286663, 7.3207927, -20.7410202, 7.2977414, -26.4889374, 26.5476685
42: -22.9779396, -0.2068093, -22.9879837, -0.1788533, -18.5016899, 18.4557152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=172, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5600512
time: 48.12 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5742005
time: 46.89 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.3548527, 19.0524063, -9.3464603, 19.0563278, -25.2218666, 25.2488289
1: -1.2178750, 22.8394604, -1.2072096, 22.8380642, -19.7371407, 19.7723236
2: -1.6316907, 20.9309425, -1.6187501, 20.9612293, -17.2788467, 17.2720566
3: -9.3532658, 16.5090313, -9.3303623, 16.5407677, -22.0426788, 21.9963493
4: -3.1515131, 22.2400646, -3.1251607, 22.2531605, -21.7327461, 21.7041245
5: -7.8406072, 20.6341343, -7.8247619, 20.6587830, -23.7559128, 23.7577896
6: -28.7793713, -1.3837895, -28.7997055, -1.3708935, -23.1635971, 23.1892395
7: -7.7021646, 21.6549053, -7.6794596, 21.6631908, -23.5850220, 23.5792885
8: -14.8063126, 14.7688417, -14.7773743, 14.7975655, -26.5098190, 26.4744797
9: -5.1886387, 21.2625751, -5.1761231, 21.2942524, -24.2781982, 24.2470322
10: -17.8780613, 17.4316559, -17.8775101, 17.5295982, -31.2411194, 31.1641006
11: -26.8001022, 3.5630898, -26.8259182, 3.5739365, -27.9123764, 27.9218140
12: -34.8863220, -2.3319468, -34.8880081, -2.3255186, -27.2258911, 27.2127304
13: -26.2666054, 15.7700090, -26.2481289, 15.8288937, -33.9944000, 33.9308548
14: -55.9147530, -17.6118145, -55.9300232, -17.5889168, -37.7505493, 37.7424011
15: -14.3656797, 15.5053673, -14.3692551, 15.5220881, -27.8986282, 27.8940125
16: -14.0493107, 20.8204136, -14.0556841, 20.8213348, -31.0436020, 31.0569839
17: -57.8568192, -14.3868990, -57.8722725, -14.4043045, -41.6642838, 41.6812210
18: -21.5976219, 12.1767893, -21.6803837, 12.1607485, -29.5766296, 29.6822357
19: -22.3317986, 3.5888140, -22.3566246, 3.5722282, -22.8069077, 22.8287125
20: -23.3482323, 1.3903348, -23.3724003, 1.3878345, -19.2527161, 19.2310829
21: -26.8538437, 2.4178774, -26.8935070, 2.4101119, -25.5504074, 25.5671921
22: -28.5675964, 3.3462679, -28.6080513, 3.3284097, -24.7594185, 24.7629547
23: -22.3443642, 5.7198968, -22.3687572, 5.7059531, -22.0526428, 22.0600891
24: -18.3724060, 9.4663515, -18.4207115, 9.4457083, -22.8801193, 22.9168396
25: -23.8625031, 5.4046717, -23.9206905, 5.3837290, -24.4085350, 24.4515991
26: -41.1105499, -0.4674668, -41.1266174, -0.4714456, -30.6530609, 30.6282883
27: -21.6043453, 8.5986547, -21.6695099, 8.5804682, -26.4516525, 26.5273132
28: -24.1684990, 6.0901432, -24.2038002, 6.0617075, -22.0035553, 22.0346794
29: -27.9083004, -0.2021346, -27.9413528, -0.2237651, -23.9931183, 24.0186615
30: -28.1707821, 3.7699513, -28.2258415, 3.7638915, -26.1557083, 26.1792984
31: -22.7023048, 5.0629349, -22.7523270, 5.0502901, -25.0816574, 25.1194153
32: -23.9199066, 2.3180344, -23.9304390, 2.3483932, -21.3840485, 21.3380852
33: -36.4052887, 3.6563659, -36.4285660, 3.6808066, -33.3712158, 33.3085403
34: -37.8380432, -4.7385459, -37.8709946, -4.7278609, -27.7732697, 27.7276688
35: -32.8853874, 0.3137465, -32.9166069, 0.3251595, -28.1669922, 28.1493530
36: -36.7966728, -0.6383882, -36.8333092, -0.6291900, -29.0185471, 29.0180740
37: -44.5126877, -1.7019563, -44.5693817, -1.6871924, -38.7645721, 38.7926102
38: -43.8988495, 2.8926573, -43.9574127, 2.9068036, -40.6850128, 40.6921158
39: -43.5694809, 3.0164838, -43.5904388, 3.0658607, -41.4137268, 41.3177261
40: -32.7115326, 0.0021672, -32.7430801, 0.0487185, -31.0566101, 31.0368805
41: -20.6908035, 7.2861023, -20.7146893, 7.3053246, -26.5070801, 26.4834213
42: -22.9853287, -0.2135479, -22.9826965, -0.1897359, -18.4775467, 18.4523888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=172, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5263182
time: 33.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5404558
time: 40.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.4091854, 19.0755234, -9.3509245, 19.0568275, -25.2805023, 25.2829323
1: -1.2503548, 22.8706074, -1.2081332, 22.8387489, -19.7718391, 19.8164024
2: -1.6834183, 20.9459133, -1.6211131, 20.9645119, -17.3074570, 17.3298187
3: -9.3776541, 16.5239697, -9.3318672, 16.5417709, -22.0628281, 22.0362282
4: -3.1979084, 22.2478256, -3.1280489, 22.2542095, -21.7766113, 21.7560768
5: -7.8901677, 20.6518440, -7.8282566, 20.6616707, -23.7913818, 23.7893982
6: -28.8319550, -1.3055596, -28.8124352, -1.3689346, -23.2044449, 23.2824059
7: -7.7481899, 21.6702271, -7.6829386, 21.6653481, -23.6295319, 23.6184387
8: -14.8833141, 14.7989969, -14.7783947, 14.8024626, -26.5760193, 26.5392075
9: -5.2744799, 21.3091850, -5.1780910, 21.3039169, -24.3748703, 24.2923050
10: -18.0989304, 17.5944023, -17.8799438, 17.5721283, -31.5018158, 31.3070450
11: -26.8779869, 3.5896664, -26.8282394, 3.5773058, -28.0041962, 27.9512482
12: -34.9123116, -2.2930164, -34.8895683, -2.3203068, -27.2844543, 27.2477036
13: -26.3004646, 15.8039570, -26.2522659, 15.8320293, -34.0359268, 33.9679489
14: -56.0819893, -17.4857464, -55.9342117, -17.5559120, -37.9513016, 37.8498840
15: -14.4244709, 15.5224171, -14.3729134, 15.5249233, -27.9743271, 27.9152298
16: -14.1470013, 20.8635197, -14.0595045, 20.8309250, -31.1446533, 31.1000977
17: -57.9674797, -14.3304291, -57.8753471, -14.3934040, -41.7970657, 41.7462463
18: -21.6510601, 12.2136250, -21.6854382, 12.1625252, -29.6296234, 29.7635574
19: -22.3731117, 3.6013994, -22.3605423, 3.5732539, -22.8615646, 22.8459511
20: -23.3746777, 1.4042089, -23.3749771, 1.3889832, -19.2881927, 19.2479897
21: -26.9136429, 2.4382167, -26.8980465, 2.4120255, -25.6241531, 25.5928459
22: -28.6002312, 3.3836985, -28.6141739, 3.3301196, -24.8526611, 24.7723541
23: -22.3779144, 5.7370591, -22.3706703, 5.7068186, -22.0884209, 22.0848312
24: -18.4003162, 9.4764624, -18.4258194, 9.4465818, -22.9083252, 22.9349785
25: -23.8932991, 5.4324532, -23.9244080, 5.3857770, -24.4807587, 24.4799118
26: -41.1601639, -0.4268556, -41.1297836, -0.4633837, -30.7159042, 30.6663589
27: -21.6668587, 8.6765137, -21.6843605, 8.5829105, -26.5067673, 26.6227036
28: -24.1931038, 6.1317134, -24.2081146, 6.0639386, -22.0363884, 22.0669632
29: -27.9489574, -0.1658041, -27.9476662, -0.2224457, -24.1046677, 24.0182190
30: -28.1978054, 3.8012743, -28.2291546, 3.7668710, -26.2093887, 26.2174377
31: -22.7651482, 5.0810084, -22.7563934, 5.0520501, -25.1502380, 25.1413879
32: -23.9591007, 2.3683510, -23.9401207, 2.3504157, -21.4200706, 21.4019852
33: -36.4769821, 3.7598939, -36.4451447, 3.6820264, -33.4411163, 33.4427414
34: -37.8998947, -4.6334658, -37.8878937, -4.7250543, -27.8315048, 27.8593903
35: -32.9652939, 0.4357953, -32.9371567, 0.3263760, -28.2387390, 28.2944031
36: -36.8946991, -0.5050769, -36.8591728, -0.6276889, -29.1013641, 29.1781693
37: -44.6312637, -1.5801435, -44.5985641, -1.6864333, -38.8688507, 38.9602966
38: -44.0227928, 3.0625052, -43.9870491, 2.9096832, -40.7986450, 40.8907623
39: -43.6346359, 3.0881371, -43.6039772, 3.0677633, -41.4823914, 41.4166336
40: -32.7788277, 0.0913157, -32.7575607, 0.0494957, -31.1239319, 31.1454239
41: -20.7671661, 7.3833990, -20.7341709, 7.3070459, -26.5748444, 26.6041641
42: -22.9924545, -0.1857240, -22.9844780, -0.1873662, -18.5097542, 18.4838104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=172, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5375597
time: 31.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5516993
time: 33.96 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.3710318, 19.0528297, -9.4073534, 19.0673466, -25.2535896, 25.3143349
1: -1.2305646, 22.8396912, -1.2504334, 22.8610420, -19.7725029, 19.8104324
2: -1.6423030, 20.9311905, -1.6549852, 20.9813309, -17.3118629, 17.3021851
3: -9.3667793, 16.5098953, -9.3755054, 16.5752144, -22.0803604, 22.0310669
4: -3.1688781, 22.2401657, -3.1856856, 22.2723541, -21.7659264, 21.7549286
5: -7.8533459, 20.6352997, -7.8679457, 20.6959381, -23.7994690, 23.7901573
6: -28.7853012, -1.3799696, -28.8192406, -1.3386998, -23.1986618, 23.2061386
7: -7.7187443, 21.6554451, -7.7330399, 21.7025795, -23.6403427, 23.6216812
8: -14.8238430, 14.7693939, -14.8378353, 14.8218098, -26.5396729, 26.5271454
9: -5.1989861, 21.2629280, -5.2180104, 21.3168392, -24.3192444, 24.2916412
10: -17.8848915, 17.4326324, -17.9062386, 17.5458927, -31.2708206, 31.1911926
11: -26.8003578, 3.5671721, -26.8320618, 3.5916176, -27.9342804, 27.9374466
12: -34.8890610, -2.3270621, -34.8986931, -2.3025050, -27.2594986, 27.2270546
13: -26.2807922, 15.7710934, -26.2984600, 15.8784256, -34.0618744, 33.9759903
14: -55.9173050, -17.6088295, -55.9477005, -17.5760288, -37.7918777, 37.7602997
15: -14.3707390, 15.5084743, -14.3984261, 15.5335979, -27.9125824, 27.9265518
16: -14.0586710, 20.8206234, -14.0920343, 20.8608246, -31.0932541, 31.0856247
17: -57.8577461, -14.3841219, -57.8766975, -14.3866940, -41.7206039, 41.6835785
18: -21.5987854, 12.1885719, -21.7338371, 12.2016554, -29.6110687, 29.7517700
19: -22.3326950, 3.6003687, -22.3929100, 3.6101623, -22.8358688, 22.8764534
20: -23.3486900, 1.3991604, -23.4045029, 1.4181142, -19.2786598, 19.2753334
21: -26.8543949, 2.4283569, -26.9244003, 2.4459865, -25.5794373, 25.6054688
22: -28.5682335, 3.3604326, -28.6646461, 3.3743818, -24.7916641, 24.8347015
23: -22.3447704, 5.7338548, -22.4125443, 5.7526565, -22.0875397, 22.1197166
24: -18.3734741, 9.4788685, -18.4708004, 9.4870853, -22.9138870, 22.9819221
25: -23.8634930, 5.4193249, -23.9686871, 5.4333715, -24.4481277, 24.5163803
26: -41.1115227, -0.4501204, -41.1920929, -0.4147825, -30.6919403, 30.7169724
27: -21.6052113, 8.6116161, -21.7126884, 8.6253815, -26.4912109, 26.5844040
28: -24.1690369, 6.1069040, -24.2547188, 6.1166019, -22.0427094, 22.1041412
29: -27.9086761, -0.1891037, -27.9908066, -0.1793482, -24.0259705, 24.0810623
30: -28.1711922, 3.7794681, -28.2534389, 3.8000009, -26.1840668, 26.2155685
31: -22.7036018, 5.0755415, -22.7907124, 5.0934658, -25.1221008, 25.1722717
32: -23.9260864, 2.3210630, -23.9548492, 2.3764293, -21.4173317, 21.3598366
33: -36.4081116, 3.6574726, -36.4465904, 3.6896601, -33.3868713, 33.3277664
34: -37.8386765, -4.7378130, -37.8803062, -4.7216287, -27.7800598, 27.7407761
35: -32.8869553, 0.3153372, -32.9299698, 0.3334093, -28.1774063, 28.1664886
36: -36.7977753, -0.6360054, -36.8432579, -0.6196594, -29.0335617, 29.0355835
37: -44.5159454, -1.7021627, -44.5907059, -1.6850748, -38.7840576, 38.8105850
38: -43.9010696, 2.8966823, -43.9776230, 2.9253368, -40.7154083, 40.7298355
39: -43.5775871, 3.0166273, -43.6277008, 3.0851793, -41.4376221, 41.3508301
40: -32.7188911, 0.0035474, -32.7798309, 0.0796015, -31.0944519, 31.0699387
41: -20.6933746, 7.2880650, -20.7258415, 7.3173800, -26.5210266, 26.4957428
42: -22.9877262, -0.2114332, -22.9911613, -0.1747775, -18.4959717, 18.4605141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=172, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5493827
time: 24.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5635265
time: 28.44 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.4253712, 19.0759506, -9.4118862, 19.0678635, -25.3121948, 25.3484344
1: -1.2630630, 22.8708267, -1.2513504, 22.8617325, -19.8072014, 19.8545151
2: -1.6940539, 20.9461937, -1.6573596, 20.9846230, -17.3404694, 17.3599548
3: -9.3911495, 16.5248375, -9.3770094, 16.5761967, -22.1004562, 22.0709763
4: -3.2152934, 22.2478943, -3.1886096, 22.2734070, -21.8097954, 21.8069000
5: -7.9028668, 20.6529846, -7.8714237, 20.6988373, -23.8349342, 23.8217583
6: -28.8378830, -1.3017235, -28.8319550, -1.3367224, -23.2395477, 23.2992630
7: -7.7647371, 21.6707153, -7.7365618, 21.7047672, -23.6848373, 23.6608047
8: -14.9009027, 14.7995434, -14.8388891, 14.8267069, -26.6058311, 26.5919266
9: -5.2848501, 21.3095016, -5.2200027, 21.3264923, -24.4159317, 24.3369064
10: -18.1058006, 17.5954113, -17.9086571, 17.5884209, -31.5315475, 31.3340836
11: -26.8782501, 3.5936666, -26.8343735, 3.5949755, -28.0260315, 27.9669189
12: -34.9150620, -2.2881947, -34.9002533, -2.2972441, -27.3180389, 27.2620049
13: -26.3146648, 15.8050804, -26.3026009, 15.8815289, -34.1034164, 34.0131607
14: -56.0844994, -17.4827919, -55.9518547, -17.5430908, -37.9925613, 37.8677597
15: -14.4295254, 15.5255289, -14.4021187, 15.5364780, -27.9883118, 27.9478378
16: -14.1563988, 20.8637810, -14.0958834, 20.8704300, -31.1942902, 31.1287460
17: -57.9684296, -14.3277416, -57.8797913, -14.3757992, -41.8534241, 41.7485657
18: -21.6521702, 12.2254000, -21.7388725, 12.2033939, -29.6640930, 29.8330536
19: -22.3740120, 3.6129837, -22.3968906, 3.6112397, -22.8905334, 22.8936958
20: -23.3751144, 1.4130101, -23.4070568, 1.4192395, -19.3141098, 19.2922401
21: -26.9142494, 2.4486773, -26.9289436, 2.4479034, -25.6531982, 25.6311417
22: -28.6008186, 3.3979123, -28.6707363, 3.3761063, -24.8848991, 24.8441086
23: -22.3783398, 5.7509956, -22.4144478, 5.7535481, -22.1232910, 22.1444702
24: -18.4013557, 9.4890413, -18.4759331, 9.4879131, -22.9420776, 23.0000801
25: -23.8942890, 5.4471369, -23.9724083, 5.4354544, -24.5203247, 24.5446663
26: -41.1611252, -0.4094558, -41.1952133, -0.4067812, -30.7547760, 30.7550278
27: -21.6677055, 8.6894979, -21.7275429, 8.6278400, -26.5463791, 26.6797867
28: -24.1936073, 6.1484575, -24.2590771, 6.1187849, -22.0755844, 22.1364174
29: -27.9493256, -0.1527513, -27.9970894, -0.1780524, -24.1375351, 24.0806274
30: -28.1981163, 3.8108027, -28.2567196, 3.8030026, -26.2377243, 26.2536926
31: -22.7664604, 5.0935965, -22.7947578, 5.0952377, -25.1907120, 25.1942406
32: -23.9652901, 2.3713434, -23.9645233, 2.3784654, -21.4533157, 21.4237175
33: -36.4798317, 3.7610297, -36.4631805, 3.6908941, -33.4567871, 33.4619446
34: -37.9005775, -4.6327577, -37.8972359, -4.7187638, -27.8382721, 27.8725204
35: -32.9668198, 0.4373894, -32.9505005, 0.3346219, -28.2491226, 28.3115463
36: -36.8957863, -0.5027242, -36.8691788, -0.6181760, -29.1164246, 29.1956482
37: -44.6345062, -1.5803623, -44.6198273, -1.6843014, -38.8883667, 38.9783020
38: -44.0249252, 3.0666270, -44.0073471, 2.9281812, -40.8290558, 40.9284439
39: -43.6427917, 3.0882406, -43.6412392, 3.0870018, -41.5063324, 41.4498596
40: -32.7861862, 0.0927052, -32.7943230, 0.0803905, -31.1617889, 31.1784439
41: -20.7697601, 7.3853474, -20.7453117, 7.3191381, -26.5888367, 26.6165085
42: -22.9948502, -0.1835752, -22.9929695, -0.1723936, -18.5281868, 18.4919434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=172, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5606881
time: 25.15 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5748328
time: 38.96 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 66.45 seconds
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 66.45
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5600329
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 66.45
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5760283
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 66.45
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5607146
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 66.45
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5766147
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 66.45
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5600512
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 66.45
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5742005
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 66.45
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5263182
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 66.45
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5404558
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 66.45
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5375597
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 66.45
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5516993
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 66.45
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5493827
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 66.45
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5635265
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 66.45
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5606881
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 66.45
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5748328

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.2260561, 18.9990883, -9.3824682, 19.1061440, -25.1942024, 25.2344131
1: -1.1262407, 22.7948112, -1.2286277, 22.9016724, -19.7554283, 19.7486496
2: -1.5923331, 20.8938980, -1.6401570, 21.0166092, -17.2952538, 17.2860336
3: -9.3031006, 16.4439812, -9.3556871, 16.6192856, -22.0548248, 21.9614983
4: -3.0777869, 22.2148514, -3.1661711, 22.2945061, -21.6994934, 21.7598915
5: -7.7765255, 20.5641212, -7.8441730, 20.7442398, -23.7764206, 23.7053909
6: -28.7875404, -1.4028072, -28.8358765, -1.3547831, -23.1534576, 23.2067757
7: -7.6324840, 21.6043167, -7.7159443, 21.7402592, -23.6078377, 23.5727768
8: -14.7174110, 14.7034101, -14.7958698, 14.8609781, -26.4817200, 26.4585800
9: -5.1807899, 21.2560844, -5.1985655, 21.3470707, -24.3557892, 24.2309036
10: -17.9748249, 17.5249653, -17.8820114, 17.6169891, -31.4343643, 31.2326584
11: -26.7895412, 3.5160389, -26.8893356, 3.5826411, -27.9233322, 27.9529495
12: -34.8724442, -2.3969622, -34.8996468, -2.2793818, -27.2963333, 27.1460648
13: -26.2437897, 15.6611462, -26.2998562, 15.9512558, -34.1085434, 33.8545380
14: -55.9260750, -17.6052170, -55.9303741, -17.4791451, -37.9083862, 37.6130371
15: -14.2955666, 15.4658079, -14.3721819, 15.5411758, -27.8706589, 27.8587570
16: -14.0490608, 20.7888927, -14.0912123, 20.9234276, -31.1615295, 31.0573120
17: -57.8383904, -14.4704418, -57.8723907, -14.2988758, -41.8008423, 41.5178986
18: -21.5498810, 12.1413784, -21.7736473, 12.1860695, -29.5335388, 29.7807617
19: -22.2400856, 3.5027642, -22.4683094, 3.5896521, -22.7340546, 22.8697586
20: -23.2412415, 1.2934313, -23.4820824, 1.3952675, -19.1548004, 19.2995338
21: -26.7704391, 2.3305264, -27.0069351, 2.4230940, -25.4835129, 25.6136169
22: -28.4486160, 3.2960870, -28.7686653, 3.3567219, -24.7111816, 24.8814468
23: -22.2691956, 5.6495318, -22.4858341, 5.7378545, -21.9985161, 22.1415024
24: -18.2476082, 9.3827438, -18.5696220, 9.4651690, -22.7652130, 23.0235977
25: -23.7969589, 5.3299580, -24.0325279, 5.4156914, -24.4079552, 24.5157127
26: -41.0156212, -0.5456729, -41.2909164, -0.4346561, -30.5797806, 30.7646637
27: -21.5366917, 8.5941696, -21.8160000, 8.6114578, -26.3884430, 26.6835403
28: -24.0711727, 6.0198078, -24.3378201, 6.0928164, -21.9269981, 22.1158981
29: -27.8134232, -0.2106369, -28.0903130, -0.1841135, -23.9888191, 24.1181641
30: -28.0827217, 3.6927047, -28.3302364, 3.7828822, -26.1153107, 26.2466545
31: -22.6476803, 4.9872942, -22.8459473, 5.0729375, -25.0461349, 25.1609840
32: -23.9032879, 2.2914412, -23.9775810, 2.3630776, -21.3637085, 21.3858376
33: -36.3435516, 3.5965905, -36.4939995, 3.6428018, -33.2745514, 33.3803101
34: -37.7936745, -4.7911506, -37.9335632, -4.7488427, -27.6984940, 27.8054581
35: -32.8527222, 0.2740159, -32.9824371, 0.2975802, -28.0994949, 28.2146378
36: -36.7765656, -0.6728992, -36.9136658, -0.6530099, -28.9626160, 29.0988464
37: -44.4931145, -1.6936440, -44.6655807, -1.7112246, -38.7150726, 38.9087906
38: -43.8757820, 2.8574018, -44.0461578, 2.8878231, -40.6253967, 40.7995605
39: -43.4938164, 2.9237580, -43.6574936, 3.0483751, -41.3047638, 41.3712845
40: -32.6908035, 0.0251305, -32.8079529, 0.0809267, -31.0574646, 31.1193466
41: -20.6879711, 7.2944126, -20.7727470, 7.2996082, -26.4302597, 26.5534668
42: -22.9695625, -0.2215343, -22.9937630, -0.1764638, -18.4952660, 18.4462013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=171, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=198, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5741336
time: 26.30 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5241337, upper bound: 11.5741336
time: 34.04 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.3564415, 19.0344620, -9.4245300, 19.1075096, -25.2909164, 25.3172989
1: -1.2132435, 22.8279037, -1.2577896, 22.9026699, -19.8118019, 19.8154449
2: -1.6527531, 20.9115105, -1.6599681, 21.0174904, -17.3461418, 17.3242722
3: -9.3603306, 16.4732513, -9.3753834, 16.6221619, -22.1151810, 22.0209198
4: -3.1631694, 22.2240887, -3.1936746, 22.2955933, -21.7820282, 21.7949028
5: -7.8619909, 20.6006393, -7.8728476, 20.7462749, -23.8498764, 23.7705650
6: -28.8253880, -1.3237319, -28.8392677, -1.3287439, -23.2217560, 23.2832603
7: -7.7161026, 21.6310215, -7.7433672, 21.7414188, -23.6826820, 23.6279869
8: -14.8449726, 14.7573795, -14.8390713, 14.8634491, -26.5911942, 26.5553589
9: -5.2560577, 21.2829933, -5.2235932, 21.3492470, -24.4165421, 24.2825928
10: -18.0723991, 17.5562782, -17.9142570, 17.6206837, -31.5344009, 31.2989426
11: -26.8132019, 3.5599532, -26.8947296, 3.5969710, -27.9601059, 28.0019836
12: -34.8883324, -2.3366141, -34.9042969, -2.2610173, -27.3317947, 27.2179947
13: -26.2604485, 15.7136898, -26.3040504, 15.9652252, -34.1409225, 33.9224625
14: -56.0183067, -17.5603008, -55.9592209, -17.4741554, -38.0045319, 37.7942200
15: -14.4023991, 15.5117626, -14.4065771, 15.5435734, -27.9651947, 27.9394073
16: -14.1065836, 20.8073311, -14.1099567, 20.9241486, -31.2083740, 31.0838394
17: -57.9090805, -14.4239922, -57.8950539, -14.2917528, -41.8861313, 41.6702423
18: -21.6008530, 12.1922693, -21.7794342, 12.2031298, -29.6101685, 29.8436356
19: -22.2896404, 3.5648327, -22.4733944, 3.6105859, -22.8055267, 22.9310646
20: -23.2944183, 1.3610020, -23.4844093, 1.4177766, -19.2330818, 19.3341560
21: -26.8234062, 2.3981364, -27.0125961, 2.4456139, -25.5587769, 25.6774712
22: -28.4941998, 3.3474307, -28.7717571, 3.3733845, -24.7745743, 24.9101753
23: -22.3005390, 5.6974974, -22.4885292, 5.7534928, -22.0444107, 22.1773758
24: -18.3002968, 9.4436569, -18.5719833, 9.4861279, -22.8388443, 23.0600395
25: -23.8263969, 5.3981185, -24.0351620, 5.4371562, -24.4544983, 24.5709000
26: -41.0580597, -0.4766784, -41.2935028, -0.4110508, -30.6497879, 30.8062210
27: -21.5689907, 8.6377802, -21.8199310, 8.6262360, -26.4449463, 26.7294464
28: -24.1095276, 6.0923781, -24.3400002, 6.1166964, -21.9896011, 22.1772995
29: -27.8461609, -0.1909823, -28.0952339, -0.1787755, -24.0345268, 24.1506424
30: -28.1196060, 3.7583337, -28.3322411, 3.8037474, -26.1615105, 26.2903481
31: -22.7016506, 5.0535851, -22.8512154, 5.0951910, -25.1239700, 25.2167358
32: -23.9435616, 2.3555171, -23.9804764, 2.3841538, -21.4296227, 21.4317284
33: -36.4353447, 3.7412400, -36.4981995, 3.6928034, -33.4163437, 33.4684753
34: -37.8582916, -4.6773047, -37.9355812, -4.7102013, -27.8049393, 27.8753128
35: -32.9248810, 0.4074488, -32.9854279, 0.3436260, -28.2168274, 28.3179550
36: -36.8418961, -0.5486655, -36.9165878, -0.6106229, -29.0693436, 29.2022934
37: -44.5675659, -1.6068416, -44.6722336, -1.6815839, -38.8241425, 39.0018539
38: -43.9704590, 3.0179424, -44.0520477, 2.9428339, -40.7723236, 40.9271774
39: -43.6018753, 3.0709152, -43.6641426, 3.0992498, -41.4621277, 41.4659271
40: -32.7486267, 0.0726926, -32.8144836, 0.0967624, -31.1361694, 31.1760712
41: -20.7291069, 7.3589773, -20.7770119, 7.3209977, -26.5301514, 26.6222763
42: -22.9865131, -0.1983089, -22.9987450, -0.1700020, -18.5217781, 18.4823799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=171, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=198, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5318518, upper bound: 11.5747583
time: 32.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5459945, upper bound: 11.5747583
time: 25.51 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.2943659, 19.0404358, -9.3765316, 19.0820122, -25.2339325, 25.2619019
1: -1.1745281, 22.8374596, -1.2282915, 22.8752060, -19.7711868, 19.7932777
2: -1.6329865, 20.9283199, -1.6411116, 21.0148411, -17.3217812, 17.3188591
3: -9.3334360, 16.4952888, -9.3593235, 16.6109333, -22.0811157, 22.0036583
4: -3.1292229, 22.2384071, -3.1674409, 22.3051033, -21.7611961, 21.7682877
5: -7.8119216, 20.6161480, -7.8361654, 20.7080727, -23.7721100, 23.7647438
6: -28.7977257, -1.3811307, -28.8276348, -1.3477540, -23.1952362, 23.2206612
7: -7.6780920, 21.6438675, -7.7164288, 21.7013226, -23.6069870, 23.6297913
8: -14.7717514, 14.7451935, -14.7948866, 14.8310308, -26.5028267, 26.4939957
9: -5.2090540, 21.2817497, -5.2217817, 21.3251286, -24.3440628, 24.3139458
10: -18.0073471, 17.5629063, -17.9502373, 17.5863132, -31.4324646, 31.3416824
11: -26.8539886, 3.5491300, -26.8973083, 3.5799198, -27.9848328, 27.9873047
12: -34.8989868, -2.3518095, -34.9237061, -2.3174839, -27.2727585, 27.2320251
13: -26.2977161, 15.7519627, -26.3045158, 15.8767204, -34.0788651, 33.9632111
14: -55.9919891, -17.5337009, -55.9933205, -17.5566959, -37.8647156, 37.7788162
15: -14.3223763, 15.4783134, -14.3828983, 15.5341682, -27.8943100, 27.8829422
16: -14.0979910, 20.8434601, -14.1345892, 20.8669472, -31.1343079, 31.1583557
17: -57.8975220, -14.3809395, -57.9283447, -14.3930454, -41.7454147, 41.6944427
18: -21.6006508, 12.1740561, -21.7427139, 12.1933613, -29.6191254, 29.7549362
19: -22.3240662, 3.5486684, -22.4141521, 3.5864284, -22.8144150, 22.8595352
20: -23.3216648, 1.3446026, -23.4241276, 1.3961718, -19.2300034, 19.2770958
21: -26.8607502, 2.3801656, -26.9593048, 2.4233019, -25.5723801, 25.6092224
22: -28.5549011, 3.3430340, -28.6785011, 3.3546553, -24.8235931, 24.8315506
23: -22.3465328, 5.6984358, -22.4180336, 5.7300878, -22.0714188, 22.1296082
24: -18.3483009, 9.4249210, -18.4853706, 9.4610271, -22.8724365, 22.9738541
25: -23.8645287, 5.3782682, -24.0088177, 5.4185276, -24.4703102, 24.5312271
26: -41.1182747, -0.4860749, -41.1908455, -0.4450274, -30.7008514, 30.7043076
27: -21.6347198, 8.6456480, -21.7281799, 8.6153240, -26.4953156, 26.6350479
28: -24.1550102, 6.0721884, -24.2627831, 6.0886803, -22.0182457, 22.0806732
29: -27.9162350, -0.1730689, -28.0393772, -0.1834747, -24.0824203, 24.1003799
30: -28.1607304, 3.7441077, -28.3203850, 3.7809625, -26.1754646, 26.2766953
31: -22.7120571, 5.0264292, -22.8032551, 5.0726414, -25.0993958, 25.1705933
32: -23.9241714, 2.3070271, -23.9615936, 2.3904314, -21.4213715, 21.3694763
33: -36.3873024, 3.6158338, -36.4596176, 3.6751914, -33.3513184, 33.3715363
34: -37.8353004, -4.7472777, -37.8951797, -4.7287140, -27.7849808, 27.7785416
35: -32.8941879, 0.3034143, -32.9496994, 0.3235822, -28.1803131, 28.1945038
36: -36.8298988, -0.6275415, -36.8678207, -0.6267419, -29.0527344, 29.0810242
37: -44.5590744, -1.6736445, -44.6290512, -1.7217398, -38.8000488, 38.8771973
38: -43.9293900, 2.9053130, -44.0042343, 2.9528384, -40.7613525, 40.8010864
39: -43.5338669, 2.9406228, -43.6381912, 3.0744991, -41.3871613, 41.3441162
40: -32.7274818, 0.0443761, -32.7964401, 0.0767176, -31.1075211, 31.1115456
41: -20.7275543, 7.3205376, -20.7420788, 7.3333731, -26.5238724, 26.5417099
42: -22.9776001, -0.2070882, -22.9911346, -0.1662707, -18.5105362, 18.4700584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1789

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5269404, upper bound: 11.5729292
time: 27.02 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5517287, upper bound: 11.5728847
time: 34.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.3500109, 19.0519810, -9.3316288, 19.0550156, -25.2138786, 25.2281799
1: -1.2137222, 22.8384418, -1.1948977, 22.8350067, -19.7301598, 19.7588844
2: -1.6242924, 20.9295635, -1.5952096, 20.9569454, -17.2682724, 17.2469940
3: -9.3452291, 16.5070896, -9.3058338, 16.5346985, -22.0285492, 21.9673805
4: -3.1423788, 22.2386990, -3.0959630, 22.2488956, -21.7200394, 21.6733742
5: -7.8387947, 20.6323471, -7.8200665, 20.6531906, -23.7483902, 23.7507706
6: -28.7746391, -1.3848653, -28.7853642, -1.3741140, -23.1549072, 23.1718369
7: -7.7002163, 21.6538181, -7.6733398, 21.6598625, -23.5782280, 23.5691910
8: -14.8040848, 14.7658348, -14.7705250, 14.7886114, -26.4960098, 26.4585114
9: -5.1861897, 21.2584057, -5.1688371, 21.2811508, -24.2547684, 24.2310448
10: -17.8749180, 17.4188538, -17.8680801, 17.4898739, -31.1974106, 31.1435776
11: -26.7976170, 3.5544820, -26.8179893, 3.5476408, -27.8821487, 27.9056473
12: -34.8849792, -2.3370938, -34.8836746, -2.3414440, -27.2001801, 27.2022820
13: -26.2646523, 15.7678776, -26.2424812, 15.8222437, -33.9850769, 33.9226761
14: -55.9121475, -17.6310844, -55.9218369, -17.6497364, -37.6751404, 37.7140198
15: -14.3641300, 15.5019073, -14.3644390, 15.5112038, -27.8824158, 27.8846893
16: -14.0452337, 20.8141232, -14.0431728, 20.8010216, -31.0150070, 31.0383835
17: -57.8539810, -14.3986330, -57.8633499, -14.4416428, -41.6097565, 41.6594162
18: -21.5945053, 12.1741657, -21.6706753, 12.1528492, -29.5562744, 29.6514168
19: -22.3285084, 3.5874865, -22.3463669, 3.5688443, -22.7957382, 22.8166351
20: -23.3465538, 1.3863292, -23.3671150, 1.3756902, -19.2379074, 19.2213898
21: -26.8495979, 2.4127209, -26.8801327, 2.3944533, -25.5279388, 25.5493011
22: -28.5637398, 3.3436446, -28.5957890, 3.3209593, -24.7476044, 24.7510414
23: -22.3425961, 5.7184067, -22.3631935, 5.7017579, -22.0422287, 22.0519333
24: -18.3699741, 9.4656734, -18.4132195, 9.4438343, -22.8735657, 22.9078979
25: -23.8600235, 5.3953881, -23.9129391, 5.3553343, -24.3737411, 24.4341660
26: -41.1078835, -0.4713283, -41.1181488, -0.4834437, -30.6328735, 30.6078949
27: -21.6002598, 8.5974998, -21.6565495, 8.5769796, -26.4428024, 26.5118027
28: -24.1670265, 6.0886250, -24.1989479, 6.0568705, -21.9970856, 22.0275078
29: -27.9042454, -0.2087885, -27.9286880, -0.2450850, -23.9675751, 24.0031738
30: -28.1678696, 3.7556980, -28.2167587, 3.7201266, -26.1075058, 26.1568565
31: -22.6990967, 5.0603542, -22.7425003, 5.0429325, -25.0579376, 25.1015587
32: -23.9110985, 2.3168530, -23.9034615, 2.3448343, -21.3718338, 21.3092079
33: -36.4018288, 3.6554785, -36.4179077, 3.6781254, -33.3631821, 33.2925110
34: -37.8334045, -4.7416630, -37.8573990, -4.7374954, -27.7535324, 27.6931610
35: -32.8803482, 0.3120432, -32.9018402, 0.3198819, -28.1532211, 28.1217117
36: -36.7907333, -0.6398821, -36.8151093, -0.6337647, -29.0046997, 28.9889374
37: -44.5080261, -1.7046146, -44.5549469, -1.6955233, -38.7563629, 38.7753830
38: -43.8868141, 2.8899317, -43.9186554, 2.8984842, -40.6642456, 40.6498489
39: -43.5648193, 3.0157914, -43.5764618, 3.0635910, -41.4034576, 41.2941360
40: -32.7075653, 0.0013821, -32.7312050, 0.0462472, -31.0493317, 31.0216370
41: -20.6806774, 7.2852077, -20.6840019, 7.3024759, -26.4939499, 26.4506378
42: -22.9836884, -0.2148125, -22.9775810, -0.1936765, -18.4678116, 18.4404602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1789

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5475778, upper bound: 11.5250955
time: 34.19 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5724299, upper bound: 11.5250499
time: 32.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.3541965, 19.0522614, -9.3531895, 19.0718422, -25.2402878, 25.2452087
1: -1.2163463, 22.8392067, -1.2132831, 22.8525124, -19.7575073, 19.7779160
2: -1.6310737, 20.9307289, -1.6223669, 20.9923401, -17.3110313, 17.2692108
3: -9.3527718, 16.5087395, -9.3323421, 16.5784111, -22.0836868, 21.9884262
4: -3.1508546, 22.2397881, -3.1314516, 22.2859268, -21.7667313, 21.7005501
5: -7.8351226, 20.6337891, -7.8181987, 20.6701031, -23.7665977, 23.7659645
6: -28.7770195, -1.3841619, -28.7987690, -1.3558512, -23.1875610, 23.1871834
7: -7.6991854, 21.6547756, -7.6868544, 21.6608849, -23.5820694, 23.6035156
8: -14.8047237, 14.7684803, -14.7765732, 14.8044167, -26.5164185, 26.4733276
9: -5.1880999, 21.2617588, -5.2029653, 21.2950783, -24.2670822, 24.2757263
10: -17.8771133, 17.4304123, -17.9513741, 17.5311737, -31.2421341, 31.2380447
11: -26.7995319, 3.5625091, -26.8942795, 3.5731711, -27.9079361, 27.9913330
12: -34.8861618, -2.3353319, -34.9162025, -2.3274426, -27.2161407, 27.2546616
13: -26.2663727, 15.7693901, -26.2542000, 15.8380661, -34.0022736, 33.9488678
14: -55.9143181, -17.6178493, -56.0003510, -17.5975723, -37.7188644, 37.8346176
15: -14.3653517, 15.5041122, -14.3844194, 15.5221701, -27.8991928, 27.9097366
16: -14.0484047, 20.8185539, -14.1132126, 20.8185463, -31.0305786, 31.1131210
17: -57.8565712, -14.3936710, -57.9435196, -14.4144859, -41.6415863, 41.7795258
18: -21.5970898, 12.1764240, -21.6900291, 12.1677818, -29.6082916, 29.6669998
19: -22.3313751, 3.5865757, -22.3789597, 3.5683563, -22.8022461, 22.8559265
20: -23.3479576, 1.3895514, -23.3917770, 1.3873267, -19.2469101, 19.2505379
21: -26.8532562, 2.4169846, -26.9295635, 2.4080844, -25.5448837, 25.6092300
22: -28.5672932, 3.3428047, -28.6189899, 3.3236310, -24.7615967, 24.7793274
23: -22.3438435, 5.7152586, -22.3750000, 5.6980872, -22.0466690, 22.0810699
24: -18.3720360, 9.4630985, -18.4325409, 9.4397717, -22.8841019, 22.9271164
25: -23.8622169, 5.4039421, -23.9597397, 5.3882179, -24.4050941, 24.4935074
26: -41.1100693, -0.4750032, -41.1248703, -0.4860859, -30.6691360, 30.6191330
27: -21.6036510, 8.5984287, -21.6740265, 8.5827560, -26.4571991, 26.5284882
28: -24.1682949, 6.0863948, -24.2097034, 6.0554209, -22.0088196, 22.0403481
29: -27.9079361, -0.2028644, -27.9886398, -0.2238404, -23.9837189, 24.0709534
30: -28.1702595, 3.7688854, -28.2915497, 3.7627285, -26.1397400, 26.2460442
31: -22.7018967, 5.0620146, -22.7661591, 5.0499086, -25.0681610, 25.1515579
32: -23.9190865, 2.3177798, -23.9303932, 2.3814847, -21.4181175, 21.3297234
33: -36.4044800, 3.6558323, -36.4292145, 3.7150412, -33.4075317, 33.3062592
34: -37.8373871, -4.7392230, -37.8709946, -4.6990857, -27.8264008, 27.7035828
35: -32.8849335, 0.3131638, -32.9187813, 0.3601718, -28.2155075, 28.1355591
36: -36.7961197, -0.6390252, -36.8350296, -0.5953608, -29.0616150, 29.0068893
37: -44.5117416, -1.7085013, -44.5852776, -1.6949692, -38.7853699, 38.7846451
38: -43.8979301, 2.8917761, -43.9601593, 2.9865670, -40.7641907, 40.6923141
39: -43.5685806, 3.0158916, -43.5941010, 3.1042848, -41.4518280, 41.3066406
40: -32.7106743, 0.0013461, -32.7517509, 0.0608673, -31.0811310, 31.0266418
41: -20.6896935, 7.2858829, -20.7157211, 7.3409853, -26.5419540, 26.4774704
42: -22.9849777, -0.2138309, -22.9858742, -0.1771343, -18.4863892, 18.4667206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1789

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5475778, upper bound: 11.5392302
time: 37.06 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5724299, upper bound: 11.5391846
time: 28.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.4043627, 19.0751305, -9.3361511, 19.0555038, -25.2725029, 25.2622948
1: -1.2462301, 22.8696022, -1.1958752, 22.8356857, -19.7648354, 19.8029518
2: -1.6760561, 20.9445152, -1.5975592, 20.9602203, -17.2968674, 17.3047714
3: -9.3695793, 16.5220222, -9.3073378, 16.5357056, -22.0486755, 22.0072899
4: -3.1887598, 22.2464676, -3.0989122, 22.2499447, -21.7639313, 21.7253304
5: -7.8883057, 20.6500568, -7.8235445, 20.6560497, -23.7838402, 23.7824097
6: -28.8272018, -1.3066211, -28.7980423, -1.3722000, -23.1957474, 23.2649765
7: -7.7462554, 21.6691170, -7.6768231, 21.6620426, -23.6227341, 23.6083488
8: -14.8811331, 14.7959652, -14.7715769, 14.7934971, -26.5621872, 26.5233231
9: -5.2720575, 21.3050346, -5.1708083, 21.2908173, -24.3515167, 24.2763367
10: -18.0958595, 17.5816555, -17.8705254, 17.5324020, -31.4581375, 31.2864532
11: -26.8754826, 3.5810523, -26.8203030, 3.5510149, -27.9739456, 27.9351044
12: -34.9109573, -2.2981739, -34.8852310, -2.3361826, -27.2587509, 27.2372437
13: -26.2984772, 15.8018894, -26.2466202, 15.8253708, -34.0266113, 33.9598160
14: -56.0793991, -17.5050735, -55.9259605, -17.6167583, -37.8758545, 37.8214798
15: -14.4228973, 15.5189476, -14.3680954, 15.5140781, -27.9580917, 27.9059143
16: -14.1429281, 20.8572235, -14.0470724, 20.8106670, -31.1161041, 31.0815048
17: -57.9646797, -14.3421364, -57.8664780, -14.4307461, -41.7425613, 41.7244415
18: -21.6478806, 12.2110071, -21.6757164, 12.1545982, -29.6092911, 29.7327423
19: -22.3698158, 3.6000848, -22.3503189, 3.5698721, -22.8504333, 22.8338928
20: -23.3729839, 1.4001880, -23.3696785, 1.3768466, -19.2733536, 19.2382774
21: -26.9093971, 2.4330764, -26.8846741, 2.3963819, -25.6017227, 25.5749969
22: -28.5963097, 3.3810899, -28.6018772, 3.3226657, -24.8408737, 24.7604179
23: -22.3761559, 5.7355928, -22.3651695, 5.7026110, -22.0779610, 22.0766792
24: -18.3978767, 9.4758081, -18.4182968, 9.4446993, -22.9017639, 22.9260483
25: -23.8908195, 5.4232111, -23.9166832, 5.3573904, -24.4459686, 24.4624329
26: -41.1575012, -0.4306750, -41.1212921, -0.4753890, -30.6957245, 30.6459732
27: -21.6627598, 8.6753654, -21.6714134, 8.5793991, -26.4979324, 26.6071243
28: -24.1915836, 6.1301446, -24.2032585, 6.0591145, -22.0299492, 22.0597916
29: -27.9449196, -0.1724281, -27.9349747, -0.2438061, -24.0791512, 24.0026894
30: -28.1948948, 3.7869997, -28.2200680, 3.7231100, -26.1611938, 26.1949692
31: -22.7619629, 5.0784712, -22.7465439, 5.0447140, -25.1265259, 25.1235085
32: -23.9503002, 2.3671715, -23.9131165, 2.3468559, -21.4078293, 21.3730927
33: -36.4735870, 3.7590466, -36.4344788, 3.6793180, -33.4330597, 33.4266739
34: -37.8953247, -4.6365790, -37.8742828, -4.7346959, -27.8117752, 27.8248978
35: -32.9602318, 0.4341249, -32.9223480, 0.3211346, -28.2249451, 28.2667923
36: -36.8886948, -0.5065689, -36.8410110, -0.6322260, -29.0875473, 29.1490021
37: -44.6265831, -1.5828495, -44.5840988, -1.6948271, -38.8606262, 38.9430542
38: -44.0108490, 3.0598783, -43.9483719, 2.9013491, -40.7778473, 40.8485107
39: -43.6299286, 3.0874062, -43.5899734, 3.0653558, -41.4720917, 41.3930969
40: -32.7748337, 0.0905371, -32.7456589, 0.0470173, -31.1166382, 31.1301651
41: -20.7570496, 7.3824682, -20.7034988, 7.3042412, -26.5617218, 26.5713730
42: -22.9908409, -0.1870122, -22.9793415, -0.1913025, -18.5000305, 18.4718971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1789

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5475778, upper bound: 11.5363042
time: 41.25 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5724299, upper bound: 11.5362585
time: 31.95 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.4085617, 19.0754089, -9.3577013, 19.0723324, -25.2989120, 25.2793045
1: -1.2488661, 22.8703384, -1.2142172, 22.8532066, -19.7922134, 19.8220139
2: -1.6828079, 20.9456673, -1.6246829, 20.9956264, -17.3396225, 17.3269577
3: -9.3771534, 16.5236855, -9.3338470, 16.5793991, -22.1038208, 22.0283203
4: -3.1972675, 22.2475567, -3.1343775, 22.2870007, -21.8105965, 21.7525024
5: -7.8846846, 20.6515007, -7.8216372, 20.6729698, -23.8020401, 23.7975845
6: -28.8296089, -1.3059020, -28.8114738, -1.3539166, -23.2284393, 23.2802963
7: -7.7451830, 21.6700897, -7.6903563, 21.6630840, -23.6265411, 23.6426506
8: -14.8817682, 14.7986059, -14.7776327, 14.8092861, -26.5825424, 26.5381012
9: -5.2739315, 21.3083572, -5.2049041, 21.3047161, -24.3637772, 24.3209686
10: -18.0980625, 17.5931835, -17.9538002, 17.5737343, -31.5028610, 31.3809433
11: -26.8773670, 3.5890946, -26.8965816, 3.5765648, -27.9997177, 28.0207672
12: -34.9121437, -2.2963958, -34.9177399, -2.3221393, -27.2747040, 27.2896461
13: -26.3002014, 15.8034029, -26.2583542, 15.8412142, -34.0438309, 33.9859848
14: -56.0816498, -17.4917984, -56.0045357, -17.5646057, -37.9196014, 37.9420853
15: -14.4241238, 15.5211601, -14.3881092, 15.5250511, -27.9748840, 27.9309692
16: -14.1461477, 20.8617210, -14.1170063, 20.8281384, -31.1316071, 31.1562729
17: -57.9672089, -14.3372345, -57.9465790, -14.4036341, -41.7744446, 41.8445282
18: -21.6504707, 12.2132301, -21.6950874, 12.1695480, -29.6613083, 29.7482986
19: -22.3726845, 3.5991683, -22.3829098, 3.5694494, -22.8569183, 22.8731461
20: -23.3743649, 1.4033952, -23.3943577, 1.3884907, -19.2823524, 19.2674408
21: -26.9131451, 2.4372642, -26.9340763, 2.4099689, -25.6186600, 25.6348991
22: -28.5998478, 3.3802378, -28.6250916, 3.3252835, -24.8548431, 24.7886696
23: -22.3774223, 5.7324562, -22.3769493, 5.6989832, -22.0824203, 22.1058426
24: -18.3998947, 9.4732513, -18.4376793, 9.4406223, -22.9123306, 22.9452782
25: -23.8930130, 5.4317789, -23.9634743, 5.3903017, -24.4772835, 24.5218124
26: -41.1596947, -0.4343958, -41.1279716, -0.4781003, -30.7319946, 30.6571655
27: -21.6661854, 8.6762743, -21.6889133, 8.5851746, -26.5123062, 26.6238785
28: -24.1928864, 6.1279969, -24.2140064, 6.0576568, -22.0416641, 22.0726166
29: -27.9485703, -0.1664665, -27.9948997, -0.2225244, -24.0953178, 24.0704880
30: -28.1973190, 3.8002181, -28.2948246, 3.7656918, -26.1934013, 26.2841911
31: -22.7647362, 5.0801024, -22.7702026, 5.0517001, -25.1367798, 25.1735229
32: -23.9582767, 2.3680675, -23.9400921, 2.3835278, -21.4541092, 21.3936310
33: -36.4762192, 3.7593780, -36.4458466, 3.7163014, -33.4774399, 33.4404907
34: -37.8992310, -4.6341109, -37.8878860, -4.6962833, -27.8846436, 27.8352585
35: -32.9648438, 0.4351997, -32.9393539, 0.3614230, -28.2872391, 28.2806320
36: -36.8941307, -0.5056663, -36.8608475, -0.5938554, -29.1444931, 29.1669540
37: -44.6302605, -1.5866714, -44.6144409, -1.6942396, -38.8896179, 38.9522781
38: -44.0219002, 3.0616870, -43.9898682, 2.9893956, -40.8778687, 40.8909454
39: -43.6337662, 3.0875468, -43.6076202, 3.1061668, -41.5204773, 41.4056168
40: -32.7779388, 0.0905273, -32.7662163, 0.0616794, -31.1484680, 31.1351471
41: -20.7660561, 7.3831654, -20.7352085, 7.3427248, -26.6097794, 26.5981903
42: -22.9921303, -0.1860125, -22.9876480, -0.1747324, -18.5186005, 18.4981232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1789

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5475778, upper bound: 11.5504444
time: 30.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5724299, upper bound: 11.5503987
time: 30.30 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.3661594, 19.0524063, -9.3925943, 19.0660458, -25.2455826, 25.2937279
1: -1.2264290, 22.8386688, -1.2381372, 22.8579826, -19.7655220, 19.7969818
2: -1.6349208, 20.9298058, -1.6314771, 20.9770241, -17.3012772, 17.2771454
3: -9.3587599, 16.5079441, -9.3510180, 16.5691700, -22.0661697, 22.0021057
4: -3.1597314, 22.2388020, -3.1565762, 22.2681160, -21.7532501, 21.7242012
5: -7.8515224, 20.6335087, -7.8632331, 20.6903687, -23.7919464, 23.7831573
6: -28.7805614, -1.3809881, -28.8048782, -1.3418951, -23.1900101, 23.1887360
7: -7.7168322, 21.6543312, -7.7269416, 21.6992588, -23.6335220, 23.6115494
8: -14.8216171, 14.7663631, -14.8310099, 14.8128595, -26.5258026, 26.5112381
9: -5.1965561, 21.2587357, -5.2106934, 21.3037338, -24.2958527, 24.2756577
10: -17.8818378, 17.4198647, -17.8967838, 17.5061150, -31.2271652, 31.1706009
11: -26.7978344, 3.5585418, -26.8241215, 3.5653076, -27.9040298, 27.9212799
12: -34.8876877, -2.3322201, -34.8944092, -2.3183880, -27.2337723, 27.2165718
13: -26.2788849, 15.7689819, -26.2928123, 15.8717690, -34.0525284, 33.9678726
14: -55.9146652, -17.6281376, -55.9394913, -17.6368465, -37.7164993, 37.7319031
15: -14.3691511, 15.5050154, -14.3935986, 15.5227461, -27.8963623, 27.9172745
16: -14.0545969, 20.8143444, -14.0795498, 20.8405495, -31.0646896, 31.0670319
17: -57.8550224, -14.3958797, -57.8678017, -14.4239902, -41.6660690, 41.6617584
18: -21.5956459, 12.1859150, -21.7241821, 12.1937656, -29.5907135, 29.7209320
19: -22.3294334, 3.5990603, -22.3826752, 3.6067641, -22.8247070, 22.8644028
20: -23.3470135, 1.3951151, -23.3991776, 1.4059703, -19.2638321, 19.2656364
21: -26.8501625, 2.4231980, -26.9110069, 2.4302576, -25.5569763, 25.5875626
22: -28.5643082, 3.3578138, -28.6524200, 3.3669236, -24.7798347, 24.8227615
23: -22.3430405, 5.7323661, -22.4070206, 5.7484689, -22.0770874, 22.1115685
24: -18.3710575, 9.4781914, -18.4632893, 9.4852257, -22.9072800, 22.9729805
25: -23.8610058, 5.4100895, -23.9609489, 5.4050279, -24.4133148, 24.4989319
26: -41.1088409, -0.4539561, -41.1836090, -0.4267898, -30.6717682, 30.6965866
27: -21.6011047, 8.6104317, -21.6997299, 8.6218996, -26.4823914, 26.5688705
28: -24.1675396, 6.1053710, -24.2498608, 6.1117530, -22.0362396, 22.0969887
29: -27.9045944, -0.1957500, -27.9781494, -0.2006409, -24.0004425, 24.0655365
30: -28.1682682, 3.7652175, -28.2443314, 3.7562373, -26.1358719, 26.1931076
31: -22.7004223, 5.0729480, -22.7808762, 5.0861239, -25.0983963, 25.1543961
32: -23.9172955, 2.3198643, -23.9278965, 2.3728616, -21.4050903, 21.3309441
33: -36.4046516, 3.6566195, -36.4359360, 3.6869397, -33.3788757, 33.3117447
34: -37.8340759, -4.7409086, -37.8667336, -4.7312703, -27.7602997, 27.7062988
35: -32.8818512, 0.3136530, -32.9151611, 0.3281083, -28.1636047, 28.1388168
36: -36.7917862, -0.6374979, -36.8250275, -0.6242390, -29.0196991, 29.0064392
37: -44.5112305, -1.7048793, -44.5762253, -1.6934347, -38.7758789, 38.7934113
38: -43.8889656, 2.8939915, -43.9389420, 2.9170351, -40.6946106, 40.6876144
39: -43.5729446, 3.0159106, -43.6137390, 3.0827937, -41.4273682, 41.3273087
40: -32.7149086, 0.0027947, -32.7679825, 0.0771189, -31.0871735, 31.0546989
41: -20.6832657, 7.2871361, -20.6951942, 7.3145475, -26.5079269, 26.4629822
42: -22.9860764, -0.2126658, -22.9860344, -0.1787469, -18.4862518, 18.4485970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1789

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5486809, upper bound: 11.5481789
time: 28.53 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5735387, upper bound: 11.5481333
time: 41.96 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.3703938, 19.0526981, -9.4141598, 19.0828857, -25.2719841, 25.3106842
1: -1.2290463, 22.8394394, -1.2565174, 22.8754978, -19.7928925, 19.8160477
2: -1.6417031, 20.9309616, -1.6585889, 21.0124130, -17.3440285, 17.2993507
3: -9.3662882, 16.5096130, -9.3775330, 16.6128788, -22.1213493, 22.0231895
4: -3.1682014, 22.2398815, -3.1919913, 22.3051491, -21.7998886, 21.7513885
5: -7.8478880, 20.6349564, -7.8613763, 20.7072449, -23.8101654, 23.7983322
6: -28.7829628, -1.3803191, -28.8183136, -1.3236566, -23.2226334, 23.2040710
7: -7.7157707, 21.6553059, -7.7404037, 21.7002792, -23.6373596, 23.6458664
8: -14.8222904, 14.7689991, -14.8370247, 14.8286514, -26.5462151, 26.5260239
9: -5.1984549, 21.2621040, -5.2448206, 21.3176460, -24.3081436, 24.3203125
10: -17.8840141, 17.4314041, -17.9800434, 17.5474625, -31.2718658, 31.2651138
11: -26.7997284, 3.5665774, -26.9004116, 3.5908766, -27.9297714, 28.0069733
12: -34.8889008, -2.3304443, -34.9269180, -2.3043909, -27.2497101, 27.2689857
13: -26.2805614, 15.7705402, -26.3045502, 15.8875580, -34.0697861, 33.9940262
14: -55.9169350, -17.6148739, -56.0180473, -17.5846920, -37.7601776, 37.8524933
15: -14.3704605, 15.5072098, -14.4136000, 15.5336962, -27.9131241, 27.9423676
16: -14.0578098, 20.8187637, -14.1495123, 20.8579941, -31.0802307, 31.1417694
17: -57.8575058, -14.3909559, -57.9479141, -14.3968887, -41.6979370, 41.7818832
18: -21.5982132, 12.1881952, -21.7435036, 12.2086344, -29.6427307, 29.7365227
19: -22.3322792, 3.5981128, -22.4152985, 3.6063602, -22.8311996, 22.9036140
20: -23.3483620, 1.3983543, -23.4238853, 1.4175930, -19.2728271, 19.2948074
21: -26.8538895, 2.4274795, -26.9604301, 2.4439352, -25.5739288, 25.6474915
22: -28.5679131, 3.3569729, -28.6756020, 3.3695853, -24.7938538, 24.8510132
23: -22.3443089, 5.7292352, -22.4187698, 5.7448430, -22.0815582, 22.1407280
24: -18.3730812, 9.4756489, -18.4826260, 9.4811630, -22.9178238, 22.9921722
25: -23.8632145, 5.4186134, -24.0077171, 5.4379063, -24.4446831, 24.5582809
26: -41.1110268, -0.4577103, -41.1903076, -0.4295034, -30.7080231, 30.7078323
27: -21.6044979, 8.6113625, -21.7172222, 8.6276321, -26.4967651, 26.5855865
28: -24.1688194, 6.1032190, -24.2606258, 6.1102858, -22.0480042, 22.1098289
29: -27.9082870, -0.1898099, -28.0380440, -0.1794020, -24.0165939, 24.1333771
30: -28.1706467, 3.7784452, -28.3191299, 3.7988391, -26.1680450, 26.2823219
31: -22.7032242, 5.0746493, -22.8045082, 5.0931296, -25.1086044, 25.2043839
32: -23.9252815, 2.3207862, -23.9548149, 2.4095333, -21.4513626, 21.3514824
33: -36.4073334, 3.6569891, -36.4472351, 3.7239122, -33.4231567, 33.3255310
34: -37.8380089, -4.7384539, -37.8803024, -4.6928768, -27.8331909, 27.7166748
35: -32.8864517, 0.3147120, -32.9321861, 0.3683476, -28.2258987, 28.1526947
36: -36.7972527, -0.6366239, -36.8449287, -0.5858564, -29.0766678, 29.0243683
37: -44.5149460, -1.7087064, -44.6065521, -1.6927905, -38.8048401, 38.8026352
38: -43.9000931, 2.8958526, -43.9803848, 3.0050602, -40.7945557, 40.7300339
39: -43.5767021, 3.0160851, -43.6313324, 3.1235466, -41.4757843, 41.3398361
40: -32.7180138, 0.0027759, -32.7885056, 0.0917675, -31.1189728, 31.0596924
41: -20.6922855, 7.2878160, -20.7268810, 7.3530397, -26.5559692, 26.4898071
42: -22.9873943, -0.2116902, -22.9943295, -0.1621468, -18.5048180, 18.4748497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1789

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5486809, upper bound: 11.5623209
time: 32.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5735387, upper bound: 11.5622752
time: 33.29 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.4204855, 19.0755463, -9.3970566, 19.0665474, -25.3042145, 25.3278046
1: -1.2589216, 22.8698196, -1.2390881, 22.8586464, -19.8001747, 19.8410683
2: -1.6866753, 20.9447575, -1.6338198, 20.9803009, -17.3298721, 17.3349037
3: -9.3831120, 16.5228653, -9.3525219, 16.5701561, -22.0862999, 22.0420036
4: -3.2061467, 22.2465363, -3.1594844, 22.2691727, -21.7971191, 21.7761345
5: -7.9010353, 20.6512108, -7.8667111, 20.6932411, -23.8274078, 23.8147659
6: -28.8331127, -1.3027363, -28.8175850, -1.3399367, -23.2308655, 23.2818451
7: -7.7628317, 21.6696434, -7.7304349, 21.7014351, -23.6780319, 23.6507111
8: -14.8986797, 14.7965021, -14.8320684, 14.8177471, -26.5919724, 26.5760345
9: -5.2824287, 21.3052959, -5.2127190, 21.3133907, -24.3925552, 24.3209381
10: -18.1027374, 17.5826206, -17.8992119, 17.5486698, -31.4878922, 31.3135147
11: -26.8757229, 3.5850921, -26.8264275, 3.5686846, -27.9957886, 27.9507446
12: -34.9137115, -2.2932787, -34.8959694, -2.3131642, -27.2923355, 27.2515182
13: -26.3126945, 15.8029833, -26.2969837, 15.8748951, -34.0940933, 34.0050507
14: -56.0819054, -17.5020905, -55.9436760, -17.6039162, -37.9171448, 37.8393326
15: -14.4279537, 15.5220470, -14.3972816, 15.5255976, -27.9720688, 27.9385147
16: -14.1523600, 20.8574715, -14.0833483, 20.8501148, -31.1657867, 31.1101837
17: -57.9656677, -14.3394203, -57.8708611, -14.4131527, -41.7987823, 41.7268143
18: -21.6490479, 12.2227936, -21.7292290, 12.1955128, -29.6437225, 29.8022575
19: -22.3707314, 3.6116390, -22.3866730, 3.6077957, -22.8793716, 22.8816338
20: -23.3734493, 1.4089684, -23.4017601, 1.4070981, -19.2992744, 19.2825508
21: -26.9100094, 2.4435222, -26.9155407, 2.4322288, -25.6307526, 25.6132545
22: -28.5969143, 3.3952720, -28.6585159, 3.3686464, -24.8730927, 24.8321228
23: -22.3766232, 5.7495542, -22.4089413, 5.7493429, -22.1128387, 22.1362915
24: -18.3989353, 9.4883490, -18.4684467, 9.4860678, -22.9354935, 22.9911537
25: -23.8917732, 5.4378581, -23.9647293, 5.4070926, -24.4855118, 24.5272026
26: -41.1584778, -0.4133573, -41.1866837, -0.4187508, -30.7346268, 30.7346497
27: -21.6636162, 8.6883354, -21.7146130, 8.6243248, -26.5375137, 26.6642685
28: -24.1920967, 6.1469359, -24.2541676, 6.1139650, -22.0691223, 22.1292648
29: -27.9452667, -0.1594048, -27.9844208, -0.1993684, -24.1119881, 24.0650864
30: -28.1952744, 3.7965474, -28.2476273, 3.7592380, -26.1895561, 26.2312469
31: -22.7632809, 5.0910311, -22.7848892, 5.0878820, -25.1669922, 25.1763649
32: -23.9565086, 2.3701742, -23.9375515, 2.3748643, -21.4410858, 21.3948441
33: -36.4764175, 3.7601485, -36.4525223, 3.6881728, -33.4487381, 33.4459076
34: -37.8959961, -4.6358418, -37.8836288, -4.7284579, -27.8185577, 27.8380203
35: -32.9617538, 0.4357319, -32.9357452, 0.3293743, -28.2353287, 28.2839127
36: -36.8898392, -0.5042458, -36.8509636, -0.6227651, -29.1025696, 29.1664886
37: -44.6297913, -1.5830717, -44.6053696, -1.6926637, -38.8801727, 38.9610825
38: -44.0129662, 3.0639234, -43.9686279, 2.9199023, -40.8082275, 40.8862000
39: -43.6380119, 3.0875211, -43.6272087, 3.0846934, -41.4960327, 41.4262772
40: -32.7821655, 0.0919240, -32.7824554, 0.0778985, -31.1545258, 31.1632004
41: -20.7596321, 7.3844271, -20.7146721, 7.3163280, -26.5757141, 26.5837021
42: -22.9932327, -0.1848862, -22.9878101, -0.1763232, -18.5184708, 18.4800186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1789

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5486809, upper bound: 11.5594390
time: 31.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5735387, upper bound: 11.5593930
time: 40.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.4247351, 19.0758114, -9.4186497, 19.0833664, -25.3306274, 25.3448029
1: -1.2615433, 22.8705711, -1.2574639, 22.8761940, -19.8275681, 19.8601227
2: -1.6934276, 20.9459419, -1.6609566, 21.0157490, -17.3726387, 17.3571053
3: -9.3906565, 16.5245399, -9.3789854, 16.6138382, -22.1414490, 22.0630608
4: -3.2146564, 22.2476387, -3.1949253, 22.3061810, -21.8437576, 21.8033104
5: -7.8973646, 20.6526718, -7.8648438, 20.7101326, -23.8456078, 23.8299294
6: -28.8355122, -1.3020725, -28.8310223, -1.3216715, -23.2635040, 23.2971840
7: -7.7617254, 21.6705799, -7.7439232, 21.7024765, -23.6818542, 23.6850166
8: -14.8993673, 14.7991438, -14.8381271, 14.8335304, -26.6123276, 26.5907745
9: -5.2842999, 21.3086872, -5.2468309, 21.3273315, -24.4048309, 24.3656387
10: -18.1049118, 17.5941620, -17.9825039, 17.5899963, -31.5325546, 31.4079819
11: -26.8776474, 3.5930963, -26.9027271, 3.5942497, -28.0215759, 28.0363770
12: -34.9149246, -2.2915378, -34.9284286, -2.2991014, -27.3082657, 27.3039284
13: -26.3144035, 15.8044758, -26.3086605, 15.8907137, -34.1113205, 34.0311432
14: -56.0841675, -17.4888458, -56.0222054, -17.5517197, -37.9608841, 37.9599991
15: -14.4292107, 15.5242519, -14.4172859, 15.5365791, -27.9888382, 27.9636078
16: -14.1555223, 20.8619232, -14.1533070, 20.8676281, -31.1812744, 31.1849136
17: -57.9681816, -14.3345051, -57.9510460, -14.3860054, -41.8307037, 41.8468781
18: -21.6515732, 12.2250261, -21.7485428, 12.2104263, -29.6957397, 29.8178520
19: -22.3735828, 3.6107235, -22.4192562, 3.6074126, -22.8858566, 22.9208641
20: -23.3748112, 1.4122140, -23.4264297, 1.4187579, -19.3082657, 19.3116951
21: -26.9136906, 2.4478092, -26.9649906, 2.4458678, -25.6476593, 25.6731377
22: -28.6004829, 3.3944049, -28.6817055, 3.3713009, -24.8870850, 24.8603745
23: -22.3778496, 5.7463684, -22.4207325, 5.7457128, -22.1173096, 22.1654663
24: -18.4010029, 9.4858217, -18.4877434, 9.4820461, -22.9460907, 23.0103455
25: -23.8939781, 5.4464536, -24.0114517, 5.4399548, -24.5168762, 24.5865555
26: -41.1606827, -0.4170170, -41.1934052, -0.4213929, -30.7708588, 30.7458878
27: -21.6670494, 8.6892576, -21.7320747, 8.6301060, -26.5518723, 26.6809692
28: -24.1933689, 6.1447387, -24.2649326, 6.1125174, -22.0808258, 22.1420975
29: -27.9489498, -0.1534359, -28.0443249, -0.1781173, -24.1281395, 24.1328964
30: -28.1976204, 3.8097248, -28.3223991, 3.8018575, -26.2217598, 26.3204727
31: -22.7660313, 5.0927162, -22.8085575, 5.0948462, -25.1772385, 25.2263718
32: -23.9644775, 2.3710570, -23.9644947, 2.4115810, -21.4873581, 21.4154053
33: -36.4790039, 3.7605104, -36.4638443, 3.7251225, -33.4931030, 33.4596863
34: -37.8998833, -4.6333785, -37.8971901, -4.6900959, -27.8914185, 27.8483963
35: -32.9663429, 0.4368038, -32.9526634, 0.3696246, -28.2976303, 28.2977448
36: -36.8952599, -0.5033340, -36.8707695, -0.5843887, -29.1594543, 29.1844482
37: -44.6334839, -1.5869250, -44.6356812, -1.6920485, -38.9091034, 38.9703217
38: -44.0240288, 3.0657883, -44.0100784, 3.0079141, -40.9082336, 40.9286575
39: -43.6418762, 3.0877390, -43.6448822, 3.1254230, -41.5444336, 41.4388046
40: -32.7852859, 0.0919609, -32.8029785, 0.0925138, -31.1862717, 31.1681976
41: -20.7686539, 7.3851104, -20.7463646, 7.3547912, -26.6237717, 26.6105423
42: -22.9945145, -0.1838837, -22.9960976, -0.1597707, -18.5370255, 18.5062714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=171, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1789

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5486809, upper bound: 11.5735847
time: 32.36 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5735387, upper bound: 11.5735387
time: 36.50 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 71.29 seconds
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5741336
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5241337, upper bound: 11.5741336
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5318518, upper bound: 11.5747583
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5459945, upper bound: 11.5747583
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5269404, upper bound: 11.5729292
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5517287, upper bound: 11.5728847
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5475778, upper bound: 11.5250955
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5724299, upper bound: 11.5250499
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5475778, upper bound: 11.5392302
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5724299, upper bound: 11.5391846
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5475778, upper bound: 11.5363042
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5724299, upper bound: 11.5362585
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5475778, upper bound: 11.5504444
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5724299, upper bound: 11.5503987
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5486809, upper bound: 11.5481789
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5735387, upper bound: 11.5481333
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5486809, upper bound: 11.5623209
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5735387, upper bound: 11.5622752
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5486809, upper bound: 11.5594390
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5735387, upper bound: 11.5593930
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5486809, upper bound: 11.5735847
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 71.29
Output dim: 2, lower bound: -11.5735387, upper bound: 11.5735387

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.2112160, 18.9977741, -9.3776159, 19.1057453, -25.1735764, 25.2264099
1: -1.1139188, 22.7917328, -1.2244997, 22.9006748, -19.7419319, 19.7416306
2: -1.5687544, 20.8895626, -1.6327729, 21.0152245, -17.2702065, 17.2754288
3: -9.2786016, 16.4378777, -9.3476734, 16.6173363, -22.0258942, 21.9473572
4: -3.0486417, 22.2106133, -3.1570301, 22.2931252, -21.6687469, 21.7472115
5: -7.7718415, 20.5585136, -7.8423400, 20.7424164, -23.7694244, 23.6978531
6: -28.7732010, -1.4060087, -28.8311100, -1.3558340, -23.1360321, 23.1980286
7: -7.6263680, 21.6009827, -7.7139654, 21.7391376, -23.5976944, 23.5659714
8: -14.7105713, 14.6944332, -14.7936306, 14.8579273, -26.4657898, 26.4447937
9: -5.1735516, 21.2429886, -5.1961331, 21.3428440, -24.3398743, 24.2075081
10: -17.9655247, 17.4852028, -17.8789444, 17.6042061, -31.4138794, 31.1889801
11: -26.7816372, 3.4897137, -26.8867874, 3.5740948, -27.9071884, 27.9226913
12: -34.8681335, -2.4127908, -34.8982582, -2.2844849, -27.2858734, 27.1202583
13: -26.2380772, 15.6545372, -26.2979050, 15.9492302, -34.1004257, 33.8452225
14: -55.9178810, -17.6660614, -55.9277420, -17.4984856, -37.8799133, 37.5376053
15: -14.2907591, 15.4549351, -14.3706131, 15.5376921, -27.8613205, 27.8425598
16: -14.0367489, 20.7685795, -14.0871201, 20.9171467, -31.1429901, 31.0287399
17: -57.8295059, -14.5077562, -57.8695526, -14.3105536, -41.7790833, 41.4633179
18: -21.5402279, 12.1335249, -21.7704754, 12.1834164, -29.5027466, 29.7604713
19: -22.2298279, 3.4993706, -22.4650555, 3.5882878, -22.7220459, 22.8585968
20: -23.2359428, 1.2813120, -23.4804001, 1.3912239, -19.1451111, 19.2847176
21: -26.7571316, 2.3148472, -27.0027046, 2.4179063, -25.4657745, 25.5911484
22: -28.4363861, 3.2886858, -28.7647419, 3.3540971, -24.6992989, 24.8696365
23: -22.2636642, 5.6453152, -22.4840870, 5.7363486, -21.9903717, 22.1310654
24: -18.2400856, 9.3809137, -18.5671749, 9.4644785, -22.7562790, 23.0170517
25: -23.7892284, 5.3015485, -24.0300560, 5.4063873, -24.3905106, 24.4808464
26: -41.0071487, -0.5576935, -41.2882729, -0.4385109, -30.5594330, 30.7444763
27: -21.5237846, 8.5907307, -21.8119240, 8.6103163, -26.3729782, 26.6747437
28: -24.0663033, 6.0150251, -24.3363400, 6.0912924, -21.9198227, 22.1094551
29: -27.8007278, -0.2319264, -28.0862217, -0.1907309, -23.9732628, 24.0926285
30: -28.0736465, 3.6489520, -28.3273411, 3.7686198, -26.0929108, 26.1984406
31: -22.6378937, 4.9799099, -22.8427773, 5.0703764, -25.0283127, 25.1372757
32: -23.8763275, 2.2879031, -23.9688015, 2.3619180, -21.3348274, 21.3735695
33: -36.3329163, 3.5938015, -36.4905472, 3.6419725, -33.2585449, 33.3722763
34: -37.7801056, -4.8008056, -37.9289703, -4.7519183, -27.6640091, 27.7856979
35: -32.8379745, 0.2688212, -32.9773865, 0.2959080, -28.0718765, 28.2008514
36: -36.7583160, -0.6774473, -36.9076309, -0.6544385, -28.9334869, 29.0849380
37: -44.4786072, -1.7019968, -44.6609421, -1.7138996, -38.6978149, 38.9006424
38: -43.8371048, 2.8491530, -44.0342178, 2.8851352, -40.5831909, 40.7788086
39: -43.4797821, 2.9214454, -43.6527863, 3.0476160, -41.2811584, 41.3610382
40: -32.6789398, 0.0226741, -32.8039703, 0.0801480, -31.0421906, 31.1120529
41: -20.6573238, 7.2915382, -20.7626190, 7.2986689, -26.3974762, 26.5403595
42: -22.9644318, -0.2254522, -22.9921551, -0.1777616, -18.4834366, 18.4364853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=171, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=198, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1789

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5074963, upper bound: 11.5480066
time: 26.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5074533, upper bound: 11.5728183
time: 37.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.2327566, 19.0145988, -9.3818035, 19.1060066, -25.1905518, 25.2528305
1: -1.1322937, 22.8092747, -1.2270942, 22.9014359, -19.7609673, 19.7689743
2: -1.5958731, 20.9249935, -1.6394966, 21.0163708, -17.2924004, 17.3181801
3: -9.3051014, 16.4816399, -9.3551922, 16.6190376, -22.0469513, 22.0025330
4: -3.0840516, 22.2476521, -3.1655087, 22.2942314, -21.6958847, 21.7938499
5: -7.7699642, 20.5754242, -7.8386679, 20.7438736, -23.7845840, 23.7160606
6: -28.7866402, -1.3877082, -28.8335571, -1.3551407, -23.1513901, 23.2307663
7: -7.6398635, 21.6020565, -7.7129126, 21.7400970, -23.6320038, 23.5698509
8: -14.7165699, 14.7102575, -14.7943096, 14.8605766, -26.4805679, 26.4652100
9: -5.2076406, 21.2568703, -5.1980333, 21.3462334, -24.3844681, 24.2198181
10: -18.0487328, 17.5265331, -17.8811646, 17.6157684, -31.5083008, 31.2337036
11: -26.8579750, 3.5152507, -26.8886986, 3.5820622, -27.9928436, 27.9484482
12: -34.9006233, -2.3987837, -34.8994865, -2.2827315, -27.3382721, 27.1362572
13: -26.2498016, 15.6703434, -26.2996101, 15.9506989, -34.1265335, 33.8624039
14: -55.9964180, -17.6139050, -55.9300156, -17.4851913, -38.0005646, 37.5813217
15: -14.3107691, 15.4658947, -14.3719091, 15.5398979, -27.8864441, 27.8593674
16: -14.1066189, 20.7860794, -14.0903330, 20.9216003, -31.2176666, 31.0442429
17: -57.9096985, -14.4806681, -57.8721466, -14.3056412, -41.8992004, 41.4951782
18: -21.5595856, 12.1484070, -21.7730236, 12.1857147, -29.5184326, 29.8124313
19: -22.2624741, 3.4989159, -22.4678974, 3.5874362, -22.7613297, 22.8650589
20: -23.2606544, 1.2929187, -23.4817924, 1.3944485, -19.1742935, 19.2936897
21: -26.8065853, 2.3284740, -27.0064011, 2.4221556, -25.5257187, 25.6080589
22: -28.4597931, 3.2912796, -28.7682743, 3.3532410, -24.7277412, 24.8836327
23: -22.2754059, 5.6417017, -22.4853439, 5.7332253, -22.0194931, 22.1355476
24: -18.2595177, 9.3768272, -18.5692158, 9.4619188, -22.7756805, 23.0275803
25: -23.8360195, 5.3343720, -24.0322418, 5.4149828, -24.4498749, 24.5121460
26: -41.0138779, -0.5603004, -41.2904663, -0.4422135, -30.5706940, 30.7807693
27: -21.5412312, 8.5964680, -21.8152847, 8.6112490, -26.3896790, 26.6890488
28: -24.0770416, 6.0135112, -24.3376160, 6.0891023, -21.9326744, 22.1211777
29: -27.8606529, -0.2106564, -28.0899277, -0.1848074, -24.0411453, 24.1087646
30: -28.1484756, 3.6915231, -28.3297443, 3.7818220, -26.1820831, 26.2306633
31: -22.6615658, 4.9868832, -22.8455448, 5.0720625, -25.0783157, 25.1474876
32: -23.9032192, 2.3245318, -23.9767609, 2.3627863, -21.3553543, 21.4198570
33: -36.3442116, 3.6308136, -36.4931870, 3.6422997, -33.2722778, 33.4165955
34: -37.7936897, -4.7623754, -37.9329147, -4.7494707, -27.6744003, 27.8586197
35: -32.8549728, 0.3090525, -32.9819756, 0.2969832, -28.0857391, 28.2631989
36: -36.7781296, -0.6390080, -36.9131050, -0.6535678, -28.9514008, 29.1419144
37: -44.5090103, -1.7013941, -44.6646271, -1.7177534, -38.7071838, 38.9295654
38: -43.8785248, 2.9372382, -44.0452271, 2.8869905, -40.6256714, 40.8787918
39: -43.4974365, 2.9621501, -43.6565933, 3.0477519, -41.2937317, 41.4094620
40: -32.6995087, 0.0373328, -32.8070908, 0.0801682, -31.0472412, 31.1438141
41: -20.6889439, 7.3300686, -20.7716484, 7.2993283, -26.4242859, 26.5883636
42: -22.9727306, -0.2088966, -22.9934235, -0.1767788, -18.5096130, 18.4550667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=171, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=198, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1789

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5228325, upper bound: 11.5480064
time: 49.94 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5227895, upper bound: 11.5728183
time: 39.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.3416576, 19.0331421, -9.4197121, 19.1071014, -25.2702484, 25.3093224
1: -1.2009397, 22.8248310, -1.2536211, 22.9016495, -19.7983360, 19.8084488
2: -1.6292112, 20.9071808, -1.6526070, 21.0160961, -17.3210831, 17.3136673
3: -9.3358097, 16.4671345, -9.3672848, 16.6202202, -22.0862427, 22.0067749
4: -3.1340570, 22.2198334, -3.1845255, 22.2942543, -21.7513084, 21.7822418
5: -7.8572855, 20.5950470, -7.8710556, 20.7445126, -23.8428802, 23.7630539
6: -28.8110046, -1.3269644, -28.8345280, -1.3297663, -23.2043228, 23.2745628
7: -7.7099838, 21.6277161, -7.7414441, 21.7403240, -23.6725540, 23.6211929
8: -14.8381004, 14.7483854, -14.8368378, 14.8604240, -26.5752182, 26.5415573
9: -5.2488022, 21.2699127, -5.2211986, 21.3450699, -24.4005966, 24.2592087
10: -18.0630035, 17.5165348, -17.9111862, 17.6079102, -31.5139084, 31.2552719
11: -26.8052731, 3.5336690, -26.8922119, 3.5883794, -27.9439468, 27.9717712
12: -34.8840408, -2.3525028, -34.9029617, -2.2661347, -27.3213272, 27.1922531
13: -26.2547646, 15.7070656, -26.3020840, 15.9631424, -34.1327896, 33.9131393
14: -56.0101700, -17.6211319, -55.9566078, -17.4934902, -37.9761734, 37.7188034
15: -14.3975954, 15.5008583, -14.4049797, 15.5400982, -27.9558868, 27.9231873
16: -14.0941887, 20.7870522, -14.1058903, 20.9178143, -31.1898499, 31.0552902
17: -57.9001846, -14.4613209, -57.8922729, -14.3034782, -41.8643875, 41.6156464
18: -21.5912113, 12.1843805, -21.7763672, 12.2004986, -29.5793991, 29.8233032
19: -22.2793751, 3.5614109, -22.4701214, 3.6092660, -22.7934723, 22.9199333
20: -23.2890835, 1.3488612, -23.4827118, 1.4137719, -19.2233772, 19.3193283
21: -26.8100548, 2.3824337, -27.0083370, 2.4404964, -25.5409241, 25.6550217
22: -28.4819374, 3.3400147, -28.7678814, 3.3707440, -24.7626419, 24.8983765
23: -22.2949924, 5.6932912, -22.4867840, 5.7520194, -22.0362625, 22.1669235
24: -18.2927532, 9.4417934, -18.5695362, 9.4854584, -22.8299026, 23.0534630
25: -23.8186569, 5.3697820, -24.0326843, 5.4278774, -24.4369888, 24.5360870
26: -41.0495262, -0.4886398, -41.2908669, -0.4149437, -30.6294250, 30.7860413
27: -21.5560284, 8.6342773, -21.8158302, 8.6251078, -26.4293900, 26.7206345
28: -24.1046696, 6.0875268, -24.3384342, 6.1151443, -21.9824219, 22.1708755
29: -27.8335171, -0.2123175, -28.0911636, -0.1853771, -24.0190163, 24.1251068
30: -28.1105156, 3.7145700, -28.3293228, 3.7894962, -26.1390343, 26.2421494
31: -22.6918449, 5.0462089, -22.8480377, 5.0926161, -25.1061325, 25.1930122
32: -23.9165764, 2.3519588, -23.9716778, 2.3830001, -21.4007568, 21.4194756
33: -36.4246826, 3.7385454, -36.4948311, 3.6919150, -33.4002838, 33.4604416
34: -37.8446503, -4.6869383, -37.9309769, -4.7133002, -27.7704926, 27.8555984
35: -32.9101486, 0.4022317, -32.9803963, 0.3419113, -28.1891861, 28.3041382
36: -36.8236847, -0.5532198, -36.9106216, -0.6120777, -29.0402451, 29.1884689
37: -44.5531006, -1.6152067, -44.6675835, -1.6842427, -38.8068848, 38.9936676
38: -43.9317398, 3.0096111, -44.0400467, 2.9401531, -40.7301178, 40.9064255
39: -43.5878677, 3.0685415, -43.6593781, 3.0985098, -41.4384918, 41.4556656
40: -32.7367325, 0.0702212, -32.8105087, 0.0959892, -31.1208954, 31.1687622
41: -20.6984444, 7.3561554, -20.7668915, 7.3200860, -26.4973679, 26.6091690
42: -22.9813366, -0.2022724, -22.9971199, -0.1713049, -18.5099258, 18.4726562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=171, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=196, inp2_unstable=198, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1789

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5305969, upper bound: 11.5486064
time: 29.30 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5305513, upper bound: 11.5575204
time: 221.23 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.3631639, 19.0499992, -9.4239159, 19.1073723, -25.2872391, 25.3357544
1: -1.2192922, 22.8423805, -1.2562580, 22.9024258, -19.8173599, 19.8358040
2: -1.6563678, 20.9425926, -1.6593604, 21.0172653, -17.3432808, 17.3564339
3: -9.3623133, 16.5108948, -9.3748302, 16.6219063, -22.1072693, 22.0619431
4: -3.1694703, 22.2568588, -3.1930132, 22.2953358, -21.7784500, 21.8288727
5: -7.8554406, 20.6119690, -7.8673840, 20.7459717, -23.8580704, 23.7812881
6: -28.8244076, -1.3086638, -28.8369331, -1.3290911, -23.2196732, 23.3072510
7: -7.7235208, 21.6287708, -7.7403822, 21.7412910, -23.7068405, 23.6250381
8: -14.8441830, 14.7641945, -14.8375244, 14.8630848, -26.5900116, 26.5619659
9: -5.2828979, 21.2838001, -5.2230654, 21.3484459, -24.4452057, 24.2714539
10: -18.1462498, 17.5578022, -17.9133682, 17.6194649, -31.6083221, 31.2999649
11: -26.8815784, 3.5592113, -26.8941193, 3.5963798, -28.0296021, 27.9975281
12: -34.9165192, -2.3384204, -34.9041519, -2.2643929, -27.3737488, 27.2082405
13: -26.2665119, 15.7229137, -26.3037910, 15.9646854, -34.1589050, 33.9303894
14: -56.0886650, -17.5690041, -55.9588852, -17.4801846, -38.0968170, 37.7625122
15: -14.4176102, 15.5118208, -14.4062653, 15.5422983, -27.9810333, 27.9400101
16: -14.1641369, 20.8045025, -14.1090822, 20.9222946, -31.2645035, 31.0708008
17: -57.9803810, -14.4342136, -57.8948135, -14.2985601, -41.9845581, 41.6475296
18: -21.6105671, 12.1993093, -21.7788773, 12.2027407, -29.5950165, 29.8753052
19: -22.3120079, 3.5609746, -22.4729767, 3.6083751, -22.8327713, 22.9263802
20: -23.3137875, 1.3604937, -23.4840813, 1.4169657, -19.2525406, 19.3283195
21: -26.8594856, 2.3960969, -27.0120468, 2.4447265, -25.6009216, 25.6719627
22: -28.5052490, 3.3426499, -28.7714386, 3.3699038, -24.7910957, 24.9123535
23: -22.3068047, 5.6896739, -22.4880409, 5.7488747, -22.0654068, 22.1714134
24: -18.3122101, 9.4377317, -18.5715637, 9.4829168, -22.8493118, 23.0639915
25: -23.8654518, 5.4026003, -24.0348587, 5.4364405, -24.4963989, 24.5673866
26: -41.0562668, -0.4913192, -41.2930603, -0.4186053, -30.6407394, 30.8222504
27: -21.5735035, 8.6400318, -21.8192253, 8.6259995, -26.4461441, 26.7349930
28: -24.1154194, 6.0860872, -24.3397274, 6.1129513, -21.9952698, 22.1825714
29: -27.8934326, -0.1910570, -28.0948715, -0.1794313, -24.0868454, 24.1412659
30: -28.1853371, 3.7571387, -28.3317604, 3.8027074, -26.2282562, 26.2743454
31: -22.7155037, 5.0531631, -22.8508186, 5.0942664, -25.1561279, 25.2032776
32: -23.9434853, 2.3885920, -23.9796791, 2.3839037, -21.4212456, 21.4657593
33: -36.4359436, 3.7755413, -36.4974060, 3.6922884, -33.4140472, 33.5047684
34: -37.8582458, -4.6485372, -37.9349289, -4.7108455, -27.7808228, 27.9284821
35: -32.9270744, 0.4425173, -32.9849739, 0.3430581, -28.2030640, 28.3665009
36: -36.8434982, -0.5147800, -36.9160385, -0.6112137, -29.0581284, 29.2453842
37: -44.5834541, -1.6145835, -44.6712532, -1.6880808, -38.8162842, 39.0226440
38: -43.9732170, 3.0977283, -44.0511436, 2.9420166, -40.7725525, 41.0063171
39: -43.6054878, 3.1092806, -43.6632347, 3.0986919, -41.4510803, 41.5040512
40: -32.7572937, 0.0848610, -32.8136520, 0.0959911, -31.1259460, 31.2005539
41: -20.7300873, 7.3946190, -20.7758884, 7.3207588, -26.5241699, 26.6571732
42: -22.9896603, -0.1857283, -22.9984150, -0.1702940, -18.5361443, 18.4912529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=171, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=198, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1789

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5447391, upper bound: 11.5486065
time: 30.32 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5446937, upper bound: 11.5734616
time: 33.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.2539787, 19.0391617, -9.3765316, 19.0820122, -25.1931763, 25.2606888
1: -1.1435833, 22.8370247, -1.2282915, 22.8752060, -19.7398262, 19.7928734
2: -1.6125767, 20.9280281, -1.6411116, 21.0148411, -17.3015671, 17.3151665
3: -9.3154640, 16.4937172, -9.3593235, 16.6109333, -22.0632324, 22.0018806
4: -3.1045542, 22.2380409, -3.1674409, 22.3051033, -21.7371254, 21.7612610
5: -7.7850304, 20.6142330, -7.8361654, 20.7080727, -23.7452011, 23.7626190
6: -28.7961845, -1.4051771, -28.8276348, -1.3477540, -23.1918411, 23.1957741
7: -7.6485538, 21.6429234, -7.7164288, 21.7013226, -23.5771027, 23.6287460
8: -14.7469521, 14.7429085, -14.7948866, 14.8310308, -26.4783669, 26.4854202
9: -5.1876879, 21.2801075, -5.2217817, 21.3251286, -24.3224030, 24.3122635
10: -17.9811096, 17.5602074, -17.9502373, 17.5863132, -31.4062271, 31.3390198
11: -26.8442249, 3.5351529, -26.8973083, 3.5799198, -27.9662094, 27.9739685
12: -34.8959236, -2.3732057, -34.9237061, -2.3174839, -27.2541962, 27.2193146
13: -26.2791061, 15.7484026, -26.3045158, 15.8767204, -34.0589218, 33.9600601
14: -55.9584007, -17.5371456, -55.9933205, -17.5566959, -37.8305054, 37.7756958
15: -14.3050632, 15.4774456, -14.3828983, 15.5341682, -27.8766861, 27.8775787
16: -14.0632486, 20.8427029, -14.1345892, 20.8669472, -31.1000671, 31.1574936
17: -57.8463097, -14.3864202, -57.9283447, -14.3930454, -41.6899796, 41.6913910
18: -21.5916901, 12.1662560, -21.7427139, 12.1933613, -29.6133881, 29.7409515
19: -22.3194962, 3.5304368, -22.4141521, 3.5864284, -22.8095932, 22.8412933
20: -23.3202400, 1.3227987, -23.4241276, 1.3961718, -19.2285538, 19.2558823
21: -26.8549614, 2.3629353, -26.9593048, 2.4233019, -25.5588303, 25.5936317
22: -28.5503616, 3.3273058, -28.6785011, 3.3546553, -24.8093948, 24.8162270
23: -22.3434010, 5.6808577, -22.4180336, 5.7300878, -22.0683060, 22.1119156
24: -18.3456631, 9.4133015, -18.4853706, 9.4610271, -22.8700027, 22.9618340
25: -23.8621712, 5.3651609, -24.0088177, 5.4185276, -24.4623375, 24.5188065
26: -41.1150818, -0.5147991, -41.1908455, -0.4450274, -30.6979370, 30.6755829
27: -21.6312618, 8.6291456, -21.7281799, 8.6153240, -26.4925613, 26.6176529
28: -24.1535187, 6.0514216, -24.2627831, 6.0886803, -22.0168114, 22.0597115
29: -27.9087009, -0.1823857, -28.0393772, -0.1834747, -24.0614548, 24.0972481
30: -28.1563282, 3.7384665, -28.3203850, 3.7809625, -26.1623497, 26.2717209
31: -22.7080460, 5.0094891, -22.8032551, 5.0726414, -25.0951309, 25.1535683
32: -23.9228630, 2.2764518, -23.9615936, 2.3904314, -21.4201279, 21.3387413
33: -36.3835144, 3.5833726, -36.4596176, 3.6751914, -33.3474121, 33.3390503
34: -37.8330460, -4.7833419, -37.8951797, -4.7287140, -27.7829971, 27.7421646
35: -32.8910217, 0.2713032, -32.9496994, 0.3235822, -28.1774979, 28.1621628
36: -36.8268623, -0.6656160, -36.8678207, -0.6267419, -29.0505447, 29.0426636
37: -44.5534859, -1.7081909, -44.6290512, -1.7217398, -38.7950134, 38.8427429
38: -43.9243431, 2.8531289, -44.0042343, 2.9528384, -40.7579498, 40.7486725
39: -43.5274200, 2.9017448, -43.6381912, 3.0744991, -41.3813324, 41.3053589
40: -32.7232437, 0.0111141, -32.7964401, 0.0767176, -31.1034927, 31.0784187
41: -20.7237282, 7.2896390, -20.7420788, 7.3333731, -26.5205765, 26.5106049
42: -22.9753265, -0.2317715, -22.9911346, -0.1662707, -18.5056076, 18.4452248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1789

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5269404, upper bound: 11.5480690
time: 29.16 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5269404, upper bound: 11.5728847
time: 39.40 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.2929935, 19.0953522, -9.3729467, 19.0818882, -25.2347717, 25.3162689
1: -1.1758070, 22.8780136, -1.2256289, 22.8750305, -19.7738380, 19.8339996
2: -1.6313772, 20.9625702, -1.6393347, 21.0146904, -17.3235626, 17.3258972
3: -9.3315973, 16.5239716, -9.3577785, 16.6106682, -22.0805511, 22.0307236
4: -3.1325359, 22.2549133, -3.1652923, 22.3050079, -21.7682304, 21.7572327
5: -7.8094621, 20.6709595, -7.8337712, 20.7078114, -23.7725220, 23.8178101
6: -28.8368359, -1.3794022, -28.8273621, -1.3497972, -23.2077408, 23.2238693
7: -7.6782379, 21.6845551, -7.7139263, 21.7010689, -23.6083107, 23.6695023
8: -14.7734060, 14.7564240, -14.7927885, 14.8305235, -26.5054398, 26.4906540
9: -5.2084022, 21.3271961, -5.2198348, 21.3249969, -24.3437653, 24.3572197
10: -18.0086861, 17.6012478, -17.9477940, 17.5859795, -31.4355850, 31.3775101
11: -26.8588028, 3.5485573, -26.8962402, 3.5782814, -27.9790955, 27.9876938
12: -34.9100647, -2.3495708, -34.9233093, -2.3212929, -27.2510986, 27.2558479
13: -26.2983532, 15.7781582, -26.3026123, 15.8762465, -34.0763702, 33.9935989
14: -55.9929886, -17.4460697, -55.9903870, -17.5572071, -37.8591003, 37.8639984
15: -14.3300924, 15.4808321, -14.3811378, 15.5339622, -27.9014587, 27.8796463
16: -14.0953188, 20.9167671, -14.1314306, 20.8668461, -31.1377106, 31.2244949
17: -57.9019623, -14.3081532, -57.9240532, -14.3937397, -41.7386475, 41.7682724
18: -21.6065636, 12.1743269, -21.7402668, 12.1924801, -29.6383667, 29.7432022
19: -22.3625603, 3.5456007, -22.4135609, 3.5848019, -22.8467026, 22.8580360
20: -23.3779030, 1.3425801, -23.4239197, 1.3942738, -19.2856903, 19.2790527
21: -26.8865376, 2.3797121, -26.9586792, 2.4217367, -25.5724792, 25.6113701
22: -28.5852661, 3.3424313, -28.6780243, 3.3531470, -24.8231392, 24.8337021
23: -22.3805885, 5.6979847, -22.4176922, 5.7284451, -22.1037064, 22.1297951
24: -18.3727226, 9.4246082, -18.4849739, 9.4598970, -22.8964157, 22.9726562
25: -23.8818989, 5.3800964, -24.0084343, 5.4167523, -24.4708138, 24.5354004
26: -41.2003326, -0.4922476, -41.1903801, -0.4476218, -30.7826843, 30.7026672
27: -21.6629562, 8.6455956, -21.7277946, 8.6136684, -26.5282898, 26.6328201
28: -24.2013130, 6.0717163, -24.2625275, 6.0867333, -22.0634842, 22.0822678
29: -27.9230843, -0.1723582, -28.0373821, -0.1843743, -24.0680695, 24.1222763
30: -28.1597061, 3.7457044, -28.3178787, 3.7792139, -26.1642761, 26.3008194
31: -22.7372284, 5.0257902, -22.8025894, 5.0711021, -25.1180191, 25.1701508
32: -23.9903679, 2.3068464, -23.9612312, 2.3877714, -21.4910812, 21.3729668
33: -36.4616394, 3.6117358, -36.4591522, 3.6723690, -33.4239120, 33.3702087
34: -37.9187775, -4.7503562, -37.8949471, -4.7318149, -27.8657227, 27.7764969
35: -32.9637299, 0.3005557, -32.9493332, 0.3208556, -28.2477570, 28.1924438
36: -36.9250488, -0.6319728, -36.8675652, -0.6299744, -29.1467896, 29.0792465
37: -44.6483040, -1.6802979, -44.6281967, -1.7245703, -38.8874359, 38.8732834
38: -44.0578537, 2.8983150, -44.0036621, 2.9484429, -40.8885803, 40.7979431
39: -43.6174316, 2.9343514, -43.6374130, 3.0712709, -41.4686127, 41.3395081
40: -32.7834167, 0.0411520, -32.7955704, 0.0739465, -31.1609955, 31.1094246
41: -20.8112335, 7.3167229, -20.7415752, 7.3307714, -26.6060028, 26.5395660
42: -23.0315723, -0.2080445, -22.9907837, -0.1684504, -18.5390129, 18.4733543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.4968370, upper bound: 11.5440198
time: 32.44 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5227894, upper bound: 11.5447281
time: 36.26 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.3647871, 19.1073341, -9.3889751, 19.0659065, -25.2463837, 25.3481178
1: -1.2276368, 22.8792076, -1.2355008, 22.8578110, -19.7680511, 19.8377075
2: -1.6333284, 20.9640388, -1.6296856, 20.9768620, -17.3030777, 17.2841682
3: -9.3568697, 16.5366173, -9.3494654, 16.5688572, -22.0656128, 22.0292435
4: -3.1630177, 22.2553062, -3.1544328, 22.2680168, -21.7601929, 21.7131424
5: -7.8490353, 20.6883125, -7.8608565, 20.6901016, -23.7923279, 23.8362503
6: -28.8196697, -1.3792629, -28.8045921, -1.3439560, -23.2024612, 23.1919403
7: -7.7169619, 21.6950493, -7.7243948, 21.6989937, -23.6347694, 23.6512985
8: -14.8232403, 14.7775421, -14.8288679, 14.8123875, -26.5284004, 26.5079498
9: -5.1958861, 21.3042068, -5.2087488, 21.3035698, -24.2955551, 24.3189316
10: -17.8831329, 17.4582329, -17.8943329, 17.5057774, -31.2302399, 31.2063980
11: -26.8027382, 3.5579200, -26.8230476, 3.5636954, -27.8983688, 27.9216232
12: -34.8987923, -2.3300261, -34.8939209, -2.3221912, -27.2121429, 27.2402267
13: -26.2795448, 15.7951012, -26.2909279, 15.8713379, -34.0500565, 33.9983597
14: -55.9157181, -17.5405560, -55.9365540, -17.6373577, -37.7108459, 37.8171005
15: -14.3769112, 15.5075712, -14.3918610, 15.5225191, -27.9035263, 27.9140015
16: -14.0519304, 20.8876057, -14.0763903, 20.8404312, -31.0681305, 31.1331711
17: -57.8594322, -14.3231544, -57.8634911, -14.4246101, -41.6592712, 41.7355347
18: -21.6017132, 12.1861515, -21.7217388, 12.1928282, -29.6099701, 29.7091904
19: -22.3680038, 3.5959632, -22.3821068, 3.6050763, -22.8570709, 22.8628807
20: -23.4032631, 1.3931060, -23.3989754, 1.4040349, -19.3195419, 19.2676125
21: -26.8761826, 2.4227452, -26.9104481, 2.4287500, -25.5573730, 25.5896835
22: -28.5948353, 3.3572121, -28.6518555, 3.3654938, -24.7796097, 24.8249321
23: -22.3771229, 5.7318892, -22.4066544, 5.7467918, -22.1094284, 22.1117401
24: -18.3955078, 9.4779320, -18.4628410, 9.4840889, -22.9313126, 22.9717712
25: -23.8784447, 5.4118705, -23.9605446, 5.4032550, -24.4139252, 24.5030212
26: -41.1909599, -0.4602513, -41.1831894, -0.4294748, -30.7537384, 30.6949387
27: -21.6292915, 8.6103487, -21.6993885, 8.6202555, -26.5154343, 26.5666504
28: -24.2137890, 6.1049490, -24.2496262, 6.1098876, -22.0815163, 22.0985870
29: -27.9114666, -0.1950428, -27.9761353, -0.2015386, -23.9860992, 24.0874596
30: -28.1672935, 3.7667561, -28.2417717, 3.7544553, -26.1247787, 26.2171478
31: -22.7257462, 5.0722847, -22.7801952, 5.0846119, -25.1172256, 25.1539383
32: -23.9834881, 2.3196864, -23.9275055, 2.3701851, -21.4748154, 21.3344116
33: -36.4789886, 3.6524663, -36.4354935, 3.6841793, -33.4513931, 33.3103485
34: -37.9175301, -4.7440429, -37.8665314, -4.7344136, -27.8410492, 27.7042465
35: -32.9514046, 0.3108344, -32.9148865, 0.3253703, -28.2310638, 28.1367569
36: -36.8869705, -0.6420031, -36.8247375, -0.6275120, -29.1137772, 29.0046844
37: -44.6005096, -1.7115283, -44.5753746, -1.6962934, -38.8632660, 38.7894821
38: -44.0174255, 2.8869758, -43.9383698, 2.9126129, -40.8218842, 40.6844330
39: -43.6564865, 3.0096006, -43.6128540, 3.0795455, -41.5088501, 41.3227158
40: -32.7707939, -0.0004702, -32.7670746, 0.0743296, -31.1405945, 31.0525818
41: -20.7668934, 7.2832937, -20.6946831, 7.3119230, -26.5900040, 26.4608002
42: -23.0400181, -0.2136350, -22.9856758, -0.1808701, -18.5147324, 18.4518929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5734619, upper bound: 11.5193256
time: 31.04 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5446936, upper bound: 11.5199073
time: 37.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.3689804, 19.1075974, -9.4105225, 19.0827408, -25.2728119, 25.3650780
1: -1.2302527, 22.8799706, -1.2538862, 22.8753376, -19.7954292, 19.8567772
2: -1.6400988, 20.9651909, -1.6568425, 21.0122643, -17.3458138, 17.3063965
3: -9.3644772, 16.5383034, -9.3759146, 16.6125927, -22.1207733, 22.0502625
4: -3.1715183, 22.2563934, -3.1898708, 22.3050423, -21.8068466, 21.7402954
5: -7.8453526, 20.6898117, -7.8590260, 20.7070103, -23.8105240, 23.8514252
6: -28.8220787, -1.3785973, -28.8179989, -1.3256912, -23.2351151, 23.2072754
7: -7.7158751, 21.6960144, -7.7379069, 21.7000427, -23.6386261, 23.6856194
8: -14.8239002, 14.7802191, -14.8349247, 14.8281870, -26.5488052, 26.5227051
9: -5.1977997, 21.3075447, -5.2428932, 21.3175011, -24.3078537, 24.3636093
10: -17.8853359, 17.4697323, -17.9776192, 17.5471306, -31.2749329, 31.3008804
11: -26.8046551, 3.5659342, -26.8993359, 3.5892668, -27.9241409, 28.0072784
12: -34.9000130, -2.3282003, -34.9264565, -2.3081708, -27.2281036, 27.2926674
13: -26.2812538, 15.7966690, -26.3026810, 15.8871269, -34.0672836, 34.0245361
14: -55.9179955, -17.5272923, -56.0150566, -17.5851593, -37.7545624, 37.9377441
15: -14.3781319, 15.5097685, -14.4118490, 15.5334997, -27.9202805, 27.9390564
16: -14.0550957, 20.8920841, -14.1463537, 20.8579121, -31.0836487, 31.2079010
17: -57.8619537, -14.3181925, -57.9436760, -14.3975334, -41.6911545, 41.8556519
18: -21.6042404, 12.1883945, -21.7410183, 12.2077780, -29.6620255, 29.7247696
19: -22.3708439, 3.5950317, -22.4147129, 3.6047449, -22.8636017, 22.9021492
20: -23.4046249, 1.3963444, -23.4236336, 1.4156733, -19.3285484, 19.2967758
21: -26.8799419, 2.4270144, -26.9598713, 2.4423671, -25.5742950, 25.6496201
22: -28.5984192, 3.3563683, -28.6750946, 3.3681092, -24.7936249, 24.8532333
23: -22.3783875, 5.7287502, -22.4184265, 5.7431788, -22.1138992, 22.1408997
24: -18.3975296, 9.4753513, -18.4821892, 9.4800262, -22.9418869, 22.9909592
25: -23.8806801, 5.4204803, -24.0073185, 5.4361668, -24.4452934, 24.5624008
26: -41.1931610, -0.4638858, -41.1899071, -0.4320979, -30.7900314, 30.7062073
27: -21.6327095, 8.6112862, -21.7168522, 8.6260109, -26.5298157, 26.5833435
28: -24.2151203, 6.1027670, -24.2603951, 6.1083865, -22.0932541, 22.1114082
29: -27.9151344, -0.1890819, -28.0360355, -0.1803375, -24.0022507, 24.1553040
30: -28.1697083, 3.7799664, -28.3166008, 3.7970798, -26.1569824, 26.3063736
31: -22.7285252, 5.0739841, -22.8038292, 5.0915813, -25.1274567, 25.2038803
32: -23.9914742, 2.3205948, -23.9544544, 2.4068480, -21.5210800, 21.3549805
33: -36.4816055, 3.6528735, -36.4467926, 3.7211394, -33.4957275, 33.3241501
34: -37.9214554, -4.7415810, -37.8800964, -4.6959772, -27.9139404, 27.7146225
35: -32.9560051, 0.3119049, -32.9318466, 0.3656683, -28.2933731, 28.1506577
36: -36.8924255, -0.6410751, -36.8445969, -0.5891042, -29.1706848, 29.0225983
37: -44.6042366, -1.7153492, -44.6057053, -1.6957107, -38.8922272, 38.7986832
38: -44.0284843, 2.8887849, -43.9798431, 3.0006332, -40.9218750, 40.7268753
39: -43.6602592, 3.0097651, -43.6305313, 3.1202884, -41.5572052, 41.3351898
40: -32.7739410, -0.0004430, -32.7876129, 0.0889845, -31.1723938, 31.0576210
41: -20.7759285, 7.2839804, -20.7263985, 7.3504486, -26.6380310, 26.4876556
42: -23.0413017, -0.2126048, -22.9939690, -0.1643302, -18.5333252, 18.4781647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5734619, upper bound: 11.5334629
time: 31.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5446936, upper bound: 11.5340500
time: 31.30 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.4191256, 19.1304512, -9.3934479, 19.0663929, -25.3050461, 25.3821793
1: -1.2601738, 22.9103584, -1.2364430, 22.8585110, -19.8028069, 19.8817711
2: -1.6850681, 20.9790020, -1.6320589, 20.9801540, -17.3316422, 17.3419342
3: -9.3812761, 16.5515442, -9.3509998, 16.5698624, -22.0857086, 22.0691147
4: -3.2094488, 22.2630577, -3.1573176, 22.2690697, -21.8041229, 21.7650948
5: -7.8985620, 20.7060394, -7.8643198, 20.6929569, -23.8277588, 23.8678207
6: -28.8722343, -1.3010273, -28.8172703, -1.3420038, -23.2433395, 23.2850342
7: -7.7629309, 21.7103195, -7.7279005, 21.7012196, -23.6793289, 23.6904221
8: -14.9002390, 14.8077307, -14.8299131, 14.8172770, -26.5945435, 26.5726776
9: -5.2817569, 21.3507671, -5.2107697, 21.3132401, -24.3922424, 24.3642044
10: -18.1040497, 17.6210117, -17.8967571, 17.5483475, -31.4909286, 31.3493347
11: -26.8805637, 3.5844989, -26.8253403, 3.5670495, -27.9901199, 27.9510727
12: -34.9247856, -2.2910433, -34.8954849, -2.3169370, -27.2706604, 27.2751465
13: -26.3133698, 15.8290424, -26.2950668, 15.8744345, -34.0915985, 34.0353851
14: -56.0829163, -17.4144859, -55.9407349, -17.6043377, -37.9115601, 37.9246140
15: -14.4356241, 15.5245857, -14.3955479, 15.5253906, -27.9791718, 27.9352341
16: -14.1496592, 20.9307880, -14.0801516, 20.8500443, -31.1691818, 31.1762848
17: -57.9700165, -14.2667561, -57.8665924, -14.4138031, -41.7920532, 41.8005829
18: -21.6550808, 12.2229481, -21.7267838, 12.1946259, -29.6629257, 29.7905121
19: -22.4092770, 3.6085663, -22.3860798, 3.6061635, -22.9116364, 22.8801422
20: -23.4296894, 1.4069505, -23.4015617, 1.4051607, -19.3549576, 19.2845192
21: -26.9359760, 2.4430954, -26.9149380, 2.4306817, -25.6310349, 25.6153717
22: -28.6273556, 3.3946767, -28.6580048, 3.3671985, -24.8728256, 24.8343201
23: -22.4106522, 5.7490549, -22.4085655, 5.7477050, -22.1451416, 22.1364975
24: -18.4233932, 9.4880695, -18.4679871, 9.4849300, -22.9595261, 22.9899292
25: -23.9092102, 5.4397359, -23.9642601, 5.4053173, -24.4861069, 24.5313568
26: -41.2405624, -0.4196267, -41.1863289, -0.4213767, -30.8164444, 30.7330093
27: -21.6918736, 8.6882353, -21.7142334, 8.6227112, -26.5706100, 26.6619873
28: -24.2383671, 6.1464891, -24.2539558, 6.1120996, -22.1143723, 22.1308441
29: -27.9520874, -0.1586869, -27.9824047, -0.2002525, -24.0976295, 24.0869980
30: -28.1942558, 3.7980990, -28.2451172, 3.7574868, -26.1784515, 26.2553558
31: -22.7884789, 5.0904360, -22.7842255, 5.0863857, -25.1856918, 25.1759109
32: -24.0226860, 2.3699906, -23.9371624, 2.3722064, -21.5108032, 21.3983116
33: -36.5507431, 3.7560439, -36.4520721, 3.6854124, -33.5213318, 33.4445648
34: -37.9794312, -4.6389923, -37.8834076, -4.7315722, -27.8992844, 27.8359756
35: -33.0312843, 0.4328690, -32.9354057, 0.3266206, -28.3027878, 28.2818451
36: -36.9850044, -0.5087042, -36.8506432, -0.6260061, -29.1966095, 29.1647644
37: -44.7190285, -1.5897164, -44.6044807, -1.6955557, -38.9674683, 38.9571457
38: -44.1414261, 3.0568523, -43.9680710, 2.9154000, -40.9355316, 40.8830032
39: -43.7216110, 3.0812368, -43.6263809, 3.0813823, -41.5775604, 41.4216232
40: -32.8381042, 0.0886517, -32.7815628, 0.0751233, -31.2079163, 31.1611099
41: -20.8432808, 7.3805914, -20.7141457, 7.3137312, -26.6578293, 26.5815353
42: -23.0471611, -0.1857669, -22.9874477, -0.1784909, -18.5469055, 18.4833336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5734619, upper bound: 11.5305510
time: 30.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5446936, upper bound: 11.5311629
time: 34.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.3843040, 19.0745296, -9.4186497, 19.0833664, -25.2898407, 25.3436050
1: -1.2305760, 22.8701572, -1.2574639, 22.8761940, -19.7962189, 19.8597107
2: -1.6730084, 20.9456291, -1.6609566, 21.0157490, -17.3524323, 17.3534088
3: -9.3726797, 16.5229950, -9.3789854, 16.6138382, -22.1235657, 22.0612717
4: -3.1899333, 22.2472534, -3.1949253, 22.3061810, -21.8196640, 21.7962685
5: -7.8704677, 20.6507664, -7.8648438, 20.7101326, -23.8187065, 23.8278084
6: -28.8340073, -1.3261118, -28.8310223, -1.3216715, -23.2601166, 23.2722626
7: -7.7322073, 21.6696548, -7.7439232, 21.7024765, -23.6519623, 23.6839828
8: -14.8745747, 14.7968979, -14.8381271, 14.8335304, -26.5878677, 26.5822144
9: -5.2629766, 21.3070717, -5.2468309, 21.3273315, -24.3831635, 24.3639450
10: -18.0787048, 17.5915031, -17.9825039, 17.5899963, -31.5063095, 31.4053574
11: -26.8678570, 3.5791106, -26.9027271, 3.5942497, -28.0029297, 28.0230179
12: -34.9118347, -2.3129506, -34.9284286, -2.2991014, -27.2897034, 27.2911415
13: -26.2957478, 15.8009253, -26.3086605, 15.8907137, -34.0913620, 34.0279999
14: -56.0505829, -17.4922752, -56.0222054, -17.5517197, -37.9266891, 37.9569168
15: -14.4118891, 15.5234041, -14.4172859, 15.5365791, -27.9711914, 27.9582367
16: -14.1208038, 20.8611622, -14.1533070, 20.8676281, -31.1470337, 31.1840591
17: -57.9169998, -14.3399563, -57.9510460, -14.3860054, -41.7752533, 41.8437958
18: -21.6426582, 12.2171593, -21.7485428, 12.2104263, -29.6900101, 29.8038864
19: -22.3690376, 3.5924902, -22.4192562, 3.6074126, -22.8810501, 22.9025955
20: -23.3733845, 1.3904028, -23.4264297, 1.4187579, -19.3068390, 19.2905159
21: -26.9079685, 2.4305224, -26.9649906, 2.4458678, -25.6341705, 25.6575699
22: -28.5959473, 3.3786576, -28.6817055, 3.3713009, -24.8729248, 24.8450508
23: -22.3747311, 5.7288036, -22.4207325, 5.7457128, -22.1141968, 22.1477852
24: -18.3984261, 9.4741974, -18.4877434, 9.4820461, -22.9436493, 22.9983063
25: -23.8916473, 5.4333572, -24.0114517, 5.4399548, -24.5089188, 24.5741119
26: -41.1575127, -0.4457893, -41.1934052, -0.4213929, -30.7679672, 30.7171249
27: -21.6636181, 8.6727228, -21.7320747, 8.6301060, -26.5491486, 26.6636124
28: -24.1918583, 6.1239767, -24.2649326, 6.1125174, -22.0794067, 22.1211395
29: -27.9413929, -0.1628218, -28.0443249, -0.1781173, -24.1071358, 24.1297760
30: -28.1932812, 3.8040719, -28.3223991, 3.8018575, -26.2086678, 26.3154755
31: -22.7620106, 5.0757747, -22.8085575, 5.0948462, -25.1729965, 25.2093353
32: -23.9631519, 2.3405027, -23.9644947, 2.4115810, -21.4861145, 21.3846359
33: -36.4752541, 3.7280183, -36.4638443, 3.7251225, -33.4892120, 33.4272079
34: -37.8976173, -4.6695108, -37.8971901, -4.6900959, -27.8894577, 27.8120117
35: -32.9632378, 0.4047213, -32.9526634, 0.3696246, -28.2948608, 28.2654343
36: -36.8922195, -0.5414524, -36.8707695, -0.5843887, -29.1573029, 29.1460876
37: -44.6279221, -1.6214652, -44.6356812, -1.6920485, -38.9040527, 38.9358292
38: -44.0189438, 3.0135479, -44.0100784, 3.0079141, -40.9048767, 40.8762894
39: -43.6354828, 3.0487909, -43.6448822, 3.1254230, -41.5386810, 41.4000702
40: -32.7810631, 0.0586250, -32.8029785, 0.0925138, -31.1822433, 31.1351242
41: -20.7648258, 7.3542109, -20.7463646, 7.3547912, -26.6204834, 26.5794601
42: -22.9922791, -0.2085629, -22.9960976, -0.1597707, -18.5320969, 18.4814339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1789

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5486809, upper bound: 11.5486807
time: 32.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5486809, upper bound: 11.5735387
time: 40.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.4233532, 19.1307068, -9.4150372, 19.0832520, -25.3314667, 25.3991890
1: -1.2627983, 22.9111118, -1.2548094, 22.8760223, -19.8302002, 19.9008369
2: -1.6918402, 20.9801674, -1.6592062, 21.0155563, -17.3744164, 17.3641396
3: -9.3888311, 16.5532265, -9.3774204, 16.6135712, -22.1409225, 22.0901337
4: -3.2179089, 22.2641373, -3.1928020, 22.3060989, -21.8507690, 21.7922668
5: -7.8949351, 20.7074738, -7.8624778, 20.7098827, -23.8459740, 23.8830109
6: -28.8746452, -1.3003550, -28.8306923, -1.3237648, -23.2760239, 23.3004074
7: -7.7618532, 21.7113094, -7.7413831, 21.7022362, -23.6831322, 23.7247429
8: -14.9009333, 14.8103809, -14.8359852, 14.8330498, -26.6149139, 26.5874329
9: -5.2836218, 21.3541679, -5.2448678, 21.3271999, -24.4045486, 24.4088821
10: -18.1061878, 17.6325397, -17.9800301, 17.5896473, -31.5356445, 31.4438248
11: -26.8824978, 3.5924835, -26.9016399, 3.5926218, -28.0158615, 28.0367355
12: -34.9259834, -2.2892675, -34.9279861, -2.3029480, -27.2866516, 27.3275375
13: -26.3150635, 15.8305740, -26.3067780, 15.8902512, -34.1088181, 34.0615463
14: -56.0851936, -17.4012432, -56.0192451, -17.5522003, -37.9552383, 38.0452118
15: -14.4369020, 15.5268230, -14.4155388, 15.5363731, -27.9959641, 27.9603119
16: -14.1528139, 20.9352264, -14.1501932, 20.8675117, -31.1846771, 31.2510223
17: -57.9725876, -14.2617722, -57.9467354, -14.3866348, -41.8239517, 41.9206467
18: -21.6576080, 12.2252140, -21.7461357, 12.2095375, -29.7149506, 29.8060913
19: -22.4121208, 3.6076441, -22.4186420, 3.6057644, -22.9181213, 22.9193649
20: -23.4310608, 1.4101813, -23.4262314, 1.4168227, -19.3639603, 19.3136711
21: -26.9396667, 2.4473348, -26.9643669, 2.4442902, -25.6479797, 25.6753120
22: -28.6309185, 3.3937464, -28.6811543, 3.3697968, -24.8868179, 24.8625793
23: -22.4119339, 5.7459259, -22.4203758, 5.7440486, -22.1495857, 22.1656456
24: -18.4254303, 9.4854994, -18.4873161, 9.4808626, -22.9701157, 23.0091248
25: -23.9113846, 5.4482961, -24.0110588, 5.4382210, -24.5174561, 24.5906906
26: -41.2427979, -0.4233022, -41.1930046, -0.4239964, -30.8527527, 30.7442551
27: -21.6953030, 8.6891918, -21.7317009, 8.6284389, -26.5849915, 26.6787567
28: -24.2396507, 6.1442647, -24.2647095, 6.1106024, -22.1261063, 22.1436577
29: -27.9557781, -0.1527395, -28.0423298, -0.1790231, -24.1137848, 24.1548309
30: -28.1966991, 3.8113022, -28.3198872, 3.8000734, -26.2106552, 26.3445320
31: -22.7912655, 5.0920553, -22.8079071, 5.0933251, -25.1959305, 25.2259064
32: -24.0306816, 2.3708777, -23.9641380, 2.4088869, -21.5570793, 21.4188690
33: -36.5533600, 3.7563791, -36.4633789, 3.7223506, -33.5656433, 33.4583206
34: -37.9833908, -4.6365414, -37.8969765, -4.6931558, -27.9721985, 27.8463516
35: -33.0359268, 0.4339819, -32.9523621, 0.3668737, -28.3651276, 28.2956772
36: -36.9904594, -0.5077701, -36.8704643, -0.5876174, -29.2535172, 29.1826859
37: -44.7227554, -1.5935330, -44.6348190, -1.6949530, -38.9964905, 38.9663544
38: -44.1524620, 3.0587101, -44.0095520, 3.0034804, -41.0355225, 40.9254913
39: -43.7254486, 3.0813947, -43.6440048, 3.1221089, -41.6259308, 41.4341431
40: -32.8412209, 0.0887024, -32.8020821, 0.0898077, -31.2397003, 31.1660957
41: -20.8522987, 7.3812599, -20.7458496, 7.3521843, -26.7058640, 26.6083527
42: -23.0484581, -0.1847715, -22.9957581, -0.1619468, -18.5654755, 18.5095901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=171, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5734619, upper bound: 11.5446935
time: 37.24 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5446936, upper bound: 11.5453115
time: 31.28 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 70.96 seconds
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5074963, upper bound: 11.5480066
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5074533, upper bound: 11.5728183
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5228325, upper bound: 11.5480064
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5227895, upper bound: 11.5728183
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5305969, upper bound: 11.5486064
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5305513, upper bound: 11.5575204
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5447391, upper bound: 11.5486065
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5446937, upper bound: 11.5734616
IS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5269404, upper bound: 11.5480690
IS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5269404, upper bound: 11.5728847
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.4968370, upper bound: 11.5440198
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5227894, upper bound: 11.5447281
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5734619, upper bound: 11.5193256
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5446936, upper bound: 11.5199073
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5734619, upper bound: 11.5334629
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5446936, upper bound: 11.5340500
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5734619, upper bound: 11.5305510
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5446936, upper bound: 11.5311629
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5486809, upper bound: 11.5486807
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5486809, upper bound: 11.5735387
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5734619, upper bound: 11.5446935
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 70.96
Output dim: 2, lower bound: -11.5446936, upper bound: 11.5453115

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.3595562, 19.0498581, -9.4224739, 19.1623039, -25.3416481, 25.3365326
1: -1.2166677, 22.8422050, -1.2574649, 22.9429817, -19.8580666, 19.8383484
2: -1.6545675, 20.9424400, -1.6577444, 21.0514641, -17.3502998, 17.3582306
3: -9.3607950, 16.5106258, -9.3730354, 16.6505795, -22.1344414, 22.0614014
4: -3.1673460, 22.2567711, -3.1961942, 22.3118343, -21.7673759, 21.8357925
5: -7.8530998, 20.6116829, -7.8648267, 20.8007927, -23.9111710, 23.7816315
6: -28.8241386, -1.3107386, -28.8760319, -1.3273253, -23.2229767, 23.3198013
7: -7.7209702, 21.6285439, -7.7404919, 21.7819824, -23.7465820, 23.6262970
8: -14.8420639, 14.7636938, -14.8391275, 14.8742905, -26.5867157, 26.5645142
9: -5.2809315, 21.2836418, -5.2223539, 21.3939266, -24.4884949, 24.2711792
10: -18.1437588, 17.5575085, -17.9146881, 17.6578636, -31.6442032, 31.3030472
11: -26.8804626, 3.5576010, -26.8990002, 3.5957704, -28.0299377, 27.9918289
12: -34.9161224, -2.3422189, -34.9152527, -2.2622533, -27.3977737, 27.1866264
13: -26.2645912, 15.7224770, -26.3044701, 15.9908810, -34.1893158, 33.9278946
14: -56.0857048, -17.5694962, -55.9599495, -17.3925934, -38.1819763, 37.7569351
15: -14.4158535, 15.5116110, -14.4139175, 15.5448313, -27.9776840, 27.9470978
16: -14.1609449, 20.8043919, -14.1063967, 20.9955635, -31.3306656, 31.0742111
17: -57.9760513, -14.4348631, -57.8992271, -14.2257843, -42.0583344, 41.6407852
18: -21.6080856, 12.1983700, -21.7848854, 12.2029400, -29.5832520, 29.8946075
19: -22.3114510, 3.5593429, -22.5115738, 3.6052902, -22.8312836, 22.9588737
20: -23.3135872, 1.3586099, -23.5403671, 1.4149692, -19.2545624, 19.3840752
21: -26.8589077, 2.3945243, -27.0380898, 2.4442594, -25.6030350, 25.6723099
22: -28.5047264, 3.3411565, -28.8019295, 3.3692858, -24.7932663, 24.9121475
23: -22.3064404, 5.6879969, -22.5220985, 5.7483878, -22.0655823, 22.2037239
24: -18.3117409, 9.4365997, -18.5960846, 9.4825954, -22.8480682, 23.0880737
25: -23.8650112, 5.4008455, -24.0523033, 5.4382920, -24.5005035, 24.5679703
26: -41.0559082, -0.4939437, -41.3751984, -0.4248447, -30.6391068, 30.9043045
27: -21.5731544, 8.6384268, -21.8475037, 8.6259422, -26.4439087, 26.7681503
28: -24.1151905, 6.0841827, -24.3860359, 6.1125040, -21.9968758, 22.2278557
29: -27.8914318, -0.1919816, -28.1017399, -0.1787276, -24.1088028, 24.1269379
30: -28.1828003, 3.7553940, -28.3308086, 3.8041975, -26.2523346, 26.2631798
31: -22.7148342, 5.0516748, -22.8761597, 5.0935812, -25.1556473, 25.2221146
32: -23.9431343, 2.3859320, -24.0458412, 2.3837481, -21.4248352, 21.5354881
33: -36.4354897, 3.7727485, -36.5717354, 3.6881928, -33.4126587, 33.5773392
34: -37.8580856, -4.6516032, -38.0183868, -4.7139401, -27.7787323, 28.0092239
35: -32.9267502, 0.4397073, -33.0545502, 0.3401799, -28.2010117, 28.4339066
36: -36.8431931, -0.5180483, -37.0112686, -0.6156483, -29.0563431, 29.3394394
37: -44.5825920, -1.6174378, -44.7605133, -1.6947274, -38.8123322, 39.1100006
38: -43.9726486, 3.0932002, -44.1796036, 2.9349685, -40.7694397, 41.1336823
39: -43.6046410, 3.1059971, -43.7467728, 3.0923843, -41.4463806, 41.5856323
40: -32.7564354, 0.0821378, -32.8695755, 0.0927541, -31.1238861, 31.2539825
41: -20.7295818, 7.3919916, -20.8595924, 7.3169298, -26.5220032, 26.7392883
42: -22.9892750, -0.1878519, -23.0524254, -0.1712227, -18.5394745, 18.5197144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=170, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=198, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1646

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5436207, upper bound: 11.5510285
time: 57.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5436207, upper bound: 11.5723909
time: 33.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2539787, 19.0391617, -9.3751202, 19.1369476, -25.2512283, 25.2584152
1: -1.1435833, 22.8370247, -1.2295270, 22.9157467, -19.7832260, 19.7935562
2: -1.6125767, 20.9280281, -1.6395259, 21.0490856, -17.3090935, 17.3179054
3: -9.3154640, 16.4937172, -9.3574867, 16.6396046, -22.0919304, 22.0003090
4: -3.1045542, 22.2380409, -3.1706939, 22.3216019, -21.7275543, 21.7688026
5: -7.7850304, 20.6142330, -7.8336453, 20.7628899, -23.8006058, 23.7607346
6: -28.7961845, -1.4051771, -28.8667717, -1.3459616, -23.1956787, 23.2096519
7: -7.6485538, 21.6429234, -7.7165518, 21.7419796, -23.6193695, 23.6278534
8: -14.7469521, 14.7429085, -14.7965450, 14.8422508, -26.4771461, 26.4882126
9: -5.1876879, 21.2801075, -5.2210956, 21.3705940, -24.3676453, 24.3109894
10: -17.9811096, 17.5602074, -17.9515533, 17.6246738, -31.4445343, 31.3408051
11: -26.8442249, 3.5351529, -26.9021702, 3.5792913, -27.9661789, 27.9697723
12: -34.8959236, -2.3732057, -34.9348221, -2.3153396, -27.2418060, 27.2079315
13: -26.2791061, 15.7484026, -26.3051186, 15.9028826, -34.0884857, 33.9588394
14: -55.9584007, -17.5371456, -55.9944077, -17.4691010, -37.9187164, 37.7722321
15: -14.3050632, 15.4774456, -14.3906059, 15.5367279, -27.8751373, 27.8850021
16: -14.0632486, 20.8427029, -14.1319036, 20.9402332, -31.1693115, 31.1572495
17: -57.8463097, -14.3864202, -57.9326553, -14.3203144, -41.7664642, 41.6857300
18: -21.5916901, 12.1662560, -21.7486610, 12.1935501, -29.6075516, 29.7413025
19: -22.3194962, 3.5304368, -22.4526463, 3.5833497, -22.8071442, 22.8753357
20: -23.3202400, 1.3227987, -23.4803658, 1.3942335, -19.2304459, 19.3129883
21: -26.8549614, 2.3629353, -26.9852715, 2.4228368, -25.5606461, 25.5951920
22: -28.5503616, 3.3273058, -28.7089329, 3.3540819, -24.8115463, 24.8167191
23: -22.3434010, 5.6808577, -22.4520626, 5.7295785, -22.0673523, 22.1459084
24: -18.3456631, 9.4133015, -18.5097694, 9.4607182, -22.8692627, 22.9869461
25: -23.8621712, 5.3651609, -24.0262566, 5.4203520, -24.4669647, 24.5203667
26: -41.1150818, -0.5147991, -41.2729263, -0.4512196, -30.6928406, 30.7600555
27: -21.6312618, 8.6291456, -21.7564125, 8.6152611, -26.4914932, 26.6496735
28: -24.1535187, 6.0514216, -24.3090420, 6.0882092, -22.0165367, 22.1068764
29: -27.9087009, -0.1823857, -28.0461922, -0.1827770, -24.0501900, 24.0903168
30: -28.1563282, 3.7384665, -28.3194046, 3.7825270, -26.1608391, 26.2681503
31: -22.7080460, 5.0094891, -22.8285561, 5.0719652, -25.0942383, 25.1738853
32: -23.9228630, 2.2764518, -24.0277786, 2.3902822, -21.4244995, 21.4101448
33: -36.3835144, 3.5833726, -36.5339203, 3.6710320, -33.3440552, 33.4143524
34: -37.8330460, -4.7833419, -37.9786682, -4.7317662, -27.7787933, 27.8260040
35: -32.8910217, 0.2713032, -33.0192451, 0.3207211, -28.1739960, 28.2323761
36: -36.8268623, -0.6656160, -36.9630470, -0.6312203, -29.0453568, 29.1399841
37: -44.5534859, -1.7081909, -44.7182693, -1.7283583, -38.7889252, 38.9329071
38: -43.9243431, 2.8531289, -44.1326332, 2.9457831, -40.7507782, 40.8804321
39: -43.5274200, 2.9017448, -43.7218018, 3.0681529, -41.3759308, 41.3900604
40: -32.7232437, 0.0111141, -32.8523788, 0.0734687, -31.1007614, 31.1345749
41: -20.7237282, 7.2896390, -20.8257294, 7.3295445, -26.5164413, 26.5953751
42: -22.9753265, -0.2317715, -23.0451584, -0.1671705, -18.5097733, 18.4747391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=170, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1646

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5258510, upper bound: 11.5504959
time: 33.32 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5258510, upper bound: 11.5718431
time: 32.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.3647871, 19.1073341, -9.3329010, 19.0640602, -25.2533913, 25.2899971
1: -1.2276368, 22.8792076, -1.1921010, 22.8558540, -19.7721977, 19.7997742
2: -1.6333284, 20.9640388, -1.5910788, 20.9750710, -17.3088455, 17.2489052
3: -9.3568697, 16.5366173, -9.3169689, 16.5630054, -22.0618744, 21.9976120
4: -3.1630177, 22.2553062, -3.1075687, 22.2663898, -21.7616539, 21.6718750
5: -7.8490353, 20.6883125, -7.8214531, 20.6850224, -23.7904434, 23.8016052
6: -28.8196697, -1.3792629, -28.7994423, -1.3575315, -23.1729279, 23.1881943
7: -7.7169619, 21.6950493, -7.6826630, 21.6959572, -23.6365356, 23.6148071
8: -14.8232403, 14.7775421, -14.7732334, 14.8070240, -26.5263062, 26.4606018
9: -5.1958861, 21.3042068, -5.1836729, 21.3001022, -24.2937851, 24.2671127
10: -17.8831329, 17.4582329, -17.8667984, 17.4991360, -31.2287445, 31.1753235
11: -26.8027382, 3.5579200, -26.8181934, 3.5320754, -27.8684387, 27.9206848
12: -34.8987923, -2.3300261, -34.8714066, -2.3345389, -27.2043839, 27.2178421
13: -26.2795448, 15.7951012, -26.2381439, 15.8637543, -34.0518951, 33.9451904
14: -55.9157181, -17.5405560, -55.8777199, -17.6461983, -37.7098236, 37.7564392
15: -14.3769112, 15.5075712, -14.3695221, 15.5159283, -27.8968430, 27.8895874
16: -14.0519304, 20.8876057, -14.0408258, 20.8381405, -31.0709305, 31.0996323
17: -57.8594322, -14.3231544, -57.8192825, -14.4370289, -41.6574860, 41.6915359
18: -21.6017132, 12.1861515, -21.7111549, 12.1594696, -29.5742035, 29.7028198
19: -22.3680038, 3.5959632, -22.3741913, 3.5564170, -22.8083115, 22.8643684
20: -23.4032631, 1.3931060, -23.3955212, 1.3505979, -19.2652130, 19.2844315
21: -26.8761826, 2.4227452, -26.9031219, 2.3759789, -25.5047989, 25.5950279
22: -28.5948353, 3.3572121, -28.6461983, 3.3123026, -24.7249832, 24.8370628
23: -22.3771229, 5.7318892, -22.4028606, 5.6933107, -22.0558777, 22.1206169
24: -18.3955078, 9.4779320, -18.4577179, 9.4369526, -22.8841476, 22.9775620
25: -23.8784447, 5.4118705, -23.9558411, 5.3560562, -24.3659668, 24.5120087
26: -41.1909599, -0.4602513, -41.1784019, -0.5008798, -30.6816864, 30.7145386
27: -21.6292915, 8.6103487, -21.6929531, 8.5670118, -26.4615631, 26.5701828
28: -24.2137890, 6.1049490, -24.2465286, 6.0517378, -22.0223312, 22.1141090
29: -27.9114666, -0.1950428, -27.9709721, -0.2404509, -23.9458237, 24.0958748
30: -28.1672935, 3.7667561, -28.2387257, 3.7028489, -26.0754814, 26.2287140
31: -22.7257462, 5.0722847, -22.7718029, 5.0445747, -25.0754089, 25.1518936
32: -23.9834881, 2.3196864, -23.9218025, 2.3602171, -21.4639130, 21.3296700
33: -36.4789886, 3.6524663, -36.4257889, 3.6665926, -33.4345169, 33.2928162
34: -37.9175301, -4.7440429, -37.8625984, -4.7702756, -27.8058701, 27.7097549
35: -32.9514046, 0.3108344, -32.9079056, 0.3050265, -28.2109299, 28.1317978
36: -36.8869705, -0.6420031, -36.8182373, -0.6654253, -29.0745010, 29.0041809
37: -44.6005096, -1.7115283, -44.5608215, -1.7197223, -38.8392639, 38.7731476
38: -44.0174255, 2.8869758, -43.9284821, 2.8795819, -40.7703094, 40.6785049
39: -43.6564865, 3.0096006, -43.5950356, 3.0743546, -41.4970398, 41.3061676
40: -32.7707939, -0.0004702, -32.7501221, 0.0708823, -31.1320724, 31.0332260
41: -20.7668934, 7.2832937, -20.6855812, 7.2884655, -26.5446320, 26.4535599
42: -23.0400181, -0.2136350, -22.9830685, -0.1930597, -18.5031967, 18.4479904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=170, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1646

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5723912, upper bound: 11.4968372
time: 32.04 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5723912, upper bound: 11.5182581
time: 38.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.3689804, 19.1075974, -9.3544455, 19.0808964, -25.2798042, 25.3069611
1: -1.2302527, 22.8799706, -1.2104731, 22.8733826, -19.7995949, 19.8188324
2: -1.6400988, 20.9651909, -1.6181896, 21.0104980, -17.3516083, 17.2711182
3: -9.3644772, 16.5383034, -9.3434753, 16.6066971, -22.1170349, 22.0186234
4: -3.1715183, 22.2563934, -3.1430078, 22.3034363, -21.8083229, 21.6990471
5: -7.8453526, 20.6898117, -7.8196182, 20.7019405, -23.8086395, 23.8167534
6: -28.8220787, -1.3785973, -28.8128471, -1.3392572, -23.2055664, 23.2035103
7: -7.7158751, 21.6960144, -7.6961966, 21.6970062, -23.6403656, 23.6491089
8: -14.8239002, 14.7802191, -14.7792873, 14.8228130, -26.5466919, 26.4753723
9: -5.1977997, 21.3075447, -5.2178173, 21.3140564, -24.3061066, 24.3117752
10: -17.8853359, 17.4697323, -17.9501152, 17.5404415, -31.2734375, 31.2698059
11: -26.8046551, 3.5659342, -26.8945084, 3.5576630, -27.8941879, 28.0063705
12: -34.9000130, -2.3282003, -34.9039536, -2.3205037, -27.2203522, 27.2702904
13: -26.2812538, 15.7966690, -26.2499046, 15.8795328, -34.0690842, 33.9713974
14: -55.9179955, -17.5272923, -55.9562302, -17.5940132, -37.7535553, 37.8770447
15: -14.3781319, 15.5097685, -14.3894854, 15.5268440, -27.9135971, 27.9146423
16: -14.0550957, 20.8920841, -14.1108093, 20.8556442, -31.0864639, 31.1744003
17: -57.8619537, -14.3181925, -57.8995132, -14.4098930, -41.6893768, 41.8116837
18: -21.6042404, 12.1883945, -21.7304878, 12.1744623, -29.6262436, 29.7184219
19: -22.3708439, 3.5950317, -22.4067726, 3.5559759, -22.8147888, 22.9035873
20: -23.4046249, 1.3963444, -23.4202099, 1.3622396, -19.2741966, 19.3135986
21: -26.8799419, 2.4270144, -26.9525452, 2.3896003, -25.5217438, 25.6549835
22: -28.5984192, 3.3563683, -28.6693649, 3.3149250, -24.7389984, 24.8653221
23: -22.3783875, 5.7287502, -22.4146843, 5.6896563, -22.0603447, 22.1497612
24: -18.3975296, 9.4753513, -18.4770298, 9.4329119, -22.8946991, 22.9966927
25: -23.8806801, 5.4204803, -24.0026245, 5.3889160, -24.3973389, 24.5713501
26: -41.1931610, -0.4638858, -41.1850662, -0.5035324, -30.7179871, 30.7257843
27: -21.6327095, 8.6112862, -21.7104053, 8.5727634, -26.4759140, 26.5868988
28: -24.2151203, 6.1027670, -24.2572746, 6.0502481, -22.0340347, 22.1269226
29: -27.9151344, -0.1890819, -28.0308952, -0.2191983, -23.9619751, 24.1636696
30: -28.1697083, 3.7799664, -28.3135433, 3.7454615, -26.1076584, 26.3179321
31: -22.7285252, 5.0739841, -22.7954636, 5.0515680, -25.0856247, 25.2018585
32: -23.9914742, 2.3205948, -23.9487419, 2.3969040, -21.5101929, 21.3502274
33: -36.4816055, 3.6528735, -36.4371033, 3.7035742, -33.4788589, 33.3065948
34: -37.9214554, -4.7415810, -37.8761711, -4.7318377, -27.8787842, 27.7201157
35: -32.9560051, 0.3119049, -32.9248581, 0.3453007, -28.2732544, 28.1456833
36: -36.8924255, -0.6410751, -36.8381042, -0.6269879, -29.1314087, 29.0221252
37: -44.6042366, -1.7153492, -44.5911636, -1.7191057, -38.8681641, 38.7822571
38: -44.0284843, 2.8887849, -43.9699936, 2.9676390, -40.8702240, 40.7209396
39: -43.6602592, 3.0097651, -43.6126900, 3.1150885, -41.5454559, 41.3186798
40: -32.7739410, -0.0004430, -32.7706375, 0.0855751, -31.1638947, 31.0382500
41: -20.7759285, 7.2839804, -20.7172565, 7.3269877, -26.5926514, 26.4803925
42: -23.0413017, -0.2126048, -22.9913635, -0.1765134, -18.5217819, 18.4742393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=170, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1646

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5723912, upper bound: 11.5109824
time: 36.36 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5723912, upper bound: 11.5323952
time: 38.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.4191256, 19.1304512, -9.3374023, 19.0645599, -25.3120461, 25.3240662
1: -1.2601738, 22.9103584, -1.1930408, 22.8565559, -19.8069725, 19.8438416
2: -1.6850681, 20.9790020, -1.5934374, 20.9783707, -17.3374290, 17.3066750
3: -9.3812761, 16.5515442, -9.3184843, 16.5639687, -22.0819931, 22.0374680
4: -3.2094488, 22.2630577, -3.1104503, 22.2674255, -21.8055878, 21.7238503
5: -7.8985620, 20.7060394, -7.8249602, 20.6879101, -23.8258820, 23.8331947
6: -28.8722343, -1.3010273, -28.8121243, -1.3556237, -23.2137985, 23.2812881
7: -7.7629309, 21.7103195, -7.6861749, 21.6981602, -23.6810760, 23.6539307
8: -14.9002390, 14.8077307, -14.7743120, 14.8118916, -26.5924377, 26.5253372
9: -5.2817569, 21.3507671, -5.1856785, 21.3097897, -24.3905029, 24.3123779
10: -18.1040497, 17.6210117, -17.8692474, 17.5416489, -31.4894409, 31.3182449
11: -26.8805637, 3.5844989, -26.8205261, 3.5354352, -27.9601898, 27.9501572
12: -34.9247856, -2.2910433, -34.8730011, -2.3292389, -27.2629089, 27.2527695
13: -26.3133698, 15.8290424, -26.2423172, 15.8668880, -34.0934677, 33.9822540
14: -56.0829163, -17.4144859, -55.8818741, -17.6132240, -37.9105301, 37.8639145
15: -14.4356241, 15.5245857, -14.3732090, 15.5187836, -27.9725113, 27.9108124
16: -14.1496592, 20.9307880, -14.0446692, 20.8477764, -31.1719742, 31.1427917
17: -57.9700165, -14.2667561, -57.8224335, -14.4261513, -41.7902756, 41.7566071
18: -21.6550808, 12.2229481, -21.7161942, 12.1612663, -29.6271667, 29.7841415
19: -22.4092770, 3.6085663, -22.3781586, 3.5574536, -22.8628540, 22.8816109
20: -23.4296894, 1.4069505, -23.3980751, 1.3517628, -19.3006248, 19.3013420
21: -26.9359760, 2.4430954, -26.9076385, 2.3779078, -25.5784760, 25.6207314
22: -28.6273556, 3.3946767, -28.6522923, 3.3140085, -24.8181915, 24.8464584
23: -22.4106522, 5.7490549, -22.4048042, 5.6942163, -22.0915947, 22.1453667
24: -18.4233932, 9.4880695, -18.4628220, 9.4377956, -22.9123383, 22.9957047
25: -23.9092102, 5.4397359, -23.9595680, 5.3580928, -24.4381371, 24.5403061
26: -41.2405624, -0.4196267, -41.1814575, -0.4928470, -30.7444229, 30.7526321
27: -21.6918736, 8.6882353, -21.7078266, 8.5694246, -26.5167236, 26.6655579
28: -24.2383671, 6.1464891, -24.2508240, 6.0539088, -22.0551605, 22.1463623
29: -27.9520874, -0.1586869, -27.9772606, -0.2391565, -24.0573540, 24.0954056
30: -28.1942558, 3.7980990, -28.2420082, 3.7058640, -26.1291733, 26.2668953
31: -22.7884789, 5.0904360, -22.7758904, 5.0463133, -25.1438751, 25.1738815
32: -24.0226860, 2.3699906, -23.9314728, 2.3622503, -21.4999352, 21.3935738
33: -36.5507431, 3.7560439, -36.4424095, 3.6678133, -33.5044403, 33.4270096
34: -37.9794312, -4.6389923, -37.8795280, -4.7674565, -27.8641357, 27.8414841
35: -33.0312843, 0.4328690, -32.9284210, 0.3062029, -28.2826614, 28.2768631
36: -36.9850044, -0.5087042, -36.8441277, -0.6639152, -29.1573257, 29.1642838
37: -44.7190285, -1.5897164, -44.5899887, -1.7189350, -38.9434967, 38.9408112
38: -44.1414261, 3.0568523, -43.9581757, 2.8824387, -40.8838959, 40.8770752
39: -43.7216110, 3.0812368, -43.6085129, 3.0762095, -41.5657196, 41.4050980
40: -32.8381042, 0.0886517, -32.7646141, 0.0716496, -31.1994476, 31.1417618
41: -20.8432808, 7.3805914, -20.7050552, 7.2902350, -26.6124573, 26.5742950
42: -23.0471611, -0.1857669, -22.9848461, -0.1906774, -18.5353546, 18.4794388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=170, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=196, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1646

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5723912, upper bound: 11.5081110
time: 35.21 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5723912, upper bound: 11.5294784
time: 31.07 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.3843040, 19.0745296, -9.4172163, 19.1382961, -25.3478966, 25.3413391
1: -1.2305760, 22.8701572, -1.2586951, 22.9167404, -19.8396072, 19.8603783
2: -1.6730084, 20.9456291, -1.6593471, 21.0499325, -17.3599548, 17.3561554
3: -9.3726797, 16.5229950, -9.3771706, 16.6425209, -22.1522560, 22.0597229
4: -3.1899333, 22.2472534, -3.1982088, 22.3227043, -21.8100891, 21.8038483
5: -7.8704677, 20.6507664, -7.8623524, 20.7649632, -23.8741035, 23.8259201
6: -28.8340073, -1.3261118, -28.8701305, -1.3199282, -23.2639618, 23.2861519
7: -7.7322073, 21.6696548, -7.7440443, 21.7431717, -23.6942215, 23.6830673
8: -14.8745747, 14.7968979, -14.8397465, 14.8447399, -26.5866547, 26.5850067
9: -5.2629766, 21.3070717, -5.2461290, 21.3728008, -24.4284134, 24.3626556
10: -18.0787048, 17.5915031, -17.9837761, 17.6283493, -31.5446167, 31.4071350
11: -26.8678570, 3.5791106, -26.9075909, 3.5935993, -28.0029373, 28.0188217
12: -34.9118347, -2.3129506, -34.9395065, -2.2969279, -27.2773209, 27.2797661
13: -26.2957478, 15.8009253, -26.3093395, 15.9167824, -34.1208801, 34.0267410
14: -56.0505829, -17.4922752, -56.0232391, -17.4641438, -38.0148773, 37.9534302
15: -14.4118891, 15.5234041, -14.4249964, 15.5391197, -27.9696884, 27.9656601
16: -14.1208038, 20.8611622, -14.1506071, 20.9409275, -31.2162323, 31.1837997
17: -57.9169998, -14.3399563, -57.9552841, -14.3132505, -41.8517227, 41.8380966
18: -21.6426582, 12.2171593, -21.7545624, 12.2106123, -29.6841354, 29.8042450
19: -22.3690376, 3.5924902, -22.4577255, 3.6043205, -22.8785858, 22.9366150
20: -23.3733845, 1.3904028, -23.4826660, 1.4167304, -19.3087196, 19.3476067
21: -26.9079685, 2.4305224, -26.9908810, 2.4454396, -25.6359863, 25.6591263
22: -28.5959473, 3.3786576, -28.7120972, 3.3706987, -24.8750916, 24.8455124
23: -22.3747311, 5.7288036, -22.4547329, 5.7452092, -22.1132431, 22.1817551
24: -18.3984261, 9.4741974, -18.5121613, 9.4816771, -22.9429321, 23.0234108
25: -23.8916473, 5.4333572, -24.0289211, 5.4418068, -24.5135574, 24.5755768
26: -41.1575127, -0.4457893, -41.2755051, -0.4276934, -30.7628555, 30.8016357
27: -21.6636181, 8.6727228, -21.7603321, 8.6300468, -26.5480881, 26.6956253
28: -24.1918583, 6.1239767, -24.3111973, 6.1120744, -22.0791283, 22.1683006
29: -27.9413929, -0.1628218, -28.0511646, -0.1774136, -24.0958786, 24.1228333
30: -28.1932812, 3.8040719, -28.3214378, 3.8033855, -26.2071114, 26.3118210
31: -22.7620106, 5.0757747, -22.8338509, 5.0942144, -25.1720963, 25.2296524
32: -23.9631519, 2.3405027, -24.0306911, 2.4114125, -21.4904938, 21.4560204
33: -36.4752541, 3.7280183, -36.5381012, 3.7209873, -33.4858322, 33.5025024
34: -37.8976173, -4.6695108, -37.9806442, -4.6931872, -27.8852386, 27.8958588
35: -32.9632378, 0.4047213, -33.0222473, 0.3667779, -28.2913437, 28.3356857
36: -36.8922195, -0.5414524, -36.9659767, -0.5888581, -29.1521225, 29.2434311
37: -44.6279221, -1.6214652, -44.7248764, -1.6987305, -38.8979645, 39.0259781
38: -44.0189438, 3.0135479, -44.1385193, 3.0008073, -40.8976746, 41.0080109
39: -43.6354828, 3.0487909, -43.7284317, 3.1191015, -41.5332031, 41.4847412
40: -32.7810631, 0.0586250, -32.8588791, 0.0893431, -31.1794891, 31.1912918
41: -20.7648258, 7.3542109, -20.8300095, 7.3509722, -26.6163025, 26.6641846
42: -22.9922791, -0.2085629, -23.0501175, -0.1606674, -18.5362854, 18.5109482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=170, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1646

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5476146, upper bound: 11.5511491
time: 29.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5476146, upper bound: 11.5725155
time: 30.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.4233532, 19.1307068, -9.3589630, 19.0813904, -25.3384781, 25.3410835
1: -1.2627983, 22.9111118, -1.2114358, 22.8740540, -19.8343582, 19.8629036
2: -1.6918402, 20.9801674, -1.6205618, 21.0138054, -17.3801880, 17.3288879
3: -9.3888311, 16.5532265, -9.3449688, 16.6076813, -22.1371689, 22.0585175
4: -3.2179089, 22.2641373, -3.1459289, 22.3044853, -21.8522415, 21.7510338
5: -7.8949351, 20.7074738, -7.8230753, 20.7048035, -23.8440704, 23.8483658
6: -28.8746452, -1.3003550, -28.8255501, -1.3373194, -23.2464600, 23.2966347
7: -7.7618532, 21.7113094, -7.6996717, 21.6991920, -23.6848984, 23.6882401
8: -14.9009333, 14.8103809, -14.7803516, 14.8276882, -26.6128044, 26.5401001
9: -5.2836218, 21.3541679, -5.2198277, 21.3237209, -24.4027939, 24.3570671
10: -18.1061878, 17.6325397, -17.9525070, 17.5829849, -31.5341339, 31.4127426
11: -26.8824978, 3.5924835, -26.8968182, 3.5610075, -27.9859543, 28.0358429
12: -34.9259834, -2.2892675, -34.9054909, -2.3152485, -27.2789001, 27.3051758
13: -26.3150635, 15.8305740, -26.2540131, 15.8826361, -34.1106491, 34.0083923
14: -56.0851936, -17.4012432, -55.9604492, -17.5610390, -37.9542236, 37.9845505
15: -14.4369020, 15.5268230, -14.3931828, 15.5297136, -27.9892883, 27.9358826
16: -14.1528139, 20.9352264, -14.1146278, 20.8652477, -31.1874771, 31.2175064
17: -57.9725876, -14.2617722, -57.9025459, -14.3990002, -41.8222046, 41.8766479
18: -21.6576080, 12.2252140, -21.7355175, 12.1761885, -29.6791840, 29.7997208
19: -22.4121208, 3.6076441, -22.4107151, 3.5570486, -22.8693466, 22.9208183
20: -23.4310608, 1.4101813, -23.4227562, 1.3633516, -19.3096313, 19.3304901
21: -26.9396667, 2.4473348, -26.9570465, 2.3915222, -25.5954285, 25.6806488
22: -28.6309185, 3.3937464, -28.6754398, 3.3166075, -24.8321838, 24.8746414
23: -22.4119339, 5.7459259, -22.4166164, 5.6905613, -22.0960350, 22.1745186
24: -18.4254303, 9.4854994, -18.4821758, 9.4337311, -22.9229355, 23.0148392
25: -23.9113846, 5.4482961, -24.0063629, 5.3909788, -24.4695129, 24.5996704
26: -41.2427979, -0.4233022, -41.1881447, -0.4955578, -30.7806931, 30.7638321
27: -21.6953030, 8.6891918, -21.7252941, 8.5752134, -26.5311279, 26.6822968
28: -24.2396507, 6.1442647, -24.2615814, 6.0524230, -22.0668793, 22.1591568
29: -27.9557781, -0.1527395, -28.0371666, -0.2179075, -24.0735245, 24.1631889
30: -28.1966991, 3.8113022, -28.3168201, 3.7484472, -26.1613731, 26.3560944
31: -22.7912655, 5.0920553, -22.7995205, 5.0533066, -25.1541290, 25.2238617
32: -24.0306816, 2.3708777, -23.9584064, 2.3989112, -21.5462265, 21.4141273
33: -36.5533600, 3.7563791, -36.4536819, 3.7048116, -33.5487671, 33.4407959
34: -37.9833908, -4.6365414, -37.8930817, -4.7290015, -27.9370117, 27.8518448
35: -33.0359268, 0.4339819, -32.9454422, 0.3464980, -28.3449936, 28.2907181
36: -36.9904594, -0.5077701, -36.8639679, -0.6254778, -29.2142715, 29.1822052
37: -44.7227554, -1.5935330, -44.6202316, -1.7183948, -38.9724731, 38.9499512
38: -44.1524620, 3.0587101, -43.9996643, 2.9704981, -40.9839172, 40.9195557
39: -43.7254486, 3.0813947, -43.6261673, 3.1169624, -41.6141052, 41.4176025
40: -32.8412209, 0.0887024, -32.7851143, 0.0863357, -31.2312317, 31.1467323
41: -20.8522987, 7.3812599, -20.7367859, 7.3287439, -26.6604614, 26.6011353
42: -23.0484581, -0.1847715, -22.9931526, -0.1741407, -18.5539474, 18.5056915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=170, inp2_unstable=170, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=198, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1646

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5723912, upper bound: 11.5222587
time: 31.20 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5723912, upper bound: 11.5436204
time: 34.92 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 68.66 seconds
IS_A1_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 68.66
Output dim: 2, lower bound: -11.5436207, upper bound: 11.5510285
IS_A1_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 68.66
Output dim: 2, lower bound: -11.5436207, upper bound: 11.5723909
IS_A2_B2_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 68.66
Output dim: 2, lower bound: -11.5258510, upper bound: 11.5504959
IS_A2_B2_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 68.66
Output dim: 2, lower bound: -11.5258510, upper bound: 11.5718431
IS_A2_B2_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 68.66
Output dim: 2, lower bound: -11.5723912, upper bound: 11.4968372
IS_A2_B2_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 68.66
Output dim: 2, lower bound: -11.5723912, upper bound: 11.5182581
IS_A2_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 68.66
Output dim: 2, lower bound: -11.5723912, upper bound: 11.5109824
IS_A2_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 68.66
Output dim: 2, lower bound: -11.5723912, upper bound: 11.5323952
IS_A2_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 68.66
Output dim: 2, lower bound: -11.5723912, upper bound: 11.5081110
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 68.66
Output dim: 2, lower bound: -11.5723912, upper bound: 11.5294784
IS_A2_B2_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 68.66
Output dim: 2, lower bound: -11.5476146, upper bound: 11.5511491
IS_A2_B2_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 68.66
Output dim: 2, lower bound: -11.5476146, upper bound: 11.5725155
IS_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 68.66
Output dim: 2, lower bound: -11.5723912, upper bound: 11.5222587
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 68.66
Output dim: 2, lower bound: -11.5723912, upper bound: 11.5436204

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 47.58 + 3815.91 = 3863.49 seconds
