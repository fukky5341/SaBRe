## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 7200 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.64 + 42.61 = 45.25 seconds
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1748

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5536042, upper bound: 11.5825229
time: 30.81 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5826125, upper bound: 11.5826126
time: 29.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 59.95 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 59.95
Output dim: 2, lower bound: -11.5536042, upper bound: 11.5825229
IS_A2, status: Status.UNKNOWN, split count: 1, time: 59.95
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

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1731

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5519104, upper bound: 11.5444519
time: 29.93 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5519104, upper bound: 11.5806461
time: 31.75 seconds

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

Time for backsubstitution: 2.12 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1731

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5807192, upper bound: 11.5445066
time: 28.70 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5807192, upper bound: 11.5807191
time: 34.08 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 65.05 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 65.05
Output dim: 2, lower bound: -11.5519104, upper bound: 11.5444519
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 65.05
Output dim: 2, lower bound: -11.5519104, upper bound: 11.5806461
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 65.05
Output dim: 2, lower bound: -11.5807192, upper bound: 11.5445066
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 65.05
Output dim: 2, lower bound: -11.5807192, upper bound: 11.5807191

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.3360195, 19.0136089, -9.3484716, 19.0141907, -25.2022095, 25.2144928
1: -1.2051058, 22.8009167, -1.2175608, 22.8014297, -19.7303505, 19.7410545
2: -1.6127644, 20.9129181, -1.6159568, 20.9132576, -17.2494240, 17.2512131
3: -9.3453665, 16.4746552, -9.3461800, 16.4753838, -21.9861679, 21.9861221
4: -3.1329985, 22.2229939, -3.1415668, 22.2229004, -21.7104568, 21.7162819
5: -7.8316870, 20.6016045, -7.8417392, 20.6020451, -23.7337646, 23.7445221
6: -28.8241997, -1.3906941, -28.8251839, -1.3852835, -23.1911697, 23.1897888
7: -7.6893530, 21.6245003, -7.6985312, 21.6258087, -23.5669250, 23.5740852
8: -14.7791157, 14.7498779, -14.7885761, 14.7519131, -26.4722023, 26.4756622
9: -5.1912518, 21.2770119, -5.1973343, 21.2779655, -24.2526398, 24.2579613
10: -17.8766060, 17.5536728, -17.8875103, 17.5559692, -31.2683411, 31.2803726
11: -26.7486992, 3.5639129, -26.7495117, 3.5703039, -27.8607559, 27.8555527
12: -34.8714905, -2.3507152, -34.8825226, -2.3488269, -27.1953049, 27.2056389
13: -26.2459335, 15.6952019, -26.2510128, 15.6974716, -33.8524475, 33.8561783
14: -55.8749084, -17.5586643, -55.9147453, -17.5583019, -37.7348785, 37.7669754
15: -14.3631382, 15.5088387, -14.3638792, 15.5099716, -27.8950424, 27.8940353
16: -14.0390425, 20.8054638, -14.0544052, 20.8061161, -31.0317459, 31.0460281
17: -57.8175812, -14.4304028, -57.8481522, -14.4272537, -41.5875244, 41.6092224
18: -21.5682068, 12.1706629, -21.5728130, 12.1614132, -29.5606613, 29.5706100
19: -22.2672081, 3.5637093, -22.2691727, 3.5762458, -22.7468948, 22.7363853
20: -23.2786865, 1.3664956, -23.2795849, 1.3775544, -19.1767502, 19.1659317
21: -26.7840939, 2.3966708, -26.7860298, 2.4041390, -25.4736633, 25.4685440
22: -28.4864502, 3.3216124, -28.4884434, 3.3302672, -24.6797981, 24.6719704
23: -22.2768288, 5.7023787, -22.2770920, 5.7165570, -21.9971466, 21.9804840
24: -18.2937031, 9.4411001, -18.2949028, 9.4484367, -22.8111420, 22.8029099
25: -23.8119698, 5.3852992, -23.8128395, 5.3878355, -24.3679962, 24.3656158
26: -41.0207520, -0.4706354, -41.0224686, -0.4573278, -30.5800247, 30.5663681
27: -21.5622902, 8.5728245, -21.5644150, 8.5799761, -26.4060287, 26.3996658
28: -24.1064339, 6.0627627, -24.1070671, 6.0739756, -21.9565239, 21.9430542
29: -27.8338223, -0.2206564, -27.8352966, -0.2132883, -23.9315720, 23.9251709
30: -28.1072521, 3.7476025, -28.1076050, 3.7507262, -26.0808258, 26.0801620
31: -22.6578445, 5.0548410, -22.6601486, 5.0582666, -25.0417404, 25.0398560
32: -23.9446087, 2.3177238, -23.9424400, 2.3218913, -21.3790016, 21.3716278
33: -36.4312057, 3.6534142, -36.4314651, 3.6661377, -33.3851929, 33.3736420
34: -37.8592300, -4.7566819, -37.8603287, -4.7423959, -27.7677536, 27.7544556
35: -32.9235458, 0.3034945, -32.9256439, 0.3152843, -28.1930771, 28.1835785
36: -36.8416634, -0.6627622, -36.8440323, -0.6428919, -29.0488129, 29.0299530
37: -44.5629349, -1.7151337, -44.5649185, -1.6980677, -38.7975769, 38.7838287
38: -43.9592171, 2.8788586, -43.9627151, 2.8930774, -40.7169647, 40.6993484
39: -43.6026459, 3.0177627, -43.5991821, 3.0216904, -41.3859863, 41.3783722
40: -32.7405014, -0.0088863, -32.7366219, -0.0086753, -31.0239639, 31.0213890
41: -20.7316074, 7.2730265, -20.7301617, 7.2888908, -26.5016174, 26.4779282
42: -22.9909973, -0.2123787, -22.9871063, -0.2051594, -18.4633789, 18.4536934

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1731

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5134431, upper bound: 11.5444519
time: 30.86 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5134431, upper bound: 11.5444519
time: 36.36 seconds

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

Time for backsubstitution: 2.13 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1745

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5292498, upper bound: 11.5791595
time: 30.47 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5509860, upper bound: 11.5797208
time: 27.17 seconds

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

Time for backsubstitution: 2.12 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1731

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5445065, upper bound: 11.5445066
time: 31.17 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5445065, upper bound: 11.5445066
time: 34.62 seconds

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

Time for backsubstitution: 2.12 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1745

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5581328, upper bound: 11.5792189
time: 33.47 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5797930, upper bound: 11.5797931
time: 29.86 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 65.57 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 65.57
Output dim: 2, lower bound: -11.5134431, upper bound: 11.5444519
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 65.57
Output dim: 2, lower bound: -11.5134431, upper bound: 11.5444519
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 65.57
Output dim: 2, lower bound: -11.5292498, upper bound: 11.5791595
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 65.57
Output dim: 2, lower bound: -11.5509860, upper bound: 11.5797208
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 65.57
Output dim: 2, lower bound: -11.5445065, upper bound: 11.5445066
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 65.57
Output dim: 2, lower bound: -11.5445065, upper bound: 11.5445066
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 65.57
Output dim: 2, lower bound: -11.5581328, upper bound: 11.5792189
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 65.57
Output dim: 2, lower bound: -11.5797930, upper bound: 11.5797931

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.3067188, 19.0128098, -9.3484716, 19.0141907, -25.1718292, 25.2138329
1: -1.1853957, 22.7998829, -1.2175608, 22.8014297, -19.7131233, 19.7400360
2: -1.5873785, 20.9118576, -1.6159568, 20.9132576, -17.2255974, 17.2501831
3: -9.3215847, 16.4709625, -9.3461800, 16.4753838, -21.9626083, 21.9820633
4: -3.1068544, 22.2216301, -3.1415668, 22.2229004, -21.6860352, 21.7147980
5: -7.8126173, 20.5981865, -7.8417392, 20.6020451, -23.7186661, 23.7396355
6: -28.8212509, -1.3968229, -28.8251839, -1.3852835, -23.1883316, 23.1745911
7: -7.6678019, 21.6234169, -7.6985312, 21.6258087, -23.5484390, 23.5727539
8: -14.7474861, 14.7476587, -14.7885761, 14.7519131, -26.4412575, 26.4733810
9: -5.1785088, 21.2753735, -5.1973343, 21.2779655, -24.2327347, 24.2561073
10: -17.8661041, 17.5510292, -17.8875103, 17.5559692, -31.2570801, 31.2751999
11: -26.7457161, 3.5469766, -26.7495117, 3.5703039, -27.8575287, 27.8388748
12: -34.8659668, -2.3583064, -34.8825226, -2.3488269, -27.1900787, 27.1986694
13: -26.2120361, 15.6916962, -26.2510128, 15.6974716, -33.8188553, 33.8528366
14: -55.8714981, -17.5649376, -55.9147453, -17.5583019, -37.7225494, 37.7614365
15: -14.3455362, 15.5046902, -14.3638792, 15.5099716, -27.8767166, 27.8896790
16: -14.0279951, 20.8043327, -14.0544052, 20.8061161, -31.0213318, 31.0450592
17: -57.8146935, -14.4364719, -57.8481522, -14.4272537, -41.5779572, 41.6039581
18: -21.5646839, 12.1369715, -21.5728130, 12.1614132, -29.5571365, 29.5386429
19: -22.2631645, 3.5405970, -22.2691727, 3.5762458, -22.7432251, 22.7132568
20: -23.2769699, 1.3380399, -23.2795849, 1.3775544, -19.1749573, 19.1391296
21: -26.7804871, 2.3654311, -26.7860298, 2.4041390, -25.4699097, 25.4371605
22: -28.4840736, 3.2908568, -28.4884434, 3.3302672, -24.6772881, 24.6426086
23: -22.2741356, 5.6770315, -22.2770920, 5.7165570, -21.9942398, 21.9587631
24: -18.2910404, 9.4137239, -18.2949028, 9.4484367, -22.8085632, 22.7776794
25: -23.8093166, 5.3530159, -23.8128395, 5.3878355, -24.3649521, 24.3346634
26: -41.0189285, -0.5098982, -41.0224686, -0.4573278, -30.5779495, 30.5303879
27: -21.5595741, 8.5407829, -21.5644150, 8.5799761, -26.4036255, 26.3678284
28: -24.1047268, 6.0310130, -24.1070671, 6.0739756, -21.9544792, 21.9142189
29: -27.8313217, -0.2420034, -27.8352966, -0.2132883, -23.9287949, 23.9044189
30: -28.1052990, 3.7124529, -28.1076050, 3.7507262, -26.0785904, 26.0460510
31: -22.6537285, 5.0288239, -22.6601486, 5.0582666, -25.0379410, 25.0134926
32: -23.9382534, 2.3128891, -23.9424400, 2.3218913, -21.3718834, 21.3663177
33: -36.4242134, 3.6505089, -36.4314651, 3.6661377, -33.3757935, 33.3695374
34: -37.8573380, -4.7694187, -37.8603287, -4.7423959, -27.7657394, 27.7432098
35: -32.9204407, 0.2991939, -32.9256439, 0.3152843, -28.1899872, 28.1793137
36: -36.8391724, -0.6707854, -36.8440323, -0.6428919, -29.0467758, 29.0214005
37: -44.5538826, -1.7171316, -44.5649185, -1.6980677, -38.7864532, 38.7796402
38: -43.9553032, 2.8679633, -43.9627151, 2.8930774, -40.7137451, 40.6867294
39: -43.5858803, 3.0161948, -43.5991821, 3.0216904, -41.3710480, 41.3764954
40: -32.7237091, -0.0114102, -32.7366219, -0.0086753, -31.0079422, 31.0187187
41: -20.7233181, 7.2699695, -20.7301617, 7.2888908, -26.4924698, 26.4730835
42: -22.9850540, -0.2157252, -22.9871063, -0.2051594, -18.4577866, 18.4501228

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1745

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5341474, upper bound: 11.5218984
time: 34.99 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5347685, upper bound: 11.5435465
time: 31.35 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.3568544, 19.0566006, -9.3484716, 19.0141907, -25.2231445, 25.2588425
1: -1.2139897, 22.8382607, -1.2175608, 22.8014297, -19.7415047, 19.7792511
2: -1.6201277, 20.9654961, -1.6159568, 20.9132576, -17.2581787, 17.3029633
3: -9.3469591, 16.5430851, -9.3461800, 16.4753838, -21.9885178, 22.0519943
4: -3.1416936, 22.2540512, -3.1415668, 22.2229004, -21.7201996, 21.7441330
5: -7.8338943, 20.6638451, -7.8417392, 20.6020451, -23.7306213, 23.7939072
6: -28.8302402, -1.3674622, -28.8251839, -1.3852835, -23.1975327, 23.2004623
7: -7.6984968, 21.6648483, -7.6985312, 21.6258087, -23.5765457, 23.6128044
8: -14.7848434, 14.8013144, -14.7885761, 14.7519131, -26.4796333, 26.5273285
9: -5.1961622, 21.3042450, -5.1973343, 21.2779655, -24.2572327, 24.2981796
10: -17.8859653, 17.5747833, -17.8875103, 17.5559692, -31.2771606, 31.2966003
11: -26.8266163, 3.5693851, -26.7495117, 3.5703039, -27.9430771, 27.8614426
12: -34.8781204, -2.3108048, -34.8825226, -2.3488269, -27.2027359, 27.2466965
13: -26.2521820, 15.8299246, -26.2510128, 15.6974716, -33.8618927, 33.9950562
14: -55.8881989, -17.5483112, -55.9147453, -17.5583019, -37.7357635, 37.7802124
15: -14.3711576, 15.5317411, -14.3638792, 15.5099716, -27.9052887, 27.9179916
16: -14.0630016, 20.8326359, -14.0544052, 20.8061161, -31.0528641, 31.0790405
17: -57.8384857, -14.3867722, -57.8481522, -14.4272537, -41.6074677, 41.6611023
18: -21.6802025, 12.1735620, -21.5728130, 12.1614132, -29.6754532, 29.5760269
19: -22.3578320, 3.5658658, -22.2691727, 3.5762458, -22.8370209, 22.7389641
20: -23.3739567, 1.3702793, -23.2795849, 1.3775544, -19.2748108, 19.1732864
21: -26.8948956, 2.3989692, -26.7860298, 2.4041390, -25.5837784, 25.4705391
22: -28.6126232, 3.3260703, -28.4884434, 3.3302672, -24.8078346, 24.6760025
23: -22.3697052, 5.7052789, -22.2770920, 5.7165570, -22.0884933, 21.9826088
24: -18.4262371, 9.4428396, -18.2949028, 9.4484367, -22.9452209, 22.8064384
25: -23.9247208, 5.3907452, -23.8128395, 5.3878355, -24.4819336, 24.3732986
26: -41.1290817, -0.4699316, -41.0224686, -0.4573278, -30.6871262, 30.5669327
27: -21.6832428, 8.5753469, -21.5644150, 8.5799761, -26.5308914, 26.4035568
28: -24.2092285, 6.0632868, -24.1070671, 6.0739756, -22.0564232, 21.9437561
29: -27.9468822, -0.2155149, -27.8352966, -0.2132883, -24.0483093, 23.9314575
30: -28.2288589, 3.7523618, -28.1076050, 3.7507262, -26.2038116, 26.0896988
31: -22.7546196, 5.0589657, -22.6601486, 5.0582666, -25.1383438, 25.0442963
32: -23.9585266, 2.3529706, -23.9424400, 2.3218913, -21.3946877, 21.4056015
33: -36.4492264, 3.6724138, -36.4314651, 3.6661377, -33.3982697, 33.3955994
34: -37.8889313, -4.7467809, -37.8603287, -4.7423959, -27.7961349, 27.7665787
35: -32.9386559, 0.3228903, -32.9256439, 0.3152843, -28.2071304, 28.2056580
36: -36.8602524, -0.6473083, -36.8440323, -0.6428919, -29.0683594, 29.0462799
37: -44.6015739, -1.7005639, -44.5649185, -1.6980677, -38.8354034, 38.7990570
38: -43.9890060, 2.9032393, -43.9627151, 2.8930774, -40.7610474, 40.7243881
39: -43.6242905, 3.0672779, -43.5991821, 3.0216904, -41.4042206, 41.4288940
40: -32.7712326, 0.0527761, -32.7366219, -0.0086753, -31.0507278, 31.0828896
41: -20.7397003, 7.2923174, -20.7301617, 7.2888908, -26.5100937, 26.4936066
42: -22.9926033, -0.1901507, -22.9871063, -0.2051594, -18.4664459, 18.4791832

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1745

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5341474, upper bound: 11.5218984
time: 28.48 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5347685, upper bound: 11.5435465
time: 29.20 seconds

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

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1741

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5274331, upper bound: 11.5553538
time: 34.04 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5286038, upper bound: 11.5785170
time: 45.21 seconds

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

Time for backsubstitution: 2.14 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1741

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5492303, upper bound: 11.5559611
time: 29.87 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5503477, upper bound: 11.5790874
time: 33.35 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.3756618, 19.0543156, -9.3628283, 19.0146866, -25.2343826, 25.2721558
1: -1.2352605, 22.8427944, -1.2288017, 22.8018913, -19.7521172, 19.7958069
2: -1.6286895, 20.9465561, -1.6260252, 20.9136963, -17.2561684, 17.2981377
3: -9.3524075, 16.5226154, -9.3540754, 16.4768906, -21.9911613, 22.0374641
4: -3.1589170, 22.2454643, -3.1536984, 22.2232609, -21.7287216, 21.7425957
5: -7.8534589, 20.6506100, -7.8520026, 20.6033497, -23.7508621, 23.8007126
6: -28.8337688, -1.3748431, -28.8264809, -1.3831568, -23.2154694, 23.2017136
7: -7.7163877, 21.6632004, -7.7094879, 21.6265488, -23.5887146, 23.6235847
8: -14.8032885, 14.7898655, -14.8030310, 14.7530880, -26.4849586, 26.5188293
9: -5.2072911, 21.3018799, -5.2035484, 21.2788258, -24.2625122, 24.3148041
10: -17.8996849, 17.5901642, -17.8939362, 17.5577221, -31.2914734, 31.3251343
11: -26.8108349, 3.5806642, -26.7505989, 3.5786185, -27.9330902, 27.8681946
12: -34.8927689, -2.3097644, -34.8886871, -2.3458242, -27.2179489, 27.2539749
13: -26.2662354, 15.7831783, -26.2648048, 15.6993599, -33.8675003, 33.9607162
14: -55.9376373, -17.4873943, -55.9303665, -17.5561714, -37.7829742, 37.8553772
15: -14.3725433, 15.5184460, -14.3678417, 15.5114250, -27.9075546, 27.9087601
16: -14.0778923, 20.8608608, -14.0637474, 20.8066769, -31.0631714, 31.1108780
17: -57.8741531, -14.3401232, -57.8589783, -14.4240742, -41.6395416, 41.7157593
18: -21.6160374, 12.1701279, -21.5754013, 12.1703568, -29.6207123, 29.5732574
19: -22.3476086, 3.5887322, -22.2711296, 3.5893285, -22.8414383, 22.7550774
20: -23.3577423, 1.3900504, -23.2804871, 1.3914757, -19.2746277, 19.1757545
21: -26.8714409, 2.4159393, -26.7878780, 2.4182198, -25.5764389, 25.4765320
22: -28.5908184, 3.3413234, -28.4898567, 3.3440156, -24.8033218, 24.6809120
23: -22.3519287, 5.7306023, -22.2779083, 5.7305822, -22.0897789, 22.0017014
24: -18.3922195, 9.4591064, -18.2961922, 9.4608564, -22.9248047, 22.8150826
25: -23.8772869, 5.4020061, -23.8140602, 5.4002705, -24.4493256, 24.3742142
26: -41.1219940, -0.4426603, -41.0237579, -0.4384727, -30.7037659, 30.5792313
27: -21.6583691, 8.5925064, -21.5660458, 8.5940685, -26.5189819, 26.4133759
28: -24.1888466, 6.0870891, -24.1078510, 6.0891986, -22.0578499, 21.9557877
29: -27.9345493, -0.2037940, -27.8365479, -0.2031064, -24.0460358, 23.9343033
30: -28.1838589, 3.7649460, -28.1083527, 3.7641497, -26.1731033, 26.0845566
31: -22.7186203, 5.0688772, -22.6622448, 5.0688767, -25.1163254, 25.0516624
32: -23.9600296, 2.3287024, -23.9438858, 2.3229604, -21.4034805, 21.3842812
33: -36.4687691, 3.6702166, -36.4338799, 3.6680822, -33.4208221, 33.3978729
34: -37.8996506, -4.7248616, -37.8612747, -4.7336016, -27.8222504, 27.7834473
35: -32.9624557, 0.3289561, -32.9274254, 0.3197064, -28.2379456, 28.2105255
36: -36.8931808, -0.6249933, -36.8457108, -0.6327519, -29.1129303, 29.0663528
37: -44.6208992, -1.6907425, -44.5684738, -1.6936407, -38.8597717, 38.8119354
38: -44.0099335, 2.9165349, -43.9652405, 2.9010420, -40.7943268, 40.7441559
39: -43.6271935, 3.0335827, -43.6036072, 3.0214810, -41.4267426, 41.3989105
40: -32.7611847, 0.0086136, -32.7407303, -0.0079274, -31.0493851, 31.0433044
41: -20.7641373, 7.2962494, -20.7323608, 7.2942576, -26.5577011, 26.5062485
42: -22.9934158, -0.2010398, -22.9876213, -0.2034159, -18.4690781, 18.4666023

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1745

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5638339, upper bound: 11.5219665
time: 39.94 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5644191, upper bound: 11.5435922
time: 42.07 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.4256420, 19.0980949, -9.3628283, 19.0146866, -25.2855453, 25.3162041
1: -1.2638097, 22.8811398, -1.2288017, 22.8018913, -19.7804604, 19.8340607
2: -1.6613636, 21.0001602, -1.6260252, 20.9136963, -17.2885818, 17.3499794
3: -9.3777876, 16.5949364, -9.3540754, 16.4768906, -22.0170708, 22.1072502
4: -3.1936378, 22.2778587, -3.1536984, 22.2232609, -21.7627792, 21.7719765
5: -7.8747320, 20.7163658, -7.8520026, 20.6033497, -23.7627029, 23.8546066
6: -28.8427086, -1.3458548, -28.8264809, -1.3831568, -23.2245941, 23.2271004
7: -7.7470846, 21.7045326, -7.7094879, 21.6265488, -23.6165543, 23.6630287
8: -14.8406096, 14.8434410, -14.8030310, 14.7530880, -26.5232430, 26.5714569
9: -5.2248583, 21.3304653, -5.2035484, 21.2788258, -24.2868729, 24.3565903
10: -17.9191170, 17.6137123, -17.8939362, 17.5577221, -31.3110962, 31.3465729
11: -26.8917847, 3.6030407, -26.7505989, 3.5786185, -28.0186691, 27.8907013
12: -34.9047089, -2.2622948, -34.8886871, -2.3458242, -27.2305679, 27.3017616
13: -26.3063316, 15.9211750, -26.2648048, 15.6993599, -33.9104385, 34.1007538
14: -55.9544411, -17.4705467, -55.9303665, -17.5561714, -37.7961273, 37.8742752
15: -14.3979950, 15.5454865, -14.3678417, 15.5114250, -27.9358673, 27.9370880
16: -14.1127071, 20.8885880, -14.0637474, 20.8066769, -31.0946198, 31.1449280
17: -57.8979874, -14.2903147, -57.8589783, -14.4240742, -41.6690521, 41.7729187
18: -21.7313919, 12.2066660, -21.5754013, 12.1703568, -29.7376175, 29.6104584
19: -22.4422588, 3.6139731, -22.2711296, 3.5893285, -22.9348679, 22.7807350
20: -23.4547691, 1.4222739, -23.2804871, 1.3914757, -19.3724365, 19.2099266
21: -26.9858513, 2.4494786, -26.7878780, 2.4182198, -25.6892776, 25.5099640
22: -28.7193222, 3.3765488, -28.4898567, 3.3440156, -24.9317398, 24.7137794
23: -22.4475574, 5.7587214, -22.2779083, 5.7305822, -22.1822472, 22.0251007
24: -18.5273781, 9.4882030, -18.2961922, 9.4608564, -23.0592575, 22.8435364
25: -23.9921417, 5.4396620, -23.8140602, 5.4002705, -24.5653000, 24.4121628
26: -41.2321892, -0.4027643, -41.0237579, -0.4384727, -30.8104324, 30.6144867
27: -21.7820415, 8.6270390, -21.5660458, 8.5940685, -26.6444473, 26.4491043
28: -24.2933121, 6.1193314, -24.1078510, 6.0891986, -22.1576653, 21.9847412
29: -28.0501823, -0.1773484, -27.8365479, -0.2031064, -24.1639862, 23.9611282
30: -28.3074379, 3.8046980, -28.1083527, 3.7641497, -26.2959976, 26.1280289
31: -22.8194675, 5.0989442, -22.6622448, 5.0688767, -25.2163544, 25.0824356
32: -23.9801846, 2.3686450, -23.9438858, 2.3229604, -21.4263992, 21.4234657
33: -36.4938812, 3.6918592, -36.4338799, 3.6680822, -33.4434357, 33.4242401
34: -37.9311905, -4.7023230, -37.8612747, -4.7336016, -27.8518677, 27.8062820
35: -32.9805374, 0.3522482, -32.9274254, 0.3197064, -28.2550201, 28.2362137
36: -36.9142227, -0.6019154, -36.8457108, -0.6327519, -29.1341858, 29.0908127
37: -44.6685333, -1.6743331, -44.5684738, -1.6936407, -38.9088440, 38.8309326
38: -44.0436172, 2.9509354, -43.9652405, 2.9010420, -40.8414307, 40.7813110
39: -43.6650162, 3.0847378, -43.6036072, 3.0214810, -41.4601440, 41.4510345
40: -32.8084106, 0.0726302, -32.7407303, -0.0079274, -31.0920944, 31.1074142
41: -20.7804546, 7.3176131, -20.7323608, 7.2942576, -26.5753937, 26.5264282
42: -23.0010128, -0.1755774, -22.9876213, -0.2034159, -18.4777222, 18.4955635

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1745

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5638339, upper bound: 11.5219665
time: 34.38 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5644191, upper bound: 11.5435922
time: 32.38 seconds

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

Time for backsubstitution: 2.13 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1741

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5563451, upper bound: 11.5554252
time: 35.25 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5574892, upper bound: 11.5785817
time: 39.77 seconds

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

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1741

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5780694, upper bound: 11.5560426
time: 34.68 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5791628, upper bound: 11.5791627
time: 31.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 68.45 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5341474, upper bound: 11.5218984
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5347685, upper bound: 11.5435465
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5341474, upper bound: 11.5218984
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5347685, upper bound: 11.5435465
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5274331, upper bound: 11.5553538
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5286038, upper bound: 11.5785170
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5492303, upper bound: 11.5559611
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5503477, upper bound: 11.5790874
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5638339, upper bound: 11.5219665
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5644191, upper bound: 11.5435922
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5638339, upper bound: 11.5219665
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5644191, upper bound: 11.5435922
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5563451, upper bound: 11.5554252
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5574892, upper bound: 11.5785817
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5780694, upper bound: 11.5560426
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 68.45
Output dim: 2, lower bound: -11.5791628, upper bound: 11.5791627

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.2619057, 19.0113411, -9.2099638, 18.9784756, -25.0869865, 25.0740128
1: -1.1543126, 22.7987480, -1.1247091, 22.7679443, -19.6448250, 19.6458511
2: -1.5662446, 20.9108486, -1.5519528, 20.8952637, -17.1858292, 17.1854134
3: -9.3006706, 16.4677048, -9.2851305, 16.4450684, -21.9051170, 21.9162636
4: -3.0777612, 22.2203503, -3.0516801, 22.2131233, -21.6464996, 21.6230087
5: -7.7820306, 20.5958843, -7.7506614, 20.5647049, -23.6507797, 23.6464081
6: -28.8174896, -1.4240484, -28.7862110, -1.4676924, -23.0975189, 23.1033287
7: -7.6384583, 21.6221809, -7.6093922, 21.5987396, -23.4908524, 23.4803619
8: -14.7016373, 14.7447500, -14.6530638, 14.6968813, -26.3407211, 26.3353653
9: -5.1516542, 21.2729988, -5.1167264, 21.2504864, -24.1790237, 24.1750908
10: -17.8312721, 17.5469933, -17.7824326, 17.5236435, -31.1884613, 31.1668701
11: -26.7395840, 3.5311084, -26.7238426, 3.5223441, -27.8008041, 27.7983093
12: -34.8606071, -2.3780303, -34.8649101, -2.4128733, -27.1182022, 27.1577377
13: -26.2072315, 15.6764450, -26.2327194, 15.6411381, -33.7503738, 33.8090668
14: -55.8399506, -17.5703545, -55.8158607, -17.6045799, -37.5850220, 37.6464081
15: -14.3092823, 15.5020037, -14.2521553, 15.4631672, -27.7929306, 27.7738190
16: -14.0067072, 20.8035774, -13.9898920, 20.7874336, -30.9819183, 30.9822311
17: -57.7894897, -14.4442320, -57.7714157, -14.4757347, -41.4563751, 41.5015945
18: -21.5583019, 12.1185665, -21.5203876, 12.1065111, -29.4886169, 29.4610596
19: -22.2575035, 3.5183544, -22.2179794, 3.5106246, -22.6718216, 22.6391296
20: -23.2743950, 1.3140326, -23.2257500, 1.3062720, -19.1014633, 19.0591125
21: -26.7742195, 2.3414400, -26.7312717, 2.3329551, -25.3882370, 25.3586617
22: -28.4805565, 3.2729912, -28.4418297, 3.2752361, -24.6187592, 24.5770874
23: -22.2711029, 5.6603217, -22.2447815, 5.6652889, -21.9397697, 21.9107132
24: -18.2884598, 9.3913031, -18.2414818, 9.3832397, -22.7418442, 22.7021561
25: -23.8063622, 5.3302040, -23.7824059, 5.3159266, -24.2901535, 24.2857056
26: -41.0160713, -0.5354342, -40.9794312, -0.5320115, -30.5012131, 30.4579544
27: -21.5553322, 8.5250854, -21.5311909, 8.5337744, -26.3528214, 26.3134308
28: -24.1023254, 6.0056105, -24.0678978, 5.9967566, -21.8752441, 21.8492393
29: -27.8250008, -0.2478092, -27.7986813, -0.2342975, -23.8948631, 23.8551407
30: -28.1029892, 3.6899033, -28.0698586, 3.6807022, -26.0120583, 25.9962044
31: -22.6478920, 5.0055599, -22.6045170, 4.9891143, -24.9613342, 24.9332199
32: -23.9350662, 2.2903781, -23.9012280, 2.2536697, -21.2985229, 21.2978210
33: -36.4195633, 3.5973439, -36.3382950, 3.5115590, -33.2165680, 33.2232361
34: -37.8550606, -4.8107738, -37.7950363, -4.8645773, -27.6375275, 27.6330948
35: -32.9171028, 0.2500873, -32.8523178, 0.1724210, -28.0437698, 28.0578079
36: -36.8359146, -0.7160587, -36.7776489, -0.7758284, -28.9105835, 28.9108276
37: -44.5465736, -1.7487993, -44.4882393, -1.7907724, -38.6857147, 38.6692657
38: -43.9489670, 2.8090520, -43.8666534, 2.7207203, -40.5361176, 40.5347595
39: -43.5784950, 2.9619679, -43.4887581, 2.8643003, -41.2061157, 41.2136688
40: -32.7166748, -0.0284090, -32.6769257, -0.0597186, -30.9457779, 30.9366760
41: -20.7186470, 7.2471590, -20.6877174, 7.2200165, -26.4142380, 26.3874359
42: -22.9793701, -0.2234409, -22.9681568, -0.2320774, -18.4213295, 18.4202461

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1741

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5406721
time: 40.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5334983, upper bound: 11.5418773
time: 30.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.3040304, 19.0127144, -9.3403683, 19.0138931, -25.1699181, 25.1706772
1: -1.1834908, 22.7997360, -1.2117133, 22.8010521, -19.7117119, 19.7021942
2: -1.5861011, 20.9117355, -1.6124015, 20.9128532, -17.2241058, 17.2361794
3: -9.3203535, 16.4706306, -9.3423758, 16.4743710, -21.9645195, 21.9765968
4: -3.1052041, 22.2214527, -3.1370063, 22.2223740, -21.6815071, 21.7055016
5: -7.8107791, 20.5979748, -7.8361025, 20.6012840, -23.7160416, 23.7198677
6: -28.8208809, -1.3979349, -28.8240585, -1.3886433, -23.1739883, 23.1717148
7: -7.6659465, 21.6233444, -7.6930199, 21.6254921, -23.5460663, 23.5551758
8: -14.7448540, 14.7472801, -14.7805443, 14.7508345, -26.4374847, 26.4447174
9: -5.1766710, 21.2752094, -5.1920033, 21.2774849, -24.2307663, 24.2358551
10: -17.8635082, 17.5506744, -17.8799477, 17.5549583, -31.2548065, 31.2669373
11: -26.7449837, 3.5454459, -26.7474957, 3.5662756, -27.8498764, 27.8350983
12: -34.8653069, -2.3595352, -34.8808365, -2.3525310, -27.1900940, 27.1933403
13: -26.2114639, 15.6904488, -26.2494049, 15.6935616, -33.8183517, 33.8414764
14: -55.8687973, -17.5653934, -55.9080887, -17.5596104, -37.7662048, 37.7426224
15: -14.3436909, 15.5044317, -14.3589230, 15.5091219, -27.8736038, 27.8682861
16: -14.0254297, 20.8042355, -14.0474062, 20.8058739, -31.0085144, 31.0291595
17: -57.8121872, -14.4371262, -57.8420181, -14.4292221, -41.6088257, 41.5868607
18: -21.5641727, 12.1356564, -21.5713978, 12.1574192, -29.5515289, 29.5377083
19: -22.2625961, 3.5393276, -22.2675896, 3.5727024, -22.7331314, 22.7105942
20: -23.2767086, 1.3365798, -23.2789192, 1.3738618, -19.1360626, 19.1374168
21: -26.7798882, 2.3639877, -26.7843494, 2.4005690, -25.4521561, 25.4340591
22: -28.4837227, 3.2896321, -28.4874153, 3.3265643, -24.6475792, 24.6405754
23: -22.2737751, 5.6759629, -22.2761326, 5.7132525, -21.9755936, 21.9566727
24: -18.2908325, 9.4122877, -18.2941895, 9.4440918, -22.7782516, 22.7758102
25: -23.8090019, 5.3516302, -23.8119202, 5.3840055, -24.3453293, 24.3323517
26: -41.0186920, -0.5118098, -41.0218353, -0.4630947, -30.5427322, 30.5279312
27: -21.5592651, 8.5398388, -21.5634995, 8.5773478, -26.3987503, 26.3700333
28: -24.1044922, 6.0294814, -24.1063232, 6.0693002, -21.9365921, 21.9118538
29: -27.8299923, -0.2424591, -27.8314209, -0.2146490, -23.9274483, 23.9006958
30: -28.1050110, 3.7107856, -28.1067753, 3.7463343, -26.0558090, 26.0424919
31: -22.6531448, 5.0278249, -22.6585426, 5.0554152, -25.0170670, 25.0110855
32: -23.9379139, 2.3115537, -23.9415035, 2.3176980, -21.3443985, 21.3638840
33: -36.4237862, 3.6473303, -36.4300919, 3.6562834, -33.3046722, 33.3650818
34: -37.8571129, -4.7721648, -37.8596382, -4.7506866, -27.7074280, 27.7396164
35: -32.9201050, 0.2961454, -32.9244995, 0.3058343, -28.1470413, 28.1752014
36: -36.8388824, -0.6736646, -36.8430672, -0.6516628, -29.0139771, 29.0176773
37: -44.5531769, -1.7190995, -44.5627823, -1.7040801, -38.7787933, 38.7783966
38: -43.9548645, 2.8641272, -43.9614296, 2.8811960, -40.6636963, 40.6817703
39: -43.5850830, 3.0128975, -43.5967789, 3.0113735, -41.3006744, 41.3710785
40: -32.7231178, -0.0125506, -32.7347298, -0.0121706, -31.0024567, 31.0154572
41: -20.7228966, 7.2685885, -20.7288437, 7.2846155, -26.4830170, 26.4873810
42: -22.9843292, -0.2169297, -22.9850445, -0.2088299, -18.4575424, 18.4468002

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1491
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1741

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5625290
time: 48.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5341288, upper bound: 11.5636658
time: 34.48 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.3120489, 19.0551186, -9.2099638, 18.9784756, -25.1383400, 25.1190186
1: -1.1828718, 22.8371353, -1.1247091, 22.7679443, -19.6732674, 19.6850548
2: -1.5989573, 20.9644775, -1.5519528, 20.8952637, -17.2183990, 17.2381783
3: -9.3260651, 16.5398407, -9.2851305, 16.4450684, -21.9310188, 21.9862022
4: -3.1125503, 22.2527790, -3.0516801, 22.2131233, -21.6806717, 21.6523476
5: -7.8033786, 20.6615353, -7.7506614, 20.5647049, -23.6627274, 23.7007103
6: -28.8264809, -1.3946981, -28.7862110, -1.4676924, -23.1067047, 23.1292000
7: -7.6691489, 21.6635704, -7.6093922, 21.5987396, -23.5189743, 23.5204239
8: -14.7390070, 14.7984390, -14.6530638, 14.6968813, -26.3791199, 26.3892899
9: -5.1692867, 21.3018951, -5.1167264, 21.2504864, -24.2035217, 24.2171783
10: -17.8511581, 17.5707340, -17.7824326, 17.5236435, -31.2085419, 31.1882706
11: -26.8204803, 3.5535221, -26.7238426, 3.5223441, -27.8863449, 27.8208847
12: -34.8728027, -2.3304119, -34.8649101, -2.4128733, -27.1308365, 27.2058182
13: -26.2473812, 15.8146782, -26.2327194, 15.6411381, -33.7934494, 33.9513245
14: -55.8566742, -17.5537300, -55.8158607, -17.6045799, -37.5982361, 37.6651917
15: -14.3348885, 15.5290451, -14.2521553, 15.4631672, -27.8214722, 27.8022003
16: -14.0417223, 20.8318710, -13.9898920, 20.7874336, -31.0134201, 31.0162430
17: -57.8132820, -14.3944740, -57.7714157, -14.4757347, -41.4858856, 41.5587387
18: -21.6738167, 12.1551352, -21.5203876, 12.1065111, -29.6069336, 29.4984055
19: -22.3521957, 3.5436409, -22.2179794, 3.5106246, -22.7656326, 22.6647835
20: -23.3714237, 1.3462934, -23.2257500, 1.3062720, -19.2013245, 19.0932617
21: -26.8886223, 2.3749690, -26.7312717, 2.3329551, -25.5021057, 25.3920784
22: -28.6091099, 3.3082321, -28.4418297, 3.2752361, -24.7493362, 24.6104889
23: -22.3667641, 5.6885796, -22.2447815, 5.6652889, -22.0340233, 21.9345589
24: -18.4236488, 9.4204865, -18.2414818, 9.3832397, -22.8785172, 22.7308960
25: -23.9218006, 5.3679004, -23.7824059, 5.3159266, -24.4071579, 24.3243561
26: -41.1262321, -0.4954472, -40.9794312, -0.5320115, -30.6103897, 30.4944763
27: -21.6790009, 8.5596418, -21.5311909, 8.5337744, -26.4800491, 26.3491669
28: -24.2068367, 6.0379181, -24.0678978, 5.9967566, -21.9772110, 21.8787956
29: -27.9405766, -0.2213125, -27.7986813, -0.2342975, -24.0144005, 23.8821030
30: -28.2265739, 3.7298007, -28.0698586, 3.6807022, -26.1372643, 26.0398560
31: -22.7487755, 5.0356951, -22.6045170, 4.9891143, -25.0617523, 24.9639969
32: -23.9552898, 2.3304703, -23.9012280, 2.2536697, -21.3212891, 21.3371658
33: -36.4445877, 3.6192522, -36.3382950, 3.5115590, -33.2389984, 33.2493362
34: -37.8866653, -4.7881937, -37.7950363, -4.8645773, -27.6679382, 27.6564255
35: -32.9352341, 0.2737613, -32.8523178, 0.1724210, -28.0609283, 28.0841980
36: -36.8569756, -0.6926122, -36.7776489, -0.7758284, -28.9321747, 28.9357147
37: -44.5942001, -1.7321630, -44.4882393, -1.7907724, -38.7346649, 38.6887360
38: -43.9826584, 2.8443480, -43.8666534, 2.7207203, -40.5834351, 40.5725403
39: -43.6168671, 3.0130620, -43.4887581, 2.8643003, -41.2391968, 41.2660904
40: -32.7641144, 0.0357924, -32.6769257, -0.0597186, -30.9885178, 31.0009155
41: -20.7349701, 7.2695274, -20.6877174, 7.2200165, -26.4318085, 26.4079666
42: -22.9869442, -0.1978426, -22.9681568, -0.2320774, -18.4299965, 18.4493828

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1741

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5200724
time: 35.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5497468, upper bound: 11.5212424
time: 34.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.3541422, 19.0565090, -9.3403683, 19.0138931, -25.2212334, 25.2156944
1: -1.2120385, 22.8381290, -1.2117133, 22.8010521, -19.7400856, 19.7413902
2: -1.6188033, 20.9653702, -1.6124015, 20.9128532, -17.2566681, 17.2889557
3: -9.3457203, 16.5427647, -9.3423758, 16.4743710, -21.9904442, 22.0465240
4: -3.1400170, 22.2538643, -3.1370063, 22.2223740, -21.7156487, 21.7348404
5: -7.8320661, 20.6636009, -7.8361025, 20.6012840, -23.7279968, 23.7741394
6: -28.8298340, -1.3685846, -28.8240585, -1.3886433, -23.1831665, 23.1975784
7: -7.6965866, 21.6647282, -7.6930199, 21.6254921, -23.5741882, 23.5952301
8: -14.7822304, 14.8009310, -14.7805443, 14.7508345, -26.4758911, 26.4986572
9: -5.1943378, 21.3040924, -5.1920033, 21.2774849, -24.2552719, 24.2779312
10: -17.8833809, 17.5744591, -17.8799477, 17.5549583, -31.2749100, 31.2883530
11: -26.8258934, 3.5678468, -26.7474957, 3.5662756, -27.9353714, 27.8576355
12: -34.8774986, -2.3120222, -34.8808365, -2.3525310, -27.2027283, 27.2413673
13: -26.2515450, 15.8287144, -26.2494049, 15.6935616, -33.8613815, 33.9836960
14: -55.8855553, -17.5487118, -55.9080887, -17.5596104, -37.7794342, 37.7614212
15: -14.3693008, 15.5314617, -14.3589230, 15.5091219, -27.9021683, 27.8966446
16: -14.0604973, 20.8325958, -14.0474062, 20.8058739, -31.0400391, 31.0631104
17: -57.8359299, -14.3874378, -57.8420181, -14.4292221, -41.6382751, 41.6440048
18: -21.6796722, 12.1721992, -21.5713978, 12.1574192, -29.6698151, 29.5750618
19: -22.3572731, 3.5645561, -22.2675896, 3.5727024, -22.8269501, 22.7362518
20: -23.3737526, 1.3688474, -23.2789192, 1.3738618, -19.2359085, 19.1715546
21: -26.8943119, 2.3975530, -26.7843494, 2.4005690, -25.5660477, 25.4674568
22: -28.6122494, 3.3248200, -28.4874153, 3.3265643, -24.7780952, 24.6739883
23: -22.3694191, 5.7041855, -22.2761326, 5.7132525, -22.0698547, 21.9804955
24: -18.4259758, 9.4414711, -18.2941895, 9.4440918, -22.9149094, 22.8045349
25: -23.9244347, 5.3893290, -23.8119202, 5.3840055, -24.4623108, 24.3709564
26: -41.1288261, -0.4718828, -41.0218353, -0.4630947, -30.6518936, 30.5644531
27: -21.6829491, 8.5744238, -21.5634995, 8.5773478, -26.5259705, 26.4057541
28: -24.2089481, 6.0617442, -24.1063232, 6.0693002, -22.0385666, 21.9413948
29: -27.9455376, -0.2159789, -27.8314209, -0.2146490, -24.0469398, 23.9277344
30: -28.2285881, 3.7506537, -28.1067753, 3.7463343, -26.1809692, 26.0861282
31: -22.7540836, 5.0579023, -22.6585426, 5.0554152, -25.1175079, 25.0418701
32: -23.9581661, 2.3516123, -23.9415035, 2.3176980, -21.3671799, 21.4031563
33: -36.4488144, 3.6691384, -36.4300919, 3.6562834, -33.3271484, 33.3911514
34: -37.8886795, -4.7495322, -37.8596382, -4.7506866, -27.7378235, 27.7629547
35: -32.9382935, 0.3197865, -32.9244995, 0.3058343, -28.1642075, 28.2015686
36: -36.8599548, -0.6502628, -36.8430672, -0.6516628, -29.0355835, 29.0425644
37: -44.6008186, -1.7025218, -44.5627823, -1.7040801, -38.8277435, 38.7978210
38: -43.9885788, 2.8993626, -43.9614296, 2.8811960, -40.7109528, 40.7194748
39: -43.6234894, 3.0640273, -43.5967789, 3.0113735, -41.3338318, 41.4234695
40: -32.7706146, 0.0516655, -32.7347298, -0.0121706, -31.0452118, 31.0796432
41: -20.7392387, 7.2909546, -20.7288437, 7.2846155, -26.5006409, 26.5078964
42: -22.9919128, -0.1913652, -22.9850445, -0.2088299, -18.4662018, 18.4758644

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1491
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1741

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5272032, upper bound: 11.5418027
time: 31.22 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5503475, upper bound: 11.5429131
time: 42.05 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.1817398, 18.9774971, -9.2982359, 19.0550804, -25.0885544, 25.1177292
1: -1.0990663, 22.7672157, -1.1723056, 22.8378048, -19.6604919, 19.6585236
2: -1.5390494, 20.8946457, -1.5917101, 20.9648590, -17.2259369, 17.2041168
3: -9.2716351, 16.4434242, -9.3047838, 16.5411530, -21.9739609, 21.9080276
4: -3.0260358, 22.2131577, -3.0893102, 22.2536087, -21.6299286, 21.6548805
5: -7.7275376, 20.5630550, -7.7898464, 20.6612587, -23.6731071, 23.6590691
6: -28.7783585, -1.4771852, -28.8099995, -1.3966212, -23.1105118, 23.1051712
7: -7.5834818, 21.5968647, -7.6451492, 21.6640587, -23.4941635, 23.4941177
8: -14.6270390, 14.6940918, -14.7210236, 14.8006058, -26.3687935, 26.3525467
9: -5.0970731, 21.2492008, -5.1488228, 21.3033981, -24.2116013, 24.1810760
10: -17.7608337, 17.5203209, -17.8433990, 17.5723228, -31.1662369, 31.2071838
11: -26.7223854, 3.5115676, -26.8227596, 3.5597486, -27.8208923, 27.8765335
12: -34.8507919, -2.4204445, -34.8796768, -2.3404975, -27.1795349, 27.1236191
13: -26.2141209, 15.6376877, -26.2361069, 15.8165941, -33.9338379, 33.7730942
14: -55.7730141, -17.6092892, -55.8907776, -17.5584316, -37.6462784, 37.5987015
15: -14.2462902, 15.4584560, -14.3355026, 15.5220871, -27.7903900, 27.8160934
16: -13.9632034, 20.7865372, -14.0343256, 20.8327789, -30.9883194, 31.0133667
17: -57.7391663, -14.4835844, -57.8425903, -14.4001884, -41.5424271, 41.4754257
18: -21.5142708, 12.1051006, -21.6778183, 12.1385727, -29.4712448, 29.6049728
19: -22.2149239, 3.4875135, -22.3548183, 3.5409734, -22.6593475, 22.7443428
20: -23.2243690, 1.2877140, -23.3723755, 1.3566518, -19.0925446, 19.1827965
21: -26.7285919, 2.3164968, -26.8918076, 2.3786950, -25.3913040, 25.4880600
22: -28.4390221, 3.2535801, -28.6111031, 3.3008001, -24.5982094, 24.7275314
23: -22.2439404, 5.6379223, -22.3679428, 5.6823716, -21.9267082, 22.0036659
24: -18.2389774, 9.3645391, -18.4236526, 9.4137754, -22.7155380, 22.8559990
25: -23.7803383, 5.2998538, -23.9216118, 5.3536530, -24.3024521, 24.3892975
26: -40.9764786, -0.5611081, -41.1264114, -0.5008283, -30.4822235, 30.5780411
27: -21.5281734, 8.5150757, -21.6808205, 8.5554981, -26.3391113, 26.4617081
28: -24.0666561, 5.9700603, -24.2072029, 6.0254340, -21.8628159, 21.9479713
29: -27.7965717, -0.2542839, -27.9429684, -0.2375371, -23.8611069, 23.9960403
30: -28.0688934, 3.6697233, -28.2273941, 3.7354341, -26.0383568, 26.1284332
31: -22.6008644, 4.9740820, -22.7503452, 5.0230732, -24.9464722, 25.0471916
32: -23.8954124, 2.2460768, -23.9376392, 2.3286772, -21.3291626, 21.2955093
33: -36.3348083, 3.4974461, -36.4420204, 3.6305804, -33.2574997, 33.2202148
34: -37.7932396, -4.8827357, -37.8873520, -4.7708230, -27.6783295, 27.6494217
35: -32.8486938, 0.1566477, -32.9352455, 0.2765751, -28.0825653, 28.0442123
36: -36.7741089, -0.8015246, -36.8579941, -0.6794910, -28.9455414, 28.9056854
37: -44.4829102, -1.8099380, -44.5937157, -1.7195916, -38.6957092, 38.7098923
38: -43.8608513, 2.6978679, -43.9826012, 2.8484163, -40.5729828, 40.5538254
39: -43.4812813, 2.8599348, -43.5989227, 3.0176787, -41.2619934, 41.2285004
40: -32.6733398, -0.0614455, -32.7500916, 0.0332575, -30.9923782, 30.9775772
41: -20.6844940, 7.2020693, -20.7309799, 7.2806892, -26.4209061, 26.4122314
42: -22.9685326, -0.2416790, -22.9797668, -0.1948681, -18.4506721, 18.4133797

Time for backsubstitution: 2.34 seconds

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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5416904
time: 32.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5528301
time: 33.66 seconds

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

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5648322
time: 27.99 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5760283
time: 32.23 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.3121166, 19.0128880, -9.3403015, 19.0564842, -25.1852341, 25.2006302
1: -1.1860647, 22.8003273, -1.2014737, 22.8388138, -19.7168617, 19.7253456
2: -1.5994580, 20.9122429, -1.6115420, 20.9657440, -17.2767906, 17.2423859
3: -9.3289013, 16.4727154, -9.3244438, 16.5440483, -22.0343094, 21.9674377
4: -3.1113124, 22.2223816, -3.1167550, 22.2546844, -21.7124481, 21.6899033
5: -7.8129835, 20.5996075, -7.8185902, 20.6633148, -23.7465820, 23.7242889
6: -28.8162155, -1.3981514, -28.8133545, -1.3705549, -23.1788940, 23.1816711
7: -7.6671009, 21.6235962, -7.6726303, 21.6652107, -23.5690002, 23.5493393
8: -14.7545576, 14.7480288, -14.7641945, 14.8031044, -26.4782600, 26.4493332
9: -5.1723251, 21.2761993, -5.1738548, 21.3055763, -24.2723694, 24.2327843
10: -17.8583908, 17.5516663, -17.8756142, 17.5760117, -31.2663116, 31.2735596
11: -26.7460728, 3.5554910, -26.8282204, 3.5740719, -27.8576965, 27.9255524
12: -34.8667107, -2.3601499, -34.8843460, -2.3221369, -27.2150650, 27.1954765
13: -26.2307987, 15.6901188, -26.2402992, 15.8305969, -33.9662399, 33.8409882
14: -55.8652725, -17.5643768, -55.9196320, -17.5534306, -37.7425232, 37.7799911
15: -14.3530483, 15.5043802, -14.3699160, 15.5244837, -27.8848267, 27.8967590
16: -14.0207624, 20.8049927, -14.0530529, 20.8334599, -31.0352707, 31.0399094
17: -57.8097916, -14.4370651, -57.8652458, -14.3931398, -41.6277542, 41.6278381
18: -21.5652962, 12.1560135, -21.6837101, 12.1556358, -29.5479126, 29.6678009
19: -22.2644920, 3.5495334, -22.3598919, 3.5619228, -22.7308044, 22.8056793
20: -23.2775726, 1.3552804, -23.3746758, 1.3792112, -19.1708183, 19.2174454
21: -26.7816601, 2.3840592, -26.8975220, 2.4012656, -25.4666595, 25.5519562
22: -28.4846001, 3.3049257, -28.6142616, 3.3174520, -24.6617355, 24.7562752
23: -22.2752800, 5.6859035, -22.3705921, 5.6979795, -21.9726067, 22.0395279
24: -18.2916985, 9.4254780, -18.4259701, 9.4347506, -22.7891998, 22.8924065
25: -23.8098259, 5.3679228, -23.9242592, 5.3751268, -24.3490372, 24.4444580
26: -41.0189476, -0.4921594, -41.1290016, -0.4772992, -30.5522308, 30.6196060
27: -21.5604744, 8.5586376, -21.6847076, 8.5702133, -26.3957748, 26.5075989
28: -24.1050491, 6.0426111, -24.2093544, 6.0492525, -21.9254303, 22.0093727
29: -27.8293266, -0.2346623, -27.9478855, -0.2321784, -23.9067841, 24.0285950
30: -28.1058006, 3.7353451, -28.2293472, 3.7562866, -26.0845680, 26.1721497
31: -22.6548042, 5.0403662, -22.7556286, 5.0452895, -25.0243301, 25.1029739
32: -23.9357071, 2.3101208, -23.9405231, 2.3498147, -21.3951454, 21.3413773
33: -36.4265900, 3.6421099, -36.4462814, 3.6805325, -33.3992767, 33.3083420
34: -37.8578377, -4.7688956, -37.8893738, -4.7321777, -27.7847748, 27.7193375
35: -32.9208527, 0.2900767, -32.9383049, 0.3226128, -28.1999512, 28.1474915
36: -36.8395920, -0.6773028, -36.8609238, -0.6370869, -29.0523224, 29.0091553
37: -44.5574226, -1.7231379, -44.6003571, -1.6899562, -38.8048248, 38.8029404
38: -43.9556503, 2.8582516, -43.9885406, 2.9034176, -40.7199860, 40.6814423
39: -43.5893860, 3.0070090, -43.6054955, 3.0685382, -41.4193878, 41.3231964
40: -32.7311478, -0.0139022, -32.7565880, 0.0491323, -31.0711060, 31.0342407
41: -20.7256298, 7.2666278, -20.7352753, 7.3021250, -26.5208054, 26.4810410
42: -22.9854202, -0.2184279, -22.9847431, -0.1883845, -18.4772034, 18.4495697

Time for backsubstitution: 2.14 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5467291, upper bound: 11.5422664
time: 28.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5467291, upper bound: 11.5534672
time: 26.22 seconds

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

Time for backsubstitution: 2.21 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5653754
time: 35.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5766147
time: 32.90 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.3308678, 19.0528259, -9.2243185, 18.9789658, -25.1495514, 25.1323166
1: -1.2041717, 22.8416729, -1.1359878, 22.7683887, -19.6838188, 19.7016144
2: -1.6075218, 20.9455452, -1.5619984, 20.8956909, -17.2164078, 17.2333679
3: -9.3315105, 16.5193901, -9.2929993, 16.4465656, -21.9336243, 21.9716797
4: -3.1297708, 22.2441711, -3.0637784, 22.2135201, -21.6891403, 21.6507683
5: -7.8228884, 20.6482983, -7.7609286, 20.5660305, -23.6830139, 23.7074509
6: -28.8300171, -1.4020720, -28.7874813, -1.4655600, -23.1246185, 23.1304207
7: -7.6870317, 21.6619148, -7.6203709, 21.5994759, -23.5310974, 23.5312195
8: -14.7575073, 14.7869720, -14.6675014, 14.6980400, -26.3844070, 26.3807907
9: -5.1804352, 21.2995758, -5.1229224, 21.2513561, -24.2088165, 24.2338104
10: -17.8648338, 17.5861454, -17.7888947, 17.5253544, -31.2228012, 31.2167969
11: -26.8046532, 3.5648260, -26.7248993, 3.5306649, -27.8763657, 27.8276062
12: -34.8874626, -2.3294587, -34.8710251, -2.4098482, -27.1460495, 27.2130814
13: -26.2615089, 15.7679529, -26.2465324, 15.6429977, -33.7991104, 33.9170074
14: -55.9061127, -17.4927349, -55.8314438, -17.6023693, -37.6454163, 37.7403412
15: -14.3362484, 15.5157290, -14.2561283, 15.4646492, -27.8237762, 27.7929001
16: -14.0565758, 20.8600597, -13.9992456, 20.7880211, -31.0237961, 31.0480881
17: -57.8489532, -14.3478508, -57.7822227, -14.4725399, -41.5180054, 41.6134109
18: -21.6096992, 12.1517076, -21.5229168, 12.1154022, -29.5522003, 29.4956856
19: -22.3419571, 3.5664840, -22.2199707, 3.5237134, -22.7700500, 22.6809235
20: -23.3551826, 1.3660421, -23.2266541, 1.3202195, -19.2011414, 19.0957565
21: -26.8651657, 2.3919778, -26.7331429, 2.3470430, -25.4948044, 25.3980751
22: -28.5872955, 3.3234847, -28.4432259, 3.2889719, -24.7447548, 24.6153870
23: -22.3489666, 5.7138453, -22.2455845, 5.6792955, -22.0353127, 21.9536743
24: -18.3895988, 9.4367046, -18.2427711, 9.3956652, -22.8580551, 22.7395248
25: -23.8743324, 5.3792019, -23.7836571, 5.3283682, -24.3745003, 24.3252792
26: -41.1192245, -0.4681659, -40.9806709, -0.5132127, -30.6270599, 30.5067368
27: -21.6541100, 8.5768242, -21.5328293, 8.5478592, -26.4681931, 26.3590088
28: -24.1864338, 6.0617118, -24.0686874, 6.0118961, -21.9786148, 21.8908424
29: -27.9282627, -0.2096252, -27.7998581, -0.2240391, -24.0121078, 23.8850098
30: -28.1814995, 3.7423706, -28.0705891, 3.6941209, -26.1065521, 26.0346985
31: -22.7127724, 5.0455837, -22.6066017, 4.9997616, -25.0397110, 24.9713631
32: -23.9568481, 2.3061492, -23.9026680, 2.2546980, -21.3301048, 21.3157959
33: -36.4641647, 3.6170702, -36.3407249, 3.5135407, -33.2616272, 33.2516098
34: -37.8974533, -4.7662387, -37.7959938, -4.8558216, -27.6940155, 27.6733322
35: -32.9590416, 0.2798424, -32.8541031, 0.1768961, -28.0917587, 28.0890503
36: -36.8898659, -0.6702595, -36.7793503, -0.7657084, -28.9767380, 28.9557648
37: -44.6136017, -1.7223010, -44.4918365, -1.7863522, -38.7590179, 38.7015457
38: -44.0036469, 2.8576345, -43.8691940, 2.7286472, -40.6166382, 40.5921631
39: -43.6198502, 2.9793425, -43.4930954, 2.8640547, -41.2617493, 41.2361603
40: -32.7541161, -0.0083811, -32.6810455, -0.0589056, -30.9872208, 30.9613266
41: -20.7594395, 7.2734399, -20.6899071, 7.2254009, -26.4794769, 26.4205856
42: -22.9877319, -0.2087364, -22.9686642, -0.2303760, -18.4326439, 18.4367332

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1741

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5399635, upper bound: 11.5408114
time: 31.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5631843, upper bound: 11.5420139
time: 35.96 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.3730068, 19.0542183, -9.3547134, 19.0143776, -25.2324677, 25.2290115
1: -1.2333140, 22.8426552, -1.2229733, 22.8015041, -19.7506828, 19.7579498
2: -1.6273696, 20.9464035, -1.6224320, 20.9133129, -17.2546692, 17.2841492
3: -9.3512068, 16.5222874, -9.3502474, 16.4758530, -21.9930649, 22.0320053
4: -3.1572852, 22.2452774, -3.1491771, 22.2227287, -21.7241631, 21.7332916
5: -7.8516316, 20.6503506, -7.8463659, 20.6025639, -23.7482681, 23.7809601
6: -28.8333893, -1.3760185, -28.8253212, -1.3865061, -23.2011337, 23.1987915
7: -7.7144880, 21.6630936, -7.7040014, 21.6262188, -23.5863495, 23.6060562
8: -14.8007088, 14.7894716, -14.7950268, 14.7520103, -26.4812241, 26.4901657
9: -5.2054672, 21.3017445, -5.1982098, 21.2783031, -24.2605286, 24.2945824
10: -17.8970490, 17.5898514, -17.8864059, 17.5566998, -31.2891922, 31.3168716
11: -26.8101215, 3.5791283, -26.7485867, 3.5745649, -27.9254074, 27.8643875
12: -34.8921509, -2.3110094, -34.8869629, -2.3495374, -27.2179794, 27.2486610
13: -26.2656727, 15.7819138, -26.2632256, 15.6954422, -33.8670273, 33.9494171
14: -55.9349747, -17.4878387, -55.9236984, -17.5574608, -37.8266220, 37.8365402
15: -14.3706264, 15.5181475, -14.3629436, 15.5106297, -27.9044647, 27.8873749
16: -14.0753479, 20.8607731, -14.0567493, 20.8064671, -31.0503464, 31.0949554
17: -57.8716240, -14.3407784, -57.8528366, -14.4260960, -41.6703644, 41.6986542
18: -21.6155739, 12.1687975, -21.5739288, 12.1663294, -29.6150589, 29.5723152
19: -22.3470573, 3.5874429, -22.2695637, 3.5857468, -22.8313828, 22.7523575
20: -23.3575153, 1.3885806, -23.2798157, 1.3878212, -19.2357330, 19.1740494
21: -26.8708515, 2.4145470, -26.7861366, 2.4146268, -25.5587387, 25.4734612
22: -28.5904274, 3.3400955, -28.4887791, 3.3403113, -24.7735596, 24.6789093
23: -22.3516216, 5.7294888, -22.2769451, 5.7272673, -22.0711288, 21.9996033
24: -18.3919353, 9.4576912, -18.2955132, 9.4565220, -22.8944702, 22.8131866
25: -23.8769646, 5.4006119, -23.8131409, 5.3964596, -24.4296341, 24.3719063
26: -41.1218338, -0.4445887, -41.0230560, -0.4442592, -30.6685181, 30.5767593
27: -21.6580696, 8.5915871, -21.5651207, 8.5914097, -26.5140915, 26.4155884
28: -24.1885872, 6.0855579, -24.1071014, 6.0844936, -22.0399818, 21.9534454
29: -27.9332504, -0.2042592, -27.8326607, -0.2044110, -24.0446739, 23.9305611
30: -28.1835461, 3.7632716, -28.1074944, 3.7597542, -26.1502571, 26.0809975
31: -22.7180805, 5.0678568, -22.6606045, 5.0660205, -25.0954742, 25.0492249
32: -23.9596825, 2.3273549, -23.9429531, 2.3187456, -21.3759613, 21.3818245
33: -36.4683495, 3.6670251, -36.4325333, 3.6582394, -33.3497162, 33.3934479
34: -37.8994446, -4.7275496, -37.8606033, -4.7419491, -27.7639389, 27.7798233
35: -32.9620819, 0.3258758, -32.9262772, 0.3103251, -28.1949921, 28.2064209
36: -36.8928642, -0.6278138, -36.8447685, -0.6415730, -29.0801468, 29.0626144
37: -44.6202087, -1.6926475, -44.5663452, -1.6995950, -38.8520813, 38.8107300
38: -44.0095634, 2.9126797, -43.9639664, 2.8892260, -40.7442169, 40.7392120
39: -43.6263885, 3.0302467, -43.6011810, 3.0111713, -41.3563538, 41.3935165
40: -32.7605667, 0.0074756, -32.7388573, -0.0113842, -31.0438766, 31.0400810
41: -20.7636795, 7.2948890, -20.7310410, 7.2899680, -26.5482559, 26.5205765
42: -22.9927044, -0.2021921, -22.9855499, -0.2071061, -18.4688416, 18.4632912

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1491
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1741

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5406293, upper bound: 11.5626578
time: 29.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5637882, upper bound: 11.5637884
time: 30.97 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.3808022, 19.0966320, -9.2243185, 18.9789658, -25.2007523, 25.1763535
1: -1.2327075, 22.8800278, -1.1359878, 22.7683887, -19.7122269, 19.7398529
2: -1.6402123, 20.9991322, -1.5619984, 20.8956909, -17.2488403, 17.2852173
3: -9.3568802, 16.5916843, -9.2929993, 16.4465656, -21.9595413, 22.0414734
4: -3.1645060, 22.2766056, -3.0637784, 22.2135201, -21.7232056, 21.6801453
5: -7.8441978, 20.7140160, -7.7609286, 20.5660305, -23.6948318, 23.7613831
6: -28.8389244, -1.3731022, -28.7874813, -1.4655600, -23.1337585, 23.1558228
7: -7.7177238, 21.7032452, -7.6203709, 21.5994759, -23.5589523, 23.5706711
8: -14.7948332, 14.8405695, -14.6675014, 14.6980400, -26.4227219, 26.4334106
9: -5.1979513, 21.3281174, -5.1229224, 21.2513561, -24.2331543, 24.2756004
10: -17.8842812, 17.6096916, -17.7888947, 17.5253544, -31.2424316, 31.2382202
11: -26.8856754, 3.5871644, -26.7248993, 3.5306649, -27.9619751, 27.8501434
12: -34.8994026, -2.2818608, -34.8710251, -2.4098482, -27.1586609, 27.2609253
13: -26.3015404, 15.9060612, -26.2465324, 15.6429977, -33.8419952, 34.0570297
14: -55.9229202, -17.4759293, -55.8314438, -17.6023693, -37.6585846, 37.7591782
15: -14.3617191, 15.5427513, -14.2561283, 15.4646492, -27.8520660, 27.8212585
16: -14.0913773, 20.8878269, -13.9992456, 20.7880211, -31.0552292, 31.0821304
17: -57.8727951, -14.2980385, -57.7822227, -14.4725399, -41.5475159, 41.6705704
18: -21.7250595, 12.1882820, -21.5229168, 12.1154022, -29.6691132, 29.5328941
19: -22.4366627, 3.5917225, -22.2199707, 3.5237134, -22.8635101, 22.7065887
20: -23.4522610, 1.3983133, -23.2266541, 1.3202195, -19.2989349, 19.1299477
21: -26.9796295, 2.4254656, -26.7331429, 2.3470430, -25.6076508, 25.4315338
22: -28.7158279, 3.3586390, -28.4432259, 3.2889719, -24.8732033, 24.6482468
23: -22.4445496, 5.7419925, -22.2455845, 5.6792955, -22.1278038, 21.9770660
24: -18.5247898, 9.4658403, -18.2427711, 9.3956652, -22.9925232, 22.7680359
25: -23.9892311, 5.4168086, -23.7836571, 5.3283682, -24.4904671, 24.3632507
26: -41.2293625, -0.4282703, -40.9806709, -0.5132127, -30.7337265, 30.5420074
27: -21.7778130, 8.6113787, -21.5328293, 8.5478592, -26.5936508, 26.3947449
28: -24.2908611, 6.0939426, -24.0686874, 6.0118961, -22.0784302, 21.9197884
29: -28.0438881, -0.1831535, -27.7998581, -0.2240391, -24.1300888, 23.9117622
30: -28.3051529, 3.7821193, -28.0705891, 3.6941209, -26.2294388, 26.0782318
31: -22.8136253, 5.0756855, -22.6066017, 4.9997616, -25.1397552, 25.0021172
32: -23.9769497, 2.3461781, -23.9026680, 2.2546980, -21.3530159, 21.3550606
33: -36.4892464, 3.6387453, -36.3407249, 3.5135407, -33.2842255, 33.2779922
34: -37.9289169, -4.7436962, -37.7959938, -4.8558216, -27.7236710, 27.6961823
35: -32.9771500, 0.3031044, -32.8541031, 0.1768961, -28.1088257, 28.1147308
36: -36.9109383, -0.6471438, -36.7793503, -0.7657084, -28.9980240, 28.9802551
37: -44.6612396, -1.7059307, -44.4918365, -1.7863522, -38.8080902, 38.7206039
38: -44.0373306, 2.8920827, -43.8691940, 2.7286472, -40.6637573, 40.6294098
39: -43.6575890, 3.0305676, -43.4930954, 2.8640547, -41.2951660, 41.2882767
40: -32.8012772, 0.0556803, -32.6810455, -0.0589056, -31.0298386, 31.0254440
41: -20.7757416, 7.2948456, -20.6899071, 7.2254009, -26.4971161, 26.4408035
42: -22.9953442, -0.1832500, -22.9686642, -0.2303760, -18.4412727, 18.4657593

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1475
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
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1741

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5554250, upper bound: 11.5201512
time: 37.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5785815, upper bound: 11.5213154
time: 28.08 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.4229364, 19.0979958, -9.3547134, 19.0143776, -25.2836304, 25.2730408
1: -1.2618461, 22.8810101, -1.2229733, 22.8015041, -19.7790565, 19.7961998
2: -1.6600692, 21.0000153, -1.6224320, 20.9133129, -17.2870560, 17.3359871
3: -9.3765574, 16.5945663, -9.3502474, 16.4758530, -22.0189743, 22.1017914
4: -3.1919689, 22.2776814, -3.1491771, 22.2227287, -21.7582283, 21.7626686
5: -7.8728952, 20.7161064, -7.8463659, 20.6025639, -23.7600784, 23.8348694
6: -28.8422966, -1.3470130, -28.8253212, -1.3865061, -23.2102661, 23.2242165
7: -7.7451944, 21.7044048, -7.7040014, 21.6262188, -23.6141891, 23.6454887
8: -14.8380070, 14.8430719, -14.7950268, 14.7520103, -26.5195084, 26.5428162
9: -5.2230434, 21.3303356, -5.1982098, 21.2783031, -24.2848969, 24.3363647
10: -17.9165192, 17.6133823, -17.8864059, 17.5566998, -31.3088303, 31.3383179
11: -26.8910923, 3.6014805, -26.7485867, 3.5745649, -28.0110168, 27.8868866
12: -34.9040871, -2.2635136, -34.8869629, -2.3495374, -27.2305984, 27.2964516
13: -26.3057270, 15.9199657, -26.2632256, 15.6954422, -33.9098892, 34.0894241
14: -55.9517784, -17.4709911, -55.9236984, -17.5574608, -37.8397751, 37.8554153
15: -14.3961372, 15.5451832, -14.3629436, 15.5106297, -27.9327621, 27.9157333
16: -14.1101618, 20.8885021, -14.0567493, 20.8064671, -31.0818176, 31.1289444
17: -57.8954582, -14.2909660, -57.8528366, -14.4260960, -41.6998901, 41.7557983
18: -21.7308769, 12.2053480, -21.5739288, 12.1663294, -29.7319717, 29.6095200
19: -22.4417305, 3.6127090, -22.2695637, 3.5857468, -22.9248276, 22.7780304
20: -23.4545631, 1.4208310, -23.2798157, 1.3878212, -19.3335342, 19.2082100
21: -26.9852886, 2.4480658, -26.7861366, 2.4146268, -25.6715546, 25.5068855
22: -28.7189827, 3.3753181, -28.4887791, 3.3403113, -24.9019470, 24.7117653
23: -22.4472237, 5.7576060, -22.2769451, 5.7272673, -22.1636200, 22.0229836
24: -18.5271473, 9.4867907, -18.2955132, 9.4565220, -23.0289154, 22.8416443
25: -23.9918671, 5.4382563, -23.8131409, 5.3964596, -24.5455933, 24.4098358
26: -41.2319374, -0.4046206, -41.0230560, -0.4442592, -30.7751999, 30.6120224
27: -21.7817631, 8.6261568, -21.5651207, 8.5914097, -26.6395798, 26.4513016
28: -24.2930355, 6.1177764, -24.1071014, 6.0844936, -22.1397972, 21.9823952
29: -28.0488853, -0.1777875, -27.8326607, -0.2044110, -24.1626472, 23.9573898
30: -28.3071709, 3.8030012, -28.1074944, 3.7597542, -26.2731285, 26.1244888
31: -22.8188992, 5.0979214, -22.6606045, 5.0660205, -25.1955109, 25.0799561
32: -23.9798470, 2.3672969, -23.9429531, 2.3187456, -21.3988800, 21.4210320
33: -36.4934883, 3.6886945, -36.4325333, 3.6582394, -33.3723297, 33.4198227
34: -37.9309349, -4.7050500, -37.8606033, -4.7419491, -27.7935867, 27.8026886
35: -32.9801788, 0.3491378, -32.9262772, 0.3103251, -28.2120743, 28.2320938
36: -36.9138908, -0.6047564, -36.8447685, -0.6415730, -29.1014328, 29.0870743
37: -44.6678352, -1.6763053, -44.5663452, -1.6995950, -38.9011688, 38.8297195
38: -44.0431557, 2.9470677, -43.9639664, 2.8892260, -40.7913055, 40.7763977
39: -43.6642075, 3.0814447, -43.6011810, 3.0111713, -41.3897552, 41.4456482
40: -32.8077927, 0.0714946, -32.7388573, -0.0113842, -31.0865402, 31.1041832
41: -20.7800350, 7.3162661, -20.7310410, 7.2899680, -26.5659485, 26.5407486
42: -23.0002918, -0.1767843, -22.9855499, -0.2071061, -18.4774933, 18.4922638

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1475
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1741

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5560424, upper bound: 11.5418624
time: 48.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5791626, upper bound: 11.5429610
time: 29.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.2506857, 19.0189934, -9.3125620, 19.0555954, -25.1511536, 25.1760254
1: -1.1489143, 22.8101368, -1.1835594, 22.8382320, -19.6994476, 19.7140961
2: -1.5803423, 20.9293404, -1.6017690, 20.9652748, -17.2563744, 17.2519989
3: -9.3024797, 16.4950600, -9.3126745, 16.5426331, -22.0024872, 21.9632683
4: -3.0780339, 22.2369614, -3.1014338, 22.2539806, -21.6726456, 21.6822166
5: -7.7684717, 20.6154213, -7.8001528, 20.6625366, -23.7052879, 23.7198868
6: -28.7908401, -1.4551539, -28.8112030, -1.3945284, -23.1376343, 23.1323166
7: -7.6320829, 21.6365662, -7.6561232, 21.6647720, -23.5342293, 23.5447350
8: -14.6829185, 14.7362700, -14.7355042, 14.8017035, -26.4125061, 26.3979950
9: -5.1258354, 21.2757454, -5.1550083, 21.3042221, -24.2413101, 24.2398987
10: -17.7942352, 17.5594749, -17.8495483, 17.5740795, -31.2003021, 31.2570419
11: -26.7874908, 3.5452905, -26.8238220, 3.5679989, -27.8963242, 27.9058914
12: -34.8774796, -2.3718972, -34.8855591, -2.3375893, -27.2073212, 27.1788483
13: -26.2683392, 15.7291193, -26.2498722, 15.8184547, -33.9823914, 33.8804626
14: -55.8392448, -17.5317059, -55.9064178, -17.5562115, -37.7067261, 37.6925430
15: -14.2733326, 15.4721756, -14.3394814, 15.5235510, -27.8212204, 27.8351517
16: -14.0130539, 20.8429642, -14.0435057, 20.8333416, -31.0301819, 31.0791168
17: -57.7985725, -14.3871975, -57.8533859, -14.3971195, -41.6035919, 41.5871582
18: -21.5656567, 12.1382627, -21.6803017, 12.1474991, -29.5347977, 29.6393280
19: -22.2993813, 3.5356188, -22.3567390, 3.5539823, -22.7575073, 22.7861328
20: -23.3051338, 1.3397565, -23.3732376, 1.3705895, -19.1918259, 19.2194557
21: -26.8194923, 2.3670182, -26.8936214, 2.3926890, -25.4977417, 25.5274620
22: -28.5456619, 3.3040237, -28.6124725, 3.3145809, -24.7241058, 24.7654114
23: -22.3217831, 5.6914258, -22.3687744, 5.6963072, -22.0219612, 22.0462341
24: -18.3400993, 9.4098969, -18.4249306, 9.4261789, -22.8310852, 22.8930931
25: -23.8482361, 5.3488235, -23.9228325, 5.3660297, -24.3867416, 24.4281540
26: -41.0795975, -0.4938540, -41.1276321, -0.4820404, -30.6079712, 30.6257248
27: -21.6268806, 8.5667686, -21.6824512, 8.5694923, -26.4540710, 26.5072174
28: -24.1507530, 6.0261555, -24.2079697, 6.0405774, -21.9661636, 21.9890175
29: -27.8998299, -0.2160861, -27.9441643, -0.2273223, -23.9782562, 24.0256882
30: -28.1474171, 3.7221656, -28.2281723, 3.7487564, -26.1326523, 26.1668625
31: -22.6656761, 5.0141087, -22.7523727, 5.0336380, -25.0247574, 25.0853271
32: -23.9170952, 2.2619076, -23.9390583, 2.3297250, -21.3605728, 21.3134918
33: -36.3793869, 3.5172706, -36.4444084, 3.6325469, -33.3024139, 33.2485428
34: -37.8355789, -4.8381395, -37.8883133, -4.7620707, -27.7347565, 27.6896286
35: -32.8906364, 0.1866326, -32.9370117, 0.2810407, -28.1303940, 28.0757217
36: -36.8280869, -0.7556167, -36.8596306, -0.6694150, -29.0115433, 28.9507217
37: -44.5498734, -1.7834420, -44.5972824, -1.7151866, -38.7685394, 38.7423859
38: -43.9155502, 2.7466221, -43.9850998, 2.8563204, -40.6535339, 40.6112289
39: -43.5221786, 2.8773251, -43.6032639, 3.0173979, -41.3174744, 41.2509537
40: -32.7107811, -0.0413990, -32.7541428, 0.0340364, -31.0338516, 31.0021400
41: -20.7252178, 7.2284055, -20.7331696, 7.2860355, -26.4859619, 26.4454956
42: -22.9768734, -0.2269394, -22.9802895, -0.1931462, -18.4617081, 18.4298782

Time for backsubstitution: 2.16 seconds

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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5537957, upper bound: 11.5417594
time: 31.84 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5537957, upper bound: 11.5529023
time: 29.37 seconds

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

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5549674, upper bound: 11.5648912
time: 30.89 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5549674, upper bound: 11.5760878
time: 37.03 seconds

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

Time for backsubstitution: 2.15 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5755830, upper bound: 11.5423393
time: 34.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5755830, upper bound: 11.5535544
time: 28.56 seconds

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

Time for backsubstitution: 2.20 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5766894, upper bound: 11.5654349
time: 39.11 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5766894, upper bound: 11.5766893
time: 31.37 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 72.81 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5406721
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5334983, upper bound: 11.5418773
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5625290
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5341288, upper bound: 11.5636658
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5200724
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5497468, upper bound: 11.5212424
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5272032, upper bound: 11.5418027
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5503475, upper bound: 11.5429131
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5416904
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5528301
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5648322
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5760283
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5467291, upper bound: 11.5422664
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5467291, upper bound: 11.5534672
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5653754
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5766147
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5399635, upper bound: 11.5408114
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5631843, upper bound: 11.5420139
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5406293, upper bound: 11.5626578
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5637882, upper bound: 11.5637884
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5554250, upper bound: 11.5201512
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5785815, upper bound: 11.5213154
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5560424, upper bound: 11.5418624
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5791626, upper bound: 11.5429610
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5537957, upper bound: 11.5417594
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5537957, upper bound: 11.5529023
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5549674, upper bound: 11.5648912
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5549674, upper bound: 11.5760878
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5755830, upper bound: 11.5423393
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5755830, upper bound: 11.5535544
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5766894, upper bound: 11.5654349
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 72.81
Output dim: 2, lower bound: -11.5766894, upper bound: 11.5766893

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.2063732, 19.0099411, -9.1924267, 18.9780483, -25.0251312, 25.0533371
1: -1.1115837, 22.7979202, -1.1110706, 22.7676811, -19.6012077, 19.6309853
2: -1.5303900, 20.9098835, -1.5405705, 20.8949547, -17.1483192, 17.1727066
3: -9.2548180, 16.4646072, -9.2707500, 16.4440880, -21.8592377, 21.8984604
4: -3.0197849, 22.2199345, -3.0331159, 22.2129993, -21.5900497, 21.6041183
5: -7.7394385, 20.5917835, -7.7371006, 20.5634232, -23.6092529, 23.6275291
6: -28.7971001, -1.4374967, -28.7795944, -1.4719076, -23.0694351, 23.0749168
7: -7.5836987, 21.6203079, -7.5917196, 21.5981770, -23.4365311, 23.4611244
8: -14.6425600, 14.7426920, -14.6342831, 14.6962290, -26.2814484, 26.3143463
9: -5.1123695, 21.2718658, -5.1043615, 21.2501583, -24.1344757, 24.1600342
10: -17.8021545, 17.5435581, -17.7730446, 17.5225639, -31.1572037, 31.1535721
11: -26.7381058, 3.5140023, -26.7233887, 3.5168743, -27.7921371, 27.7770767
12: -34.8508682, -2.3975620, -34.8618164, -2.4189982, -27.0951614, 27.1350327
13: -26.1570263, 15.6725559, -26.2169476, 15.6399345, -33.6972351, 33.7891998
14: -55.8308105, -17.5816479, -55.8129654, -17.6083107, -37.5474091, 37.6212769
15: -14.2914839, 15.4898863, -14.2465668, 15.4593372, -27.7708054, 27.7558441
16: -13.9729033, 20.8027744, -13.9791327, 20.7871304, -30.9473801, 30.9701538
17: -57.7853546, -14.4592619, -57.7701416, -14.4807196, -41.4206390, 41.4819717
18: -21.5541573, 12.0775719, -21.5190697, 12.0934572, -29.4708710, 29.4158554
19: -22.2540855, 3.4800372, -22.2169418, 3.4984250, -22.6560745, 22.6003151
20: -23.2727165, 1.2849021, -23.2252350, 1.2969208, -19.0895920, 19.0263100
21: -26.7718601, 2.3063812, -26.7304993, 2.3218417, -25.3737640, 25.3221626
22: -28.4781914, 3.2262394, -28.4410782, 3.2602484, -24.6007843, 24.5287094
23: -22.2693634, 5.6145992, -22.2442036, 5.6504970, -21.9232025, 21.8632431
24: -18.2846012, 9.3498755, -18.2402725, 9.3699780, -22.7239609, 22.6568642
25: -23.8026619, 5.2812128, -23.7812767, 5.3003645, -24.2698517, 24.2341995
26: -41.0126419, -0.5933504, -40.9783440, -0.5504580, -30.4783554, 30.3962631
27: -21.5523071, 8.4816666, -21.5302258, 8.5199928, -26.3365326, 26.2677689
28: -24.1003342, 5.9501433, -24.0672951, 5.9790392, -21.8556404, 21.7927513
29: -27.8234253, -0.2926691, -27.7981796, -0.2483656, -23.8786011, 23.8093872
30: -28.1014786, 3.6573029, -28.0693722, 3.6704557, -25.9992905, 25.9648285
31: -22.6430321, 4.9635005, -22.6030083, 4.9757371, -24.9430847, 24.8888474
32: -23.9131241, 2.2796180, -23.8942986, 2.2502832, -21.2719727, 21.2781487
33: -36.4097137, 3.5930996, -36.3351860, 3.5102606, -33.2008057, 33.2132339
34: -37.8528214, -4.8206067, -37.7942963, -4.8677044, -27.6317596, 27.6220322
35: -32.9118881, 0.2369657, -32.8506813, 0.1683431, -28.0337296, 28.0419464
36: -36.8320694, -0.7307568, -36.7764511, -0.7807522, -28.9015579, 28.8906708
37: -44.5350342, -1.7553062, -44.4846039, -1.7928205, -38.6651001, 38.6535721
38: -43.9414597, 2.7879834, -43.8643494, 2.7137561, -40.5213928, 40.5050964
39: -43.5471344, 2.9610600, -43.4787865, 2.8639832, -41.1759338, 41.2032547
40: -32.6897964, -0.0336306, -32.6683884, -0.0613039, -30.9184418, 30.9234467
41: -20.7077713, 7.2402115, -20.6842613, 7.2178650, -26.4003296, 26.3739548
42: -22.9701157, -0.2309551, -22.9652061, -0.2344453, -18.4085579, 18.4094467

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1778
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
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1777
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
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1756
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
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1556
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
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5240669
time: 42.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5406721
time: 39.24 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.2673492, 19.0209541, -9.2086000, 18.9784431, -25.0906601, 25.0850525
1: -1.1548123, 22.8208809, -1.1237421, 22.7678871, -19.6393204, 19.6663437
2: -1.5666842, 20.9299583, -1.5511827, 20.8952026, -17.1784515, 17.2057228
3: -9.2999401, 16.4990730, -9.2842855, 16.4449348, -21.8939667, 21.9360733
4: -3.0803304, 22.2391434, -3.0504827, 22.2130871, -21.6408310, 21.6372643
5: -7.7825866, 20.6289253, -7.7498040, 20.5645638, -23.6416245, 23.6710510
6: -28.8166275, -1.4052849, -28.7854958, -1.4680691, -23.0862961, 23.1100655
7: -7.6372886, 21.6596794, -7.6082625, 21.5986919, -23.4789162, 23.5164032
8: -14.7029724, 14.7669344, -14.6517773, 14.6967659, -26.3340950, 26.3440857
9: -5.1542592, 21.2944412, -5.1147242, 21.2504578, -24.1790543, 24.2010727
10: -17.8308296, 17.5598392, -17.7798786, 17.5235558, -31.1842346, 31.1832428
11: -26.7442722, 3.5316596, -26.7236366, 3.5209365, -27.8077698, 27.7989273
12: -34.8615494, -2.3745136, -34.8645554, -2.4140954, -27.1094894, 27.1686668
13: -26.2073650, 15.7220850, -26.2311268, 15.6410017, -33.7424088, 33.8567200
14: -55.8484764, -17.5689030, -55.8155785, -17.6053543, -37.5653076, 37.6625977
15: -14.3206930, 15.5014095, -14.2516460, 15.4624424, -27.8034363, 27.7697983
16: -14.0092649, 20.8422699, -13.9885302, 20.7873802, -30.9760208, 31.0198135
17: -57.7897949, -14.4416151, -57.7710686, -14.4779367, -41.4229736, 41.5382690
18: -21.6075935, 12.1184702, -21.5202065, 12.1051731, -29.5403366, 29.4502945
19: -22.2904282, 3.5179753, -22.2178574, 3.5099783, -22.7038422, 22.6292915
20: -23.3048134, 1.3151670, -23.2256927, 1.3057399, -19.1338577, 19.0522346
21: -26.8027496, 2.3422437, -26.7311287, 2.3323219, -25.4120560, 25.3511963
22: -28.5348167, 3.2722063, -28.4417381, 3.2743921, -24.6725159, 24.5609818
23: -22.3131447, 5.6613183, -22.2446365, 5.6644487, -21.9828148, 21.8981361
24: -18.3346825, 9.3912649, -18.2413597, 9.3825083, -22.7890625, 22.6906433
25: -23.8506889, 5.3308249, -23.7822800, 5.3150368, -24.3346329, 24.2738113
26: -41.0780640, -0.5367780, -40.9792862, -0.5331397, -30.5670166, 30.4351730
27: -21.5954781, 8.5266008, -21.5310936, 8.5329533, -26.3936157, 26.3073502
28: -24.1512527, 6.0050354, -24.0678272, 5.9957500, -21.9250984, 21.8319206
29: -27.8728142, -0.2482924, -27.7985096, -0.2353137, -23.9409943, 23.8422470
30: -28.1290398, 3.6934481, -28.0697441, 3.6799934, -26.0355377, 25.9931641
31: -22.6814308, 5.0066919, -22.6043491, 4.9883575, -24.9959412, 24.9293251
32: -23.9375572, 2.3076944, -23.9005356, 2.2533221, -21.2937469, 21.3114624
33: -36.4277802, 3.6019440, -36.3379745, 3.5113478, -33.2200165, 33.2289658
34: -37.8621445, -4.8143415, -37.7949562, -4.8669491, -27.6448822, 27.6288376
35: -32.9252472, 0.2452154, -32.8521652, 0.1699162, -28.0508270, 28.0523758
36: -36.8419838, -0.7212434, -36.7775269, -0.7784367, -28.9190445, 28.9056702
37: -44.5563126, -1.7531285, -44.4878540, -1.7930670, -38.6830902, 38.6730804
38: -43.9617615, 2.8065495, -43.8664780, 2.7178006, -40.5591583, 40.5354919
39: -43.5843658, 2.9803271, -43.4869385, 2.8641157, -41.2090149, 41.2272491
40: -32.7265778, -0.0027471, -32.6757355, -0.0599208, -30.9515076, 30.9613190
41: -20.7189178, 7.2522879, -20.6868553, 7.2198095, -26.4126434, 26.3879776
42: -22.9785881, -0.2159836, -22.9676018, -0.2323072, -18.4167023, 18.4278946

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1778
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
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1777
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
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1756
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
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1556
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
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 523
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
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5334983, upper bound: 11.5252685
time: 29.84 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5334983, upper bound: 11.5418773
time: 31.48 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.2484674, 19.0112972, -9.3227797, 19.0134411, -25.1080284, 25.1500168
1: -1.1407447, 22.7988892, -1.1980643, 22.8008003, -19.6680832, 19.6873322
2: -1.5502589, 20.9107494, -1.6009912, 20.9125423, -17.1865692, 17.2234840
3: -9.2744789, 16.4675083, -9.3279648, 16.4733944, -21.9186707, 21.9587898
4: -3.0472531, 22.2210293, -3.1184392, 22.2222347, -21.6250229, 21.6865997
5: -7.7681179, 20.5938587, -7.8225141, 20.5999241, -23.6744690, 23.7010193
6: -28.8004780, -1.4114428, -28.8174400, -1.3929033, -23.1458817, 23.1433105
7: -7.6111898, 21.6214828, -7.6753235, 21.6248951, -23.4917717, 23.5359344
8: -14.6857147, 14.7452021, -14.7617407, 14.7501755, -26.3782310, 26.4237061
9: -5.1373758, 21.2740803, -5.1796350, 21.2771187, -24.1861725, 24.2208061
10: -17.8343811, 17.5472736, -17.8705444, 17.5539074, -31.2235641, 31.2536240
11: -26.7435760, 3.5283146, -26.7470360, 3.5608048, -27.8411560, 27.8138199
12: -34.8555222, -2.3791471, -34.8777695, -2.3586869, -27.1670685, 27.1706581
13: -26.1612396, 15.6865864, -26.2336369, 15.6923170, -33.7651215, 33.8216171
14: -55.8596878, -17.5767345, -55.9052429, -17.5633984, -37.7286224, 37.7175522
15: -14.3258848, 15.4922485, -14.3533497, 15.5053320, -27.8514862, 27.8503113
16: -13.9916716, 20.8034344, -14.0366211, 20.8055992, -30.9739685, 31.0170746
17: -57.8080444, -14.4521456, -57.8407211, -14.4342041, -41.5730438, 41.5672455
18: -21.5600510, 12.0946255, -21.5700855, 12.1443386, -29.5337448, 29.4924774
19: -22.2592049, 3.5009766, -22.2665291, 3.5604641, -22.7174149, 22.6717796
20: -23.2750568, 1.3074064, -23.2783985, 1.3645315, -19.1241875, 19.1046181
21: -26.7775154, 2.3289642, -26.7835693, 2.3894413, -25.4376526, 25.3975487
22: -28.4813461, 3.2428470, -28.4866829, 3.3115427, -24.6295776, 24.5922356
23: -22.2720432, 5.6302152, -22.2755814, 5.6984630, -21.9590073, 21.9091873
24: -18.2869606, 9.3708630, -18.2929993, 9.4308195, -22.7603760, 22.7305374
25: -23.8053493, 5.3025761, -23.8107758, 5.3684559, -24.3249931, 24.2808456
26: -41.0152550, -0.5698357, -41.0207520, -0.4814777, -30.5198212, 30.4662704
27: -21.5562439, 8.4964380, -21.5625267, 8.5635386, -26.3824463, 26.3243942
28: -24.1024475, 5.9740438, -24.1056747, 6.0515842, -21.9169922, 21.8553734
29: -27.8283691, -0.2873600, -27.8309002, -0.2287180, -23.9111786, 23.8549385
30: -28.1035061, 3.6781602, -28.1062927, 3.7360735, -26.0429688, 26.0111313
31: -22.6483688, 4.9857397, -22.6570206, 5.0420256, -24.9988327, 24.9666977
32: -23.9159737, 2.3007953, -23.9346142, 2.3143339, -21.3178406, 21.3442154
33: -36.4139519, 3.6430521, -36.4269867, 3.6549301, -33.2889099, 33.3551178
34: -37.8548279, -4.7819209, -37.8589172, -4.7538157, -27.7016373, 27.7285690
35: -32.9149094, 0.2830176, -32.9228668, 0.3017511, -28.1369705, 28.1593246
36: -36.8349991, -0.6883264, -36.8418808, -0.6566238, -29.0049591, 28.9974747
37: -44.5416794, -1.7256160, -44.5591164, -1.7060905, -38.7582092, 38.7627182
38: -43.9473877, 2.8429961, -43.9590912, 2.8741860, -40.6490173, 40.6521301
39: -43.5537033, 3.0119572, -43.5868835, 3.0110841, -41.2704926, 41.3606262
40: -32.6962280, -0.0177767, -32.7262115, -0.0137668, -30.9750595, 31.0022659
41: -20.7120190, 7.2616577, -20.7253895, 7.2824316, -26.4691086, 26.4739227
42: -22.9750824, -0.2244649, -22.9820747, -0.2112200, -18.4447823, 18.4359894

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
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

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5459994
time: 32.15 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5625290
time: 40.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.3094263, 19.0223446, -9.3389826, 19.0138683, -25.1735611, 25.1817169
1: -1.1839499, 22.8218784, -1.2107410, 22.8010101, -19.7061691, 19.7226944
2: -1.5865116, 20.9308510, -1.6116226, 20.9127922, -17.2167053, 17.2564888
3: -9.3196135, 16.5019894, -9.3414822, 16.4742317, -21.9533958, 21.9963799
4: -3.1077814, 22.2402534, -3.1357932, 22.2223015, -21.6758461, 21.7197571
5: -7.8112931, 20.6309929, -7.8352504, 20.6011009, -23.7068481, 23.7445145
6: -28.8199749, -1.3791966, -28.8233566, -1.3890076, -23.1627731, 23.1783676
7: -7.6647873, 21.6608391, -7.6919155, 21.6254177, -23.5341034, 23.5912323
8: -14.7461700, 14.7694292, -14.7792826, 14.7507191, -26.4308853, 26.4534454
9: -5.1793318, 21.2966423, -5.1899962, 21.2774658, -24.2307968, 24.2618561
10: -17.8630676, 17.5635033, -17.8774090, 17.5548706, -31.2505951, 31.2833405
11: -26.7496834, 3.5459962, -26.7472858, 3.5648575, -27.8567963, 27.8357010
12: -34.8662415, -2.3560624, -34.8805084, -2.3537846, -27.1813660, 27.2042274
13: -26.2115593, 15.7361221, -26.2478600, 15.6934891, -33.8103027, 33.8891068
14: -55.8773956, -17.5639000, -55.9077606, -17.5604229, -37.7465591, 37.7588425
15: -14.3550549, 15.5038090, -14.3584414, 15.5083942, -27.8840866, 27.8642731
16: -14.0280380, 20.8429489, -14.0460396, 20.8058529, -31.0026398, 31.0667267
17: -57.8124809, -14.4345665, -57.8416977, -14.4314632, -41.5753555, 41.6235733
18: -21.6135235, 12.1355219, -21.5712242, 12.1560555, -29.6032181, 29.5269737
19: -22.2955437, 3.5389671, -22.2674217, 3.5720093, -22.7651672, 22.7007408
20: -23.3071461, 1.3376918, -23.2788429, 1.3733177, -19.1684418, 19.1305466
21: -26.8084106, 2.3648283, -26.7842026, 2.3999064, -25.4759598, 25.4265823
22: -28.5379715, 3.2888021, -28.4872627, 3.3257358, -24.7013016, 24.6244583
23: -22.3158283, 5.6769590, -22.2760162, 5.7124166, -22.0186386, 21.9440994
24: -18.3370590, 9.4122562, -18.2940521, 9.4433699, -22.8254547, 22.7643013
25: -23.8533497, 5.3522863, -23.8117638, 5.3831277, -24.3897705, 24.3204231
26: -41.0807419, -0.5132408, -41.0217209, -0.4641232, -30.6085358, 30.5051804
27: -21.5993900, 8.5413465, -21.5634346, 8.5765114, -26.4394913, 26.3639603
28: -24.1534023, 6.0288744, -24.1062260, 6.0683546, -21.9864540, 21.8945274
29: -27.8778172, -0.2429261, -27.8312302, -0.2156839, -23.9735069, 23.8878059
30: -28.1310558, 3.7143140, -28.1066513, 3.7456088, -26.0792351, 26.0394478
31: -22.6867371, 5.0289145, -22.6583633, 5.0546227, -25.0517044, 25.0072021
32: -23.9404335, 2.3288608, -23.9407997, 2.3173633, -21.3396034, 21.3774910
33: -36.4319878, 3.6519065, -36.4298058, 3.6560836, -33.3081284, 33.3708038
34: -37.8641777, -4.7757144, -37.8595390, -4.7530627, -27.7147827, 27.7353439
35: -32.9282532, 0.2912912, -32.9243774, 0.3033123, -28.1540909, 28.1697464
36: -36.8449669, -0.6788483, -36.8429642, -0.6543179, -29.0224686, 29.0125351
37: -44.5629501, -1.7234468, -44.5623283, -1.7063313, -38.7761536, 38.7822342
38: -43.9676590, 2.8615336, -43.9612236, 2.8782735, -40.6867371, 40.6824799
39: -43.5908890, 3.0312405, -43.5950012, 3.0112433, -41.3036957, 41.3845978
40: -32.7330208, 0.0131361, -32.7335663, -0.0124376, -31.0081635, 31.0401611
41: -20.7231541, 7.2737265, -20.7279682, 7.2844124, -26.4814606, 26.4879456
42: -22.9835491, -0.2094553, -22.9844856, -0.2090540, -18.4529037, 18.4544334

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5341288, upper bound: 11.5471347
time: 36.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5341288, upper bound: 11.5636658
time: 35.32 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.2565184, 19.0537300, -9.1924267, 18.9780483, -25.0764618, 25.0983505
1: -1.1401849, 22.8363152, -1.1110706, 22.7676811, -19.6296272, 19.6701889
2: -1.5631137, 20.9635143, -1.5405705, 20.8949547, -17.1808968, 17.2254791
3: -9.2801924, 16.5367470, -9.2707500, 16.4440880, -21.8851547, 21.9683990
4: -3.0545883, 22.2523422, -3.0331159, 22.2129993, -21.6242065, 21.6334496
5: -7.7607207, 20.6574383, -7.7371006, 20.5634232, -23.6212311, 23.6818466
6: -28.8060684, -1.4081569, -28.7795944, -1.4719076, -23.0786209, 23.1008110
7: -7.6144161, 21.6616745, -7.5917196, 21.5981770, -23.4646378, 23.5011864
8: -14.6798630, 14.7963724, -14.6342831, 14.6962290, -26.3198700, 26.3682785
9: -5.1299772, 21.3007622, -5.1043615, 21.2501583, -24.1589813, 24.2021370
10: -17.8220291, 17.5673332, -17.7730446, 17.5225639, -31.1773148, 31.1749802
11: -26.8190022, 3.5364008, -26.7233887, 3.5168743, -27.8776703, 27.7996521
12: -34.8630257, -2.3498716, -34.8618164, -2.4189982, -27.1078110, 27.1832619
13: -26.1971283, 15.8108597, -26.2169476, 15.6399345, -33.7402878, 33.9314423
14: -55.8475609, -17.5650749, -55.8129654, -17.6083107, -37.5606842, 37.6400909
15: -14.3170967, 15.5169382, -14.2465668, 15.4593372, -27.7993317, 27.7842178
16: -14.0079498, 20.8310528, -13.9791327, 20.7871304, -30.9788818, 31.0041885
17: -57.8091278, -14.4094658, -57.7701416, -14.4807196, -41.4501495, 41.5391235
18: -21.6696892, 12.1141310, -21.5190697, 12.0934572, -29.5891800, 29.4532204
19: -22.3487854, 3.5052924, -22.2169418, 3.4984250, -22.7499008, 22.6260185
20: -23.3697662, 1.3171625, -23.2252350, 1.2969208, -19.1894531, 19.0604744
21: -26.8862591, 2.3399384, -26.7304993, 2.3218417, -25.4876328, 25.3555489
22: -28.6067219, 3.2614939, -28.4410782, 3.2602484, -24.7313766, 24.5621147
23: -22.3649673, 5.6428256, -22.2442036, 5.6504970, -22.0174713, 21.8870697
24: -18.4197769, 9.3790398, -18.2402725, 9.3699780, -22.8606262, 22.6856384
25: -23.9181023, 5.3188610, -23.7812767, 5.3003645, -24.3868561, 24.2728424
26: -41.1228638, -0.5534782, -40.9783440, -0.5504580, -30.5875320, 30.4328384
27: -21.6759911, 8.5162449, -21.5302258, 8.5199928, -26.4637680, 26.3035202
28: -24.2048607, 5.9824505, -24.0672951, 5.9790392, -21.9576149, 21.8222733
29: -27.9389801, -0.2662175, -27.7981796, -0.2483656, -23.9981003, 23.8363495
30: -28.2250423, 3.6971815, -28.0693722, 3.6704557, -26.1244888, 26.0085335
31: -22.7439690, 4.9936371, -22.6030083, 4.9757371, -25.0435181, 24.9196320
32: -23.9333534, 2.3197634, -23.8942986, 2.2502832, -21.2947464, 21.3175468
33: -36.4347839, 3.6149964, -36.3351860, 3.5102606, -33.2232361, 33.2393570
34: -37.8843765, -4.7979460, -37.7942963, -4.8677044, -27.6621933, 27.6453857
35: -32.9300613, 0.2606473, -32.8506813, 0.1683431, -28.0508499, 28.0683289
36: -36.8531342, -0.7073350, -36.7764511, -0.7807522, -28.9231567, 28.9155502
37: -44.5826645, -1.7386346, -44.4846039, -1.7928205, -38.7140503, 38.6729889
38: -43.9751892, 2.8232946, -43.8643494, 2.7137561, -40.5687256, 40.5428009
39: -43.5854225, 3.0122123, -43.4787865, 2.8639832, -41.2090607, 41.2556305
40: -32.7372017, 0.0305841, -32.6683884, -0.0613039, -30.9612122, 30.9877090
41: -20.7240868, 7.2626228, -20.6842613, 7.2178650, -26.4178772, 26.3945389
42: -22.9777031, -0.2053428, -22.9652061, -0.2344453, -18.4172173, 18.4386177

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1778
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
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1777
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
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1756
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
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1556
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
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5023389
time: 26.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5200724
time: 32.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.3174362, 19.0647659, -9.2086000, 18.9784431, -25.1419449, 25.1300621
1: -1.1833863, 22.8592701, -1.1237421, 22.7678871, -19.6677322, 19.7055511
2: -1.5993886, 20.9835892, -1.5511827, 20.8952026, -17.2110367, 17.2584839
3: -9.3253279, 16.5711937, -9.2842855, 16.4449348, -21.9198685, 22.0060959
4: -3.1151271, 22.2715588, -3.0504827, 22.2130871, -21.6749496, 21.6666336
5: -7.8038692, 20.6946163, -7.7498040, 20.5645638, -23.6535797, 23.7254524
6: -28.8255920, -1.3758707, -28.7854958, -1.4680691, -23.0954895, 23.1359825
7: -7.6680098, 21.7011070, -7.6082625, 21.5986919, -23.5070076, 23.5564804
8: -14.7403011, 14.8205967, -14.6517773, 14.6967659, -26.3724861, 26.3980713
9: -5.1718302, 21.3233719, -5.1147242, 21.2504578, -24.2035294, 24.2431641
10: -17.8507385, 17.5835819, -17.7798786, 17.5235558, -31.2043533, 31.2046967
11: -26.8251591, 3.5540614, -26.7236366, 3.5209365, -27.8932877, 27.8214645
12: -34.8737335, -2.3268366, -34.8645554, -2.4140954, -27.1221008, 27.2168579
13: -26.2474594, 15.8603668, -26.2311268, 15.6410017, -33.7854385, 33.9989548
14: -55.8652344, -17.5522709, -55.8155785, -17.6053543, -37.5785522, 37.6813889
15: -14.3463001, 15.5284595, -14.2516460, 15.4624424, -27.8319550, 27.7981644
16: -14.0443001, 20.8705616, -13.9885302, 20.7873802, -31.0075302, 31.0538635
17: -57.8135834, -14.3918943, -57.7710686, -14.4779367, -41.4525146, 41.5954514
18: -21.7231846, 12.1550388, -21.5202065, 12.1051731, -29.6586761, 29.4876480
19: -22.3851318, 3.5432329, -22.2178574, 3.5099783, -22.7976685, 22.6549606
20: -23.4018250, 1.3474107, -23.2256927, 1.3057399, -19.2337189, 19.0863914
21: -26.9171753, 2.3757572, -26.7311287, 2.3323219, -25.5259552, 25.3845749
22: -28.6633472, 3.3073809, -28.4417381, 3.2743921, -24.8031387, 24.5943718
23: -22.4087791, 5.6895695, -22.2446365, 5.6644487, -22.0770760, 21.9219513
24: -18.4698982, 9.4204254, -18.2413597, 9.3825083, -22.9257355, 22.7193909
25: -23.9661140, 5.3685150, -23.7822800, 5.3150368, -24.4516678, 24.3124275
26: -41.1882591, -0.4968157, -40.9792862, -0.5331397, -30.6762314, 30.4717255
27: -21.7191811, 8.5611830, -21.5310936, 8.5329533, -26.5208817, 26.3430710
28: -24.2558022, 6.0373020, -24.0678272, 5.9957500, -22.0270729, 21.8614616
29: -27.9884300, -0.2217860, -27.7985096, -0.2353137, -24.0605392, 23.8691788
30: -28.2526283, 3.7332819, -28.0697441, 3.6799934, -26.1607513, 26.0368652
31: -22.7823677, 5.0367870, -22.6043491, 4.9883575, -25.0963821, 24.9600716
32: -23.9577770, 2.3478413, -23.9005356, 2.2533221, -21.3164978, 21.3508682
33: -36.4527512, 3.6238065, -36.3379745, 3.5113478, -33.2424469, 33.2550430
34: -37.8937149, -4.7917171, -37.7949562, -4.8669491, -27.6753006, 27.6521683
35: -32.9434433, 0.2689242, -32.8521652, 0.1699162, -28.0679855, 28.0787506
36: -36.8630524, -0.6978121, -36.7775269, -0.7784367, -28.9406738, 28.9305496
37: -44.6039810, -1.7365370, -44.4878540, -1.7930670, -38.7320709, 38.6925125
38: -43.9954758, 2.8417602, -43.8664780, 2.7178006, -40.6064911, 40.5731812
39: -43.6226578, 3.0315104, -43.4869385, 2.8641157, -41.2421265, 41.2796860
40: -32.7739182, 0.0614955, -32.6757355, -0.0599208, -30.9942169, 31.0255890
41: -20.7352238, 7.2747049, -20.6868553, 7.2198095, -26.4301987, 26.4085464
42: -22.9861679, -0.1903675, -22.9676018, -0.2323072, -18.4253387, 18.4570465

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1778
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
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1777
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
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1756
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
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1556
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
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 523
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
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5497468, upper bound: 11.5035148
time: 47.28 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5497468, upper bound: 11.5212424
time: 31.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.2985821, 19.0550957, -9.3227797, 19.0134411, -25.1593513, 25.1950302
1: -1.1693192, 22.8372631, -1.1980643, 22.8008003, -19.6964798, 19.7265472
2: -1.5829828, 20.9643745, -1.6009912, 20.9125423, -17.2191467, 17.2762566
3: -9.2998543, 16.5396500, -9.3279648, 16.4733944, -21.9445572, 22.0287361
4: -3.0820456, 22.2534389, -3.1184392, 22.2222347, -21.6591797, 21.7159348
5: -7.7894573, 20.6595135, -7.8225141, 20.5999241, -23.6864548, 23.7553062
6: -28.8094368, -1.3820839, -28.8174400, -1.3929033, -23.1550674, 23.1691742
7: -7.6418381, 21.6628571, -7.6753235, 21.6248951, -23.5198631, 23.5759964
8: -14.7230463, 14.7988625, -14.7617407, 14.7501755, -26.4166374, 26.4776459
9: -5.1550279, 21.3029709, -5.1796350, 21.2771187, -24.2106628, 24.2628860
10: -17.8542538, 17.5710106, -17.8705444, 17.5539074, -31.2436447, 31.2750015
11: -26.8244400, 3.5506926, -26.7470360, 3.5608048, -27.9266739, 27.8363800
12: -34.8677216, -2.3315024, -34.8777695, -2.3586869, -27.1797180, 27.2187958
13: -26.2013206, 15.8247986, -26.2336369, 15.6923170, -33.8081512, 33.9638672
14: -55.8764534, -17.5600586, -55.9052429, -17.5633984, -37.7418671, 37.7363281
15: -14.3514748, 15.5193253, -14.3533497, 15.5053320, -27.8799973, 27.8786545
16: -14.0266914, 20.8317432, -14.0366211, 20.8055992, -31.0054855, 31.0510559
17: -57.8317719, -14.4023495, -57.8407211, -14.4342041, -41.6025391, 41.6243896
18: -21.6755543, 12.1311798, -21.5700855, 12.1443386, -29.6520538, 29.5298233
19: -22.3538876, 3.5262339, -22.2665291, 3.5604641, -22.8112259, 22.6974487
20: -23.3720779, 1.3397045, -23.2783985, 1.3645315, -19.2240410, 19.1387634
21: -26.8919506, 2.3624873, -26.7835693, 2.3894413, -25.5515594, 25.4309425
22: -28.6098862, 3.2780457, -28.4866829, 3.3115427, -24.7601242, 24.6256256
23: -22.3676643, 5.6584582, -22.2755814, 5.6984630, -22.0532684, 21.9329910
24: -18.4221458, 9.4000206, -18.2929993, 9.4308195, -22.8970490, 22.7592621
25: -23.9207363, 5.3402729, -23.8107758, 5.3684559, -24.4419746, 24.3194351
26: -41.1253929, -0.5299230, -41.0207520, -0.4814777, -30.6289902, 30.5028534
27: -21.6799202, 8.5310287, -21.5625267, 8.5635386, -26.5096664, 26.3601151
28: -24.2070160, 6.0062914, -24.1056747, 6.0515842, -22.0189438, 21.8848991
29: -27.9439793, -0.2608554, -27.8309002, -0.2287180, -24.0306549, 23.8819656
30: -28.2270813, 3.7180386, -28.1062927, 3.7360735, -26.1681442, 26.0547905
31: -22.7492561, 5.0158672, -22.6570206, 5.0420256, -25.0992432, 24.9974823
32: -23.9362335, 2.3408840, -23.9346142, 2.3143339, -21.3406372, 21.3835335
33: -36.4389725, 3.6649399, -36.4269867, 3.6549301, -33.3114014, 33.3811874
34: -37.8864365, -4.7593002, -37.8589172, -4.7538157, -27.7320709, 27.7518921
35: -32.9330940, 0.3067102, -32.9228668, 0.3017511, -28.1541290, 28.1856995
36: -36.8560486, -0.6649508, -36.8418808, -0.6566238, -29.0265656, 29.0223389
37: -44.5893326, -1.7089877, -44.5591164, -1.7060905, -38.8072052, 38.7821426
38: -43.9810753, 2.8782792, -43.9590912, 2.8741860, -40.6963196, 40.6897736
39: -43.5921021, 3.0631447, -43.5868835, 3.0110841, -41.3037109, 41.4130173
40: -32.7436905, 0.0464218, -32.7262115, -0.0137668, -31.0178909, 31.0664749
41: -20.7283688, 7.2840366, -20.7253895, 7.2824316, -26.4867096, 26.4944763
42: -22.9826736, -0.1988482, -22.9820747, -0.2112200, -18.4534340, 18.4650993

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
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

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5272032, upper bound: 11.5241521
time: 33.21 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5272032, upper bound: 11.5418027
time: 57.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.3595324, 19.0661354, -9.3389826, 19.0138683, -25.2248611, 25.2267380
1: -1.2125177, 22.8602600, -1.2107410, 22.8010101, -19.7345695, 19.7619171
2: -1.6192417, 20.9844704, -1.6116226, 20.9127922, -17.2492867, 17.3092422
3: -9.3449879, 16.5741005, -9.3414822, 16.4742317, -21.9792900, 22.0663872
4: -3.1425781, 22.2726593, -3.1357932, 22.2223015, -21.7099724, 21.7490997
5: -7.8325787, 20.6966915, -7.8352504, 20.6011009, -23.7188187, 23.7988853
6: -28.8289471, -1.3498068, -28.8233566, -1.3890076, -23.1719589, 23.2043228
7: -7.6954312, 21.7022781, -7.6919155, 21.6254177, -23.5622406, 23.6312943
8: -14.7834930, 14.8231163, -14.7792826, 14.7507191, -26.4692535, 26.5074081
9: -5.1968937, 21.3255615, -5.1899962, 21.2774658, -24.2552719, 24.3039436
10: -17.8829460, 17.5873089, -17.8774090, 17.5548706, -31.2706985, 31.3047791
11: -26.8305492, 3.5683737, -26.7472858, 3.5648575, -27.9422913, 27.8582382
12: -34.8784142, -2.3084846, -34.8805084, -2.3537846, -27.1940155, 27.2523346
13: -26.2516479, 15.8743744, -26.2478600, 15.6934891, -33.8533478, 34.0313492
14: -55.8941040, -17.5472584, -55.9077606, -17.5604229, -37.7597885, 37.7776031
15: -14.3806839, 15.5308771, -14.3584414, 15.5083942, -27.9126434, 27.8926010
16: -14.0630322, 20.8712502, -14.0460396, 20.8058529, -31.0341644, 31.1007004
17: -57.8362236, -14.3847752, -57.8416977, -14.4314632, -41.6048813, 41.6807404
18: -21.7290573, 12.1720676, -21.5712242, 12.1560555, -29.7215271, 29.5643158
19: -22.3902206, 3.5641947, -22.2674217, 3.5720093, -22.8589783, 22.7264175
20: -23.4041634, 1.3699365, -23.2788429, 1.3733177, -19.2683029, 19.1646881
21: -26.9228420, 2.3983579, -26.7842026, 2.3999064, -25.5898514, 25.4599838
22: -28.6665421, 3.3240008, -28.4872627, 3.3257358, -24.8318710, 24.6578598
23: -22.4114342, 5.7051616, -22.2760162, 5.7124166, -22.1128769, 21.9678726
24: -18.4722443, 9.4413891, -18.2940521, 9.4433699, -22.9621429, 22.7930145
25: -23.9687710, 5.3899689, -23.8117638, 5.3831277, -24.5067520, 24.3590164
26: -41.1908493, -0.4732218, -41.0217209, -0.4641232, -30.7177048, 30.5417328
27: -21.7231369, 8.5759296, -21.5634346, 8.5765114, -26.5667572, 26.3996887
28: -24.2579498, 6.0611596, -24.1062260, 6.0683546, -22.0884285, 21.9240532
29: -27.9933796, -0.2164330, -27.8312302, -0.2156839, -24.0930595, 23.9148178
30: -28.2546272, 3.7541597, -28.1066513, 3.7456088, -26.2044182, 26.0830994
31: -22.7876472, 5.0589952, -22.6583633, 5.0546227, -25.1521378, 25.0379791
32: -23.9606323, 2.3689408, -23.9407997, 2.3173633, -21.3623924, 21.4168015
33: -36.4570236, 3.6738086, -36.4298058, 3.6560836, -33.3306046, 33.3968887
34: -37.8957367, -4.7530708, -37.8595390, -4.7530627, -27.7451859, 27.7586746
35: -32.9464455, 0.3149195, -32.9243774, 0.3033123, -28.1712418, 28.1960907
36: -36.8660507, -0.6554322, -36.8429642, -0.6543179, -29.0440674, 29.0373840
37: -44.6106339, -1.7069130, -44.5623283, -1.7063313, -38.8251648, 38.8016739
38: -44.0013809, 2.8968053, -43.9612236, 2.8782735, -40.7340393, 40.7201767
39: -43.6293182, 3.0823965, -43.5950012, 3.0112433, -41.3367920, 41.4370651
40: -32.7804489, 0.0773182, -32.7335663, -0.0124376, -31.0509338, 31.1043320
41: -20.7394886, 7.2960758, -20.7279682, 7.2844124, -26.4990311, 26.5084915
42: -22.9911461, -0.1838865, -22.9844856, -0.2090540, -18.4615555, 18.4835396

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5503475, upper bound: 11.5252725
time: 29.11 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5503475, upper bound: 11.5429131
time: 35.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.1555233, 18.9755402, -9.2900372, 19.0544720, -25.0625954, 25.1076469
1: -1.0810199, 22.7634315, -1.1668072, 22.8366356, -19.6417999, 19.6499176
2: -1.5299733, 20.8787003, -1.5888419, 20.9599190, -17.1976013, 17.1859207
3: -9.2652121, 16.4281445, -9.3028183, 16.5364113, -21.9538345, 21.8817062
4: -3.0140719, 22.2070084, -3.0855494, 22.2516727, -21.6074905, 21.6417923
5: -7.7143126, 20.5452461, -7.7858315, 20.6554184, -23.6502609, 23.6317902
6: -28.7290688, -1.4848466, -28.7951088, -1.3990898, -23.0680695, 23.0856400
7: -7.5699267, 21.5884647, -7.6410375, 21.6613140, -23.4700813, 23.4735146
8: -14.6228886, 14.6727352, -14.7197781, 14.7939682, -26.3566437, 26.3322372
9: -5.0845995, 21.2090912, -5.1448870, 21.2912445, -24.1876831, 24.1365395
10: -17.7471333, 17.3612518, -17.8391380, 17.5242157, -31.1069946, 31.0479965
11: -26.7114315, 3.4854312, -26.8194447, 3.5513449, -27.8002090, 27.8433990
12: -34.8437424, -2.4407721, -34.8774605, -2.3468838, -27.1626129, 27.0856438
13: -26.1956978, 15.6261559, -26.2301693, 15.8130875, -33.9134750, 33.7556381
14: -55.7562408, -17.7342567, -55.8855133, -17.5960293, -37.5939026, 37.4673386
15: -14.2317944, 15.4457216, -14.3308868, 15.5182304, -27.7732925, 27.7942657
16: -13.9420004, 20.7455215, -14.0277233, 20.8200951, -30.9548645, 30.9646835
17: -57.7268143, -14.5296764, -57.8388252, -14.4144802, -41.5178986, 41.4171906
18: -21.4953480, 12.0927992, -21.6720619, 12.1347971, -29.4365158, 29.5850029
19: -22.1978569, 3.4786320, -22.3495731, 3.5382268, -22.6373749, 22.7255936
20: -23.2143459, 1.2707748, -23.3692207, 1.3513756, -19.0751648, 19.1597977
21: -26.7099609, 2.2997236, -26.8860531, 2.3735499, -25.3686905, 25.4638405
22: -28.4154129, 3.2444489, -28.6035633, 3.2979929, -24.5700912, 24.6962776
23: -22.2351742, 5.6184068, -22.3652859, 5.6763291, -21.9114799, 21.9816742
24: -18.2186165, 9.3601055, -18.4170780, 9.4123745, -22.6909790, 22.8432693
25: -23.7651634, 5.2875462, -23.9168701, 5.3498850, -24.2777557, 24.3574142
26: -40.9650650, -0.6036148, -41.1229095, -0.5138764, -30.4573364, 30.5390396
27: -21.4733315, 8.5033512, -21.6639156, 8.5516415, -26.2801971, 26.4358826
28: -24.0460625, 5.9615297, -24.2008801, 6.0226793, -21.8376312, 21.9322395
29: -27.7723026, -0.2599726, -27.9352131, -0.2393308, -23.8302155, 23.9565430
30: -28.0553474, 3.6518497, -28.2230949, 3.7296703, -26.0151825, 26.0971603
31: -22.5835304, 4.9565954, -22.7450066, 5.0175085, -24.9255600, 25.0255127
32: -23.8579655, 2.2381995, -23.9261360, 2.3261724, -21.2866974, 21.2742271
33: -36.2689590, 3.4918923, -36.4219360, 3.6288919, -33.1844330, 33.1920624
34: -37.7311058, -4.8969851, -37.8680267, -4.7751760, -27.6103668, 27.6176376
35: -32.7712860, 0.1503525, -32.9118233, 0.2746615, -28.0018311, 28.0146027
36: -36.6773148, -0.8084855, -36.8287277, -0.6816401, -28.8457642, 28.8696060
37: -44.3712997, -1.8151970, -44.5592422, -1.7212009, -38.5826416, 38.6670227
38: -43.7495041, 2.6833291, -43.9489708, 2.8438783, -40.4575500, 40.5070648
39: -43.4205284, 2.8519344, -43.5794106, 3.0152025, -41.2010040, 41.2006149
40: -32.6162186, -0.0654266, -32.7325211, 0.0320876, -30.9363708, 30.9556808
41: -20.6089745, 7.1951685, -20.7081757, 7.2785707, -26.3420868, 26.3813324
42: -22.9600487, -0.2514954, -22.9772263, -0.1979239, -18.4399796, 18.3996811

Time for backsubstitution: 2.20 seconds

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
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 974
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5257199
time: 31.27 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5416904
time: 32.51 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.2098713, 18.9986382, -9.2945251, 19.0549564, -25.1212502, 25.1417427
1: -1.1135478, 22.7945862, -1.1677380, 22.8373432, -19.6765289, 19.6940308
2: -1.5817018, 20.8936348, -1.5912211, 20.9632149, -17.2261887, 17.2437172
3: -9.2895918, 16.4431057, -9.3043604, 16.5373688, -21.9739532, 21.9215889
4: -3.0604315, 22.2147942, -3.0884576, 22.2527142, -21.6513367, 21.6937637
5: -7.7638421, 20.5629578, -7.7892752, 20.6583233, -23.6857300, 23.6633987
6: -28.7816544, -1.4066167, -28.8078270, -1.3971329, -23.1090393, 23.1787643
7: -7.6159267, 21.6037865, -7.6445484, 21.6634941, -23.5146179, 23.5127335
8: -14.6998444, 14.7028732, -14.7207851, 14.7988405, -26.4227905, 26.3969955
9: -5.1704626, 21.2557278, -5.1468792, 21.3008652, -24.2843475, 24.1818695
10: -17.9679832, 17.5239964, -17.8415833, 17.5667591, -31.3676682, 31.1909485
11: -26.7892876, 3.5119934, -26.8217373, 3.5547285, -27.8920059, 27.8728409
12: -34.8696938, -2.4018121, -34.8789825, -2.3416300, -27.2211609, 27.1206131
13: -26.2295284, 15.6600323, -26.2343254, 15.8161469, -33.9550018, 33.7927017
14: -55.9235229, -17.6081791, -55.8897247, -17.5630589, -37.7947006, 37.5749359
15: -14.2905197, 15.4627104, -14.3345661, 15.5211020, -27.8489151, 27.8154831
16: -14.0397263, 20.7886772, -14.0315657, 20.8297043, -31.0558624, 31.0077896
17: -57.8374481, -14.4731941, -57.8419495, -14.4036255, -41.6508102, 41.4821854
18: -21.5487537, 12.1296282, -21.6770592, 12.1365433, -29.4895477, 29.6663094
19: -22.2391891, 3.4912229, -22.3535480, 3.5392423, -22.6920166, 22.7428436
20: -23.2407799, 1.2846255, -23.3717670, 1.3525329, -19.1106377, 19.1767120
21: -26.7698421, 2.3200853, -26.8905849, 2.3754892, -25.4424438, 25.4894905
22: -28.4479961, 3.2818491, -28.6095982, 3.2996874, -24.6633263, 24.7056541
23: -22.2687435, 5.6355677, -22.3671875, 5.6772132, -21.9472466, 22.0064354
24: -18.2465229, 9.3702164, -18.4222145, 9.4131718, -22.7191696, 22.8614082
25: -23.7959290, 5.3152905, -23.9205914, 5.3519316, -24.3499680, 24.3856773
26: -41.0146790, -0.5630426, -41.1259804, -0.5058355, -30.5200806, 30.5771179
27: -21.5358276, 8.5811806, -21.6787891, 8.5540848, -26.3353119, 26.5312347
28: -24.0706501, 6.0030813, -24.2051601, 6.0248814, -21.8704681, 21.9645004
29: -27.8130894, -0.2236538, -27.9414539, -0.2380662, -23.9418068, 23.9560814
30: -28.0823593, 3.6832268, -28.2263699, 3.7326794, -26.0688629, 26.1353149
31: -22.6463585, 4.9746780, -22.7490940, 5.0192623, -24.9941254, 25.0475197
32: -23.8971214, 2.2884605, -23.9357815, 2.3282256, -21.3226357, 21.3381081
33: -36.3407669, 3.5954566, -36.4385414, 3.6301112, -33.2544098, 33.3262558
34: -37.7930603, -4.7918706, -37.8849220, -4.7723970, -27.6686325, 27.7493744
35: -32.8512268, 0.2724652, -32.9323502, 0.2758946, -28.0735779, 28.1596298
36: -36.7753830, -0.6751909, -36.8545914, -0.6801357, -28.9286499, 29.0297012
37: -44.4898758, -1.6934419, -44.5884056, -1.7204418, -38.6869507, 38.8347015
38: -43.8736267, 2.8533735, -43.9786911, 2.8467112, -40.5712891, 40.7057800
39: -43.4857140, 2.9236293, -43.5929604, 3.0170708, -41.2695618, 41.2995529
40: -32.6834564, 0.0237596, -32.7469826, 0.0328622, -31.0036469, 31.0642052
41: -20.6853828, 7.2924652, -20.7277069, 7.2803373, -26.4098816, 26.5020981
42: -22.9671974, -0.2236602, -22.9789963, -0.1955578, -18.4721985, 18.4310951

Time for backsubstitution: 2.18 seconds

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
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 974
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5367906
time: 34.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5528301
time: 29.18 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.1716919, 18.9759502, -9.3509884, 19.0655136, -25.0942917, 25.1731529
1: -1.0937209, 22.7636490, -1.2100005, 22.8596230, -19.6771851, 19.6880264
2: -1.5405695, 20.8789330, -1.6251070, 20.9800205, -17.2306252, 17.2160568
3: -9.2787542, 16.4290161, -9.3479567, 16.5708694, -21.9914856, 21.9164238
4: -3.0314469, 22.2070923, -3.1461172, 22.2709026, -21.6406860, 21.6925659
5: -7.7269917, 20.5463829, -7.8289742, 20.6925735, -23.6938629, 23.6641502
6: -28.7349777, -1.4810424, -28.8146362, -1.3668575, -23.1032410, 23.1025085
7: -7.5864744, 21.5889797, -7.6946125, 21.7007141, -23.5253906, 23.5158844
8: -14.6403866, 14.6732578, -14.7801762, 14.8182125, -26.3864670, 26.3849258
9: -5.0949602, 21.2094440, -5.1867676, 21.3137836, -24.2287140, 24.1811714
10: -17.7539997, 17.3622112, -17.8678169, 17.5404606, -31.1367111, 31.0750427
11: -26.7116375, 3.4894867, -26.8255882, 3.5690379, -27.8220749, 27.8590317
12: -34.8464661, -2.4358397, -34.8881454, -2.3238659, -27.1961823, 27.0999451
13: -26.2099285, 15.6272345, -26.2804985, 15.8626032, -33.9809341, 33.8008347
14: -55.7587662, -17.7313194, -55.9032173, -17.5832348, -37.6352692, 37.4852524
15: -14.2368698, 15.4488106, -14.3600445, 15.5297737, -27.7872391, 27.8269043
16: -13.9513807, 20.7457676, -14.0640841, 20.8595753, -31.0045547, 30.9933548
17: -57.7277527, -14.5269127, -57.8432770, -14.3969116, -41.5742416, 41.4195251
18: -21.4964905, 12.1045551, -21.7255173, 12.1756859, -29.4709930, 29.6545258
19: -22.1987228, 3.4901814, -22.3859291, 3.5761716, -22.6663208, 22.7733574
20: -23.2148151, 1.2795639, -23.4013138, 1.3816562, -19.1010590, 19.2040749
21: -26.7106018, 2.3101892, -26.9169464, 2.4093990, -25.3977203, 25.5021591
22: -28.4160385, 3.2586412, -28.6601677, 3.3439746, -24.6023483, 24.7680435
23: -22.2356472, 5.6323566, -22.4090519, 5.7230868, -21.9463921, 22.0413170
24: -18.2196922, 9.3726091, -18.4671822, 9.4537296, -22.7246933, 22.9083481
25: -23.7661476, 5.3021903, -23.9648457, 5.3995538, -24.3173447, 24.4222031
26: -40.9660034, -0.5862594, -41.1883011, -0.4572506, -30.4962158, 30.6277161
27: -21.4742031, 8.5162811, -21.7071209, 8.5965500, -26.3197937, 26.4929352
28: -24.0465908, 5.9782887, -24.2518005, 6.0775547, -21.8767662, 22.0017204
29: -27.7726421, -0.2469485, -27.9846077, -0.1948984, -23.8630638, 24.0189323
30: -28.0557442, 3.6613984, -28.2506866, 3.7657826, -26.0435333, 26.1334229
31: -22.5848827, 4.9691768, -22.7834415, 5.0606823, -24.9660263, 25.0784149
32: -23.8641605, 2.2411690, -23.9505444, 2.3542628, -21.3199921, 21.2959518
33: -36.2718048, 3.4930334, -36.4399719, 3.6377149, -33.2001419, 33.2112656
34: -37.7317734, -4.8962450, -37.8773575, -4.7689414, -27.6171417, 27.6307449
35: -32.7727737, 0.1519384, -32.9251785, 0.2828879, -28.0122299, 28.0316925
36: -36.6784897, -0.8061571, -36.8386307, -0.6721349, -28.8607788, 28.8870850
37: -44.3745041, -1.8154426, -44.5805283, -1.7190638, -38.6022339, 38.6849976
38: -43.7517014, 2.6874976, -43.9692612, 2.8624125, -40.4879456, 40.5447998
39: -43.4286346, 2.8521247, -43.6166458, 3.0344887, -41.2249298, 41.2337036
40: -32.6235733, -0.0639992, -32.7692413, 0.0629816, -30.9742279, 30.9887009
41: -20.6115627, 7.1970987, -20.7193527, 7.2906227, -26.3560867, 26.3936768
42: -22.9624653, -0.2493412, -22.9856968, -0.1829646, -18.4584084, 18.4078140

Time for backsubstitution: 2.16 seconds

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
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5488710
time: 33.12 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5648322
time: 31.14 seconds

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

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5600329
time: 36.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5760283
time: 31.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.2858915, 19.0109272, -9.3321428, 19.0558128, -25.1593246, 25.1905441
1: -1.1680636, 22.7965393, -1.1959658, 22.8376522, -19.6981964, 19.7167511
2: -1.5903780, 20.8963070, -1.6087086, 20.9607792, -17.2484703, 17.2241592
3: -9.3224106, 16.4574299, -9.3224897, 16.5392990, -22.0141983, 21.9410896
4: -3.0994186, 22.2162266, -3.1130176, 22.2527790, -21.6900177, 21.6767769
5: -7.7997503, 20.5817642, -7.8145018, 20.6575069, -23.7237663, 23.6969910
6: -28.7669086, -1.4057970, -28.7984657, -1.3730164, -23.1364899, 23.1621170
7: -7.6535349, 21.6151962, -7.6684828, 21.6624565, -23.5449371, 23.5287209
8: -14.7503834, 14.7266779, -14.7629070, 14.7964554, -26.4661102, 26.4290009
9: -5.1598501, 21.2360840, -5.1699219, 21.2934227, -24.2484665, 24.1882210
10: -17.8446102, 17.3925133, -17.8713341, 17.5278893, -31.2070389, 31.1143494
11: -26.7350864, 3.5293846, -26.8248672, 3.5656881, -27.8369904, 27.8924332
12: -34.8596268, -2.3804440, -34.8821259, -2.3284941, -27.1981201, 27.1575394
13: -26.2123928, 15.6785707, -26.2343922, 15.8270750, -33.9458618, 33.8235397
14: -55.8484802, -17.6893425, -55.9143982, -17.5910568, -37.6901398, 37.6486206
15: -14.3386030, 15.4916544, -14.3652687, 15.5206470, -27.8677521, 27.8749084
16: -13.9995308, 20.7639656, -14.0464897, 20.8207741, -31.0018158, 30.9912415
17: -57.7974358, -14.4831829, -57.8614578, -14.4074287, -41.6033478, 41.5695496
18: -21.5463142, 12.1436796, -21.6779060, 12.1518364, -29.5131226, 29.6478767
19: -22.2474194, 3.5406616, -22.3546600, 3.5591629, -22.7087936, 22.7869415
20: -23.2674980, 1.3383570, -23.3715343, 1.3739252, -19.1534576, 19.1944389
21: -26.7629433, 2.3673396, -26.8917542, 2.3960989, -25.4439392, 25.5277557
22: -28.4609642, 3.2958012, -28.6067047, 3.3146126, -24.6334915, 24.7250900
23: -22.2665405, 5.6663599, -22.3679447, 5.6919613, -21.9574318, 22.0175476
24: -18.2712803, 9.4209690, -18.4194431, 9.4333439, -22.7645950, 22.8797073
25: -23.7945900, 5.3556385, -23.9194679, 5.3713083, -24.3242683, 24.4126472
26: -41.0074501, -0.5346851, -41.1254463, -0.4902763, -30.5272980, 30.5805893
27: -21.5055981, 8.5469141, -21.6678524, 8.5663948, -26.3367081, 26.4817581
28: -24.0844307, 6.0340853, -24.2030106, 6.0465441, -21.9002266, 21.9936523
29: -27.8051186, -0.2403812, -27.9401855, -0.2339587, -23.8759499, 23.9890518
30: -28.0922356, 3.7174957, -28.2250938, 3.7505391, -26.0613556, 26.1408844
31: -22.6375122, 5.0228963, -22.7503433, 5.0397344, -25.0033875, 25.0813065
32: -23.8982239, 2.3022308, -23.9289970, 2.3473411, -21.3526611, 21.3200912
33: -36.3607368, 3.6365652, -36.4261436, 3.6788578, -33.3262558, 33.2801819
34: -37.7957153, -4.7830992, -37.8700600, -4.7365570, -27.7168579, 27.6875229
35: -32.8434525, 0.2837930, -32.9148483, 0.3206997, -28.1192093, 28.1178894
36: -36.7427750, -0.6842723, -36.8316727, -0.6392775, -28.9525375, 28.9730988
37: -44.4457741, -1.7284265, -44.5659027, -1.6915207, -38.6917572, 38.7600861
38: -43.8442764, 2.8438773, -43.9548798, 2.8989120, -40.6044922, 40.6347656
39: -43.5286751, 2.9991274, -43.5860291, 3.0661635, -41.3583069, 41.2952423
40: -32.6740570, -0.0178781, -32.7389908, 0.0479286, -31.0151215, 31.0123444
41: -20.6501350, 7.2597399, -20.7124729, 7.2999907, -26.4420242, 26.4501266
42: -22.9769554, -0.2282691, -22.9821835, -0.1914325, -18.4664955, 18.4358635

Time for backsubstitution: 2.15 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5467291, upper bound: 11.5263603
time: 28.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5467291, upper bound: 11.5422664
time: 30.05 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.3402634, 19.0340405, -9.3366117, 19.0563393, -25.2179146, 25.2246628
1: -1.2005243, 22.8276825, -1.1969028, 22.8383217, -19.7329025, 19.7608566
2: -1.6421335, 20.9112587, -1.6110353, 20.9641075, -17.2770538, 17.2819672
3: -9.3467617, 16.4723663, -9.3239651, 16.5402889, -22.0342865, 21.9810257
4: -3.1457915, 22.2239971, -3.1159215, 22.2538185, -21.7338676, 21.7287560
5: -7.8492718, 20.5994873, -7.8180227, 20.6603794, -23.7592010, 23.7286110
6: -28.8194427, -1.3275747, -28.8111763, -1.3710451, -23.1773834, 23.2552719
7: -7.6995683, 21.6305199, -7.6720085, 21.6646576, -23.5894318, 23.5679245
8: -14.8274164, 14.7568293, -14.7639971, 14.8013287, -26.5322342, 26.4937744
9: -5.2456779, 21.2826977, -5.1719298, 21.3030739, -24.3451309, 24.2335548
10: -18.0655212, 17.5552940, -17.8738003, 17.5704098, -31.4677124, 31.2572479
11: -26.8129196, 3.5559440, -26.8272076, 3.5690532, -27.9287415, 27.9218826
12: -34.8856239, -2.3414974, -34.8837013, -2.3232422, -27.2566986, 27.1925316
13: -26.2462349, 15.7125454, -26.2385025, 15.8301811, -33.9874039, 33.8606873
14: -56.0157852, -17.5632668, -55.9185905, -17.5581093, -37.8908997, 37.7561264
15: -14.3973541, 15.5086508, -14.3689404, 15.5234747, -27.9434128, 27.8961334
16: -14.0972366, 20.8071041, -14.0503340, 20.8303490, -31.1028366, 31.0343399
17: -57.9080811, -14.4267235, -57.8646011, -14.3965359, -41.7361145, 41.6345978
18: -21.5997276, 12.1805077, -21.6829433, 12.1536226, -29.5661469, 29.7291946
19: -22.2887268, 3.5532498, -22.3586178, 3.5602119, -22.7634735, 22.8041840
20: -23.2939377, 1.3522229, -23.3740807, 1.3750741, -19.1889267, 19.2113724
21: -26.8228207, 2.3876634, -26.8962631, 2.3980298, -25.5177460, 25.5534058
22: -28.4935474, 3.3332334, -28.6127930, 3.3163056, -24.7267609, 24.7344170
23: -22.3001137, 5.6835246, -22.3698711, 5.6928220, -21.9931564, 22.0422935
24: -18.2991982, 9.4311028, -18.4245453, 9.4341173, -22.7928467, 22.8978615
25: -23.8253918, 5.3834562, -23.9232140, 5.3733993, -24.3965187, 24.4409294
26: -41.0571136, -0.4940724, -41.1286011, -0.4822755, -30.5901260, 30.6186829
27: -21.5681343, 8.6248302, -21.6827126, 8.5688534, -26.3917999, 26.5771332
28: -24.1090050, 6.0755758, -24.2073345, 6.0487165, -21.9331017, 22.0259285
29: -27.8458252, -0.2040722, -27.9464340, -0.2326846, -23.9875488, 23.9885864
30: -28.1192455, 3.7488146, -28.2283630, 3.7535167, -26.1150513, 26.1789932
31: -22.7003231, 5.0410013, -22.7543640, 5.0415077, -25.0720062, 25.1032639
32: -23.9373589, 2.3525288, -23.9386806, 2.3493457, -21.3886108, 21.3840027
33: -36.4324760, 3.7401447, -36.4427414, 3.6800604, -33.3961945, 33.4144135
34: -37.8576317, -4.6779790, -37.8869438, -4.7337704, -27.7751083, 27.8192596
35: -32.9234085, 0.4058609, -32.9353790, 0.3219147, -28.1909409, 28.2629547
36: -36.8407974, -0.5510292, -36.8575745, -0.6377487, -29.0353775, 29.1331863
37: -44.5643349, -1.6066761, -44.5950394, -1.6908226, -38.7960205, 38.9278030
38: -43.9683151, 3.0138850, -43.9845619, 2.9017525, -40.7182007, 40.8333893
39: -43.5937576, 3.0707269, -43.5996017, 3.0679550, -41.4269104, 41.3942413
40: -32.7412872, 0.0713272, -32.7534790, 0.0487192, -31.0824127, 31.1208801
41: -20.7264709, 7.3570375, -20.7319717, 7.3017426, -26.5097809, 26.5708771
42: -22.9841137, -0.2004638, -22.9839554, -0.1890557, -18.4987106, 18.4672699

Time for backsubstitution: 2.15 seconds

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
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 974
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
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5467291, upper bound: 11.5375187
time: 35.11 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5467291, upper bound: 11.5534672
time: 23.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.3020859, 19.0113087, -9.3930511, 19.0668926, -25.1910095, 25.2560349
1: -1.1807547, 22.7967720, -1.2391806, 22.8605919, -19.7335815, 19.7548523
2: -1.6009955, 20.8965416, -1.6449616, 20.9809055, -17.2814674, 17.2543030
3: -9.3359299, 16.4583054, -9.3676071, 16.5737534, -22.0518570, 21.9758301
4: -3.1167755, 22.2163563, -3.1735864, 22.2719898, -21.7232056, 21.7275810
5: -7.8124647, 20.5829277, -7.8577013, 20.6946716, -23.7673264, 23.7293625
6: -28.7728062, -1.4019737, -28.8180046, -1.3407907, -23.1715851, 23.1789932
7: -7.6701055, 21.6157455, -7.7220726, 21.7018757, -23.6002426, 23.5711060
8: -14.7679272, 14.7272100, -14.8233681, 14.8207026, -26.4958878, 26.4816971
9: -5.1702042, 21.2363930, -5.2118063, 21.3160000, -24.2895279, 24.2328568
10: -17.8515301, 17.3935394, -17.9000454, 17.5441666, -31.2367401, 31.1413727
11: -26.7353134, 3.5334268, -26.8310051, 3.5833311, -27.8588333, 27.9081116
12: -34.8623734, -2.3755255, -34.8928146, -2.3054748, -27.2317200, 27.1718941
13: -26.2266178, 15.6797066, -26.2847195, 15.8765450, -34.0133286, 33.8686981
14: -55.8510132, -17.6863785, -55.9321022, -17.5781994, -37.7314377, 37.6665115
15: -14.3436680, 15.4947586, -14.3944416, 15.5321913, -27.8816986, 27.9075623
16: -14.0088673, 20.7642021, -14.0828209, 20.8602810, -31.0514603, 31.0198975
17: -57.7983932, -14.4804697, -57.8659286, -14.3898067, -41.6596222, 41.5718689
18: -21.5474319, 12.1554356, -21.7314072, 12.1927414, -29.5475616, 29.7173843
19: -22.2482986, 3.5522487, -22.3909874, 3.5971055, -22.7377396, 22.8346748
20: -23.2679482, 1.3471565, -23.4035854, 1.4042046, -19.1793594, 19.2386856
21: -26.7635612, 2.3778048, -26.9225922, 2.4319515, -25.4729691, 25.5660172
22: -28.4616013, 3.3099663, -28.6633339, 3.3606291, -24.6657257, 24.7967911
23: -22.2669716, 5.6803231, -22.4117260, 5.7387204, -21.9922791, 22.0771713
24: -18.2723598, 9.4334679, -18.4695282, 9.4747009, -22.7983475, 22.9447861
25: -23.7956123, 5.3703499, -23.9674759, 5.4209776, -24.3638535, 24.4774399
26: -41.0083847, -0.5173378, -41.1908913, -0.4336452, -30.5661850, 30.6693039
27: -21.5064754, 8.5598907, -21.7110710, 8.6113415, -26.3762665, 26.5388489
28: -24.0849781, 6.0508137, -24.2539482, 6.1014061, -21.9393730, 22.0631065
29: -27.8054848, -0.2273376, -27.9896069, -0.1895382, -23.9087791, 24.0514526
30: -28.0925903, 3.7270045, -28.2526894, 3.7866471, -26.0896912, 26.1771545
31: -22.6387749, 5.0354743, -22.7886868, 5.0829120, -25.0438004, 25.1341515
32: -23.9044151, 2.3052166, -23.9534073, 2.3753898, -21.3859253, 21.3418350
33: -36.3635635, 3.6376987, -36.4441910, 3.6876945, -33.3419647, 33.2994232
34: -37.7963791, -4.7823434, -37.8794136, -4.7303348, -27.7236099, 27.7006454
35: -32.8449860, 0.2854171, -32.9282036, 0.3289366, -28.1295853, 28.1350098
36: -36.7438889, -0.6819639, -36.8415909, -0.6297455, -28.9675827, 28.9905930
37: -44.4489975, -1.7286673, -44.5871468, -1.6894059, -38.7112579, 38.7781067
38: -43.8464279, 2.8480000, -43.9751320, 2.9173555, -40.6349182, 40.6724472
39: -43.5367508, 2.9992285, -43.6232719, 3.0854058, -41.3822632, 41.3284378
40: -32.6813889, -0.0164425, -32.7757645, 0.0787940, -31.0529709, 31.0453911
41: -20.6527195, 7.2616911, -20.7236481, 7.3120785, -26.4560089, 26.4624481
42: -22.9793644, -0.2261059, -22.9906654, -0.1764770, -18.4849014, 18.4439926

Time for backsubstitution: 2.17 seconds

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

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5494866
time: 32.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5653754
time: 33.78 seconds

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

Time for backsubstitution: 2.15 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5607146
time: 32.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5766147
time: 26.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.2753057, 19.0514336, -9.2068005, 18.9785233, -25.0876846, 25.1116447
1: -1.1614232, 22.8408127, -1.1223569, 22.7681332, -19.6402054, 19.6867447
2: -1.5716820, 20.9445534, -1.5506065, 20.8953838, -17.1788712, 17.2206497
3: -9.2856607, 16.5163097, -9.2786150, 16.4455662, -21.8877563, 21.9538994
4: -3.0718198, 22.2437515, -3.0452461, 22.2133656, -21.6326942, 21.6318588
5: -7.7802315, 20.6441803, -7.7473350, 20.5647049, -23.6414490, 23.6886330
6: -28.8096123, -1.4155359, -28.7808704, -1.4698062, -23.0965500, 23.1019897
7: -7.6323295, 21.6600304, -7.6027288, 21.5989151, -23.4768219, 23.5119743
8: -14.6983290, 14.7848787, -14.6486588, 14.6973877, -26.3251152, 26.3597794
9: -5.1411405, 21.2984314, -5.1105504, 21.2510147, -24.1642303, 24.2187347
10: -17.8356743, 17.5827579, -17.7794914, 17.5243416, -31.1915512, 31.2034912
11: -26.8032093, 3.5476627, -26.7244587, 3.5252233, -27.8676529, 27.8063812
12: -34.8776894, -2.3489690, -34.8679886, -2.4160037, -27.1229935, 27.1904488
13: -26.2112083, 15.7641048, -26.2307587, 15.6418133, -33.7458801, 33.8971634
14: -55.8969345, -17.5041332, -55.8285980, -17.6061478, -37.6078033, 37.7152557
15: -14.3184414, 15.5036011, -14.2505989, 15.4608269, -27.8015976, 27.7749100
16: -14.0227709, 20.8592720, -13.9884844, 20.7877388, -30.9892349, 31.0360184
17: -57.8448563, -14.3628273, -57.7809067, -14.4775219, -41.4822083, 41.5937347
18: -21.6054955, 12.1106873, -21.5216465, 12.1023169, -29.5344009, 29.4504471
19: -22.3385372, 3.5281799, -22.2188873, 3.5115221, -22.7543488, 22.6421242
20: -23.3535233, 1.3368902, -23.2261486, 1.3108702, -19.1892853, 19.0629692
21: -26.8628139, 2.3569398, -26.7323570, 2.3358912, -25.4803543, 25.3615723
22: -28.5848923, 3.2766659, -28.4424324, 3.2740161, -24.7267838, 24.5670090
23: -22.3471870, 5.6681170, -22.2450409, 5.6645045, -22.0187225, 21.9061928
24: -18.3857536, 9.3952637, -18.2415695, 9.3823729, -22.8401871, 22.6942825
25: -23.8706264, 5.3301229, -23.7824879, 5.3128195, -24.3542061, 24.2737732
26: -41.1157990, -0.5261412, -40.9795532, -0.5316138, -30.6041946, 30.4451523
27: -21.6510830, 8.5333977, -21.5318527, 8.5340433, -26.4518814, 26.3133392
28: -24.1844578, 6.0062189, -24.0681152, 5.9942083, -21.9590149, 21.8343277
29: -27.9266853, -0.2545035, -27.7993736, -0.2381301, -23.9958458, 23.8392448
30: -28.1800041, 3.7098038, -28.0701237, 3.6838961, -26.0937462, 26.0033493
31: -22.7079659, 5.0035172, -22.6050816, 4.9863524, -25.0214844, 24.9269753
32: -23.9349289, 2.2953999, -23.8957882, 2.2513437, -21.3035698, 21.2961121
33: -36.4544220, 3.6128201, -36.3375854, 3.5122285, -33.2458572, 33.2416534
34: -37.8951454, -4.7760558, -37.7952347, -4.8589106, -27.6882782, 27.6622467
35: -32.9538689, 0.2667432, -32.8524590, 0.1728063, -28.0817032, 28.0731812
36: -36.8860359, -0.6849651, -36.7781258, -0.7706733, -28.9677353, 28.9356155
37: -44.6021118, -1.7288351, -44.4881668, -1.7884150, -38.7384491, 38.6858826
38: -43.9962006, 2.8365126, -43.8668442, 2.7216907, -40.6019745, 40.5625610
39: -43.5884933, 2.9784117, -43.4832077, 2.8637862, -41.2316284, 41.2256927
40: -32.7272110, -0.0136273, -32.6724930, -0.0605497, -30.9599152, 30.9481201
41: -20.7485867, 7.2664890, -20.6864529, 7.2232175, -26.4656143, 26.4071426
42: -22.9784813, -0.2162871, -22.9657059, -0.2327120, -18.4198608, 18.4259109

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1778
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
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1777
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
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1756
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
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1556
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
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 981
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
type: B, layer: 1, pos: 1653

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5263508, upper bound: 11.5382248
time: 28.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5374069, upper bound: 11.5382249
time: 39.98 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.3362465, 19.0624866, -9.2229404, 18.9789619, -25.1531677, 25.1433411
1: -1.2046528, 22.8638000, -1.1350460, 22.7683487, -19.6782951, 19.7220840
2: -1.6079416, 20.9646454, -1.5612047, 20.8956547, -17.2090302, 17.2536469
3: -9.3307743, 16.5507355, -9.2921238, 16.4464474, -21.9224930, 21.9915085
4: -3.1323619, 22.2629547, -3.0626121, 22.2134590, -21.6834564, 21.6650467
5: -7.8234339, 20.6813240, -7.7600646, 20.5658875, -23.6738205, 23.7321854
6: -28.8291645, -1.3833194, -28.7867966, -1.4659424, -23.1134338, 23.1371689
7: -7.6858621, 21.6994362, -7.6192713, 21.5994492, -23.5191650, 23.5672684
8: -14.7588081, 14.8091459, -14.6661911, 14.6979179, -26.3777618, 26.3895340
9: -5.1830091, 21.3210068, -5.1209269, 21.2513237, -24.2088165, 24.2597771
10: -17.8643894, 17.5990181, -17.7863636, 17.5252762, -31.2185822, 31.2332153
11: -26.8093624, 3.5653353, -26.7247200, 3.5292501, -27.8832855, 27.8282394
12: -34.8883705, -2.3259158, -34.8707161, -2.4111328, -27.1373367, 27.2240448
13: -26.2616043, 15.8135643, -26.2449379, 15.6429052, -33.7910538, 33.9646225
14: -55.9146233, -17.4912758, -55.8311462, -17.6031551, -37.6257324, 37.7565231
15: -14.3476648, 15.5151472, -14.2556372, 15.4639502, -27.8342667, 27.7888489
16: -14.0591269, 20.8987904, -13.9978743, 20.7879944, -31.0178833, 31.0856857
17: -57.8492775, -14.3452148, -57.7818642, -14.4747915, -41.4845428, 41.6500320
18: -21.6589870, 12.1515894, -21.5227852, 12.1141129, -29.6039047, 29.4848938
19: -22.3748798, 3.5660911, -22.2197857, 3.5230551, -22.8020935, 22.6710815
20: -23.3856277, 1.3671587, -23.2265701, 1.3196740, -19.2335243, 19.0888710
21: -26.8937054, 2.3927879, -26.7329674, 2.3463807, -25.5186386, 25.3906021
22: -28.6415749, 3.3226600, -28.4430752, 3.2881920, -24.7985306, 24.5992432
23: -22.3909702, 5.7148647, -22.2454948, 5.6784272, -22.0783501, 21.9410782
24: -18.4358521, 9.4366245, -18.2426376, 9.3949509, -22.9052734, 22.7280273
25: -23.9186554, 5.3797836, -23.7834816, 5.3274798, -24.4189987, 24.3133507
26: -41.1812744, -0.4695110, -40.9805679, -0.5142145, -30.6928864, 30.4840240
27: -21.6942806, 8.5783339, -21.5327415, 8.5469923, -26.5089569, 26.3529282
28: -24.2354317, 6.0610666, -24.0686131, 6.0109501, -22.0284958, 21.8735008
29: -27.9761353, -0.2100790, -27.7996883, -0.2250690, -24.0582657, 23.8720932
30: -28.2075844, 3.7459030, -28.0704613, 3.6933906, -26.1300430, 26.0317078
31: -22.7463245, 5.0466928, -22.6064129, 4.9989510, -25.0743408, 24.9674263
32: -23.9593506, 2.3235173, -23.9019547, 2.2543557, -21.3253403, 21.3294067
33: -36.4724426, 3.6216369, -36.3404465, 3.5133448, -33.2650909, 33.2573547
34: -37.9044952, -4.7698064, -37.7958984, -4.8581915, -27.7013855, 27.6690903
35: -32.9672241, 0.2749910, -32.8539734, 0.1743774, -28.0988083, 28.0835800
36: -36.8959923, -0.6754403, -36.7792740, -0.7683764, -28.9852371, 28.9506073
37: -44.6233292, -1.7267008, -44.4914207, -1.7886295, -38.7564697, 38.7053757
38: -44.0164604, 2.8550186, -43.8690300, 2.7258239, -40.6396790, 40.5928650
39: -43.6257172, 2.9976983, -43.4912949, 2.8639159, -41.2647400, 41.2496948
40: -32.7640114, 0.0172782, -32.6798477, -0.0591531, -30.9929504, 30.9859467
41: -20.7597122, 7.2785606, -20.6890335, 7.2251616, -26.4779205, 26.4211273
42: -22.9869499, -0.2012875, -22.9680958, -0.2305694, -18.4279900, 18.4443779

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1778
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
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1777
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
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1756
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
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1556
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
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 523
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
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1653

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5495173, upper bound: 11.5394490
time: 30.88 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5606478, upper bound: 11.5394490
time: 35.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.3174067, 19.0527954, -9.3371468, 19.0139256, -25.1705780, 25.2083206
1: -1.1906004, 22.8418217, -1.2093163, 22.8012466, -19.7070580, 19.7430916
2: -1.5915406, 20.9454384, -1.6110258, 20.9129982, -17.2171478, 17.2714310
3: -9.3052893, 16.5191917, -9.3358307, 16.4748650, -21.9472046, 22.0142479
4: -3.0992818, 22.2448502, -3.1305723, 22.2226086, -21.6676865, 21.7143822
5: -7.8089681, 20.6462669, -7.8327637, 20.6012630, -23.7066994, 23.7620964
6: -28.8129826, -1.3894529, -28.8187103, -1.3907528, -23.1729965, 23.1703873
7: -7.6597462, 21.6612167, -7.6863155, 21.6256561, -23.5320358, 23.5868225
8: -14.7415752, 14.7873964, -14.7762060, 14.7513332, -26.4218979, 26.4691772
9: -5.1661601, 21.3006306, -5.1858282, 21.2779732, -24.2159195, 24.2795296
10: -17.8679237, 17.5864334, -17.8769684, 17.5556355, -31.2579269, 31.3035431
11: -26.8086472, 3.5619879, -26.7480984, 3.5691476, -27.9167023, 27.8431396
12: -34.8823471, -2.3305340, -34.8838882, -2.3557138, -27.1949615, 27.2259827
13: -26.2154312, 15.7780352, -26.2474213, 15.6942091, -33.8137741, 33.9295044
14: -55.9258232, -17.4991283, -55.9208107, -17.5611744, -37.7890625, 37.8114166
15: -14.3528681, 15.5059958, -14.3573284, 15.5068188, -27.8823013, 27.8694077
16: -14.0415602, 20.8599472, -14.0459538, 20.8061962, -31.0158081, 31.0829010
17: -57.8674583, -14.3557262, -57.8514977, -14.4310427, -41.6346359, 41.6789932
18: -21.6114006, 12.1277485, -21.5726299, 12.1532297, -29.5972900, 29.5270920
19: -22.3436661, 3.5491095, -22.2684746, 3.5735414, -22.8156662, 22.7135773
20: -23.3558445, 1.3594341, -23.2792740, 1.3784585, -19.2238770, 19.1412544
21: -26.8684692, 2.3795152, -26.7854271, 2.4035046, -25.5442276, 25.4369392
22: -28.5880604, 3.2932930, -28.4880600, 3.3253520, -24.7555466, 24.6305313
23: -22.3498611, 5.6837654, -22.2763977, 5.7124810, -22.0545502, 21.9521255
24: -18.3881283, 9.4162321, -18.2942696, 9.4432602, -22.8766022, 22.7679291
25: -23.8732986, 5.3515711, -23.8119888, 5.3809557, -24.4093246, 24.3203888
26: -41.1183815, -0.5026088, -41.0220032, -0.4626231, -30.6456680, 30.5151062
27: -21.6550179, 8.5481663, -21.5641937, 8.5776291, -26.4977875, 26.3699493
28: -24.1866302, 6.0300941, -24.1064758, 6.0667610, -22.0203934, 21.8969612
29: -27.9316483, -0.2491772, -27.8321495, -0.2185309, -24.0284119, 23.8848419
30: -28.1820183, 3.7306485, -28.1070404, 3.7495182, -26.1374512, 26.0496178
31: -22.7132854, 5.0257540, -22.6591110, 5.0526323, -25.0772018, 25.0048370
32: -23.9377861, 2.3165777, -23.9360619, 2.3154247, -21.3494339, 21.3621292
33: -36.4585953, 3.6627893, -36.4294052, 3.6568847, -33.3339844, 33.3835068
34: -37.8971901, -4.7373886, -37.8598328, -4.7450991, -27.7581711, 27.7687988
35: -32.9568863, 0.3128052, -32.9246292, 0.3061843, -28.1849594, 28.1905670
36: -36.8890228, -0.6425562, -36.8435669, -0.6465001, -29.0711212, 29.0424423
37: -44.6087112, -1.6991744, -44.5626678, -1.7016835, -38.8315277, 38.7950134
38: -44.0021057, 2.8915825, -43.9615593, 2.8821473, -40.7295380, 40.7095642
39: -43.5950012, 3.0293489, -43.5912399, 3.0109124, -41.3262329, 41.3830338
40: -32.7336426, 0.0022726, -32.7303314, -0.0129797, -31.0165405, 31.0268974
41: -20.7528191, 7.2879171, -20.7275791, 7.2877550, -26.5343628, 26.5071030
42: -22.9834709, -0.2097459, -22.9826164, -0.2094698, -18.4560699, 18.4524727

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
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
type: B, layer: 1, pos: 1653

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5269515, upper bound: 11.5601485
time: 32.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5381161, upper bound: 11.5601485
time: 34.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.3783369, 19.0638504, -9.3533192, 19.0143509, -25.2361221, 25.2400208
1: -1.2338014, 22.8647957, -1.2220359, 22.8014641, -19.7451630, 19.7784424
2: -1.6277890, 20.9655304, -1.6216512, 20.9132462, -17.2472839, 17.3044243
3: -9.3504429, 16.5536404, -9.3493547, 16.4757423, -21.9818878, 22.0518532
4: -3.1598487, 22.2640419, -3.1479402, 22.2226791, -21.7184944, 21.7475548
5: -7.8521314, 20.6834297, -7.8454747, 20.6024361, -23.7390671, 23.8056450
6: -28.8324928, -1.3572249, -28.8246231, -1.3869133, -23.1899109, 23.2055016
7: -7.7133503, 21.7005901, -7.7028532, 21.6261616, -23.5744209, 23.6421089
8: -14.8020163, 14.8116426, -14.7937450, 14.7518826, -26.4745941, 26.4988937
9: -5.2080631, 21.3232384, -5.1961818, 21.2782707, -24.2605591, 24.3205757
10: -17.8966713, 17.6026955, -17.8838921, 17.5565891, -31.2849884, 31.3332520
11: -26.8148098, 3.5796876, -26.7483330, 3.5731726, -27.9323273, 27.8650055
12: -34.8930740, -2.3075190, -34.8866196, -2.3508086, -27.2092667, 27.2595901
13: -26.2658043, 15.8275490, -26.2616463, 15.6953325, -33.8589554, 33.9970093
14: -55.9434967, -17.4863052, -55.9233627, -17.5582314, -37.8069305, 37.8527069
15: -14.3820333, 15.5175171, -14.3624363, 15.5099144, -27.9149246, 27.8833618
16: -14.0778751, 20.8994198, -14.0554218, 20.8064251, -31.0444565, 31.1325455
17: -57.8719254, -14.3381462, -57.8524590, -14.4283314, -41.6369476, 41.7353134
18: -21.6648655, 12.1686373, -21.5737476, 12.1649923, -29.6667709, 29.5615540
19: -22.3799858, 3.5870795, -22.2693634, 3.5850670, -22.8634033, 22.7425117
20: -23.3879204, 1.3897069, -23.2797451, 1.3872375, -19.2681122, 19.1671753
21: -26.8993874, 2.4153659, -26.7860050, 2.4139566, -25.5825348, 25.4659805
22: -28.6446838, 3.3393323, -28.4886818, 3.3395305, -24.8272591, 24.6627579
23: -22.3936329, 5.7304864, -22.2768250, 5.7264438, -22.1141586, 21.9870262
24: -18.4382439, 9.4576426, -18.2953587, 9.4558144, -22.9416962, 22.8016891
25: -23.9212856, 5.4012470, -23.8129826, 5.3955765, -24.4740982, 24.3599625
26: -41.1838531, -0.4459610, -41.0229187, -0.4452958, -30.7343750, 30.5540314
27: -21.6982174, 8.5930996, -21.5650520, 8.5905743, -26.5548477, 26.4095383
28: -24.2375412, 6.0849752, -24.1069794, 6.0835171, -22.0898399, 21.9361191
29: -27.9810867, -0.2047256, -27.8324890, -0.2054570, -24.0908127, 23.9176598
30: -28.2096214, 3.7667687, -28.1074181, 3.7590175, -26.1737289, 26.0779419
31: -22.7516193, 5.0689535, -22.6604061, 5.0652313, -25.1300964, 25.0453186
32: -23.9622116, 2.3446541, -23.9422588, 2.3184073, -21.3711967, 21.3954315
33: -36.4766121, 3.6716623, -36.4322090, 3.6580262, -33.3531723, 33.3992157
34: -37.9065056, -4.7311478, -37.8605270, -4.7443357, -27.7713165, 27.7755508
35: -32.9702568, 0.3210416, -32.9261627, 0.3077908, -28.2020493, 28.2009354
36: -36.8989220, -0.6329913, -36.8446732, -0.6441488, -29.0886536, 29.0574570
37: -44.6299820, -1.6970162, -44.5659103, -1.7018900, -38.8495178, 38.8145142
38: -44.0223083, 2.9101548, -43.9637756, 2.8862467, -40.7673035, 40.7399673
39: -43.6322746, 3.0486484, -43.5994225, 3.0110140, -41.3594513, 41.4070663
40: -32.7704659, 0.0331151, -32.7376671, -0.0116186, -31.0495987, 31.0647583
41: -20.7639866, 7.3000131, -20.7301636, 7.2896862, -26.5466766, 26.5211105
42: -22.9919014, -0.1947670, -22.9850044, -0.2073181, -18.4641914, 18.4709206

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1653

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5500821, upper bound: 11.5613050
time: 33.96 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5613048, upper bound: 11.5613050
time: 29.35 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.3252850, 19.0952301, -9.2068005, 18.9785233, -25.1388550, 25.1557083
1: -1.1899672, 22.8791809, -1.1223569, 22.7681332, -19.6686096, 19.7250061
2: -1.6043940, 20.9981194, -1.5506065, 20.8953838, -17.2113152, 17.2725029
3: -9.3110151, 16.5886078, -9.2786150, 16.4455662, -21.9136887, 22.0236931
4: -3.1064825, 22.2761726, -3.0452461, 22.2133656, -21.6667595, 21.6612396
5: -7.8015509, 20.7099533, -7.7473350, 20.5647049, -23.6533051, 23.7425766
6: -28.8185234, -1.3865442, -28.7808704, -1.4698062, -23.1056824, 23.1273918
7: -7.6629734, 21.7013760, -7.6027288, 21.5989151, -23.5046234, 23.5514107
8: -14.7356596, 14.8384781, -14.6486588, 14.6973877, -26.3633919, 26.4124146
9: -5.1586461, 21.3270092, -5.1105504, 21.2510147, -24.1885605, 24.2605209
10: -17.8552055, 17.6062317, -17.7794914, 17.5243416, -31.2111893, 31.2249374
11: -26.8841972, 3.5700288, -26.7244587, 3.5252233, -27.9532471, 27.8288879
12: -34.8896141, -2.3013568, -34.8679886, -2.4160037, -27.1356354, 27.2384262
13: -26.2512970, 15.9021988, -26.2307587, 15.6418133, -33.7887878, 34.0372009
14: -55.9137802, -17.4873257, -55.8285980, -17.6061478, -37.6209869, 37.7341461
15: -14.3438816, 15.5306568, -14.2505989, 15.4608269, -27.8298798, 27.8032684
16: -14.0576353, 20.8870487, -13.9884844, 20.7877388, -31.0206833, 31.0700531
17: -57.8686295, -14.3129883, -57.7809067, -14.4775219, -41.5117493, 41.6509171
18: -21.7208862, 12.1472149, -21.5216465, 12.1023169, -29.6513290, 29.4876480
19: -22.4332581, 3.5533845, -22.2188873, 3.5115221, -22.8478241, 22.6677933
20: -23.4505692, 1.3691370, -23.2261486, 1.3108702, -19.2870789, 19.0971298
21: -26.9772739, 2.3904219, -26.7323570, 2.3358912, -25.5932007, 25.3950119
22: -28.7134418, 3.3118789, -28.4424324, 3.2740161, -24.8552551, 24.5998993
23: -22.4428082, 5.6962442, -22.2450409, 5.6645045, -22.1112289, 21.9295921
24: -18.5209293, 9.4244089, -18.2415695, 9.3823729, -22.9746704, 22.7227478
25: -23.9855499, 5.3677797, -23.7824879, 5.3128195, -24.4702110, 24.3117294
26: -41.2259521, -0.4862237, -40.9795532, -0.5316138, -30.7108612, 30.4803848
27: -21.7747765, 8.5679245, -21.5318527, 8.5340433, -26.5773544, 26.3490372
28: -24.2889061, 6.0384655, -24.0681152, 5.9942083, -22.0588455, 21.8632812
29: -28.0423374, -0.2280264, -27.7993736, -0.2381301, -24.1138268, 23.8659897
30: -28.3036289, 3.7495246, -28.0701237, 3.6838961, -26.2166405, 26.0468941
31: -22.8088150, 5.0335579, -22.6050816, 4.9863524, -25.1215286, 24.9577255
32: -23.9550209, 2.3354332, -23.8957882, 2.2513437, -21.3264732, 21.3354340
33: -36.4794769, 3.6344557, -36.3375854, 3.5122285, -33.2684402, 33.2680359
34: -37.9266281, -4.7534585, -37.7952347, -4.8589106, -27.7179260, 27.6851425
35: -32.9720001, 0.2900167, -32.8524590, 0.1728063, -28.0987625, 28.0988998
36: -36.9070930, -0.6618714, -36.7781258, -0.7706733, -28.9889908, 28.9600830
37: -44.6497192, -1.7125063, -44.4881668, -1.7884150, -38.7875671, 38.7048950
38: -44.0298538, 2.8709807, -43.8668442, 2.7216907, -40.6491089, 40.5997314
39: -43.6262054, 3.0296378, -43.4832077, 2.8637862, -41.2650757, 41.2778244
40: -32.7743454, 0.0504332, -32.6724930, -0.0605497, -31.0025787, 31.0122681
41: -20.7648926, 7.2878928, -20.6864529, 7.2232175, -26.4832153, 26.4273529
42: -22.9860821, -0.1907687, -22.9657059, -0.2327120, -18.4284897, 18.4549713

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1778
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
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1777
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
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1756
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
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1556
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
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 981
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
type: B, layer: 1, pos: 1653

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5417593, upper bound: 11.5175923
time: 31.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5529021, upper bound: 11.5175923
time: 34.93 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.3861876, 19.1062965, -9.2229404, 18.9789619, -25.2043152, 25.1873856
1: -1.2331882, 22.9021683, -1.1350460, 22.7683487, -19.7067070, 19.7603378
2: -1.6406348, 21.0182476, -1.5612047, 20.8956547, -17.2414474, 17.3054962
3: -9.3561707, 16.6230698, -9.2921238, 16.4464474, -21.9484024, 22.0613937
4: -3.1670322, 22.2953606, -3.0626121, 22.2134590, -21.7174988, 21.6944427
5: -7.8447356, 20.7471390, -7.7600646, 20.5658875, -23.6856766, 23.7862206
6: -28.8380470, -1.3542509, -28.7867966, -1.4659424, -23.1225739, 23.1626282
7: -7.7165456, 21.7407990, -7.6192713, 21.5994492, -23.5469818, 23.6067314
8: -14.7961044, 14.8627348, -14.6661911, 14.6979179, -26.4160843, 26.4421997
9: -5.2004905, 21.3495922, -5.1209269, 21.2513237, -24.2331314, 24.3015900
10: -17.8838596, 17.6225243, -17.7863636, 17.5252762, -31.2382660, 31.2546921
11: -26.8903503, 3.5876870, -26.7247200, 3.5292501, -27.9688950, 27.8507309
12: -34.9003029, -2.2782507, -34.8707161, -2.4111328, -27.1499481, 27.2720032
13: -26.3016415, 15.9516668, -26.2449379, 15.6429052, -33.8339767, 34.1046524
14: -55.9314270, -17.4745388, -55.8311462, -17.6031551, -37.6389160, 37.7754288
15: -14.3731308, 15.5421772, -14.2556372, 15.4639502, -27.8625641, 27.8171997
16: -14.0940008, 20.9265099, -13.9978743, 20.7879944, -31.0493240, 31.1197281
17: -57.8730621, -14.2954235, -57.7818642, -14.4747915, -41.5140381, 41.7072372
18: -21.7743607, 12.1881170, -21.5227852, 12.1141129, -29.7208252, 29.5220833
19: -22.4695778, 3.5913091, -22.2197857, 3.5230551, -22.8955765, 22.6967697
20: -23.4826736, 1.3994069, -23.2265701, 1.3196740, -19.3313179, 19.1230354
21: -27.0081654, 2.4262774, -26.7329674, 2.3463807, -25.6315231, 25.4240265
22: -28.7700729, 3.3578694, -28.4430752, 3.2881920, -24.9270172, 24.6321220
23: -22.4866104, 5.7429523, -22.2454948, 5.6784272, -22.1708641, 21.9644623
24: -18.5710754, 9.4657602, -18.2426376, 9.3949509, -23.0397720, 22.7565041
25: -24.0335693, 5.4174109, -23.7834816, 5.3274798, -24.5350113, 24.3513184
26: -41.2914085, -0.4296589, -40.9805679, -0.5142145, -30.7995529, 30.5192642
27: -21.8180256, 8.6128664, -21.5327415, 8.5469923, -26.6344681, 26.3886185
28: -24.3398323, 6.0933514, -24.0686131, 6.0109501, -22.1283188, 21.9024582
29: -28.0917625, -0.1836052, -27.7996883, -0.2250690, -24.1762924, 23.8988190
30: -28.3312149, 3.7856376, -28.0704613, 3.6933906, -26.2529526, 26.0752449
31: -22.8471870, 5.0767856, -22.6064129, 4.9989510, -25.1744080, 24.9981766
32: -23.9794350, 2.3635213, -23.9019547, 2.2543557, -21.3482361, 21.3687286
33: -36.4975128, 3.6432858, -36.3404465, 3.5133448, -33.2877045, 33.2837448
34: -37.9359932, -4.7472367, -37.7958984, -4.8581915, -27.7310715, 27.6919327
35: -32.9853210, 0.2983060, -32.8539734, 0.1743774, -28.1158829, 28.1092529
36: -36.9170303, -0.6523514, -36.7792740, -0.7683764, -29.0065002, 28.9751053
37: -44.6709518, -1.7103291, -44.4914207, -1.7886295, -38.8055725, 38.7244263
38: -44.0500870, 2.8894711, -43.8690300, 2.7258239, -40.6867981, 40.6300964
39: -43.6634178, 3.0489082, -43.4912949, 2.8639159, -41.2981415, 41.3017960
40: -32.8110924, 0.0813353, -32.6798477, -0.0591531, -31.0355530, 31.0500946
41: -20.7760506, 7.2999730, -20.6890335, 7.2251616, -26.4955597, 26.4413528
42: -22.9945450, -0.1758087, -22.9680958, -0.2305694, -18.4366264, 18.4734192

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1778
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
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1777
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
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1756
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
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1556
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
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 523
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
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1653

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5648911, upper bound: 11.5187798
time: 27.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5760876, upper bound: 11.5187798
time: 27.32 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.3673601, 19.0965958, -9.3371468, 19.0139256, -25.2217560, 25.2523651
1: -1.2191200, 22.8801689, -1.2093163, 22.8012466, -19.7354469, 19.7813454
2: -1.6242201, 20.9990387, -1.6110258, 20.9129982, -17.2495728, 17.3232727
3: -9.3306427, 16.5914841, -9.3358307, 16.4748650, -21.9731140, 22.0840378
4: -3.1339812, 22.2772751, -3.1305723, 22.2226086, -21.7017517, 21.7437401
5: -7.8302732, 20.7120304, -7.8327637, 20.6012630, -23.7185249, 23.8160286
6: -28.8219166, -1.3604698, -28.8187103, -1.3907528, -23.1821365, 23.1957855
7: -7.6904335, 21.7025681, -7.6863155, 21.6256561, -23.5598755, 23.6262512
8: -14.7788410, 14.8409863, -14.7762060, 14.7513332, -26.4601822, 26.5218048
9: -5.1837311, 21.3291931, -5.1858282, 21.2779732, -24.2402878, 24.3212967
10: -17.8873749, 17.6099434, -17.8769684, 17.5556355, -31.2775803, 31.3249664
11: -26.8896408, 3.5843434, -26.7480984, 3.5691476, -28.0022812, 27.8656158
12: -34.8942833, -2.2829194, -34.8838882, -2.3557138, -27.2075577, 27.2738876
13: -26.2554970, 15.9160652, -26.2474213, 15.6942091, -33.8566742, 34.0695267
14: -55.9426308, -17.4823608, -55.9208107, -17.5611744, -37.8022156, 37.8303070
15: -14.3782940, 15.5330524, -14.3573284, 15.5068188, -27.9105682, 27.8977661
16: -14.0763779, 20.8877258, -14.0459538, 20.8061962, -31.0472717, 31.1168671
17: -57.8913193, -14.3058815, -57.8514977, -14.4310427, -41.6641006, 41.7361755
18: -21.7267265, 12.1642609, -21.5726299, 12.1532297, -29.7142029, 29.5642776
19: -22.4383278, 3.5743606, -22.2684746, 3.5735414, -22.9091263, 22.7392387
20: -23.4528885, 1.3916893, -23.2792740, 1.3784585, -19.3216705, 19.1754227
21: -26.9829006, 2.4130044, -26.7854271, 2.4035046, -25.6570663, 25.4703598
22: -28.7166290, 3.3284922, -28.4880600, 3.3253520, -24.8839493, 24.6633873
23: -22.4454632, 5.7118735, -22.2763977, 5.7124810, -22.1470337, 21.9755020
24: -18.5233078, 9.4453497, -18.2942696, 9.4432602, -23.0110703, 22.7963905
25: -23.9882050, 5.3892050, -23.8119888, 5.3809557, -24.5252991, 24.3583183
26: -41.2285767, -0.4627342, -41.0220032, -0.4626231, -30.7523499, 30.5503845
27: -21.7787189, 8.5826988, -21.5641937, 8.5776291, -26.6232529, 26.4056549
28: -24.2910423, 6.0623279, -24.1064758, 6.0667610, -22.1202011, 21.9258995
29: -28.0472641, -0.2226751, -27.8321495, -0.2185309, -24.1463547, 23.9116669
30: -28.3056583, 3.7703867, -28.1070404, 3.7495182, -26.2603073, 26.0931320
31: -22.8141098, 5.0558305, -22.6591110, 5.0526323, -25.1772537, 25.0355988
32: -23.9579391, 2.3565843, -23.9360619, 2.3154247, -21.3723679, 21.4014091
33: -36.4836884, 3.6844163, -36.4294052, 3.6568847, -33.3565979, 33.4098663
34: -37.9286842, -4.7148480, -37.8598328, -4.7450991, -27.7878189, 27.7916412
35: -32.9749794, 0.3360848, -32.9246292, 0.3061843, -28.2020035, 28.2162476
36: -36.9100723, -0.6194525, -36.8435669, -0.6465001, -29.0923843, 29.0669022
37: -44.6562729, -1.6828260, -44.5626678, -1.7016835, -38.8806305, 38.8139954
38: -44.0357513, 2.9260240, -43.9615593, 2.8821473, -40.7766571, 40.7467041
39: -43.6328278, 3.0805464, -43.5912399, 3.0109124, -41.3597260, 41.4351959
40: -32.7808418, 0.0663192, -32.7303314, -0.0129797, -31.0592957, 31.0910072
41: -20.7691708, 7.3093081, -20.7275791, 7.2877550, -26.5520401, 26.5272903
42: -22.9910717, -0.1842918, -22.9826164, -0.2094698, -18.4646988, 18.4814758

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
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
type: B, layer: 1, pos: 1653

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5423392, upper bound: 11.5393648
time: 28.37 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5535543, upper bound: 11.5393648
time: 29.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.4282579, 19.1076393, -9.3533192, 19.0143509, -25.2872543, 25.2840805
1: -1.2623115, 22.9031334, -1.2220359, 22.8014641, -19.7735367, 19.8166809
2: -1.6604664, 21.0191097, -1.6216512, 20.9132462, -17.2797012, 17.3562737
3: -9.3757992, 16.6259556, -9.3493547, 16.4757423, -22.0078201, 22.1217041
4: -3.1944904, 22.2964859, -3.1479402, 22.2226791, -21.7525368, 21.7769356
5: -7.8734183, 20.7492161, -7.8454747, 20.6024361, -23.7509003, 23.8596497
6: -28.8414345, -1.3282251, -28.8246231, -1.3869133, -23.1990356, 23.2309303
7: -7.7440014, 21.7419605, -7.7028532, 21.6261616, -23.6022301, 23.6815643
8: -14.8393049, 14.8652229, -14.7937450, 14.7518826, -26.5128708, 26.5515518
9: -5.2255778, 21.3517723, -5.1961818, 21.2782707, -24.2848892, 24.3623619
10: -17.9160767, 17.6262302, -17.8838921, 17.5565891, -31.3046265, 31.3547211
11: -26.8957825, 3.6020198, -26.7483330, 3.5731726, -28.0179291, 27.8874893
12: -34.9049950, -2.2598987, -34.8866196, -2.3508086, -27.2218704, 27.3074722
13: -26.3058548, 15.9656200, -26.2616463, 15.6953325, -33.9018478, 34.1370239
14: -55.9603157, -17.4695587, -55.9233627, -17.5582314, -37.8201294, 37.8715897
15: -14.4075260, 15.5445747, -14.3624363, 15.5099144, -27.9432220, 27.9116974
16: -14.1127062, 20.9271908, -14.0554218, 20.8064251, -31.0758820, 31.1665497
17: -57.8957481, -14.2883167, -57.8524590, -14.4283314, -41.6664581, 41.7924957
18: -21.7802544, 12.2051735, -21.5737476, 12.1649923, -29.7837067, 29.5987358
19: -22.4746761, 3.6122851, -22.2693634, 3.5850670, -22.9568558, 22.7681847
20: -23.4849873, 1.4219308, -23.2797451, 1.3872375, -19.3659286, 19.2013245
21: -27.0138168, 2.4488306, -26.7860050, 2.4139566, -25.6953964, 25.4993668
22: -28.7732277, 3.3745015, -28.4886818, 3.3395305, -24.9556999, 24.6956253
23: -22.4892616, 5.7586293, -22.2768250, 5.7264438, -22.2066574, 22.0104065
24: -18.5734138, 9.4867201, -18.2953587, 9.4558144, -23.0761719, 22.8301048
25: -24.0362034, 5.4388800, -23.8129826, 5.3955765, -24.5900574, 24.3978996
26: -41.2939911, -0.4060082, -41.0229187, -0.4452958, -30.8410187, 30.5892715
27: -21.8219223, 8.6276455, -21.5650520, 8.5905743, -26.6803436, 26.4452438
28: -24.3419456, 6.1171851, -24.1069794, 6.0835171, -22.1896553, 21.9650688
29: -28.0966949, -0.1782357, -27.8324890, -0.2054570, -24.2087936, 23.9444656
30: -28.3332520, 3.8064668, -28.1074181, 3.7590175, -26.2966080, 26.1214523
31: -22.8525009, 5.0989966, -22.6604061, 5.0652313, -25.2301407, 25.0760460
32: -23.9823322, 2.3846483, -23.9422588, 2.3184073, -21.3941078, 21.4346657
33: -36.5017242, 3.6932292, -36.4322090, 3.6580262, -33.3758011, 33.4255295
34: -37.9380112, -4.7086082, -37.8605270, -4.7443357, -27.8009567, 27.7984161
35: -32.9883499, 0.3443389, -32.9261627, 0.3077908, -28.2191391, 28.2266235
36: -36.9199753, -0.6099348, -36.8446732, -0.6441488, -29.1099167, 29.0819016
37: -44.6775818, -1.6806846, -44.5659103, -1.7018900, -38.8986816, 38.8335266
38: -44.0559998, 2.9445171, -43.9637756, 2.8862467, -40.8143768, 40.7770844
39: -43.6700439, 3.0998020, -43.5994225, 3.0110140, -41.3928223, 41.4591827
40: -32.8176041, 0.0971923, -32.7376671, -0.0116186, -31.0922775, 31.1288910
41: -20.7803001, 7.3213525, -20.7301636, 7.2896862, -26.5643463, 26.5412903
42: -22.9995117, -0.1693001, -22.9850044, -0.2073181, -18.4728355, 18.4999275

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1741
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1653

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5654347, upper bound: 11.5404899
time: 40.27 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5766891, upper bound: 11.5404899
time: 27.91 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.2244682, 19.0170307, -9.3043394, 19.0549545, -25.1251831, 25.1659546
1: -1.1308627, 22.8063431, -1.1780553, 22.8370609, -19.6807518, 19.7054825
2: -1.5712450, 20.9133301, -1.5989368, 20.9603462, -17.2279968, 17.2337952
3: -9.2960396, 16.4797955, -9.3106766, 16.5378933, -21.9823532, 21.9369316
4: -3.0661259, 22.2308369, -3.0976343, 22.2520638, -21.6502037, 21.6690903
5: -7.7552075, 20.5976219, -7.7960377, 20.6566887, -23.6824265, 23.6926003
6: -28.7415504, -1.4628086, -28.7963486, -1.3969917, -23.0951996, 23.1127663
7: -7.6185198, 21.6281719, -7.6519585, 21.6619892, -23.5101967, 23.5240593
8: -14.6787701, 14.7149029, -14.7341957, 14.7950659, -26.4003868, 26.3776855
9: -5.1133733, 21.2356205, -5.1510472, 21.2920532, -24.2173920, 24.1953201
10: -17.7804813, 17.4002800, -17.8453026, 17.5259266, -31.1410370, 31.0977936
11: -26.7765083, 3.5191498, -26.8204708, 3.5595894, -27.8756485, 27.8727493
12: -34.8704147, -2.3922620, -34.8833160, -2.3439460, -27.1903763, 27.1408882
13: -26.2499409, 15.7175350, -26.2439499, 15.8149166, -33.9619827, 33.8630295
14: -55.8224869, -17.6567230, -55.9011230, -17.5939026, -37.6543884, 37.5611420
15: -14.2588425, 15.4594231, -14.3348360, 15.5196753, -27.8041458, 27.8132706
16: -13.9917631, 20.8019676, -14.0369129, 20.8206577, -30.9966431, 31.0304413
17: -57.7861557, -14.4333735, -57.8495865, -14.4113808, -41.5790405, 41.5288544
18: -21.5466690, 12.1259098, -21.6744843, 12.1436825, -29.5000229, 29.6193542
19: -22.2822628, 3.5267494, -22.3515434, 3.5512691, -22.7355194, 22.7673836
20: -23.2950764, 1.3227611, -23.3701134, 1.3653276, -19.1744537, 19.1964684
21: -26.8008595, 2.3502617, -26.8878975, 2.3875761, -25.4751205, 25.5032501
22: -28.5220413, 3.2949011, -28.6048679, 3.3117704, -24.6959610, 24.7341690
23: -22.3130150, 5.6719050, -22.3660965, 5.6902966, -22.0067673, 22.0242386
24: -18.3197441, 9.4054737, -18.4183807, 9.4247246, -22.8064728, 22.8803902
25: -23.8330822, 5.3365040, -23.9180489, 5.3622870, -24.3619766, 24.3963470
26: -41.0681534, -0.5364385, -41.1241264, -0.4950342, -30.5831223, 30.5867157
27: -21.5720043, 8.5550575, -21.6655960, 8.5657043, -26.3950882, 26.4814301
28: -24.1301537, 6.0175629, -24.2016315, 6.0378695, -21.9409409, 21.9732742
29: -27.8755398, -0.2218032, -27.9363842, -0.2291076, -23.9473801, 23.9861374
30: -28.1338730, 3.7043211, -28.2238541, 3.7430482, -26.1094704, 26.1355743
31: -22.6483688, 4.9966197, -22.7470512, 5.0280666, -25.0038528, 25.0636482
32: -23.8796272, 2.2540352, -23.9275475, 2.3272758, -21.3180466, 21.2922058
33: -36.3135223, 3.5116529, -36.4243126, 3.6308999, -33.2294388, 33.2204132
34: -37.7733994, -4.8524284, -37.8689651, -4.7664914, -27.6668243, 27.6577682
35: -32.8131981, 0.1803198, -32.9135666, 0.2791219, -28.0496368, 28.0460587
36: -36.7312317, -0.7625790, -36.8303833, -0.6715498, -28.9117966, 28.9146271
37: -44.4382324, -1.7886581, -44.5626793, -1.7168441, -38.6554565, 38.6995163
38: -43.8041306, 2.7321229, -43.9514732, 2.8517742, -40.5380707, 40.5645065
39: -43.4614143, 2.8693933, -43.5838394, 3.0149446, -41.2563477, 41.2230759
40: -32.6537094, -0.0453694, -32.7365837, 0.0328605, -30.9778290, 30.9802246
41: -20.6496735, 7.2215710, -20.7104015, 7.2838917, -26.4071503, 26.4146042
42: -22.9684105, -0.2367599, -22.9777222, -0.1962206, -18.4510269, 18.4161873

Time for backsubstitution: 2.18 seconds

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
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 974
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5518971, upper bound: 11.5257083
time: 51.28 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5518971, upper bound: 11.5398423
time: 38.29 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.2788458, 19.0401497, -9.3088465, 19.0554409, -25.1838646, 25.2000237
1: -1.1633992, 22.8374882, -1.1789851, 22.8377686, -19.7154846, 19.7495880
2: -1.6229851, 20.9282932, -1.6012843, 20.9636421, -17.2565994, 17.2915726
3: -9.3204117, 16.4947205, -9.3122206, 16.5388470, -22.0024719, 21.9768181
4: -3.1124983, 22.2385635, -3.1005707, 22.2531013, -21.6940460, 21.7210426
5: -7.8046942, 20.6153030, -7.7995634, 20.6595993, -23.7178497, 23.7242050
6: -28.7941399, -1.3846111, -28.8090324, -1.3950291, -23.1361237, 23.2059021
7: -7.6645269, 21.6434650, -7.6554680, 21.6641979, -23.5547028, 23.5632286
8: -14.7557678, 14.7450352, -14.7352543, 14.7999468, -26.4665298, 26.4424744
9: -5.1992359, 21.2822456, -5.1530523, 21.3017464, -24.3141174, 24.2406425
10: -18.0013962, 17.5631294, -17.8477039, 17.5684471, -31.4017334, 31.2406998
11: -26.8544025, 3.5457001, -26.8227901, 3.5629711, -27.9674530, 27.9021759
12: -34.8964043, -2.3533583, -34.8849030, -2.3386636, -27.2489395, 27.1758270
13: -26.2837906, 15.7514591, -26.2480869, 15.8180227, -34.0035553, 33.9000473
14: -55.9897842, -17.5306358, -55.9053040, -17.5609055, -37.8551102, 37.6687088
15: -14.3175926, 15.4764729, -14.3385096, 15.5225554, -27.8797684, 27.8345566
16: -14.0895395, 20.8450909, -14.0407562, 20.8302441, -31.0976715, 31.0735397
17: -57.8968048, -14.3768597, -57.8526688, -14.4004593, -41.7117767, 41.5938644
18: -21.6000557, 12.1627159, -21.6795349, 12.1454563, -29.5530167, 29.7006607
19: -22.3235626, 3.5393825, -22.3554783, 3.5523088, -22.7901611, 22.7846451
20: -23.3215256, 1.3366277, -23.3726540, 1.3664551, -19.2099152, 19.2133484
21: -26.8606663, 2.3705778, -26.8923931, 2.3894691, -25.5488663, 25.5289116
22: -28.5546227, 3.3323650, -28.6109962, 3.3134327, -24.7891388, 24.7435455
23: -22.3466129, 5.6890602, -22.3680038, 5.6912017, -22.0425224, 22.0489807
24: -18.3476105, 9.4155960, -18.4234867, 9.4255524, -22.8346939, 22.8985291
25: -23.8638153, 5.3642769, -23.9217720, 5.3642998, -24.4342003, 24.4246063
26: -41.1177902, -0.4958401, -41.1272430, -0.4870400, -30.6458969, 30.6247711
27: -21.6345406, 8.6329384, -21.6804276, 8.5681686, -26.4501877, 26.5767975
28: -24.1547413, 6.0591478, -24.2059708, 6.0400705, -21.9738159, 22.0055618
29: -27.9162884, -0.1854327, -27.9426689, -0.2278209, -24.0589828, 23.9856720
30: -28.1608963, 3.7356358, -28.2271233, 3.7460475, -26.1631317, 26.1737137
31: -22.7111282, 5.0147448, -22.7510834, 5.0298328, -25.0723877, 25.0856285
32: -23.9187889, 2.3042612, -23.9372215, 2.3292835, -21.3540421, 21.3560791
33: -36.3852921, 3.6151986, -36.4409294, 3.6320872, -33.2993546, 33.3545990
34: -37.8353615, -4.7473326, -37.8858719, -4.7636786, -27.7250443, 27.7895279
35: -32.8931313, 0.3023839, -32.9340973, 0.2803483, -28.1213760, 28.1911545
36: -36.8293076, -0.6292605, -36.8562546, -0.6701140, -28.9946289, 29.0747070
37: -44.5568047, -1.6669002, -44.5918770, -1.7161617, -38.7597809, 38.8671951
38: -43.9281998, 2.9020600, -43.9811783, 2.8546276, -40.6517792, 40.7631607
39: -43.5266190, 2.9411030, -43.5973625, 3.0168123, -41.3250427, 41.3220673
40: -32.7209702, 0.0437589, -32.7510452, 0.0336659, -31.0451660, 31.0887566
41: -20.7260609, 7.3188448, -20.7298965, 7.2856760, -26.4749680, 26.5353546
42: -22.9755478, -0.2089493, -22.9795132, -0.1938570, -18.4832458, 18.4475975

Time for backsubstitution: 2.20 seconds

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
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 974
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5518971, upper bound: 11.5368824
time: 42.26 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5518971, upper bound: 11.5510219
time: 36.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.2406368, 19.0174294, -9.3652935, 19.0660019, -25.1569023, 25.2314377
1: -1.1435428, 22.8065758, -1.2212768, 22.8600426, -19.7161102, 19.7436142
2: -1.5818508, 20.9136086, -1.6351662, 20.9804230, -17.2610245, 17.2639542
3: -9.3095617, 16.4806404, -9.3558264, 16.5723381, -22.0199890, 21.9716568
4: -3.0835085, 22.2309246, -3.1582046, 22.2712631, -21.6833916, 21.7198830
5: -7.7678580, 20.5987587, -7.8392320, 20.6938725, -23.7260056, 23.7249680
6: -28.7474575, -1.4589949, -28.8158569, -1.3647194, -23.1303482, 23.1296387
7: -7.6350889, 21.6287079, -7.7055449, 21.7014008, -23.5654984, 23.5664406
8: -14.6963034, 14.7154560, -14.7946663, 14.8193169, -26.4301796, 26.4304123
9: -5.1237330, 21.2359409, -5.1929450, 21.3146038, -24.2584381, 24.2399368
10: -17.7873592, 17.4013138, -17.8740063, 17.5421829, -31.1707535, 31.1248245
11: -26.7767563, 3.5231924, -26.8266335, 3.5772762, -27.8974838, 27.8883972
12: -34.8731461, -2.3873491, -34.8940201, -2.3208919, -27.2239609, 27.1551590
13: -26.2641335, 15.7186403, -26.2943134, 15.8644857, -34.0294266, 33.9081573
14: -55.8250351, -17.6537914, -55.9188347, -17.5810757, -37.6957092, 37.5790863
15: -14.2639198, 15.4625320, -14.3640232, 15.5312300, -27.8181152, 27.8459473
16: -14.0011358, 20.8021564, -14.0732574, 20.8601437, -31.0463257, 31.0590668
17: -57.7871628, -14.4306641, -57.8540154, -14.3937483, -41.6353073, 41.5311584
18: -21.5478134, 12.1376467, -21.7279472, 12.1845665, -29.5344620, 29.6888809
19: -22.2831554, 3.5383091, -22.3878422, 3.5892322, -22.7644806, 22.8151550
20: -23.2955456, 1.3315754, -23.4021816, 1.3955643, -19.2003784, 19.2407188
21: -26.8014450, 2.3607631, -26.9187565, 2.4233782, -25.5041428, 25.5415497
22: -28.5226955, 3.3090680, -28.6615143, 3.3577385, -24.7282181, 24.8059349
23: -22.3134537, 5.6858749, -22.4098625, 5.7370210, -22.0416489, 22.0838699
24: -18.3208084, 9.4180193, -18.4684563, 9.4661350, -22.8401871, 22.9455147
25: -23.8340607, 5.3512068, -23.9660301, 5.4119439, -24.4015732, 24.4610977
26: -41.0690842, -0.5190601, -41.1895142, -0.4383769, -30.6220245, 30.6754379
27: -21.5728912, 8.5680180, -21.7087574, 8.6106195, -26.4346619, 26.5385132
28: -24.1307220, 6.0343103, -24.2525978, 6.0927186, -21.9801331, 22.0427246
29: -27.8758659, -0.2087380, -27.9858170, -0.1846879, -23.9802322, 24.0485306
30: -28.1342278, 3.7138181, -28.2514114, 3.7791603, -26.1378059, 26.1718140
31: -22.6496811, 5.0092630, -22.7854195, 5.0712352, -25.0442963, 25.1165237
32: -23.8858147, 2.2570026, -23.9519672, 2.3553097, -21.3513298, 21.3139648
33: -36.3163605, 3.5127983, -36.4424057, 3.6396971, -33.2450714, 33.2396317
34: -37.7740555, -4.8516779, -37.8783264, -4.7602286, -27.6735611, 27.6709061
35: -32.8147278, 0.1819339, -32.9269714, 0.2873802, -28.0600510, 28.0631561
36: -36.7323875, -0.7602520, -36.8403397, -0.6620440, -28.9268341, 28.9321518
37: -44.4414444, -1.7889171, -44.5840187, -1.7146878, -38.6749725, 38.7175293
38: -43.8062897, 2.7362127, -43.9717140, 2.8703241, -40.5684204, 40.6022339
39: -43.4695129, 2.8695207, -43.6210861, 3.0342464, -41.2803040, 41.2562027
40: -32.6610489, -0.0439701, -32.7733078, 0.0637584, -31.0157013, 31.0132294
41: -20.6522751, 7.2235193, -20.7215405, 7.2959647, -26.4211502, 26.4269562
42: -22.9708099, -0.2346058, -22.9861984, -0.1812303, -18.4694748, 18.4243202

Time for backsubstitution: 2.18 seconds

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
type: B, layer: 1, pos: 529

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5488235
time: 33.33 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5629606
time: 32.19 seconds

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

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5600512
time: 45.43 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5742005
time: 43.77 seconds

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

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5263182
time: 31.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5404558
time: 39.72 seconds

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

Time for backsubstitution: 2.19 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5375597
time: 31.87 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5516993
time: 33.75 seconds

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

Time for backsubstitution: 2.20 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5493827
time: 23.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5635265
time: 28.39 seconds

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

Time for backsubstitution: 2.21 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5606881
time: 25.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5748328
time: 39.43 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 67.18 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5240669
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5406721
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5334983, upper bound: 11.5252685
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5334983, upper bound: 11.5418773
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5459994
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5625290
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5341288, upper bound: 11.5471347
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5341288, upper bound: 11.5636658
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5023389
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5103083, upper bound: 11.5200724
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5497468, upper bound: 11.5035148
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5497468, upper bound: 11.5212424
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5272032, upper bound: 11.5241521
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5272032, upper bound: 11.5418027
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5503475, upper bound: 11.5252725
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5503475, upper bound: 11.5429131
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5257199
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5416904
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5367906
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5248577, upper bound: 11.5528301
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5488710
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5648322
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5600329
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5260626, upper bound: 11.5760283
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5467291, upper bound: 11.5263603
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5467291, upper bound: 11.5422664
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5467291, upper bound: 11.5375187
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5467291, upper bound: 11.5534672
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5494866
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5653754
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5607146
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5478629, upper bound: 11.5766147
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5263508, upper bound: 11.5382248
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5374069, upper bound: 11.5382249
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5495173, upper bound: 11.5394490
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5606478, upper bound: 11.5394490
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5269515, upper bound: 11.5601485
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5381161, upper bound: 11.5601485
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5500821, upper bound: 11.5613050
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5613048, upper bound: 11.5613050
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5417593, upper bound: 11.5175923
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5529021, upper bound: 11.5175923
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5648911, upper bound: 11.5187798
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5760876, upper bound: 11.5187798
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5423392, upper bound: 11.5393648
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5535543, upper bound: 11.5393648
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5654347, upper bound: 11.5404899
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5766891, upper bound: 11.5404899
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5518971, upper bound: 11.5257083
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5518971, upper bound: 11.5398423
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5518971, upper bound: 11.5368824
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5518971, upper bound: 11.5510219
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5488235
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5629606
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5600512
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5742005
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5263182
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5404558
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5375597
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5516993
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5493827
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5635265
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5606881
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.18
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5748328

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.2063732, 19.0099411, -9.1506596, 18.9766521, -25.0239105, 25.0101013
1: -1.1115837, 22.7979202, -1.0789518, 22.7661133, -19.5996590, 19.6025200
2: -1.5303900, 20.9098835, -1.5120063, 20.8935585, -17.1467514, 17.1465378
3: -9.2548180, 16.4646072, -9.2461281, 16.4396229, -21.8542404, 21.8740387
4: -3.0197849, 22.2199345, -2.9984865, 22.2117538, -21.5873413, 21.5726662
5: -7.7394385, 20.5917835, -7.7079625, 20.5595551, -23.6040688, 23.6013794
6: -28.7971001, -1.4374967, -28.7756367, -1.4834437, -23.0516968, 23.0709038
7: -7.5836987, 21.6203079, -7.5610881, 21.5957909, -23.4338074, 23.4341202
8: -14.6425600, 14.7426920, -14.5931683, 14.6919575, -26.2770157, 26.2777977
9: -5.1123695, 21.2718658, -5.0855379, 21.2475014, -24.1306229, 24.1328468
10: -17.8021545, 17.5435581, -17.7516327, 17.5176239, -31.1519547, 31.1302795
11: -26.7381058, 3.5140023, -26.7196007, 3.4935842, -27.7697906, 27.7734070
12: -34.8508682, -2.3975620, -34.8452377, -2.4284396, -27.0871048, 27.1183014
13: -26.1570263, 15.6725559, -26.1780071, 15.6341915, -33.6919022, 33.7499008
14: -55.8308105, -17.5816479, -55.7697220, -17.6149349, -37.5416336, 37.5766678
15: -14.2914839, 15.4898863, -14.2282686, 15.4540873, -27.7645569, 27.7366638
16: -13.9729033, 20.8027744, -13.9527645, 20.7854176, -30.9457169, 30.9447708
17: -57.7853546, -14.4592619, -57.7366142, -14.4899817, -41.4144592, 41.4497604
18: -21.5541573, 12.0775719, -21.5108833, 12.0690041, -29.4445724, 29.4080658
19: -22.2540855, 3.4800372, -22.2108555, 3.4627628, -22.6203156, 22.5944748
20: -23.2727165, 1.2849021, -23.2225780, 1.2574077, -19.0511436, 19.0236816
21: -26.7718601, 2.3063812, -26.7249374, 2.2831192, -25.3351822, 25.3162880
22: -28.4781914, 3.2262394, -28.4367218, 3.2208545, -24.5614357, 24.5240250
23: -22.2693634, 5.6145992, -22.2412415, 5.6109800, -21.8846321, 21.8601456
24: -18.2846012, 9.3498755, -18.2363968, 9.3352194, -22.6895523, 22.6533241
25: -23.8026619, 5.2812128, -23.7777500, 5.2655964, -24.2351608, 24.2297325
26: -41.0126419, -0.5933504, -40.9747047, -0.6030641, -30.4270401, 30.3925171
27: -21.5523071, 8.4816666, -21.5253677, 8.4807749, -26.2966919, 26.2636719
28: -24.1003342, 5.9501433, -24.0649605, 5.9360285, -21.8128357, 21.7901802
29: -27.8234253, -0.2926691, -27.7941952, -0.2770455, -23.8502579, 23.8053589
30: -28.1014786, 3.6573029, -28.0670490, 3.6322784, -25.9641953, 25.9622726
31: -22.6430321, 4.9635005, -22.5965614, 4.9463220, -24.9123077, 24.8824692
32: -23.9131241, 2.2796180, -23.8901329, 2.2413135, -21.2624359, 21.2741318
33: -36.4097137, 3.5930996, -36.3279037, 3.4946365, -33.1855011, 33.2041779
34: -37.8528214, -4.8206067, -37.7913094, -4.8947792, -27.6065903, 27.6194000
35: -32.9118881, 0.2369657, -32.8454666, 0.1522875, -28.0181885, 28.0370102
36: -36.8320694, -0.7307568, -36.7715530, -0.8086524, -28.8723907, 28.8868484
37: -44.5350342, -1.7553062, -44.4735794, -1.8118954, -38.6455078, 38.6408005
38: -43.9414597, 2.7879834, -43.8568115, 2.6886525, -40.4885101, 40.4992676
39: -43.5471344, 2.9610600, -43.4654465, 2.8585229, -41.1682892, 41.1902008
40: -32.6897964, -0.0336306, -32.6555099, -0.0640213, -30.9137344, 30.9080048
41: -20.7077713, 7.2402115, -20.6773911, 7.1989326, -26.3736572, 26.3666611
42: -22.9701157, -0.2309551, -22.9631367, -0.2450345, -18.3980179, 18.4065628

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 974
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
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5077801, upper bound: 11.5104686
time: 32.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5077801, upper bound: 11.5215026
time: 30.85 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.2063732, 19.0099411, -9.2196369, 19.0181541, -25.0673637, 25.0814972
1: -1.1115837, 22.7979202, -1.1287818, 22.8090305, -19.6456947, 19.6476517
2: -1.5303900, 20.9098835, -1.5532911, 20.9282436, -17.1852646, 17.1849823
3: -9.2548180, 16.4646072, -9.2770100, 16.4913120, -21.9025040, 21.9036903
4: -3.0197849, 22.2199345, -3.0504842, 22.2355785, -21.6049576, 21.6199760
5: -7.7394385, 20.5917835, -7.7488227, 20.6119881, -23.6566086, 23.6372833
6: -28.7971001, -1.4374967, -28.7881546, -1.4614620, -23.0789642, 23.0860443
7: -7.5836987, 21.6203079, -7.6096668, 21.6355057, -23.4749947, 23.4791222
8: -14.6425600, 14.7426920, -14.6489830, 14.7341480, -26.3118439, 26.3239441
9: -5.1123695, 21.2718658, -5.1142893, 21.2740726, -24.1648254, 24.1646385
10: -17.8021545, 17.5435581, -17.7851295, 17.5567780, -31.1941147, 31.1683197
11: -26.7381058, 3.5140023, -26.7846832, 3.5272632, -27.8019714, 27.8414307
12: -34.8508682, -2.3975620, -34.8720627, -2.3798571, -27.1367722, 27.1463852
13: -26.1570263, 15.6725559, -26.2321320, 15.7256479, -33.7855377, 33.8067017
14: -55.8308105, -17.5816479, -55.8358688, -17.5373096, -37.6195679, 37.6416626
15: -14.2914839, 15.4898863, -14.2552776, 15.4678001, -27.7784042, 27.7667618
16: -13.9729033, 20.8027744, -14.0025921, 20.8419209, -31.0032654, 30.9909897
17: -57.7853546, -14.4592619, -57.7961121, -14.3935452, -41.5142517, 41.5152740
18: -21.5541573, 12.0775719, -21.5622597, 12.1020927, -29.4811096, 29.4621391
19: -22.2540855, 3.4800372, -22.2953205, 3.5108833, -22.6694031, 22.6797409
20: -23.2727165, 1.2849021, -23.3033485, 1.3094244, -19.1081734, 19.1068878
21: -26.7718601, 2.3063812, -26.8158894, 2.3336580, -25.3867798, 25.4088516
22: -28.4781914, 3.2262394, -28.5433578, 3.2713184, -24.6164017, 24.6347122
23: -22.2693634, 5.6145992, -22.3190861, 5.6644812, -21.9398766, 21.9404488
24: -18.2846012, 9.3498755, -18.3375359, 9.3806295, -22.7368851, 22.7561264
25: -23.8026619, 5.2812128, -23.8456726, 5.3145862, -24.2882957, 24.3003845
26: -41.0126419, -0.5933504, -41.0778503, -0.5358739, -30.4991074, 30.4975815
27: -21.5523071, 8.4816666, -21.6241379, 8.5325069, -26.3508530, 26.3646088
28: -24.1003342, 5.9501433, -24.1490879, 5.9921370, -21.8730469, 21.8767853
29: -27.8234253, -0.2926691, -27.8974724, -0.2388282, -23.8927994, 23.9104729
30: -28.1014786, 3.6573029, -28.1455593, 3.6847029, -26.0175095, 26.0421677
31: -22.6430321, 4.9635005, -22.6614799, 4.9863443, -24.9546890, 24.9498215
32: -23.9131241, 2.2796180, -23.9119072, 2.2571120, -21.2796364, 21.3040237
33: -36.4097137, 3.5930996, -36.3725700, 3.5143909, -33.2053833, 33.2478943
34: -37.8528214, -4.8206067, -37.8336716, -4.8502202, -27.6548462, 27.6657333
35: -32.9118881, 0.2369657, -32.8874550, 0.1820593, -28.0493240, 28.0798950
36: -36.8320694, -0.7307568, -36.8255501, -0.7628713, -28.9205322, 28.9425278
37: -44.5350342, -1.7553062, -44.5405579, -1.7855029, -38.6741180, 38.7094040
38: -43.9414597, 2.7879834, -43.9115601, 2.7371879, -40.5456848, 40.5611572
39: -43.5471344, 2.9610600, -43.5068169, 2.8758640, -41.1872101, 41.2415009
40: -32.6897964, -0.0336306, -32.6929588, -0.0439644, -30.9344254, 30.9457016
41: -20.7077713, 7.2402115, -20.7181892, 7.2252426, -26.4067993, 26.4129868
42: -22.9701157, -0.2309551, -22.9715023, -0.2303288, -18.4132881, 18.4164658

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 974
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
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5077801, upper bound: 11.5270717
time: 31.48 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5077801, upper bound: 11.5380848
time: 30.99 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.2673492, 19.0209541, -9.1668606, 18.9770927, -25.0894470, 25.0417900
1: -1.1548123, 22.8208809, -1.0916166, 22.7663345, -19.6377792, 19.6378822
2: -1.5666842, 20.9299583, -1.5226417, 20.8938370, -17.1768837, 17.1795616
3: -9.2999401, 16.4990730, -9.2596788, 16.4404945, -21.8889771, 21.9116211
4: -3.0803304, 22.2391434, -3.0158339, 22.2118416, -21.6381073, 21.6058311
5: -7.7825866, 20.6289253, -7.7206664, 20.5607414, -23.6364403, 23.6449127
6: -28.8166275, -1.4052849, -28.7815399, -1.4795914, -23.0685654, 23.1060562
7: -7.6372886, 21.6596794, -7.5776205, 21.5963135, -23.4762001, 23.4894066
8: -14.7029724, 14.7669344, -14.6106911, 14.6924868, -26.3296700, 26.3075333
9: -5.1542592, 21.2944412, -5.0958929, 21.2478333, -24.1752243, 24.1739044
10: -17.8308296, 17.5598392, -17.7584953, 17.5185966, -31.1789780, 31.1599503
11: -26.7442722, 3.5316596, -26.7198296, 3.4976249, -27.7854156, 27.7952728
12: -34.8615494, -2.3745136, -34.8479805, -2.4235415, -27.1014023, 27.1519394
13: -26.2073650, 15.7220850, -26.1922112, 15.6353083, -33.7370987, 33.8174133
14: -55.8484764, -17.5689030, -55.7722092, -17.6120052, -37.5595627, 37.6180038
15: -14.3206930, 15.5014095, -14.2333632, 15.4571962, -27.7972107, 27.7506104
16: -14.0092649, 20.8422699, -13.9621477, 20.7856312, -30.9743576, 30.9944153
17: -57.7897949, -14.4416151, -57.7375832, -14.4872522, -41.4167404, 41.5060730
18: -21.6075935, 12.1184702, -21.5120220, 12.0807858, -29.5140762, 29.4425049
19: -22.2904282, 3.5179753, -22.2117710, 3.4743395, -22.6680679, 22.6234703
20: -23.3048134, 1.3151670, -23.2230206, 1.2662115, -19.0953903, 19.0496140
21: -26.8027496, 2.3422437, -26.7255096, 2.2936182, -25.3734894, 25.3453484
22: -28.5348167, 3.2722063, -28.4373455, 3.2350292, -24.6331635, 24.5562820
23: -22.3131447, 5.6613183, -22.2416840, 5.6249213, -21.9442444, 21.8950462
24: -18.3346825, 9.3912649, -18.2374840, 9.3477650, -22.7546539, 22.6870728
25: -23.8506889, 5.3308249, -23.7787437, 5.2802601, -24.2999420, 24.2693253
26: -41.0780640, -0.5367780, -40.9756355, -0.5857000, -30.5157318, 30.4314194
27: -21.5954781, 8.5266008, -21.5262356, 8.4937458, -26.3537445, 26.3032761
28: -24.1512527, 6.0050354, -24.0654736, 5.9528265, -21.8822861, 21.8293648
29: -27.8728142, -0.2482924, -27.7945213, -0.2639883, -23.9126205, 23.8382225
30: -28.1290398, 3.6934481, -28.0674114, 3.6417503, -26.0004120, 25.9905930
31: -22.6814308, 5.0066919, -22.5978832, 4.9589176, -24.9651794, 24.9229660
32: -23.9375572, 2.3076944, -23.8963127, 2.2443223, -21.2842064, 21.3074303
33: -36.4277802, 3.6019440, -36.3307343, 3.4957514, -33.2047272, 33.2199249
34: -37.8621445, -4.8143415, -37.7919617, -4.8940134, -27.6197281, 27.6261826
35: -32.9252472, 0.2452154, -32.8469543, 0.1538706, -28.0352707, 28.0474472
36: -36.8419838, -0.7212434, -36.7726479, -0.8063602, -28.8898849, 28.9018784
37: -44.5563126, -1.7531285, -44.4767532, -1.8121743, -38.6634827, 38.6603165
38: -43.9617615, 2.8065495, -43.8590240, 2.6927209, -40.5262451, 40.5296631
39: -43.5843658, 2.9803271, -43.4735641, 2.8586063, -41.2014008, 41.2141724
40: -32.7265778, -0.0027471, -32.6628380, -0.0626516, -30.9468079, 30.9458847
41: -20.7189178, 7.2522879, -20.6800022, 7.2009134, -26.3859940, 26.3806915
42: -22.9785881, -0.2159836, -22.9655190, -0.2428923, -18.4061813, 18.4250031

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5309873, upper bound: 11.5116241
time: 30.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5309873, upper bound: 11.5227192
time: 32.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.2673492, 19.0209541, -9.2357893, 19.0186005, -25.1328888, 25.1132050
1: -1.1548123, 22.8208809, -1.1414814, 22.8092537, -19.6837997, 19.6829948
2: -1.5666842, 20.9299583, -1.5639107, 20.9285164, -17.2153778, 17.2179718
3: -9.2999401, 16.4990730, -9.2904797, 16.4921627, -21.9372330, 21.9412842
4: -3.0803304, 22.2391434, -3.0678463, 22.2356548, -21.6557198, 21.6531219
5: -7.7825866, 20.6289253, -7.7615509, 20.6131401, -23.6889572, 23.6808090
6: -28.8166275, -1.4052849, -28.7940865, -1.4576573, -23.0958557, 23.1211929
7: -7.6372886, 21.6596794, -7.6262102, 21.6360245, -23.5173798, 23.5344238
8: -14.7029724, 14.7669344, -14.6665306, 14.7346859, -26.3645248, 26.3537064
9: -5.1542592, 21.2944412, -5.1246405, 21.2743530, -24.2094193, 24.2057114
10: -17.8308296, 17.5598392, -17.7920532, 17.5577641, -31.2211533, 31.1980057
11: -26.7442722, 3.5316596, -26.7849426, 3.5313277, -27.8176270, 27.8633118
12: -34.8615494, -2.3745136, -34.8748131, -2.3749499, -27.1510773, 27.1800308
13: -26.2073650, 15.7220850, -26.2464180, 15.7267570, -33.8306808, 33.8741989
14: -55.8484764, -17.5689030, -55.8384247, -17.5343590, -37.6374664, 37.6829529
15: -14.3206930, 15.5014095, -14.2603168, 15.4709034, -27.8110657, 27.7807465
16: -14.0092649, 20.8422699, -14.0120029, 20.8421402, -31.0319061, 31.0406570
17: -57.7897949, -14.4416151, -57.7970428, -14.3908377, -41.5165558, 41.5715790
18: -21.6075935, 12.1184702, -21.5634308, 12.1138973, -29.5506287, 29.4965897
19: -22.2904282, 3.5179753, -22.2962303, 3.5224550, -22.7171326, 22.7087250
20: -23.3048134, 1.3151670, -23.3037910, 1.3182073, -19.1524277, 19.1328201
21: -26.8027496, 2.3422437, -26.8164673, 2.3441401, -25.4250717, 25.4379120
22: -28.5348167, 3.2722063, -28.5439987, 3.2855253, -24.6881409, 24.6669922
23: -22.3131447, 5.6613183, -22.3195496, 5.6784406, -21.9994965, 21.9753418
24: -18.3346825, 9.3912649, -18.3386059, 9.3931618, -22.8019943, 22.7898979
25: -23.8506889, 5.3308249, -23.8466778, 5.3292503, -24.3530884, 24.3399620
26: -41.0780640, -0.5367780, -41.0788040, -0.5184579, -30.5877914, 30.5364609
27: -21.5954781, 8.5266008, -21.6250114, 8.5454617, -26.4079208, 26.4042206
28: -24.1512527, 6.0050354, -24.1496048, 6.0089235, -21.9425011, 21.9159470
29: -27.8728142, -0.2482924, -27.8977795, -0.2257998, -23.9551849, 23.9433403
30: -28.1290398, 3.6934481, -28.1459045, 3.6942296, -26.0537643, 26.0704918
31: -22.6814308, 5.0066919, -22.6628151, 4.9989557, -25.0075531, 24.9903107
32: -23.9375572, 2.3076944, -23.9180832, 2.2601039, -21.3013802, 21.3373299
33: -36.4277802, 3.6019440, -36.3753510, 3.5155120, -33.2245560, 33.2636185
34: -37.8621445, -4.8143415, -37.8343048, -4.8494682, -27.6680222, 27.6725311
35: -32.9252472, 0.2452154, -32.8889885, 0.1836462, -28.0664139, 28.0903320
36: -36.8419838, -0.7212434, -36.8266907, -0.7605133, -28.9380493, 28.9575577
37: -44.5563126, -1.7531285, -44.5438232, -1.7857480, -38.6920929, 38.7289352
38: -43.9617615, 2.8065495, -43.9137383, 2.7413044, -40.5834503, 40.5915527
39: -43.5843658, 2.9803271, -43.5149384, 2.8760262, -41.2203217, 41.2654572
40: -32.7265778, -0.0027471, -32.7002792, -0.0426002, -30.9674988, 30.9835892
41: -20.7189178, 7.2522879, -20.7208309, 7.2272072, -26.4191513, 26.4269943
42: -22.9785881, -0.2159836, -22.9738960, -0.2281828, -18.4214363, 18.4349213

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5309873, upper bound: 11.5282422
time: 34.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5309873, upper bound: 11.5393120
time: 49.21 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.2484674, 19.0112972, -9.2810564, 19.0120678, -25.1068230, 25.1067886
1: -1.1407447, 22.7988892, -1.1659107, 22.7992363, -19.6665382, 19.6588516
2: -1.5502589, 20.9107494, -1.5724139, 20.9111671, -17.1849899, 17.1973152
3: -9.2744789, 16.4675083, -9.3033323, 16.4689350, -21.9136887, 21.9344025
4: -3.0472531, 22.2210293, -3.0837603, 22.2209854, -21.6223068, 21.6551514
5: -7.7681179, 20.5938587, -7.7933750, 20.5961246, -23.6693115, 23.6748428
6: -28.8004780, -1.4114428, -28.8134995, -1.4044199, -23.1281433, 23.1393051
7: -7.6111898, 21.6214828, -7.6446776, 21.6225204, -23.4890556, 23.5089226
8: -14.6857147, 14.7452021, -14.7206345, 14.7459068, -26.3737717, 26.3871231
9: -5.1373758, 21.2740803, -5.1607914, 21.2745094, -24.1823425, 24.1936340
10: -17.8343811, 17.5472736, -17.8491268, 17.5489311, -31.2183151, 31.2302628
11: -26.7435760, 3.5283146, -26.7432346, 3.5375156, -27.8188019, 27.8101425
12: -34.8555222, -2.3791471, -34.8611374, -2.3682065, -27.1589203, 27.1539116
13: -26.1612396, 15.6865864, -26.1946507, 15.6865158, -33.7597809, 33.7823029
14: -55.8596878, -17.5767345, -55.8619232, -17.5699921, -37.7229156, 37.6728973
15: -14.3258848, 15.4922485, -14.3350344, 15.5000582, -27.8452377, 27.8311157
16: -13.9916716, 20.8034344, -14.0102606, 20.8038330, -30.9722824, 30.9916687
17: -57.8080444, -14.4521456, -57.8072510, -14.4434347, -41.5668793, 41.5350647
18: -21.5600510, 12.0946255, -21.5619240, 12.1198950, -29.5074387, 29.4846954
19: -22.2592049, 3.5009766, -22.2604694, 3.5248418, -22.6816406, 22.6659698
20: -23.2750568, 1.3074064, -23.2757931, 1.3250184, -19.0857315, 19.1020012
21: -26.7775154, 2.3289642, -26.7780323, 2.3506908, -25.3990860, 25.3917351
22: -28.4813461, 3.2428470, -28.4823017, 3.2721732, -24.5901871, 24.5875702
23: -22.2720432, 5.6302152, -22.2726002, 5.6589518, -21.9204521, 21.9060898
24: -18.2869606, 9.3708630, -18.2891502, 9.3960953, -22.7259445, 22.7269821
25: -23.8053493, 5.3025761, -23.8072472, 5.3336716, -24.2902412, 24.2763901
26: -41.0152550, -0.5698357, -41.0171738, -0.5340719, -30.4685211, 30.4625015
27: -21.5562439, 8.4964380, -21.5577183, 8.5243568, -26.3425980, 26.3203583
28: -24.1024475, 5.9740438, -24.1033649, 6.0085592, -21.8741608, 21.8527908
29: -27.8283691, -0.2873600, -27.8269176, -0.2573988, -23.8827820, 23.8508987
30: -28.1035061, 3.6781602, -28.1039467, 3.6978600, -26.0078735, 26.0085564
31: -22.6483688, 4.9857397, -22.6505966, 5.0125952, -24.9680634, 24.9603500
32: -23.9159737, 2.3007953, -23.9304333, 2.3053489, -21.3082809, 21.3402061
33: -36.4139519, 3.6430521, -36.4197426, 3.6392817, -33.2736053, 33.3460693
34: -37.8548279, -4.7819209, -37.8559189, -4.7808790, -27.6764832, 27.7259064
35: -32.9149094, 0.2830176, -32.9176559, 0.2856851, -28.1214218, 28.1544037
36: -36.8349991, -0.6883264, -36.8370209, -0.6845207, -28.9757996, 28.9936981
37: -44.5416794, -1.7256160, -44.5480690, -1.7251863, -38.7386169, 38.7499313
38: -43.9473877, 2.8429961, -43.9516678, 2.8490524, -40.6161652, 40.6463470
39: -43.5537033, 3.0119572, -43.5735970, 3.0055671, -41.2628632, 41.3475952
40: -32.6962280, -0.0177767, -32.7133369, -0.0165234, -30.9703522, 30.9868393
41: -20.7120190, 7.2616577, -20.7185402, 7.2635064, -26.4424133, 26.4666595
42: -22.9750824, -0.2244649, -22.9800148, -0.2217619, -18.4342232, 18.4330940

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1491
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 974
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5084779, upper bound: 11.5323201
time: 30.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5084779, upper bound: 11.5434971
time: 30.51 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2484674, 19.0112972, -9.3499727, 19.0535736, -25.1502609, 25.1781311
1: -1.1407447, 22.7988892, -1.2157726, 22.8421459, -19.7125511, 19.7039871
2: -1.5502589, 20.9107494, -1.6136892, 20.9458427, -17.2235107, 17.2357445
3: -9.2744789, 16.4675083, -9.3341990, 16.5206566, -21.9619446, 21.9640045
4: -3.0472531, 22.2210293, -3.1357894, 22.2448101, -21.6399536, 21.7024765
5: -7.7681179, 20.5938587, -7.8342547, 20.6485443, -23.7218132, 23.7107430
6: -28.8004780, -1.4114428, -28.8260345, -1.3824897, -23.1554413, 23.1544266
7: -7.6111898, 21.6214828, -7.6932459, 21.6622810, -23.5302353, 23.5539398
8: -14.6857147, 14.7452021, -14.7764797, 14.7880993, -26.4086037, 26.4333191
9: -5.1373758, 21.2740803, -5.1895580, 21.3010750, -24.2165451, 24.2254333
10: -17.8343811, 17.5472736, -17.8826981, 17.5881214, -31.2604980, 31.2683487
11: -26.7435760, 3.5283146, -26.8083553, 3.5712209, -27.8510208, 27.8781815
12: -34.8555222, -2.3791471, -34.8879585, -2.3196335, -27.2085800, 27.1820068
13: -26.1612396, 15.6865864, -26.2489166, 15.7780437, -33.8533707, 33.8390884
14: -55.8596878, -17.5767345, -55.9280930, -17.4924355, -37.8007812, 37.7378387
15: -14.3258848, 15.4922485, -14.3620424, 15.5137939, -27.8591080, 27.8612671
16: -13.9916716, 20.8034344, -14.0601139, 20.8603516, -31.0298386, 31.0379257
17: -57.8080444, -14.4521456, -57.8666534, -14.3470964, -41.6666870, 41.6005325
18: -21.5600510, 12.0946255, -21.6133041, 12.1530151, -29.5439911, 29.5387650
19: -22.2592049, 3.5009766, -22.3449574, 3.5729671, -22.7307281, 22.7512169
20: -23.2750568, 1.3074064, -23.3565292, 1.3770134, -19.1427917, 19.1851959
21: -26.7775154, 2.3289642, -26.8689880, 2.4012463, -25.4506760, 25.4843254
22: -28.4813461, 3.2428470, -28.5890121, 3.3226502, -24.6451912, 24.6983490
23: -22.2720432, 5.6302152, -22.3504753, 5.7124805, -21.9756851, 21.9863815
24: -18.2869606, 9.3708630, -18.3902950, 9.4415131, -22.7733231, 22.8298187
25: -23.8053493, 5.3025761, -23.8752003, 5.3826580, -24.3434525, 24.3470459
26: -41.0152550, -0.5698357, -41.1203079, -0.4668851, -30.5406418, 30.5675507
27: -21.5562439, 8.4964380, -21.6564884, 8.5761013, -26.3967667, 26.4213409
28: -24.1024475, 5.9740438, -24.1874580, 6.0647078, -21.9343948, 21.9393768
29: -27.8283691, -0.2873600, -27.9301224, -0.2192006, -23.9253845, 23.9560699
30: -28.1035061, 3.6781602, -28.1825066, 3.7503300, -26.0612411, 26.0884933
31: -22.6483688, 4.9857397, -22.7155113, 5.0526280, -25.0104675, 25.0276947
32: -23.9159737, 2.3007953, -23.9522247, 2.3211098, -21.3254662, 21.3701477
33: -36.4139519, 3.6430521, -36.4643402, 3.6590195, -33.2934570, 33.3897705
34: -37.8548279, -4.7819209, -37.8982697, -4.7362905, -27.7247620, 27.7722397
35: -32.9149094, 0.2830176, -32.9596863, 0.3154778, -28.1525803, 28.1972809
36: -36.8349991, -0.6883264, -36.8910027, -0.6386991, -29.0240021, 29.0493622
37: -44.5416794, -1.7256160, -44.6151428, -1.6987367, -38.7671661, 38.8185654
38: -43.9473877, 2.8429961, -44.0063095, 2.8976250, -40.6733398, 40.7081757
39: -43.5537033, 3.0119572, -43.6148758, 3.0229349, -41.2818146, 41.3988876
40: -32.6962280, -0.0177767, -32.7507782, 0.0035167, -30.9910431, 31.0245514
41: -20.7120190, 7.2616577, -20.7593536, 7.2898011, -26.4755859, 26.5129471
42: -22.9750824, -0.2244649, -22.9884033, -0.2070537, -18.4495239, 18.4430008

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1491
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 974
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5084779, upper bound: 11.5488518
time: 27.45 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5084779, upper bound: 11.5600182
time: 32.07 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.3094263, 19.0223446, -9.2972431, 19.0124722, -25.1723518, 25.1384697
1: -1.1839499, 22.8218784, -1.1786184, 22.7994595, -19.7046318, 19.6942062
2: -1.5865116, 20.9308510, -1.5830419, 20.9114342, -17.2151375, 17.2303352
3: -9.3196135, 16.5019894, -9.3168554, 16.4698257, -21.9484100, 21.9719658
4: -3.1077814, 22.2402534, -3.1011004, 22.2210674, -21.6731339, 21.6883011
5: -7.8112931, 20.6309929, -7.8060560, 20.5972805, -23.7016678, 23.7183456
6: -28.8199749, -1.3791966, -28.8194466, -1.4006195, -23.1450195, 23.1743584
7: -7.6647873, 21.6608391, -7.6612291, 21.6230545, -23.5314026, 23.5642242
8: -14.7461700, 14.7694292, -14.7381792, 14.7464342, -26.4264603, 26.4168777
9: -5.1793318, 21.2966423, -5.1711502, 21.2748108, -24.2269821, 24.2346802
10: -17.8630676, 17.5635033, -17.8559742, 17.5499096, -31.2453842, 31.2599792
11: -26.7496834, 3.5459962, -26.7434845, 3.5415497, -27.8344498, 27.8320389
12: -34.8662415, -2.3560624, -34.8638649, -2.3632946, -27.1732559, 27.1874924
13: -26.2115593, 15.7361221, -26.2088547, 15.6877174, -33.8050003, 33.8498001
14: -55.8773956, -17.5639000, -55.8644562, -17.5670204, -37.7408066, 37.7142181
15: -14.3550549, 15.5038090, -14.3401346, 15.5031605, -27.8778534, 27.8450775
16: -14.0280380, 20.8429489, -14.0196209, 20.8040981, -31.0009155, 31.0413284
17: -57.8124809, -14.4345665, -57.8082047, -14.4407253, -41.5691528, 41.5913391
18: -21.6135235, 12.1355219, -21.5630608, 12.1316071, -29.5769501, 29.5191879
19: -22.2955437, 3.5389671, -22.2613811, 3.5363505, -22.7293854, 22.6949196
20: -23.3071461, 1.3376918, -23.2762394, 1.3338099, -19.1299896, 19.1279259
21: -26.8084106, 2.3648283, -26.7786331, 2.3612206, -25.4373856, 25.4207687
22: -28.5379715, 3.2888021, -28.4829712, 3.2863505, -24.6619110, 24.6197891
23: -22.3158283, 5.6769590, -22.2730618, 5.6728845, -21.9800797, 21.9409943
24: -18.3370590, 9.4122562, -18.2901993, 9.4086437, -22.7910309, 22.7607613
25: -23.8533497, 5.3522863, -23.8082657, 5.3483276, -24.3550186, 24.3159752
26: -41.0807419, -0.5132408, -41.0180817, -0.5167603, -30.5572052, 30.5014343
27: -21.5993900, 8.5413465, -21.5585709, 8.5373402, -26.3996124, 26.3599167
28: -24.1534023, 6.0288744, -24.1038818, 6.0253735, -21.9436302, 21.8919563
29: -27.8778172, -0.2429261, -27.8272324, -0.2443826, -23.9451065, 23.8837852
30: -28.1310558, 3.7143140, -28.1043472, 3.7073867, -26.0441055, 26.0368805
31: -22.6867371, 5.0289145, -22.6519165, 5.0252075, -25.0209274, 25.0008278
32: -23.9404335, 2.3288608, -23.9366207, 2.3083467, -21.3300591, 21.3734703
33: -36.4319878, 3.6519065, -36.4225540, 3.6404233, -33.2928162, 33.3617630
34: -37.8641777, -4.7757144, -37.8565636, -4.7801533, -27.6896210, 27.7326889
35: -32.9282532, 0.2912912, -32.9191704, 0.2872791, -28.1385269, 28.1648407
36: -36.8449669, -0.6788483, -36.8381195, -0.6821532, -28.9933167, 29.0087204
37: -44.5629501, -1.7234468, -44.5513306, -1.7253947, -38.7565308, 38.7694702
38: -43.9676590, 2.8615336, -43.9538040, 2.8531251, -40.6539001, 40.6766663
39: -43.5908890, 3.0312405, -43.5817261, 3.0056434, -41.2960510, 41.3715515
40: -32.7330208, 0.0131361, -32.7206726, -0.0150945, -31.0034561, 31.0247154
41: -20.7231541, 7.2737265, -20.7211647, 7.2654724, -26.4547882, 26.4806747
42: -22.9835491, -0.2094553, -22.9823990, -0.2196474, -18.4423485, 18.4515495

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1491
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5316438, upper bound: 11.5334295
time: 32.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5316438, upper bound: 11.5446582
time: 40.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.3094263, 19.0223446, -9.3661451, 19.0539837, -25.2157936, 25.2098351
1: -1.1839499, 22.8218784, -1.2284594, 22.8423615, -19.7506485, 19.7393417
2: -1.5865116, 20.9308510, -1.6243134, 20.9461098, -17.2536430, 17.2687607
3: -9.3196135, 16.5019894, -9.3476982, 16.5215034, -21.9966736, 22.0016251
4: -3.1077814, 22.2402534, -3.1531730, 22.2448864, -21.6907692, 21.7356491
5: -7.8112931, 20.6309929, -7.8469491, 20.6496735, -23.7541962, 23.7542915
6: -28.8199749, -1.3791966, -28.8319473, -1.3786068, -23.1723175, 23.1894913
7: -7.6647873, 21.6608391, -7.7098126, 21.6627731, -23.5725861, 23.6092529
8: -14.7461700, 14.7694292, -14.7940149, 14.7886324, -26.4613037, 26.4630508
9: -5.1793318, 21.2966423, -5.1999245, 21.3013954, -24.2611465, 24.2664986
10: -17.8630676, 17.5635033, -17.8895683, 17.5890942, -31.2875214, 31.2980499
11: -26.7496834, 3.5459962, -26.8085823, 3.5752478, -27.8666611, 27.9000549
12: -34.8662415, -2.3560624, -34.8907166, -2.3147116, -27.2228851, 27.2155952
13: -26.2115593, 15.7361221, -26.2630768, 15.7791328, -33.8985519, 33.9065857
14: -55.8773956, -17.5639000, -55.9306488, -17.4894848, -37.8186798, 37.7791595
15: -14.3550549, 15.5038090, -14.3671227, 15.5168886, -27.8917160, 27.8752060
16: -14.0280380, 20.8429489, -14.0695057, 20.8605576, -31.0585175, 31.0875854
17: -57.8124809, -14.4345665, -57.8676338, -14.3443508, -41.6689453, 41.6568604
18: -21.6135235, 12.1355219, -21.6144257, 12.1647644, -29.6135025, 29.5732422
19: -22.2955437, 3.5389671, -22.3458481, 3.5845070, -22.7784576, 22.7801590
20: -23.3071461, 1.3376918, -23.3569984, 1.3858211, -19.1870346, 19.2111282
21: -26.8084106, 2.3648283, -26.8695641, 2.4117267, -25.4889679, 25.5133591
22: -28.5379715, 3.2888021, -28.5896282, 3.3368297, -24.7169075, 24.7305603
23: -22.3158283, 5.6769590, -22.3509026, 5.7263994, -22.0353088, 22.0212975
24: -18.3370590, 9.4122562, -18.3913555, 9.4540167, -22.8384094, 22.8635635
25: -23.8533497, 5.3522863, -23.8762169, 5.3973355, -24.4082413, 24.3866196
26: -41.0807419, -0.5132408, -41.1212845, -0.4495101, -30.6293182, 30.6064758
27: -21.5993900, 8.5413465, -21.6573486, 8.5890217, -26.4538193, 26.4609299
28: -24.1534023, 6.0288744, -24.1879902, 6.0814323, -22.0038452, 21.9785385
29: -27.8778172, -0.2429261, -27.9304733, -0.2061455, -23.9877014, 23.9889145
30: -28.1310558, 3.7143140, -28.1828918, 3.7598248, -26.0974960, 26.1168251
31: -22.6867371, 5.0289145, -22.7168255, 5.0652385, -25.0633163, 25.0681648
32: -23.9404335, 2.3288608, -23.9584217, 2.3241239, -21.3472214, 21.4034004
33: -36.4319878, 3.6519065, -36.4671631, 3.6601324, -33.3126831, 33.4054489
34: -37.8641777, -4.7757144, -37.8989143, -4.7356005, -27.7378998, 27.7789917
35: -32.9282532, 0.2912912, -32.9611626, 0.3170404, -28.1697464, 28.2077026
36: -36.8449669, -0.6788483, -36.8920975, -0.6363335, -29.0414734, 29.0644226
37: -44.5629501, -1.7234468, -44.6183472, -1.6990042, -38.7851257, 38.8380585
38: -43.9676590, 2.8615336, -44.0084534, 2.9017086, -40.7110901, 40.7385330
39: -43.5908890, 3.0312405, -43.6229935, 3.0230980, -41.3150177, 41.4228363
40: -32.7330208, 0.0131361, -32.7581367, 0.0049033, -31.0241623, 31.0624352
41: -20.7231541, 7.2737265, -20.7619228, 7.2917523, -26.4879303, 26.5269623
42: -22.9835491, -0.2094553, -22.9907856, -0.2049215, -18.4576569, 18.4614487

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1491
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5316438, upper bound: 11.5499657
time: 27.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5316438, upper bound: 11.5611768
time: 27.49 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.2565184, 19.0537300, -9.1506596, 18.9766521, -25.0752487, 25.0551109
1: -1.1401849, 22.8363152, -1.0789518, 22.7661133, -19.6280861, 19.6417236
2: -1.5631137, 20.9635143, -1.5120063, 20.8935585, -17.1793365, 17.1993065
3: -9.2801924, 16.5367470, -9.2461281, 16.4396229, -21.8801575, 21.9439774
4: -3.0545883, 22.2523422, -2.9984865, 22.2117538, -21.6215057, 21.6020012
5: -7.7607207, 20.6574383, -7.7079625, 20.5595551, -23.6160469, 23.6557007
6: -28.8060684, -1.4081569, -28.7756367, -1.4834437, -23.0608978, 23.0967979
7: -7.6144161, 21.6616745, -7.5610881, 21.5957909, -23.4619141, 23.4741821
8: -14.6798630, 14.7963724, -14.5931683, 14.6919575, -26.3154297, 26.3317375
9: -5.1299772, 21.3007622, -5.0855379, 21.2475014, -24.1551208, 24.1749496
10: -17.8220291, 17.5673332, -17.7516327, 17.5176239, -31.1720581, 31.1516876
11: -26.8190022, 3.5364008, -26.7196007, 3.4935842, -27.8553162, 27.7959747
12: -34.8630257, -2.3498716, -34.8452377, -2.4284396, -27.0997543, 27.1665306
13: -26.1971283, 15.8108597, -26.1780071, 15.6341915, -33.7349625, 33.8921432
14: -55.8475609, -17.5650749, -55.7697220, -17.6149349, -37.5549088, 37.5954819
15: -14.3170967, 15.5169382, -14.2282686, 15.4540873, -27.7930832, 27.7650299
16: -14.0079498, 20.8310528, -13.9527645, 20.7854176, -30.9772186, 30.9788132
17: -57.8091278, -14.4094658, -57.7366142, -14.4899817, -41.4439697, 41.5069122
18: -21.6696892, 12.1141310, -21.5108833, 12.0690041, -29.5628815, 29.4454346
19: -22.3487854, 3.5052924, -22.2108555, 3.4627628, -22.7141418, 22.6201782
20: -23.3697662, 1.3171625, -23.2225780, 1.2574077, -19.1510048, 19.0578461
21: -26.8862591, 2.3399384, -26.7249374, 2.2831192, -25.4490585, 25.3496780
22: -28.6067219, 3.2614939, -28.4367218, 3.2208545, -24.6920280, 24.5574265
23: -22.3649673, 5.6428256, -22.2412415, 5.6109800, -21.9788933, 21.8839684
24: -18.4197769, 9.3790398, -18.2363968, 9.3352194, -22.8262177, 22.6821022
25: -23.9181023, 5.3188610, -23.7777500, 5.2655964, -24.3521652, 24.2683716
26: -41.1228638, -0.5534782, -40.9747047, -0.6030641, -30.5362167, 30.4290924
27: -21.6759911, 8.5162449, -21.5253677, 8.4807749, -26.4239273, 26.2994156
28: -24.2048607, 5.9824505, -24.0649605, 5.9360285, -21.9148026, 21.8197021
29: -27.9389801, -0.2662175, -27.7941952, -0.2770455, -23.9697495, 23.8323212
30: -28.2250423, 3.6971815, -28.0670490, 3.6322784, -26.0893936, 26.0059776
31: -22.7439690, 4.9936371, -22.5965614, 4.9463220, -25.0127411, 24.9132538
32: -23.9333534, 2.3197634, -23.8901329, 2.2413135, -21.2852173, 21.3135376
33: -36.4347839, 3.6149964, -36.3279037, 3.4946365, -33.2079468, 33.2303009
34: -37.8843765, -4.7979460, -37.7913094, -4.8947792, -27.6370239, 27.6427460
35: -32.9300613, 0.2606473, -32.8454666, 0.1522875, -28.0353088, 28.0633850
36: -36.8531342, -0.7073350, -36.7715530, -0.8086524, -28.8939972, 28.9117355
37: -44.5826645, -1.7386346, -44.4735794, -1.8118954, -38.6944733, 38.6602173
38: -43.9751892, 2.8232946, -43.8568115, 2.6886525, -40.5358276, 40.5369797
39: -43.5854225, 3.0122123, -43.4654465, 2.8585229, -41.2014465, 41.2425766
40: -32.7372017, 0.0305841, -32.6555099, -0.0640213, -30.9564896, 30.9722672
41: -20.7240868, 7.2626228, -20.6773911, 7.1989326, -26.3911972, 26.3872375
42: -22.9777031, -0.2053428, -22.9631367, -0.2450345, -18.4066849, 18.4357338

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1475
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 974
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
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5240288, upper bound: 11.4887218
time: 33.07 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5240288, upper bound: 11.4997714
time: 39.05 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.2565184, 19.0537300, -9.2196369, 19.0181541, -25.1186943, 25.1265106
1: -1.1401849, 22.8363152, -1.1287818, 22.8090305, -19.6741142, 19.6868553
2: -1.5631137, 20.9635143, -1.5532911, 20.9282436, -17.2178421, 17.2377472
3: -9.2801924, 16.5367470, -9.2770100, 16.4913120, -21.9284134, 21.9736290
4: -3.0545883, 22.2523422, -3.0504842, 22.2355785, -21.6391144, 21.6493111
5: -7.7607207, 20.6574383, -7.7488227, 20.6119881, -23.6685867, 23.6916008
6: -28.8060684, -1.4081569, -28.7881546, -1.4614620, -23.0881653, 23.1119385
7: -7.6144161, 21.6616745, -7.6096668, 21.6355057, -23.5031090, 23.5191841
8: -14.6798630, 14.7963724, -14.6489830, 14.7341480, -26.3502579, 26.3778839
9: -5.1299772, 21.3007622, -5.1142893, 21.2740726, -24.1893311, 24.2067413
10: -17.8220291, 17.5673332, -17.7851295, 17.5567780, -31.2142181, 31.1897278
11: -26.8190022, 3.5364008, -26.7846832, 3.5272632, -27.8875046, 27.8639984
12: -34.8630257, -2.3498716, -34.8720627, -2.3798571, -27.1494217, 27.1946106
13: -26.1971283, 15.8108597, -26.2321320, 15.7256479, -33.8285904, 33.9489441
14: -55.8475609, -17.5650749, -55.8358688, -17.5373096, -37.6328430, 37.6604767
15: -14.3170967, 15.5169382, -14.2552776, 15.4678001, -27.8069305, 27.7951279
16: -14.0079498, 20.8310528, -14.0025921, 20.8419209, -31.0347748, 31.0250244
17: -57.8091278, -14.4094658, -57.7961121, -14.3935452, -41.5437469, 41.5724258
18: -21.6696892, 12.1141310, -21.5622597, 12.1020927, -29.5994186, 29.4995041
19: -22.3487854, 3.5052924, -22.2953205, 3.5108833, -22.7632294, 22.7054443
20: -23.3697662, 1.3171625, -23.3033485, 1.3094244, -19.2080421, 19.1410561
21: -26.8862591, 2.3399384, -26.8158894, 2.3336580, -25.5006561, 25.4422417
22: -28.6067219, 3.2614939, -28.5433578, 3.2713184, -24.7469940, 24.6681175
23: -22.3649673, 5.6428256, -22.3190861, 5.6644812, -22.0341377, 21.9642715
24: -18.4197769, 9.3790398, -18.3375359, 9.3806295, -22.8735580, 22.7849045
25: -23.9181023, 5.3188610, -23.8456726, 5.3145862, -24.4053001, 24.3390236
26: -41.1228638, -0.5534782, -41.0778503, -0.5358739, -30.6082840, 30.5341568
27: -21.6759911, 8.5162449, -21.6241379, 8.5325069, -26.4780884, 26.4003601
28: -24.2048607, 5.9824505, -24.1490879, 5.9921370, -21.9750061, 21.9063034
29: -27.9389801, -0.2662175, -27.8974724, -0.2388282, -24.0122910, 23.9374352
30: -28.2250423, 3.6971815, -28.1455593, 3.6847029, -26.1427078, 26.0858688
31: -22.7439690, 4.9936371, -22.6614799, 4.9863443, -25.0551147, 24.9806061
32: -23.9333534, 2.3197634, -23.9119072, 2.2571120, -21.3024178, 21.3434296
33: -36.4347839, 3.6149964, -36.3725700, 3.5143909, -33.2278137, 33.2740097
34: -37.8843765, -4.7979460, -37.8336716, -4.8502202, -27.6852875, 27.6890945
35: -32.9300613, 0.2606473, -32.8874550, 0.1820593, -28.0664520, 28.1062698
36: -36.8531342, -0.7073350, -36.8255501, -0.7628713, -28.9421310, 28.9674149
37: -44.5826645, -1.7386346, -44.5405579, -1.7855029, -38.7230835, 38.7288284
38: -43.9751892, 2.8232946, -43.9115601, 2.7371879, -40.5930176, 40.5988617
39: -43.5854225, 3.0122123, -43.5068169, 2.8758640, -41.2203674, 41.2938843
40: -32.7372017, 0.0305841, -32.6929588, -0.0439644, -30.9771957, 31.0099716
41: -20.7240868, 7.2626228, -20.7181892, 7.2252426, -26.4243469, 26.4335632
42: -22.9777031, -0.2053428, -22.9715023, -0.2303288, -18.4219551, 18.4456367

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1475
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 974
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
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5240288, upper bound: 11.5064539
time: 34.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5240288, upper bound: 11.5175091
time: 35.16 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.3174362, 19.0647659, -9.1668606, 18.9770927, -25.1407242, 25.0867996
1: -1.1833863, 22.8592701, -1.0916166, 22.7663345, -19.6661911, 19.6770897
2: -1.5993886, 20.9835892, -1.5226417, 20.8938370, -17.2094765, 17.2323227
3: -9.3253279, 16.5711937, -9.2596788, 16.4404945, -21.9148712, 21.9816437
4: -3.1151271, 22.2715588, -3.0158339, 22.2118416, -21.6722260, 21.6351967
5: -7.8038692, 20.6946163, -7.7206664, 20.5607414, -23.6483955, 23.6993103
6: -28.8255920, -1.3758707, -28.7815399, -1.4795914, -23.0777435, 23.1319771
7: -7.6680098, 21.7011070, -7.5776205, 21.5963135, -23.5042915, 23.5294800
8: -14.7403011, 14.8205967, -14.6106911, 14.6924868, -26.3680611, 26.3615036
9: -5.1718302, 21.3233719, -5.0958929, 21.2478333, -24.1996994, 24.2159958
10: -17.8507385, 17.5835819, -17.7584953, 17.5185966, -31.1991043, 31.1813965
11: -26.8251591, 3.5540614, -26.7198296, 3.4976249, -27.8709335, 27.8178101
12: -34.8737335, -2.3268366, -34.8479805, -2.4235415, -27.1140060, 27.2001305
13: -26.2474594, 15.8603668, -26.1922112, 15.6353083, -33.7801285, 33.9596481
14: -55.8652344, -17.5522709, -55.7722092, -17.6120052, -37.5728073, 37.6367950
15: -14.3463001, 15.5284595, -14.2333632, 15.4571962, -27.8257370, 27.7789612
16: -14.0443001, 20.8705616, -13.9621477, 20.7856312, -31.0058670, 31.0284653
17: -57.8135834, -14.3918943, -57.7375832, -14.4872522, -41.4462967, 41.5632477
18: -21.7231846, 12.1550388, -21.5120220, 12.0807858, -29.6324158, 29.4798584
19: -22.3851318, 3.5432329, -22.2117710, 3.4743395, -22.7618866, 22.6491356
20: -23.4018250, 1.3474107, -23.2230206, 1.2662115, -19.1952591, 19.0837669
21: -26.9171753, 2.3757572, -26.7255096, 2.2936182, -25.4873886, 25.3787308
22: -28.6633472, 3.3073809, -28.4373455, 3.2350292, -24.7637863, 24.5896759
23: -22.4087791, 5.6895695, -22.2416840, 5.6249213, -22.0385056, 21.9188614
24: -18.4698982, 9.4204254, -18.2374840, 9.3477650, -22.8913269, 22.7158241
25: -23.9661140, 5.3685150, -23.7787437, 5.2802601, -24.4169769, 24.3079414
26: -41.1882591, -0.4968157, -40.9756355, -0.5857000, -30.6249466, 30.4679642
27: -21.7191811, 8.5611830, -21.5262356, 8.4937458, -26.4810028, 26.3389893
28: -24.2558022, 6.0373020, -24.0654736, 5.9528265, -21.9842529, 21.8589096
29: -27.9884300, -0.2217860, -27.7945213, -0.2639883, -24.0321732, 23.8651505
30: -28.2526283, 3.7332819, -28.0674114, 3.6417503, -26.1256332, 26.0342941
31: -22.7823677, 5.0367870, -22.5978832, 4.9589176, -25.0656204, 24.9537125
32: -23.9577770, 2.3478413, -23.8963127, 2.2443223, -21.3069496, 21.3468323
33: -36.4527512, 3.6238065, -36.3307343, 3.4957514, -33.2271729, 33.2460098
34: -37.8937149, -4.7917171, -37.7919617, -4.8940134, -27.6501465, 27.6495056
35: -32.9434433, 0.2689242, -32.8469543, 0.1538706, -28.0524216, 28.0738144
36: -36.8630524, -0.6978121, -36.7726479, -0.8063602, -28.9115219, 28.9267502
37: -44.6039810, -1.7365370, -44.4767532, -1.8121743, -38.7124634, 38.6797485
38: -43.9954758, 2.8417602, -43.8590240, 2.6927209, -40.5735626, 40.5673523
39: -43.6226578, 3.0315104, -43.4735641, 2.8586063, -41.2345123, 41.2666016
40: -32.7739182, 0.0614955, -32.6628380, -0.0626516, -30.9895172, 31.0101471
41: -20.7352238, 7.2747049, -20.6800022, 7.2009134, -26.4035416, 26.4012604
42: -22.9861679, -0.1903675, -22.9655190, -0.2428923, -18.4148178, 18.4541512

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5472419, upper bound: 11.4898627
time: 31.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5472419, upper bound: 11.5009722
time: 33.37 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.3174362, 19.0647659, -9.2357893, 19.0186005, -25.1841736, 25.1582108
1: -1.1833863, 22.8592701, -1.1414814, 22.8092537, -19.7122116, 19.7222023
2: -1.5993886, 20.9835892, -1.5639107, 20.9285164, -17.2479706, 17.2707329
3: -9.3253279, 16.5711937, -9.2904797, 16.4921627, -21.9631348, 22.0113029
4: -3.1151271, 22.2715588, -3.0678463, 22.2356548, -21.6898537, 21.6824875
5: -7.8038692, 20.6946163, -7.7615509, 20.6131401, -23.7009125, 23.7352066
6: -28.8255920, -1.3758707, -28.7940865, -1.4576573, -23.1050415, 23.1471176
7: -7.6680098, 21.7011070, -7.6262102, 21.6360245, -23.5454712, 23.5745010
8: -14.7403011, 14.8205967, -14.6665306, 14.7346859, -26.4029160, 26.4076920
9: -5.1718302, 21.3233719, -5.1246405, 21.2743530, -24.2339020, 24.2478027
10: -17.8507385, 17.5835819, -17.7920532, 17.5577641, -31.2412720, 31.2194519
11: -26.8251591, 3.5540614, -26.7849426, 3.5313277, -27.9031448, 27.8858490
12: -34.8737335, -2.3268366, -34.8748131, -2.3749499, -27.1636887, 27.2282219
13: -26.2474594, 15.8603668, -26.2464180, 15.7267570, -33.8737183, 34.0164413
14: -55.8652344, -17.5522709, -55.8384247, -17.5343590, -37.6506958, 37.7017441
15: -14.3463001, 15.5284595, -14.2603168, 15.4709034, -27.8395844, 27.8091049
16: -14.0443001, 20.8705616, -14.0120029, 20.8421402, -31.0634232, 31.0747070
17: -57.8135834, -14.3918943, -57.7970428, -14.3908377, -41.5460968, 41.6287537
18: -21.7231846, 12.1550388, -21.5634308, 12.1138973, -29.6689682, 29.5339432
19: -22.3851318, 3.5432329, -22.2962303, 3.5224550, -22.8109512, 22.7343979
20: -23.4018250, 1.3474107, -23.3037910, 1.3182073, -19.2522964, 19.1669769
21: -26.9171753, 2.3757572, -26.8164673, 2.3441401, -25.5389709, 25.4712906
22: -28.6633472, 3.3073809, -28.5439987, 3.2855253, -24.8187637, 24.7003860
23: -22.4087791, 5.6895695, -22.3195496, 5.6784406, -22.0937576, 21.9991608
24: -18.4698982, 9.4204254, -18.3386059, 9.3931618, -22.9386749, 22.8186455
25: -23.9661140, 5.3685150, -23.8466778, 5.3292503, -24.4701233, 24.3785782
26: -41.1882591, -0.4968157, -41.0788040, -0.5184579, -30.6970062, 30.5730057
27: -21.7191811, 8.5611830, -21.6250114, 8.5454617, -26.5351868, 26.4399414
28: -24.2558022, 6.0373020, -24.1496048, 6.0089235, -22.0444756, 21.9454918
29: -27.9884300, -0.2217860, -27.8977795, -0.2257998, -24.0747299, 23.9702682
30: -28.2526283, 3.7332819, -28.1459045, 3.6942296, -26.1789856, 26.1141968
31: -22.7823677, 5.0367870, -22.6628151, 4.9989557, -25.1080017, 25.0210571
32: -23.9577770, 2.3478413, -23.9180832, 2.2601039, -21.3241310, 21.3767357
33: -36.4527512, 3.6238065, -36.3753510, 3.5155120, -33.2470169, 33.2896957
34: -37.8937149, -4.7917171, -37.8343048, -4.8494682, -27.6984406, 27.6958618
35: -32.9434433, 0.2689242, -32.8889885, 0.1836462, -28.0835724, 28.1166992
36: -36.8630524, -0.6978121, -36.8266907, -0.7605133, -28.9596786, 28.9824371
37: -44.6039810, -1.7365370, -44.5438232, -1.7857480, -38.7410736, 38.7483673
38: -43.9954758, 2.8417602, -43.9137383, 2.7413044, -40.6307678, 40.6292419
39: -43.6226578, 3.0315104, -43.5149384, 2.8760262, -41.2534332, 41.3178940
40: -32.7739182, 0.0614955, -32.7002792, -0.0426002, -31.0102081, 31.0478516
41: -20.7352238, 7.2747049, -20.7208309, 7.2272072, -26.4366989, 26.4475632
42: -22.9861679, -0.1903675, -22.9738960, -0.2281828, -18.4300728, 18.4640732

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5472419, upper bound: 11.5075972
time: 35.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5472419, upper bound: 11.5187029
time: 33.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.2985821, 19.0550957, -9.2810564, 19.0120678, -25.1581383, 25.1517982
1: -1.1693192, 22.8372631, -1.1659107, 22.7992363, -19.6949348, 19.6980629
2: -1.5829828, 20.9643745, -1.5724139, 20.9111671, -17.2175751, 17.2500877
3: -9.2998543, 16.5396500, -9.3033323, 16.4689350, -21.9395676, 22.0043488
4: -3.0820456, 22.2534389, -3.0837603, 22.2209854, -21.6564713, 21.6844826
5: -7.7894573, 20.6595135, -7.7933750, 20.5961246, -23.6812897, 23.7291260
6: -28.8094368, -1.3820839, -28.8134995, -1.4044199, -23.1373291, 23.1651688
7: -7.6418381, 21.6628571, -7.6446776, 21.6225204, -23.5171394, 23.5489845
8: -14.7230463, 14.7988625, -14.7206345, 14.7459068, -26.4121780, 26.4410553
9: -5.1550279, 21.3029709, -5.1607914, 21.2745094, -24.2068405, 24.2357140
10: -17.8542538, 17.5710106, -17.8491268, 17.5489311, -31.2383881, 31.2516403
11: -26.8244400, 3.5506926, -26.7432346, 3.5375156, -27.9043274, 27.8327026
12: -34.8677216, -2.3315024, -34.8611374, -2.3682065, -27.1715698, 27.2020493
13: -26.2013206, 15.8247986, -26.1946507, 15.6865158, -33.8028107, 33.9245453
14: -55.8764534, -17.5600586, -55.8619232, -17.5699921, -37.7361603, 37.6916809
15: -14.3514748, 15.5193253, -14.3350344, 15.5000582, -27.8737488, 27.8594589
16: -14.0266914, 20.8317432, -14.0102606, 20.8038330, -31.0037994, 31.0256500
17: -57.8317719, -14.4023495, -57.8072510, -14.4434347, -41.5963593, 41.5922089
18: -21.6755543, 12.1311798, -21.5619240, 12.1198950, -29.6257477, 29.5220413
19: -22.3538876, 3.5262339, -22.2604694, 3.5248418, -22.7754517, 22.6916389
20: -23.3720779, 1.3397045, -23.2757931, 1.3250184, -19.1855774, 19.1361465
21: -26.8919506, 2.3624873, -26.7780323, 2.3506908, -25.5129929, 25.4251289
22: -28.6098862, 3.2780457, -28.4823017, 3.2721732, -24.7207336, 24.6209602
23: -22.3676643, 5.6584582, -22.2726002, 5.6589518, -22.0147133, 21.9298973
24: -18.4221458, 9.4000206, -18.2891502, 9.3960953, -22.8626175, 22.7557068
25: -23.9207363, 5.3402729, -23.8072472, 5.3336716, -24.4072227, 24.3149796
26: -41.1253929, -0.5299230, -41.0171738, -0.5340719, -30.5776901, 30.4990768
27: -21.6799202, 8.5310287, -21.5577183, 8.5243568, -26.4698257, 26.3560791
28: -24.2070160, 6.0062914, -24.1033649, 6.0085592, -21.9761124, 21.8823166
29: -27.9439793, -0.2608554, -27.8269176, -0.2573988, -24.0022659, 23.8779297
30: -28.2270813, 3.7180386, -28.1039467, 3.6978600, -26.1330490, 26.0522156
31: -22.7492561, 5.0158672, -22.6505966, 5.0125952, -25.0684738, 24.9911308
32: -23.9362335, 2.3408840, -23.9304333, 2.3053489, -21.3310776, 21.3795204
33: -36.4389725, 3.6649399, -36.4197426, 3.6392817, -33.2960815, 33.3721390
34: -37.8864365, -4.7593002, -37.8559189, -4.7808790, -27.7069168, 27.7492371
35: -32.9330940, 0.3067102, -32.9176559, 0.2856851, -28.1385727, 28.1807861
36: -36.8560486, -0.6649508, -36.8370209, -0.6845207, -28.9974060, 29.0185623
37: -44.5893326, -1.7089877, -44.5480690, -1.7251863, -38.7876129, 38.7693558
38: -43.9810753, 2.8782792, -43.9516678, 2.8490524, -40.6634674, 40.6839905
39: -43.5921021, 3.0631447, -43.5735970, 3.0055671, -41.2960815, 41.3999863
40: -32.7436905, 0.0464218, -32.7133369, -0.0165234, -31.0131836, 31.0510483
41: -20.7283688, 7.2840366, -20.7185402, 7.2635064, -26.4600220, 26.4872131
42: -22.9826736, -0.1988482, -22.9800148, -0.2217619, -18.4428749, 18.4622040

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1491
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 974
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5246966, upper bound: 11.5104733
time: 32.50 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5246966, upper bound: 11.5216577
time: 38.47 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2985821, 19.0550957, -9.3499727, 19.0535736, -25.2015762, 25.2231407
1: -1.1693192, 22.8372631, -1.2157726, 22.8421459, -19.7409554, 19.7431984
2: -1.5829828, 20.9643745, -1.6136892, 20.9458427, -17.2560883, 17.2885170
3: -9.2998543, 16.5396500, -9.3341990, 16.5206566, -21.9878235, 22.0339508
4: -3.0820456, 22.2534389, -3.1357894, 22.2448101, -21.6741028, 21.7318077
5: -7.7894573, 20.6595135, -7.8342547, 20.6485443, -23.7337990, 23.7650299
6: -28.8094368, -1.3820839, -28.8260345, -1.3824897, -23.1646271, 23.1802902
7: -7.6418381, 21.6628571, -7.6932459, 21.6622810, -23.5583267, 23.5940018
8: -14.7230463, 14.7988625, -14.7764797, 14.7880993, -26.4470177, 26.4872589
9: -5.1550279, 21.3029709, -5.1895580, 21.3010750, -24.2410431, 24.2675133
10: -17.8542538, 17.5710106, -17.8826981, 17.5881214, -31.2805786, 31.2897263
11: -26.8244400, 3.5506926, -26.8083553, 3.5712209, -27.9365463, 27.9007416
12: -34.8677216, -2.3315024, -34.8879585, -2.3196335, -27.2212372, 27.2301407
13: -26.2013206, 15.8247986, -26.2489166, 15.7780437, -33.8963928, 33.9813385
14: -55.8764534, -17.5600586, -55.9280930, -17.4924355, -37.8140106, 37.7566223
15: -14.3514748, 15.5193253, -14.3620424, 15.5137939, -27.8876190, 27.8896103
16: -14.0266914, 20.8317432, -14.0601139, 20.8603516, -31.0613556, 31.0719070
17: -57.8317719, -14.4023495, -57.8666534, -14.3470964, -41.6961670, 41.6576767
18: -21.6755543, 12.1311798, -21.6133041, 12.1530151, -29.6623001, 29.5761147
19: -22.3538876, 3.5262339, -22.3449574, 3.5729671, -22.8245392, 22.7768860
20: -23.3720779, 1.3397045, -23.3565292, 1.3770134, -19.2426453, 19.2193413
21: -26.8919506, 2.3624873, -26.8689880, 2.4012463, -25.5645752, 25.5177193
22: -28.6098862, 3.2780457, -28.5890121, 3.3226502, -24.7757301, 24.7317352
23: -22.3676643, 5.6584582, -22.3504753, 5.7124805, -22.0699463, 22.0101814
24: -18.4221458, 9.4000206, -18.3902950, 9.4415131, -22.9099960, 22.8585434
25: -23.9207363, 5.3402729, -23.8752003, 5.3826580, -24.4604340, 24.3856354
26: -41.1253929, -0.5299230, -41.1203079, -0.4668851, -30.6498108, 30.6041260
27: -21.6799202, 8.5310287, -21.6564884, 8.5761013, -26.5239944, 26.4570618
28: -24.2070160, 6.0062914, -24.1874580, 6.0647078, -22.0363464, 21.9689026
29: -27.9439793, -0.2608554, -27.9301224, -0.2192006, -24.0448608, 23.9831009
30: -28.2270813, 3.7180386, -28.1825066, 3.7503300, -26.1864090, 26.1321564
31: -22.7492561, 5.0158672, -22.7155113, 5.0526280, -25.1108780, 25.0584755
32: -23.9362335, 2.3408840, -23.9522247, 2.3211098, -21.3482704, 21.4094620
33: -36.4389725, 3.6649399, -36.4643402, 3.6590195, -33.3159332, 33.4158401
34: -37.8864365, -4.7593002, -37.8982697, -4.7362905, -27.7551956, 27.7955704
35: -32.9330940, 0.3067102, -32.9596863, 0.3154778, -28.1697388, 28.2236633
36: -36.8560486, -0.6649508, -36.8910027, -0.6386991, -29.0456085, 29.0742340
37: -44.5893326, -1.7089877, -44.6151428, -1.6987367, -38.8161621, 38.8379898
38: -43.9810753, 2.8782792, -44.0063095, 2.8976250, -40.7206421, 40.7458267
39: -43.5921021, 3.0631447, -43.6148758, 3.0229349, -41.3150330, 41.4512711
40: -32.7436905, 0.0464218, -32.7507782, 0.0035167, -31.0338745, 31.0887604
41: -20.7283688, 7.2840366, -20.7593536, 7.2898011, -26.4931946, 26.5335007
42: -22.9826736, -0.1988482, -22.9884033, -0.2070537, -18.4581833, 18.4721146

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1491
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 974
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5246966, upper bound: 11.5281074
time: 35.52 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5246966, upper bound: 11.5393035
time: 32.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.3595324, 19.0661354, -9.2972431, 19.0124722, -25.2236519, 25.1834869
1: -1.2125177, 22.8602600, -1.1786184, 22.7994595, -19.7330322, 19.7334251
2: -1.6192417, 20.9844704, -1.5830419, 20.9114342, -17.2477188, 17.2830925
3: -9.3449879, 16.5741005, -9.3168554, 16.4698257, -21.9742966, 22.0419731
4: -3.1425781, 22.2726593, -3.1011004, 22.2210674, -21.7072525, 21.7176437
5: -7.8325787, 20.6966915, -7.8060560, 20.5972805, -23.7136383, 23.7727127
6: -28.8289471, -1.3498068, -28.8194466, -1.4006195, -23.1542130, 23.2003136
7: -7.6954312, 21.7022781, -7.6612291, 21.6230545, -23.5595398, 23.6042862
8: -14.7834930, 14.8231163, -14.7381792, 14.7464342, -26.4648361, 26.4708405
9: -5.1968937, 21.3255615, -5.1711502, 21.2748108, -24.2514572, 24.2767715
10: -17.8829460, 17.5873089, -17.8559742, 17.5499096, -31.2654877, 31.2814178
11: -26.8305492, 3.5683737, -26.7434845, 3.5415497, -27.9199524, 27.8545761
12: -34.8784142, -2.3084846, -34.8638649, -2.3632946, -27.1859055, 27.2356033
13: -26.2516479, 15.8743744, -26.2088547, 15.6877174, -33.8480453, 33.9920349
14: -55.8941040, -17.5472584, -55.8644562, -17.5670204, -37.7540359, 37.7329788
15: -14.3806839, 15.5308771, -14.3401346, 15.5031605, -27.9064102, 27.8734055
16: -14.0630322, 20.8712502, -14.0196209, 20.8040981, -31.0324326, 31.0753021
17: -57.8362236, -14.3847752, -57.8082047, -14.4407253, -41.5986786, 41.6485062
18: -21.7290573, 12.1720676, -21.5630608, 12.1316071, -29.6952515, 29.5565262
19: -22.3902206, 3.5641947, -22.2613811, 3.5363505, -22.8231964, 22.7205963
20: -23.4041634, 1.3699365, -23.2762394, 1.3338099, -19.2298508, 19.1620674
21: -26.9228420, 2.3983579, -26.7786331, 2.3612206, -25.5512772, 25.4541702
22: -28.6665421, 3.3240008, -28.4829712, 3.2863505, -24.7924881, 24.6531906
23: -22.4114342, 5.7051616, -22.2730618, 5.6728845, -22.0743179, 21.9647636
24: -18.4722443, 9.4413891, -18.2901993, 9.4086437, -22.9277191, 22.7894745
25: -23.9687710, 5.3899689, -23.8082657, 5.3483276, -24.4720001, 24.3545685
26: -41.1908493, -0.4732218, -41.0180817, -0.5167603, -30.6663818, 30.5379791
27: -21.7231369, 8.5759296, -21.5585709, 8.5373402, -26.5268860, 26.3956451
28: -24.2579498, 6.0611596, -24.1038818, 6.0253735, -22.0456047, 21.9214821
29: -27.9933796, -0.2164330, -27.8272324, -0.2443826, -24.0646591, 23.9107971
30: -28.2546272, 3.7541597, -28.1043472, 3.7073867, -26.1692886, 26.0805321
31: -22.7876472, 5.0589952, -22.6519165, 5.0252075, -25.1213608, 25.0316048
32: -23.9606323, 2.3689408, -23.9366207, 2.3083467, -21.3528404, 21.4127808
33: -36.4570236, 3.6738086, -36.4225540, 3.6404233, -33.3152924, 33.3878326
34: -37.8957367, -4.7530708, -37.8565636, -4.7801533, -27.7200241, 27.7560120
35: -32.9464455, 0.3149195, -32.9191704, 0.2872791, -28.1556854, 28.1911926
36: -36.8660507, -0.6554322, -36.8381195, -0.6821532, -29.0149155, 29.0335693
37: -44.6106339, -1.7069130, -44.5513306, -1.7253947, -38.8055573, 38.7889099
38: -44.0013809, 2.8968053, -43.9538040, 2.8531251, -40.7011871, 40.7143631
39: -43.6293182, 3.0823965, -43.5817261, 3.0056434, -41.3291626, 41.4240265
40: -32.7804489, 0.0773182, -32.7206726, -0.0150945, -31.0462418, 31.0888863
41: -20.7394886, 7.2960758, -20.7211647, 7.2654724, -26.4723663, 26.5012207
42: -22.9911461, -0.1838865, -22.9823990, -0.2196474, -18.4510002, 18.4806519

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1491
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5478627, upper bound: 11.5115618
time: 31.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5478627, upper bound: 11.5227922
time: 42.92 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.3595324, 19.0661354, -9.3661451, 19.0539837, -25.2670937, 25.2548523
1: -1.2125177, 22.8602600, -1.2284594, 22.8423615, -19.7790565, 19.7785645
2: -1.6192417, 20.9844704, -1.6243134, 20.9461098, -17.2862244, 17.3215179
3: -9.3449879, 16.5741005, -9.3476982, 16.5215034, -22.0225601, 22.0716324
4: -3.1425781, 22.2726593, -3.1531730, 22.2448864, -21.7248878, 21.7649918
5: -7.8325787, 20.6966915, -7.8469491, 20.6496735, -23.7661591, 23.8086624
6: -28.8289471, -1.3498068, -28.8319473, -1.3786068, -23.1815033, 23.2154465
7: -7.6954312, 21.7022781, -7.7098126, 21.6627731, -23.6007233, 23.6493149
8: -14.7834930, 14.8231163, -14.7940149, 14.7886324, -26.4996719, 26.5170212
9: -5.1968937, 21.3255615, -5.1999245, 21.3013954, -24.2856140, 24.3085861
10: -17.8829460, 17.5873089, -17.8895683, 17.5890942, -31.3076324, 31.3194885
11: -26.8305492, 3.5683737, -26.8085823, 3.5752478, -27.9521561, 27.9225922
12: -34.8784142, -2.3084846, -34.8907166, -2.3147116, -27.2355347, 27.2637024
13: -26.2516479, 15.8743744, -26.2630768, 15.7791328, -33.9415970, 34.0488281
14: -55.8941040, -17.5472584, -55.9306488, -17.4894848, -37.8319092, 37.7979202
15: -14.3806839, 15.5308771, -14.3671227, 15.5168886, -27.9202805, 27.9035416
16: -14.0630322, 20.8712502, -14.0695057, 20.8605576, -31.0900345, 31.1215668
17: -57.8362236, -14.3847752, -57.8676338, -14.3443508, -41.6984711, 41.7140350
18: -21.7290573, 12.1720676, -21.6144257, 12.1647644, -29.7318115, 29.6105843
19: -22.3902206, 3.5641947, -22.3458481, 3.5845070, -22.8722687, 22.8058357
20: -23.4041634, 1.3699365, -23.3569984, 1.3858211, -19.2868958, 19.2452660
21: -26.9228420, 2.3983579, -26.8695641, 2.4117267, -25.6028595, 25.5467567
22: -28.6665421, 3.3240008, -28.5896282, 3.3368297, -24.8474770, 24.7639618
23: -22.4114342, 5.7051616, -22.3509026, 5.7263994, -22.1295471, 22.0450668
24: -18.4722443, 9.4413891, -18.3913555, 9.4540167, -22.9750977, 22.8922768
25: -23.9687710, 5.3899689, -23.8762169, 5.3973355, -24.5252151, 24.4252090
26: -41.1908493, -0.4732218, -41.1212845, -0.4495101, -30.7384872, 30.6430283
27: -21.7231369, 8.5759296, -21.6573486, 8.5890217, -26.5810852, 26.4966583
28: -24.2579498, 6.0611596, -24.1879902, 6.0814323, -22.1058197, 22.0080605
29: -27.9933796, -0.2164330, -27.9304733, -0.2061455, -24.1072617, 24.0159302
30: -28.2546272, 3.7541597, -28.1828918, 3.7598248, -26.2226791, 26.1604767
31: -22.7876472, 5.0589952, -22.7168255, 5.0652385, -25.1637573, 25.0989456
32: -23.9606323, 2.3689408, -23.9584217, 2.3241239, -21.3700104, 21.4427109
33: -36.4570236, 3.6738086, -36.4671631, 3.6601324, -33.3351593, 33.4315262
34: -37.8957367, -4.7530708, -37.8989143, -4.7356005, -27.7682953, 27.8023224
35: -32.9464455, 0.3149195, -32.9611626, 0.3170404, -28.1869049, 28.2340546
36: -36.8660507, -0.6554322, -36.8920975, -0.6363335, -29.0630722, 29.0892715
37: -44.6106339, -1.7069130, -44.6183472, -1.6990042, -38.8341675, 38.8574982
38: -44.0013809, 2.8968053, -44.0084534, 2.9017086, -40.7583923, 40.7762299
39: -43.6293182, 3.0823965, -43.6229935, 3.0230980, -41.3481140, 41.4752960
40: -32.7804489, 0.0773182, -32.7581367, 0.0049033, -31.0669327, 31.1266060
41: -20.7394886, 7.2960758, -20.7619228, 7.2917523, -26.5055084, 26.5475082
42: -22.9911461, -0.1838865, -22.9907856, -0.2049215, -18.4663162, 18.4905548

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1491
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5478627, upper bound: 11.5291791
time: 40.21 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5478627, upper bound: 11.5404359
time: 33.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.1555233, 18.9755402, -9.2482843, 19.0531006, -25.0613861, 25.0643921
1: -1.0810199, 22.7634315, -1.1346254, 22.8351288, -19.6402817, 19.6215706
2: -1.5299733, 20.8787003, -1.5602996, 20.9585667, -17.1960678, 17.1600876
3: -9.2652121, 16.4281445, -9.2782087, 16.5319977, -21.9488869, 21.8572273
4: -3.0140719, 22.2070084, -3.0508327, 22.2504311, -21.6047745, 21.6103249
5: -7.7143126, 20.5452461, -7.7566485, 20.6516190, -23.6451416, 23.6056519
6: -28.7290688, -1.4848466, -28.7912121, -1.4105940, -23.0504913, 23.0816460
7: -7.5699267, 21.5884647, -7.6102848, 21.6589394, -23.4674492, 23.4465065
8: -14.6228886, 14.6727352, -14.6785984, 14.7897091, -26.3522720, 26.2956238
9: -5.0845995, 21.2090912, -5.1260071, 21.2886028, -24.1838684, 24.1093445
10: -17.7471333, 17.3612518, -17.8177681, 17.5191879, -31.1017609, 31.0246124
11: -26.7114315, 3.4854312, -26.8156605, 3.5279856, -27.7777863, 27.8397369
12: -34.8437424, -2.4407721, -34.8608093, -2.3562799, -27.1547012, 27.0688896
13: -26.1956978, 15.6261559, -26.1912193, 15.8073311, -33.9081497, 33.7163620
14: -55.7562408, -17.7342567, -55.8423195, -17.6027393, -37.5882416, 37.4226990
15: -14.2317944, 15.4457216, -14.3124599, 15.5130787, -27.7672577, 27.7750092
16: -13.9420004, 20.7455215, -14.0013914, 20.8183651, -30.9532089, 30.9393616
17: -57.7268143, -14.5296764, -57.8053436, -14.4237347, -41.5117722, 41.3850708
18: -21.4953480, 12.0927992, -21.6638908, 12.1103210, -29.4102173, 29.5773010
19: -22.1978569, 3.4786320, -22.3435783, 3.5025468, -22.6015625, 22.7198601
20: -23.2143459, 1.2707748, -23.3666039, 1.3118763, -19.0369415, 19.1572037
21: -26.7099609, 2.2997236, -26.8805275, 2.3348038, -25.3300781, 25.4580383
22: -28.4154129, 3.2444489, -28.5992088, 3.2585773, -24.5307159, 24.6916504
23: -22.2351742, 5.6184068, -22.3623009, 5.6368251, -21.8729439, 21.9786224
24: -18.2186165, 9.3601055, -18.4132118, 9.3775997, -22.6565552, 22.8397293
25: -23.7651634, 5.2875462, -23.9133415, 5.3150644, -24.2434654, 24.3530197
26: -40.9650650, -0.6036148, -41.1192932, -0.5665240, -30.4060745, 30.5353546
27: -21.4733315, 8.5033512, -21.6591110, 8.5124321, -26.2403259, 26.4318542
28: -24.0460625, 5.9615297, -24.1985435, 5.9797244, -21.7951965, 21.9297218
29: -27.7723026, -0.2599726, -27.9312191, -0.2680190, -23.8020706, 23.9524918
30: -28.0553474, 3.6518497, -28.2207870, 3.6914256, -25.9803925, 26.0945816
31: -22.5835304, 4.9565954, -22.7386684, 4.9880657, -24.8947525, 25.0191879
32: -23.8579655, 2.2381995, -23.9218636, 2.3172789, -21.2773666, 21.2701836
33: -36.2689590, 3.4918923, -36.4146233, 3.6133075, -33.1688232, 33.1830292
34: -37.7311058, -4.8969851, -37.8650246, -4.8022985, -27.5853195, 27.6150208
35: -32.7712860, 0.1503525, -32.9065895, 0.2587810, -27.9866562, 28.0096664
36: -36.6773148, -0.8084855, -36.8238754, -0.7094674, -28.8168793, 28.8658447
37: -44.3712997, -1.8151970, -44.5482254, -1.7402587, -38.5630646, 38.6543045
38: -43.7495041, 2.6833291, -43.9415512, 2.8187709, -40.4246674, 40.5013428
39: -43.4205284, 2.8519344, -43.5659943, 3.0097675, -41.1934814, 41.1875763
40: -32.6162186, -0.0654266, -32.7196198, 0.0293825, -30.9317017, 30.9401932
41: -20.6089745, 7.1951685, -20.7013206, 7.2604637, -26.3156357, 26.3740692
42: -22.9600487, -0.2514954, -22.9751282, -0.2084241, -18.4297333, 18.3967781

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
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1526
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
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5237622
time: 54.20 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5237622
time: 31.33 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.1555233, 18.9755402, -9.3170700, 19.0945950, -25.1038628, 25.1348419
1: -1.0810199, 22.7634315, -1.1844749, 22.8780022, -19.6853371, 19.6664200
2: -1.5299733, 20.8787003, -1.6015260, 20.9932041, -17.2336578, 17.1981049
3: -9.2652121, 16.4281445, -9.3090267, 16.5838737, -21.9970055, 21.8868752
4: -3.0140719, 22.2070084, -3.1027446, 22.2742558, -21.6224594, 21.6571350
5: -7.7143126, 20.5452461, -7.7974877, 20.7041569, -23.6973114, 23.6414185
6: -28.7290688, -1.4848466, -28.8036537, -1.3889771, -23.0772705, 23.0967216
7: -7.5699267, 21.5884647, -7.6588154, 21.6986313, -23.5080185, 23.4912071
8: -14.6228886, 14.6727352, -14.7343521, 14.8318300, -26.3857956, 26.3411331
9: -5.0845995, 21.2090912, -5.1547151, 21.3148193, -24.2180634, 24.1409950
10: -17.7471333, 17.3612518, -17.8509140, 17.5581245, -31.1439438, 31.0626984
11: -26.7114315, 3.4854312, -26.8808517, 3.5616045, -27.8097229, 27.9078522
12: -34.8437424, -2.4407721, -34.8873672, -2.3076611, -27.2041626, 27.0967903
13: -26.1956978, 15.6261559, -26.2453899, 15.8986912, -33.9995117, 33.7723083
14: -55.7562408, -17.7342567, -55.9085159, -17.5250053, -37.6662674, 37.4875793
15: -14.2317944, 15.4457216, -14.3392792, 15.5267830, -27.7811203, 27.8048706
16: -13.9420004, 20.7455215, -14.0510378, 20.8743553, -31.0108109, 30.9855194
17: -57.7268143, -14.5296764, -57.8648682, -14.3272934, -41.6116028, 41.4505539
18: -21.4953480, 12.0927992, -21.7150631, 12.1433802, -29.4460907, 29.6299667
19: -22.1978569, 3.4786320, -22.4280567, 3.5506682, -22.6504745, 22.8047714
20: -23.2143459, 1.2707748, -23.4474335, 1.3638151, -19.0933876, 19.2383461
21: -26.7099609, 2.2997236, -26.9715137, 2.3852773, -25.3807449, 25.5495872
22: -28.4154129, 3.2444489, -28.7058868, 3.3090880, -24.5856895, 24.8002930
23: -22.2351742, 5.6184068, -22.4401321, 5.6902452, -21.9278679, 22.0571289
24: -18.2186165, 9.3601055, -18.5143890, 9.4229546, -22.7032700, 22.9403534
25: -23.7651634, 5.2875462, -23.9807739, 5.3639803, -24.2961349, 24.4226761
26: -40.9650650, -0.6036148, -41.2223892, -0.4992943, -30.4781113, 30.6379089
27: -21.4733315, 8.5033512, -21.7578888, 8.5641308, -26.2938004, 26.5311127
28: -24.0460625, 5.9615297, -24.2826080, 6.0357170, -21.8549805, 22.0141525
29: -27.7723026, -0.2599726, -28.0345459, -0.2298390, -23.8443680, 24.0561485
30: -28.0553474, 3.6518497, -28.2993507, 3.7437413, -26.0332909, 26.1722374
31: -22.5835304, 4.9565954, -22.8035316, 5.0280085, -24.9370880, 25.0861588
32: -23.8579655, 2.2381995, -23.9434986, 2.3330011, -21.2944489, 21.3001823
33: -36.2689590, 3.4918923, -36.4593620, 3.6327629, -33.1889038, 33.2269135
34: -37.7311058, -4.8969851, -37.9072723, -4.7578764, -27.6334381, 27.6605835
35: -32.7712860, 0.1503525, -32.9484901, 0.2880969, -28.0173264, 28.0524750
36: -36.6773148, -0.8084855, -36.8778534, -0.6640058, -28.8647461, 28.9212265
37: -44.3712997, -1.8151970, -44.6151505, -1.7141180, -38.5912323, 38.7230759
38: -43.7495041, 2.6833291, -43.9961967, 2.8664665, -40.4813232, 40.5631638
39: -43.4205284, 2.8519344, -43.6067352, 3.0271969, -41.2120972, 41.2392120
40: -32.6162186, -0.0654266, -32.7567406, 0.0492375, -30.9522858, 30.9777832
41: -20.6089745, 7.1951685, -20.7421112, 7.2857409, -26.3484192, 26.4204178
42: -22.9600487, -0.2514954, -22.9835205, -0.1938441, -18.4446106, 18.4066734

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
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1526
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
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5397775
time: 39.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5397775
time: 34.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.2098713, 18.9986382, -9.2528172, 19.0535927, -25.1200409, 25.0984840
1: -1.1135478, 22.7945862, -1.1356010, 22.8357811, -19.6750183, 19.6656799
2: -1.5817018, 20.8936348, -1.5626671, 20.9618683, -17.2246552, 17.2178650
3: -9.2895918, 16.4431057, -9.2797060, 16.5329704, -21.9690170, 21.8971367
4: -3.0604315, 22.2147942, -3.0537410, 22.2514515, -21.6486435, 21.6622925
5: -7.7638421, 20.5629578, -7.7601471, 20.6545124, -23.6805878, 23.6372643
6: -28.7816544, -1.4066167, -28.8038902, -1.4086704, -23.0914383, 23.1748009
7: -7.6159267, 21.6037865, -7.6138115, 21.6611328, -23.5119705, 23.4857025
8: -14.6998444, 14.7028732, -14.6796436, 14.7945900, -26.4184036, 26.3603668
9: -5.1704626, 21.2557278, -5.1280222, 21.2982616, -24.2805328, 24.1546669
10: -17.9679832, 17.5239964, -17.8202133, 17.5617599, -31.3623962, 31.1675415
11: -26.7892876, 3.5119934, -26.8179893, 3.5313730, -27.8695679, 27.8691940
12: -34.8696938, -2.4018121, -34.8623466, -2.3509860, -27.2132492, 27.1038818
13: -26.2295284, 15.6600323, -26.1953316, 15.8104143, -33.9496994, 33.7534409
14: -55.9235229, -17.6081791, -55.8465347, -17.5698051, -37.7890549, 37.5302582
15: -14.2905197, 15.4627104, -14.3161087, 15.5159369, -27.8428879, 27.7962112
16: -14.0397263, 20.7886772, -14.0051851, 20.8279781, -31.0542068, 30.9824753
17: -57.8374481, -14.4731941, -57.8084526, -14.4128647, -41.6446609, 41.4500427
18: -21.5487537, 12.1296282, -21.6689396, 12.1120844, -29.4632568, 29.6585846
19: -22.2391891, 3.4912229, -22.3474960, 3.5035760, -22.6562195, 22.7371063
20: -23.2407799, 1.2846255, -23.3691883, 1.3130031, -19.0724335, 19.1741104
21: -26.7698421, 2.3200853, -26.8850327, 2.3367233, -25.4038391, 25.4836884
22: -28.4479961, 3.2818491, -28.6052971, 3.2602880, -24.6239548, 24.7010269
23: -22.2687435, 5.6355677, -22.3642540, 5.6377187, -21.9087067, 22.0033684
24: -18.2465229, 9.3702164, -18.4183292, 9.3784132, -22.6847458, 22.8578491
25: -23.7959290, 5.3152905, -23.9170551, 5.3171043, -24.3156891, 24.3812714
26: -41.0146790, -0.5630426, -41.1223755, -0.5584888, -30.4688568, 30.5734406
27: -21.5358276, 8.5811806, -21.6739464, 8.5148678, -26.2954483, 26.5271912
28: -24.0706501, 6.0030813, -24.2028351, 5.9818974, -21.8280678, 21.9619865
29: -27.8130894, -0.2236538, -27.9375229, -0.2667246, -23.9136467, 23.9520493
30: -28.0823593, 3.6832268, -28.2240753, 3.6944411, -26.0340538, 26.1327133
31: -22.6463585, 4.9746780, -22.7426853, 4.9898243, -24.9633179, 25.0411797
32: -23.8971214, 2.2884605, -23.9314880, 2.3192720, -21.3133316, 21.3340569
33: -36.3407669, 3.5954566, -36.4312325, 3.6144800, -33.2387772, 33.3171997
34: -37.7930603, -4.7918706, -37.8819275, -4.7995057, -27.6436005, 27.7467728
35: -32.8512268, 0.2724652, -32.9271317, 0.2599797, -28.0584183, 28.1547012
36: -36.7753830, -0.6751909, -36.8497200, -0.7079806, -28.8997498, 29.0259552
37: -44.4898758, -1.6934419, -44.5773544, -1.7395611, -38.6673431, 38.8219833
38: -43.8736267, 2.8533735, -43.9712639, 2.8216314, -40.5383911, 40.6999893
39: -43.4857140, 2.9236293, -43.5795364, 3.0116329, -41.2620697, 41.2865067
40: -32.6834564, 0.0237596, -32.7340698, 0.0301716, -30.9989777, 31.0487175
41: -20.6853828, 7.2924652, -20.7208099, 7.2622232, -26.3834457, 26.4948349
42: -22.9671974, -0.2236602, -22.9769325, -0.2060535, -18.4619560, 18.4281883

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5348720
time: 25.20 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5348720
time: 39.22 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.2098713, 18.9986382, -9.3215742, 19.0951080, -25.1625099, 25.1689377
1: -1.1135478, 22.7945862, -1.1854401, 22.8786945, -19.7200623, 19.7105446
2: -1.5817018, 20.8936348, -1.6039054, 20.9965172, -17.2622490, 17.2558823
3: -9.2895918, 16.4431057, -9.3105402, 16.5848331, -22.0171356, 21.9267845
4: -3.0604315, 22.2147942, -3.1056104, 22.2753010, -21.6662979, 21.7091217
5: -7.7638421, 20.5629578, -7.8009577, 20.7070351, -23.7327881, 23.6730194
6: -28.7816544, -1.4066167, -28.8163528, -1.3870554, -23.1182327, 23.1898689
7: -7.6159267, 21.6037865, -7.6623416, 21.7008343, -23.5525398, 23.5303993
8: -14.6998444, 14.7028732, -14.7354298, 14.8367195, -26.4519386, 26.4058914
9: -5.1704626, 21.2557278, -5.1567173, 21.3244648, -24.3147049, 24.1863251
10: -17.9679832, 17.5239964, -17.8533745, 17.6006756, -31.4046097, 31.2056274
11: -26.7892876, 3.5119934, -26.8831825, 3.5649943, -27.9014893, 27.9372864
12: -34.8696938, -2.4018121, -34.8889427, -2.3024049, -27.2627335, 27.1317673
13: -26.2295284, 15.6600323, -26.2495308, 15.9017401, -34.0410461, 33.8094177
14: -55.9235229, -17.6081791, -55.9126740, -17.4919758, -37.8670197, 37.5951309
15: -14.2905197, 15.4627104, -14.3429317, 15.5296316, -27.8567276, 27.8260803
16: -14.0397263, 20.7886772, -14.0548992, 20.8839302, -31.1118088, 31.0286560
17: -57.8374481, -14.4731941, -57.8679733, -14.3163385, -41.7445221, 41.5155640
18: -21.5487537, 12.1296282, -21.7201443, 12.1451759, -29.4991074, 29.7112579
19: -22.2391891, 3.4912229, -22.4319763, 3.5517080, -22.7051163, 22.8220177
20: -23.2407799, 1.2846255, -23.4499893, 1.3650100, -19.1288834, 19.2552643
21: -26.7698421, 2.3200853, -26.9760036, 2.3872259, -25.4545059, 25.5752335
22: -28.4479961, 3.2818491, -28.7119865, 3.3107650, -24.6789398, 24.8096619
23: -22.2687435, 5.6355677, -22.4420643, 5.6911035, -21.9636230, 22.0818901
24: -18.2465229, 9.3702164, -18.5195007, 9.4238052, -22.7314606, 22.9584923
25: -23.7959290, 5.3152905, -23.9845295, 5.3660269, -24.3683624, 24.4509277
26: -41.0146790, -0.5630426, -41.2255211, -0.4912796, -30.5408401, 30.6759872
27: -21.5358276, 8.5811806, -21.7727928, 8.5665665, -26.3489304, 26.6264572
28: -24.0706501, 6.0030813, -24.2869339, 6.0379443, -21.8878326, 22.0464172
29: -27.8130894, -0.2236538, -28.0408363, -0.2285240, -23.9559479, 24.0557022
30: -28.0823593, 3.6832268, -28.3026581, 3.7467504, -26.0869446, 26.2103882
31: -22.6463585, 4.9746780, -22.8075752, 5.0298023, -25.0056534, 25.1081314
32: -23.8971214, 2.2884605, -23.9531746, 2.3349771, -21.3303642, 21.3640671
33: -36.3407669, 3.5954566, -36.4759521, 3.6339850, -33.2588425, 33.3610764
34: -37.7930603, -4.7918706, -37.9242363, -4.7550364, -27.6916885, 27.7923203
35: -32.8512268, 0.2724652, -32.9690323, 0.2893629, -28.0890808, 28.1974869
36: -36.7753830, -0.6751909, -36.9036789, -0.6625028, -28.9475784, 29.0813141
37: -44.4898758, -1.6934419, -44.6443176, -1.7134171, -38.6954956, 38.8907700
38: -43.8736267, 2.8533735, -44.0258713, 2.8693299, -40.5950775, 40.7618256
39: -43.4857140, 2.9236293, -43.6203041, 3.0290570, -41.2807922, 41.3381805
40: -32.6834564, 0.0237596, -32.7712479, 0.0500078, -31.0195465, 31.0863190
41: -20.6853828, 7.2924652, -20.7615967, 7.2875175, -26.4162521, 26.5411758
42: -22.9671974, -0.2236602, -22.9853077, -0.1914845, -18.4768066, 18.4380684

Time for backsubstitution: 2.34 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5509441
time: 34.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5229372, upper bound: 11.5509441
time: 30.07 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.1716919, 18.9759502, -9.3092051, 19.0641365, -25.0930862, 25.1298676
1: -1.0937209, 22.7636490, -1.1778760, 22.8580894, -19.6756744, 19.6596642
2: -1.5405695, 20.8789330, -1.5965447, 20.9786530, -17.2290916, 17.1902122
3: -9.2787542, 16.4290161, -9.3233757, 16.5664711, -21.9865646, 21.8919601
4: -3.0314469, 22.2070923, -3.1113338, 22.2696476, -21.6379738, 21.6610832
5: -7.7269917, 20.5463829, -7.7998052, 20.6888199, -23.6887550, 23.6380272
6: -28.7349777, -1.4810424, -28.8106956, -1.3783388, -23.0856934, 23.0985451
7: -7.5864744, 21.5889797, -7.6638713, 21.6983471, -23.5227432, 23.4888725
8: -14.6403866, 14.6732578, -14.7390270, 14.8139534, -26.3820801, 26.3482971
9: -5.0949602, 21.2094440, -5.1678467, 21.3111687, -24.2248993, 24.1539497
10: -17.7539997, 17.3622112, -17.8464775, 17.5354843, -31.1314774, 31.0516739
11: -26.7116375, 3.4894867, -26.8218117, 3.5456576, -27.7996292, 27.8553543
12: -34.8464661, -2.4358397, -34.8715019, -2.3331981, -27.1883163, 27.0831947
13: -26.2099285, 15.6272345, -26.2415009, 15.8568516, -33.9756317, 33.7614975
14: -55.7587662, -17.7313194, -55.8599930, -17.5899353, -37.6296158, 37.4405899
15: -14.2368698, 15.4488106, -14.3416405, 15.5246067, -27.7812195, 27.8076553
16: -13.9513807, 20.7457676, -14.0377369, 20.8578453, -31.0028915, 30.9680557
17: -57.7277527, -14.5269127, -57.8098373, -14.4061718, -41.5681000, 41.3873978
18: -21.4964905, 12.1045551, -21.7173958, 12.1512337, -29.4446564, 29.6467896
19: -22.1987228, 3.4901814, -22.3799438, 3.5404828, -22.6305237, 22.7676392
20: -23.2148151, 1.2795639, -23.3986912, 1.3421526, -19.0628815, 19.2014885
21: -26.7106018, 2.3101892, -26.9114037, 2.3706350, -25.3591232, 25.4963531
22: -28.4160385, 3.2586412, -28.6558075, 3.3045859, -24.5629578, 24.7633934
23: -22.2356472, 5.6323566, -22.4060955, 5.6835489, -21.9078522, 22.0382805
24: -18.2196922, 9.3726091, -18.4633408, 9.4189920, -22.6902695, 22.9048195
25: -23.7661476, 5.3021903, -23.9613419, 5.3647642, -24.2830620, 24.4178085
26: -40.9660034, -0.5862594, -41.1846848, -0.5098448, -30.4450150, 30.6240616
27: -21.4742031, 8.5162811, -21.7022934, 8.5573244, -26.2799377, 26.4888840
28: -24.0465908, 5.9782887, -24.2494640, 6.0345621, -21.8343506, 21.9992104
29: -27.7726421, -0.2469485, -27.9806786, -0.2235883, -23.8348999, 24.0149231
30: -28.0557442, 3.6613984, -28.2483444, 3.7275357, -26.0087204, 26.1308441
31: -22.5848827, 4.9691768, -22.7770500, 5.0312061, -24.9352112, 25.0721130
32: -23.8641605, 2.2411690, -23.9462280, 2.3453486, -21.3106842, 21.2919044
33: -36.2718048, 3.4930334, -36.4326935, 3.6221366, -33.1845398, 33.2022018
34: -37.7317734, -4.8962450, -37.8743782, -4.7960672, -27.5920944, 27.6281509
35: -32.7727737, 0.1519384, -32.9200020, 0.2669992, -27.9970779, 28.0267715
36: -36.6784897, -0.8061571, -36.8338127, -0.6999874, -28.8319092, 28.8833389
37: -44.3745041, -1.8154426, -44.5694923, -1.7381134, -38.5826111, 38.6722870
38: -43.7517014, 2.6874976, -43.9617996, 2.8372970, -40.4550476, 40.5390320
39: -43.4286346, 2.8521247, -43.6031761, 3.0290365, -41.2174530, 41.2206421
40: -32.6235733, -0.0639992, -32.7563477, 0.0602980, -30.9695587, 30.9732323
41: -20.6115627, 7.1970987, -20.7124577, 7.2725668, -26.3296738, 26.3864136
42: -22.9624653, -0.2493412, -22.9836082, -0.1934392, -18.4481773, 18.4049149

Time for backsubstitution: 2.27 seconds

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
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1526
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
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5099927, upper bound: 11.5469114
time: 34.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5241337, upper bound: 11.5469114
time: 37.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.1716919, 18.9759502, -9.3779945, 19.1056480, -25.1355591, 25.2002945
1: -1.0937209, 22.7636490, -1.2276945, 22.9009972, -19.7207260, 19.7045059
2: -1.5405695, 20.8789330, -1.6377804, 21.0132942, -17.2666817, 17.2282257
3: -9.2787542, 16.4290161, -9.3542061, 16.6183205, -22.0347137, 21.9216118
4: -3.0314469, 22.2070923, -3.1632843, 22.2934570, -21.6556587, 21.7079048
5: -7.7269917, 20.5463829, -7.8406582, 20.7413368, -23.7409630, 23.6737747
6: -28.7349777, -1.4810424, -28.8231583, -1.3567276, -23.1125031, 23.1136131
7: -7.5864744, 21.5889797, -7.7124538, 21.7380524, -23.5633087, 23.5335655
8: -14.6403866, 14.6732578, -14.7948179, 14.8560867, -26.4156036, 26.3937988
9: -5.0949602, 21.2094440, -5.1965523, 21.3373909, -24.2590637, 24.1856041
10: -17.7539997, 17.3622112, -17.8795891, 17.5744591, -31.1737137, 31.0897217
11: -26.7116375, 3.4894867, -26.8870049, 3.5792966, -27.8315659, 27.9234924
12: -34.8464661, -2.4358397, -34.8980751, -2.2846551, -27.2377472, 27.1110764
13: -26.2099285, 15.6272345, -26.2957001, 15.9481602, -34.0669861, 33.8174667
14: -55.7587662, -17.7313194, -55.9261971, -17.5121365, -37.7076035, 37.5055161
15: -14.2368698, 15.4488106, -14.3684883, 15.5383158, -27.7950668, 27.8375244
16: -13.9513807, 20.7457676, -14.0873671, 20.9138260, -31.0604858, 31.0141830
17: -57.7277527, -14.5269127, -57.8692780, -14.3096924, -41.6679459, 41.4528503
18: -21.4964905, 12.1045551, -21.7685623, 12.1843243, -29.4805145, 29.6994629
19: -22.1987228, 3.4901814, -22.4643631, 3.5885925, -22.6794205, 22.8525200
20: -23.2148151, 1.2795639, -23.4795094, 1.3941183, -19.1193237, 19.2826538
21: -26.7106018, 2.3101892, -27.0024166, 2.4211340, -25.4097748, 25.5879211
22: -28.4160385, 3.2586412, -28.7625561, 3.3550758, -24.6179504, 24.8720627
23: -22.2356472, 5.6323566, -22.4839268, 5.7369852, -21.9627457, 22.1167831
24: -18.2196922, 9.3726091, -18.5645103, 9.4643106, -22.7369995, 23.0054550
25: -23.7661476, 5.3021903, -24.0287895, 5.4136157, -24.3357162, 24.4874802
26: -40.9660034, -0.5862594, -41.2878647, -0.4426517, -30.5169830, 30.7265930
27: -21.4742031, 8.5162811, -21.8011475, 8.6090164, -26.3333817, 26.5881424
28: -24.0465908, 5.9782887, -24.3335056, 6.0905747, -21.8941383, 22.0836487
29: -27.7726421, -0.2469485, -28.0840282, -0.1854053, -23.8772049, 24.1185989
30: -28.0557442, 3.6613984, -28.3269558, 3.7798619, -26.0616264, 26.2085190
31: -22.5848827, 4.9691768, -22.8418999, 5.0711818, -24.9775467, 25.1390381
32: -23.8641605, 2.2411690, -23.9679012, 2.3610480, -21.3277397, 21.3219147
33: -36.2718048, 3.4930334, -36.4773979, 3.6415958, -33.2046356, 33.2460938
34: -37.7317734, -4.8962450, -37.9166603, -4.7516618, -27.6402206, 27.6736755
35: -32.7727737, 0.1519384, -32.9619064, 0.2963328, -28.0277252, 28.0695801
36: -36.6784897, -0.8061571, -36.8877716, -0.6544938, -28.8797455, 28.9387131
37: -44.3745041, -1.8154426, -44.6364365, -1.7119236, -38.6107788, 38.7411041
38: -43.7517014, 2.6874976, -44.0164528, 2.8849158, -40.5116730, 40.6008606
39: -43.4286346, 2.8521247, -43.6439667, 3.0464983, -41.2361450, 41.2722702
40: -32.6235733, -0.0639992, -32.7935028, 0.0801589, -30.9901428, 31.0108070
41: -20.6115627, 7.1970987, -20.7532310, 7.2978282, -26.3624496, 26.4327469
42: -22.9624653, -0.2493412, -22.9920063, -0.1788809, -18.4630699, 18.4147949

Time for backsubstitution: 2.26 seconds

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
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1526
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
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5099927, upper bound: 11.5628999
time: 32.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5241337, upper bound: 11.5628999
time: 33.21 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.2260561, 18.9990883, -9.3136921, 19.0646286, -25.1517334, 25.1639481
1: -1.1262407, 22.7948112, -1.1788125, 22.8587933, -19.7103729, 19.7037582
2: -1.5923331, 20.8938980, -1.5988955, 20.9819374, -17.2576675, 17.2480049
3: -9.3031006, 16.4439812, -9.3248482, 16.5674248, -22.0066833, 21.9318619
4: -3.0777869, 22.2148514, -3.1142588, 22.2706833, -21.6818008, 21.7130699
5: -7.7765255, 20.5641212, -7.8033237, 20.6916885, -23.7241898, 23.6696281
6: -28.7875404, -1.4028072, -28.8234062, -1.3763881, -23.1266327, 23.1916962
7: -7.6324840, 21.6043167, -7.6673555, 21.7005405, -23.5672607, 23.5280647
8: -14.7174110, 14.7034101, -14.7400398, 14.8188372, -26.4481964, 26.4130707
9: -5.1807899, 21.2560844, -5.1698771, 21.3208542, -24.3216095, 24.1992607
10: -17.9748249, 17.5249653, -17.8489113, 17.5780354, -31.3921356, 31.1945953
11: -26.7895412, 3.5160389, -26.8241158, 3.5490470, -27.8914108, 27.8848267
12: -34.8724442, -2.3969622, -34.8730316, -2.3279705, -27.2468643, 27.1181755
13: -26.2437897, 15.6611462, -26.2456436, 15.8599634, -34.0171890, 33.7985458
14: -55.9260750, -17.6052170, -55.8641968, -17.5569420, -37.8303986, 37.5481644
15: -14.2955666, 15.4658079, -14.3453407, 15.5274687, -27.8568420, 27.8289032
16: -14.0490608, 20.7888927, -14.0415678, 20.8675003, -31.1039200, 31.0111389
17: -57.8383904, -14.4704418, -57.8128891, -14.3953142, -41.7009811, 41.4524307
18: -21.5498810, 12.1413784, -21.7224064, 12.1529722, -29.4976807, 29.7280922
19: -22.2400856, 3.5027642, -22.3838654, 3.5415382, -22.6851807, 22.7848778
20: -23.2412415, 1.2934313, -23.4012756, 1.3432651, -19.0983543, 19.2183685
21: -26.7704391, 2.3305264, -26.9159546, 2.3725491, -25.4328537, 25.5220184
22: -28.4486160, 3.2960870, -28.6619320, 3.3062668, -24.6561813, 24.7727737
23: -22.2691956, 5.6495318, -22.4080200, 5.6844401, -21.9435921, 22.0629959
24: -18.2476082, 9.3827438, -18.4684143, 9.4197903, -22.7185059, 22.9229622
25: -23.7969589, 5.3299580, -23.9650860, 5.3667874, -24.3552856, 24.4460373
26: -41.0156212, -0.5456729, -41.1878433, -0.5019007, -30.5077667, 30.6621475
27: -21.5366917, 8.5941696, -21.7171593, 8.5597744, -26.3349762, 26.5842590
28: -24.0711727, 6.0198078, -24.2537746, 6.0367470, -21.8672218, 22.0314560
29: -27.8134232, -0.2106369, -27.9869518, -0.2223172, -23.9465103, 24.0144806
30: -28.0827217, 3.6927047, -28.2516594, 3.7305083, -26.0623932, 26.1689911
31: -22.6476803, 4.9872942, -22.7810688, 5.0329638, -25.0037994, 25.0940399
32: -23.9032879, 2.2914412, -23.9559021, 2.3473716, -21.3466263, 21.3558311
33: -36.3435516, 3.5965905, -36.4492493, 3.6233644, -33.2544403, 33.3364105
34: -37.7936745, -4.7911506, -37.8912888, -4.7932339, -27.6503525, 27.7598801
35: -32.8527222, 0.2740159, -32.9405251, 0.2682695, -28.0688248, 28.1718369
36: -36.7765656, -0.6728992, -36.8596802, -0.6985068, -28.9147720, 29.0434799
37: -44.4931145, -1.6936440, -44.5986252, -1.7373734, -38.6868896, 38.8399887
38: -43.8757820, 2.8574018, -43.9915276, 2.8401232, -40.5687866, 40.7377090
39: -43.4938164, 2.9237580, -43.6167450, 3.0308814, -41.2860718, 41.3196259
40: -32.6908035, 0.0251305, -32.7708206, 0.0610797, -31.0368652, 31.0817490
41: -20.6879711, 7.2944126, -20.7319412, 7.2743101, -26.3974686, 26.5071259
42: -22.9695625, -0.2215343, -22.9854031, -0.1910594, -18.4803963, 18.4363136

Time for backsubstitution: 2.26 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5581073
time: 33.23 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5241337, upper bound: 11.5581073
time: 31.56 seconds

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

Time for backsubstitution: 2.26 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5741336
time: 26.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5241337, upper bound: 11.5741336
time: 34.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.2858915, 19.0109272, -9.2903500, 19.0544567, -25.1581116, 25.1472702
1: -1.1680636, 22.7965393, -1.1638355, 22.8360939, -19.6966743, 19.6883888
2: -1.5903780, 20.8963070, -1.5801435, 20.9594421, -17.2469368, 17.1983261
3: -9.3224106, 16.4574299, -9.2978725, 16.5348911, -22.0092583, 21.9166412
4: -3.0994186, 22.2162266, -3.0782890, 22.2515068, -21.6873016, 21.6452942
5: -7.7997503, 20.5817642, -7.7853656, 20.6536903, -23.7186279, 23.6708450
6: -28.7669086, -1.4057970, -28.7945786, -1.3845253, -23.1188812, 23.1581459
7: -7.6535349, 21.6151962, -7.6377912, 21.6601067, -23.5423050, 23.5017128
8: -14.7503834, 14.7266779, -14.7217350, 14.7922077, -26.4617233, 26.3923798
9: -5.1598501, 21.2360840, -5.1510344, 21.2908173, -24.2446365, 24.1610565
10: -17.8446102, 17.3925133, -17.8499928, 17.5229321, -31.2018051, 31.0909576
11: -26.7350864, 3.5293846, -26.8210659, 3.5423369, -27.8145218, 27.8887863
12: -34.8596268, -2.3804440, -34.8654938, -2.3378272, -27.1902237, 27.1407967
13: -26.2123928, 15.6785707, -26.1953678, 15.8213482, -33.9405594, 33.7842255
14: -55.8484802, -17.6893425, -55.8711929, -17.5977497, -37.6845093, 37.6039581
15: -14.3386030, 15.4916544, -14.3468590, 15.5154657, -27.8617020, 27.8556824
16: -13.9995308, 20.7639656, -14.0201359, 20.8191013, -31.0001144, 30.9659042
17: -57.7974358, -14.4831829, -57.8280449, -14.4166460, -41.5972443, 41.5374832
18: -21.5463142, 12.1436796, -21.6697979, 12.1273651, -29.4868164, 29.6401443
19: -22.2474194, 3.5406616, -22.3486710, 3.5235140, -22.6730118, 22.7812042
20: -23.2674980, 1.3383570, -23.3689232, 1.3344049, -19.1152573, 19.1918373
21: -26.7629433, 2.3673396, -26.8862019, 2.3573263, -25.4053268, 25.5219612
22: -28.4609642, 3.2958012, -28.6023407, 3.2752225, -24.5940933, 24.7204514
23: -22.2665405, 5.6663599, -22.3649902, 5.6524715, -21.9188843, 22.0144882
24: -18.2712803, 9.4209690, -18.4155960, 9.3985825, -22.7301941, 22.8761749
25: -23.7945900, 5.3556385, -23.9159489, 5.3365002, -24.2899704, 24.4082146
26: -41.0074501, -0.5346851, -41.1218643, -0.5429101, -30.4760590, 30.5769196
27: -21.5055981, 8.5469141, -21.6630421, 8.5271740, -26.2968369, 26.4777298
28: -24.0844307, 6.0340853, -24.2006779, 6.0035372, -21.8578110, 21.9911385
29: -27.8051186, -0.2403812, -27.9362030, -0.2626646, -23.8477859, 23.9850502
30: -28.0922356, 3.7174957, -28.2227936, 3.7122812, -26.0265541, 26.1382980
31: -22.6375122, 5.0228963, -22.7439690, 5.0102944, -24.9725800, 25.0749626
32: -23.8982239, 2.3022308, -23.9247475, 2.3384101, -21.3433151, 21.3160591
33: -36.3607368, 3.6365652, -36.4188614, 3.6632533, -33.3106537, 33.2711487
34: -37.7957153, -4.7830992, -37.8670616, -4.7636924, -27.6918182, 27.6849365
35: -32.8434525, 0.2837930, -32.9096336, 0.3047872, -28.1040421, 28.1129532
36: -36.7427750, -0.6842723, -36.8268127, -0.6670780, -28.9236450, 28.9693451
37: -44.4457741, -1.7284265, -44.5548553, -1.7106113, -38.6722107, 38.7473831
38: -43.8442764, 2.8438773, -43.9474869, 2.8738079, -40.5716400, 40.6290131
39: -43.5286751, 2.9991274, -43.5726013, 3.0606389, -41.3507843, 41.2821884
40: -32.6740570, -0.0178781, -32.7261086, 0.0452528, -31.0104523, 30.9968796
41: -20.6501350, 7.2597399, -20.7055969, 7.2818689, -26.4155502, 26.4428635
42: -22.9769554, -0.2282691, -22.9801064, -0.2019272, -18.4562302, 18.4329605

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
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5307281, upper bound: 11.5244318
time: 29.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5448637, upper bound: 11.5244318
time: 33.47 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.2858915, 19.0109272, -9.3591747, 19.0959854, -25.2005844, 25.2177505
1: -1.1680636, 22.7965393, -1.2136288, 22.8789864, -19.7417297, 19.7332497
2: -1.5903780, 20.8963070, -1.6213915, 20.9940796, -17.2845192, 17.2363396
3: -9.3224106, 16.4574299, -9.3287125, 16.5867462, -22.0573654, 21.9462891
4: -3.0994186, 22.2162266, -3.1302304, 22.2753391, -21.7049599, 21.6921272
5: -7.7997503, 20.5817642, -7.8261867, 20.7062206, -23.7708054, 23.7066040
6: -28.7669086, -1.4057970, -28.8070335, -1.3629436, -23.1456909, 23.1732216
7: -7.6535349, 21.6151962, -7.6863089, 21.6998177, -23.5828552, 23.5463943
8: -14.7503834, 14.7266779, -14.7775555, 14.8343344, -26.4952469, 26.4379044
9: -5.1598501, 21.2360840, -5.1797562, 21.3170052, -24.2788086, 24.1927147
10: -17.8446102, 17.3925133, -17.8831215, 17.5618267, -31.2439651, 31.1290436
11: -26.7350864, 3.5293846, -26.8862610, 3.5759535, -27.8464661, 27.9569168
12: -34.8596268, -2.3804440, -34.8920555, -2.2892795, -27.2396622, 27.1686935
13: -26.2123928, 15.6785707, -26.2495918, 15.9126377, -34.0318756, 33.8401794
14: -55.8484802, -17.6893425, -55.9374008, -17.5199890, -37.7624817, 37.6688461
15: -14.3386030, 15.4916544, -14.3736982, 15.5291939, -27.8755798, 27.8855286
16: -13.9995308, 20.7639656, -14.0697985, 20.8750496, -31.0576935, 31.0120850
17: -57.7974358, -14.4831829, -57.8875237, -14.3202076, -41.6970673, 41.6029205
18: -21.5463142, 12.1436796, -21.7209625, 12.1604815, -29.5226898, 29.6928253
19: -22.2474194, 3.5406616, -22.4330997, 3.5715952, -22.7219009, 22.8661118
20: -23.2674980, 1.3383570, -23.4497566, 1.3863752, -19.1716957, 19.2729797
21: -26.7629433, 2.3673396, -26.9771690, 2.4078603, -25.4559631, 25.6134758
22: -28.4609642, 3.2958012, -28.7090683, 3.3257153, -24.6491013, 24.8290558
23: -22.2665405, 5.6663599, -22.4428043, 5.7058907, -21.9737968, 22.0929947
24: -18.2712803, 9.4209690, -18.5167465, 9.4439392, -22.7768860, 22.9767685
25: -23.7945900, 5.3556385, -23.9834156, 5.3854303, -24.3426743, 24.4778519
26: -41.0074501, -0.5346851, -41.2249985, -0.4757309, -30.5480347, 30.6794510
27: -21.5055981, 8.5469141, -21.7618256, 8.5788898, -26.3502884, 26.5769653
28: -24.0844307, 6.0340853, -24.2846985, 6.0595980, -21.9175682, 22.0755692
29: -27.8051186, -0.2403812, -28.0395222, -0.2244687, -23.8900795, 24.0886269
30: -28.0922356, 3.7174957, -28.3013725, 3.7646158, -26.0794601, 26.2159042
31: -22.6375122, 5.0228963, -22.8088303, 5.0502396, -25.0149155, 25.1419067
32: -23.8982239, 2.3022308, -23.9463596, 2.3540893, -21.3603859, 21.3460388
33: -36.3607368, 3.6365652, -36.4635887, 3.6827412, -33.3307419, 33.3150177
34: -37.7957153, -4.7830992, -37.9093628, -4.7192550, -27.7399063, 27.7304840
35: -32.8434525, 0.2837930, -32.9515495, 0.3341594, -28.1347198, 28.1557388
36: -36.7427750, -0.6842723, -36.8807755, -0.6216078, -28.9714737, 29.0246887
37: -44.4457741, -1.7284265, -44.6217613, -1.6844668, -38.7003479, 38.8161621
38: -43.8442764, 2.8438773, -44.0020828, 2.9215026, -40.6283112, 40.6908340
39: -43.5286751, 2.9991274, -43.6133614, 3.0781531, -41.3694763, 41.3338242
40: -32.6740570, -0.0178781, -32.7633057, 0.0651097, -31.0310211, 31.0344849
41: -20.6501350, 7.2597399, -20.7463779, 7.3071804, -26.4483719, 26.4892197
42: -22.9769554, -0.2282691, -22.9884872, -0.1873653, -18.4711380, 18.4428444

Time for backsubstitution: 2.32 seconds

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
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5307281, upper bound: 11.5403762
time: 77.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5448637, upper bound: 11.5403762
time: 32.50 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.3402634, 19.0340405, -9.2948732, 19.0549889, -25.2167053, 25.1813736
1: -1.2005243, 22.8276825, -1.1647663, 22.8367844, -19.7313995, 19.7324829
2: -1.6421335, 20.9112587, -1.5824885, 20.9627609, -17.2755165, 17.2561150
3: -9.3467617, 16.4723663, -9.2993908, 16.5358543, -22.0293541, 21.9565620
4: -3.1457915, 22.2239971, -3.0812006, 22.2525520, -21.7311630, 21.6972809
5: -7.8492718, 20.5994873, -7.7888603, 20.6565857, -23.7540741, 23.7024727
6: -28.8194427, -1.3275747, -28.8072548, -1.3825727, -23.1597595, 23.2513008
7: -7.6995683, 21.6305199, -7.6412797, 21.6623192, -23.5868073, 23.5409012
8: -14.8274164, 14.7568293, -14.7228050, 14.7970839, -26.5278854, 26.4571304
9: -5.2456779, 21.2826977, -5.1530609, 21.3004742, -24.3413010, 24.2063484
10: -18.0655212, 17.5552940, -17.8524513, 17.5654430, -31.4624634, 31.2338638
11: -26.8129196, 3.5559440, -26.8234100, 3.5456862, -27.9063110, 27.9182281
12: -34.8856239, -2.3414974, -34.8670654, -2.3326025, -27.2488251, 27.1757851
13: -26.2462349, 15.7125454, -26.1994743, 15.8243999, -33.9820862, 33.8213577
14: -56.0157852, -17.5632668, -55.8753586, -17.5647449, -37.8852844, 37.7114639
15: -14.3973541, 15.5086508, -14.3505440, 15.5183325, -27.9373779, 27.8768997
16: -14.0972366, 20.8071041, -14.0239553, 20.8286915, -31.1011429, 31.0090179
17: -57.9080811, -14.4267235, -57.8311272, -14.4057465, -41.7299957, 41.6024933
18: -21.5997276, 12.1805077, -21.6748199, 12.1291409, -29.5398331, 29.7214508
19: -22.2887268, 3.5532498, -22.3525925, 3.5245347, -22.7276764, 22.7984428
20: -23.2939377, 1.3522229, -23.3715210, 1.3355370, -19.1507111, 19.2087746
21: -26.8228207, 2.3876634, -26.8907280, 2.3592722, -25.4791107, 25.5475960
22: -28.4935474, 3.3332334, -28.6084251, 3.2769618, -24.6873665, 24.7297897
23: -22.3001137, 5.6835246, -22.3669071, 5.6533375, -21.9546165, 22.0392380
24: -18.2991982, 9.4311028, -18.4206905, 9.3993950, -22.7584076, 22.8943024
25: -23.8253918, 5.3834562, -23.9196911, 5.3385506, -24.3622284, 24.4364777
26: -41.0571136, -0.4940724, -41.1249771, -0.5348706, -30.5388718, 30.6150055
27: -21.5681343, 8.6248302, -21.6778793, 8.5296068, -26.3519440, 26.5731049
28: -24.1090050, 6.0755758, -24.2049980, 6.0057836, -21.8906784, 22.0233879
29: -27.8458252, -0.2040722, -27.9424477, -0.2613695, -23.9593735, 23.9845657
30: -28.1192455, 3.7488146, -28.2260780, 3.7152750, -26.0802460, 26.1764069
31: -22.7003231, 5.0410013, -22.7480068, 5.0120201, -25.0411758, 25.0969200
32: -23.9373589, 2.3525288, -23.9343929, 2.3404140, -21.3793030, 21.3799629
33: -36.4324760, 3.7401447, -36.4354477, 3.6644526, -33.3805923, 33.4053497
34: -37.8576317, -4.6779790, -37.8839874, -4.7608795, -27.7500534, 27.8166504
35: -32.9234085, 0.4058609, -32.9301796, 0.3059959, -28.1757889, 28.2580261
36: -36.8407974, -0.5510292, -36.8526917, -0.6656132, -29.0064545, 29.1294098
37: -44.5643349, -1.6066761, -44.5839958, -1.7098989, -38.7764130, 38.9151001
38: -43.9683151, 3.0138850, -43.9772034, 2.8766756, -40.6853027, 40.8276367
39: -43.5937576, 3.0707269, -43.5861435, 3.0625944, -41.4194489, 41.3812332
40: -32.7412872, 0.0713272, -32.7405777, 0.0460219, -31.0777283, 31.1054077
41: -20.7264709, 7.3570375, -20.7250767, 7.2836218, -26.4833298, 26.5636292
42: -22.9841137, -0.2004638, -22.9818993, -0.1995528, -18.4884491, 18.4643745

Time for backsubstitution: 2.27 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5307281, upper bound: 11.5356412
time: 33.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5448637, upper bound: 11.5356412
time: 32.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.3402634, 19.0340405, -9.3636227, 19.0964584, -25.2591782, 25.2518234
1: -1.2005243, 22.8276825, -1.2145510, 22.8796692, -19.7764473, 19.7773361
2: -1.6421335, 20.9112587, -1.6237350, 20.9973927, -17.3131104, 17.2941437
3: -9.3467617, 16.4723663, -9.3301945, 16.5876999, -22.0774689, 21.9861794
4: -3.1457915, 22.2239971, -3.1331153, 22.2764053, -21.7488251, 21.7441254
5: -7.8492718, 20.5994873, -7.8296747, 20.7090931, -23.8062515, 23.7382202
6: -28.8194427, -1.3275747, -28.8197174, -1.3609905, -23.1865692, 23.2663689
7: -7.6995683, 21.6305199, -7.6898079, 21.7020187, -23.6273651, 23.5856094
8: -14.8274164, 14.7568293, -14.7785873, 14.8392029, -26.5613861, 26.5027008
9: -5.2456779, 21.2826977, -5.1817379, 21.3266716, -24.3754883, 24.2380295
10: -18.0655212, 17.5552940, -17.8855553, 17.6043949, -31.5046234, 31.2719345
11: -26.8129196, 3.5559440, -26.8885841, 3.5793133, -27.9382324, 27.9863358
12: -34.8856239, -2.3414974, -34.8936157, -2.2840509, -27.2982407, 27.2036743
13: -26.2462349, 15.7125454, -26.2537270, 15.9157124, -34.0734253, 33.8773117
14: -56.0157852, -17.5632668, -55.9415817, -17.4870186, -37.9632568, 37.7763367
15: -14.3973541, 15.5086508, -14.3773594, 15.5320330, -27.9512177, 27.9067307
16: -14.0972366, 20.8071041, -14.0735989, 20.8846512, -31.1587219, 31.0551834
17: -57.9080811, -14.4267235, -57.8905869, -14.3093147, -41.8298035, 41.6679382
18: -21.5997276, 12.1805077, -21.7260094, 12.1622419, -29.5757065, 29.7741318
19: -22.2887268, 3.5532498, -22.4370499, 3.5726306, -22.7765884, 22.8833466
20: -23.2939377, 1.3522229, -23.4523315, 1.3875506, -19.2071686, 19.2899246
21: -26.8228207, 2.3876634, -26.9816971, 2.4097826, -25.5297699, 25.6391144
22: -28.4935474, 3.3332334, -28.7151508, 3.3273895, -24.7423592, 24.8384209
23: -22.3001137, 5.6835246, -22.4447689, 5.7067604, -22.0095291, 22.1177711
24: -18.2991982, 9.4311028, -18.5218582, 9.4447737, -22.8051224, 22.9949150
25: -23.8253918, 5.3834562, -23.9871368, 5.3874569, -24.4149094, 24.5061188
26: -41.0571136, -0.4940724, -41.2281189, -0.4676852, -30.6108856, 30.7175522
27: -21.5681343, 8.6248302, -21.7767067, 8.5813370, -26.4053955, 26.6723404
28: -24.1090050, 6.0755758, -24.2890739, 6.0617914, -21.9504700, 22.1078300
29: -27.8458252, -0.2040722, -28.0457573, -0.2231771, -24.0016861, 24.0881729
30: -28.1192455, 3.7488146, -28.3046722, 3.7676182, -26.1331558, 26.2540169
31: -22.7003231, 5.0410013, -22.8128643, 5.0520334, -25.0835266, 25.1638947
32: -23.9373589, 2.3525288, -23.9560719, 2.3561335, -21.3963470, 21.4099770
33: -36.4324760, 3.7401447, -36.4801865, 3.6839423, -33.4006805, 33.4492188
34: -37.8576317, -4.6779790, -37.9262543, -4.7164183, -27.7981567, 27.8621979
35: -32.9234085, 0.4058609, -32.9721069, 0.3353972, -28.2064590, 28.3008194
36: -36.8407974, -0.5510292, -36.9066544, -0.6201630, -29.0543137, 29.1847916
37: -44.5643349, -1.6066761, -44.6509247, -1.6836948, -38.8046112, 38.9838638
38: -43.9683151, 3.0138850, -44.0317841, 2.9243045, -40.7419739, 40.8894730
39: -43.5937576, 3.0707269, -43.6268997, 3.0799742, -41.4381256, 41.4328384
40: -32.7412872, 0.0713272, -32.7777481, 0.0658717, -31.0983276, 31.1430054
41: -20.7264709, 7.3570375, -20.7658710, 7.3089323, -26.5161209, 26.6099625
42: -22.9841137, -0.2004638, -22.9902706, -0.1849902, -18.5033340, 18.4742546

Time for backsubstitution: 2.27 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5307281, upper bound: 11.5516114
time: 34.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5448637, upper bound: 11.5516114
time: 33.47 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.3020859, 19.0113087, -9.3513231, 19.0655060, -25.1898117, 25.2127838
1: -1.1807547, 22.7967720, -1.2070436, 22.8590660, -19.7320633, 19.7265015
2: -1.6009955, 20.8965416, -1.6163635, 20.9795189, -17.2799301, 17.2284775
3: -9.3359299, 16.4583054, -9.3430204, 16.5693264, -22.0469627, 21.9513741
4: -3.1167755, 22.2163563, -3.1388597, 22.2707539, -21.7205048, 21.6960907
5: -7.8124647, 20.5829277, -7.8285480, 20.6908836, -23.7622147, 23.7032089
6: -28.7728062, -1.4019737, -28.8141022, -1.3522625, -23.1540375, 23.1750145
7: -7.6701055, 21.6157455, -7.6913309, 21.6995125, -23.5975952, 23.5440826
8: -14.7679272, 14.7272100, -14.7821875, 14.8164444, -26.4915123, 26.4450531
9: -5.1702042, 21.2363930, -5.1929345, 21.3133965, -24.2857285, 24.2056694
10: -17.8515301, 17.3935394, -17.8786888, 17.5391769, -31.2315063, 31.1179962
11: -26.7353134, 3.5334268, -26.8272438, 3.5599685, -27.8363800, 27.9044571
12: -34.8623734, -2.3755255, -34.8761940, -2.3148088, -27.2238083, 27.1551552
13: -26.2266178, 15.6797066, -26.2456932, 15.8708658, -34.0080109, 33.8293915
14: -55.8510132, -17.6863785, -55.8888626, -17.5849152, -37.7258148, 37.6218643
15: -14.3436680, 15.4947586, -14.3760624, 15.5269785, -27.8756485, 27.8882904
16: -14.0088673, 20.7642021, -14.0564861, 20.8585377, -31.0497665, 30.9945908
17: -57.7983932, -14.4804697, -57.8325157, -14.3990574, -41.6534958, 41.5398254
18: -21.5474319, 12.1554356, -21.7232666, 12.1682920, -29.5212479, 29.7096672
19: -22.2482986, 3.5522487, -22.3850212, 3.5614877, -22.7019348, 22.8289566
20: -23.2679482, 1.3471565, -23.4009933, 1.3646798, -19.1411476, 19.2360992
21: -26.7635612, 2.3778048, -26.9170723, 2.3931999, -25.4343719, 25.5602303
22: -28.4616013, 3.3099663, -28.6589680, 3.3211904, -24.6263237, 24.7921600
23: -22.2669716, 5.6803231, -22.4087677, 5.6991749, -21.9537430, 22.0741119
24: -18.2723598, 9.4334679, -18.4656906, 9.4399652, -22.7639008, 22.9412613
25: -23.7956123, 5.3703499, -23.9639530, 5.3861628, -24.3295670, 24.4729919
26: -41.0083847, -0.5173378, -41.1873055, -0.4863291, -30.5149155, 30.6656418
27: -21.5064754, 8.5598907, -21.7062607, 8.5720882, -26.3364258, 26.5348206
28: -24.0849781, 6.0508137, -24.2516308, 6.0583920, -21.8969345, 22.0605888
29: -27.8054848, -0.2273376, -27.9855919, -0.2182521, -23.8806076, 24.0474472
30: -28.0925903, 3.7270045, -28.2503395, 3.7483883, -26.0548859, 26.1745758
31: -22.6387749, 5.0354743, -22.7823277, 5.0534592, -25.0130081, 25.1278381
32: -23.9044151, 2.3052166, -23.9491329, 2.3664896, -21.3766060, 21.3378029
33: -36.3635635, 3.6376987, -36.4368820, 3.6720967, -33.3263550, 33.2903595
34: -37.7963791, -4.7823434, -37.8764114, -4.7574444, -27.6985703, 27.6980438
35: -32.8449860, 0.2854171, -32.9230499, 0.3130460, -28.1144333, 28.1300659
36: -36.7438889, -0.6819639, -36.8367767, -0.6575999, -28.9386673, 28.9868469
37: -44.4489975, -1.7286673, -44.5761528, -1.7084713, -38.6916809, 38.7654190
38: -43.8464279, 2.8480000, -43.9677582, 2.8923068, -40.6019897, 40.6666870
39: -43.5367508, 2.9992285, -43.6098633, 3.0799427, -41.3747559, 41.3153381
40: -32.6813889, -0.0164425, -32.7628555, 0.0761096, -31.0482941, 31.0298843
41: -20.6527195, 7.2616911, -20.7167473, 7.2939367, -26.4295731, 26.4551926
42: -22.9793644, -0.2261059, -22.9885864, -0.1869543, -18.4746475, 18.4410782

Time for backsubstitution: 2.26 seconds

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
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5318518, upper bound: 11.5475498
time: 29.46 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5459945, upper bound: 11.5475498
time: 30.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.3020859, 19.0113087, -9.4200678, 19.1070137, -25.2322807, 25.2832146
1: -1.1807547, 22.7967720, -1.2568436, 22.9019852, -19.7771149, 19.7713547
2: -1.6009955, 20.8965416, -1.6576393, 21.0141716, -17.3175278, 17.2664986
3: -9.3359299, 16.4583054, -9.3738556, 16.6212063, -22.0950966, 21.9810333
4: -3.1167755, 22.2163563, -3.1907496, 22.2945480, -21.7381592, 21.7429276
5: -7.8124647, 20.5829277, -7.8693867, 20.7433968, -23.8144455, 23.7389755
6: -28.7728062, -1.4019737, -28.8265762, -1.3306408, -23.1808472, 23.1900902
7: -7.6701055, 21.6157455, -7.7398720, 21.7392178, -23.6381569, 23.5887947
8: -14.7679272, 14.7272100, -14.8380394, 14.8585911, -26.5250397, 26.4905853
9: -5.1702042, 21.2363930, -5.2216082, 21.3396034, -24.3198776, 24.2373238
10: -17.8515301, 17.3935394, -17.9118118, 17.5781212, -31.2736893, 31.1560822
11: -26.7353134, 3.5334268, -26.8924065, 3.5936265, -27.8683243, 27.9725571
12: -34.8623734, -2.3755255, -34.9027748, -2.2662735, -27.2732391, 27.1830215
13: -26.2266178, 15.6797066, -26.2999001, 15.9620876, -34.0993347, 33.8853455
14: -55.8510132, -17.6863785, -55.9551010, -17.5071659, -37.8038025, 37.6867142
15: -14.3436680, 15.4947586, -14.4028912, 15.5407276, -27.8895187, 27.9182053
16: -14.0088673, 20.7642021, -14.1061306, 20.9145222, -31.1073608, 31.0407410
17: -57.7983932, -14.4804697, -57.8919716, -14.3026304, -41.7533340, 41.6052780
18: -21.5474319, 12.1554356, -21.7743931, 12.2013435, -29.5571289, 29.7623405
19: -22.2482986, 3.5522487, -22.4694481, 3.6095417, -22.7508316, 22.9138412
20: -23.2679482, 1.3471565, -23.4818439, 1.4166443, -19.1975822, 19.3172493
21: -26.7635612, 2.3778048, -27.0080910, 2.4437137, -25.4850311, 25.6517944
22: -28.4616013, 3.3099663, -28.7656803, 3.3716788, -24.6813507, 24.9007874
23: -22.2669716, 5.6803231, -22.4866028, 5.7526097, -22.0086517, 22.1526299
24: -18.2723598, 9.4334679, -18.5668640, 9.4852924, -22.8106232, 23.0418663
25: -23.7956123, 5.3703499, -24.0313988, 5.4350843, -24.3822632, 24.5426102
26: -41.0083847, -0.5173378, -41.2904434, -0.4190817, -30.5869217, 30.7681503
27: -21.5064754, 8.5598907, -21.8050690, 8.6238289, -26.3898621, 26.6340714
28: -24.0849781, 6.0508137, -24.3356495, 6.1144581, -21.9567337, 22.1450157
29: -27.8054848, -0.2273376, -28.0889473, -0.1800374, -23.9229202, 24.1510735
30: -28.0925903, 3.7270045, -28.3289452, 3.8007417, -26.1078262, 26.2522049
31: -22.6387749, 5.0354743, -22.8471813, 5.0934157, -25.0553513, 25.1947708
32: -23.9044151, 2.3052166, -23.9708023, 2.3821449, -21.3936501, 21.3677979
33: -36.3635635, 3.6376987, -36.4816246, 3.6915884, -33.3464203, 33.3342438
34: -37.7963791, -4.7823434, -37.9186859, -4.7130432, -27.7466812, 27.7435837
35: -32.8449860, 0.2854171, -32.9649315, 0.3424010, -28.1450729, 28.1728668
36: -36.7438889, -0.6819639, -36.8907166, -0.6121159, -28.9865036, 29.0422211
37: -44.4489975, -1.7286673, -44.6431160, -1.6823096, -38.7198792, 38.8341827
38: -43.8464279, 2.8480000, -44.0223503, 2.9399652, -40.6586609, 40.7284851
39: -43.5367508, 2.9992285, -43.6505928, 3.0973811, -41.3934631, 41.3669586
40: -32.6813889, -0.0164425, -32.8000145, 0.0959897, -31.0688934, 31.0674591
41: -20.6527195, 7.2616911, -20.7575111, 7.3192377, -26.4623566, 26.5015259
42: -22.9793644, -0.2261059, -22.9969749, -0.1723690, -18.4895630, 18.4509697

Time for backsubstitution: 2.28 seconds

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
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5318518, upper bound: 11.5634604
time: 40.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5459945, upper bound: 11.5634604
time: 35.81 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.3564415, 19.0344620, -9.3557987, 19.0659981, -25.2484550, 25.2468643
1: -1.2132435, 22.8279037, -1.2079649, 22.8597794, -19.7667465, 19.7705841
2: -1.6527531, 20.9115105, -1.6187344, 20.9828377, -17.3085518, 17.2862549
3: -9.3603306, 16.4732513, -9.3445396, 16.5703106, -22.0670319, 21.9912720
4: -3.1631694, 22.2240887, -3.1417155, 22.2717667, -21.7643471, 21.7480469
5: -7.8619909, 20.6006393, -7.8320026, 20.6937943, -23.7976494, 23.7348175
6: -28.8253880, -1.3237319, -28.8267708, -1.3503408, -23.1949310, 23.2681847
7: -7.7161026, 21.6310215, -7.6948256, 21.7017174, -23.6421013, 23.5832748
8: -14.8449726, 14.7573795, -14.7832460, 14.8213253, -26.5576630, 26.5098038
9: -5.2560577, 21.2829933, -5.1949091, 21.3230476, -24.3823776, 24.2509232
10: -18.0723991, 17.5562782, -17.8810940, 17.5817566, -31.4922028, 31.2608261
11: -26.8132019, 3.5599532, -26.8295479, 3.5633860, -27.9281540, 27.9338608
12: -34.8883324, -2.3366141, -34.8777504, -2.3095360, -27.2823563, 27.1900864
13: -26.2604485, 15.7136898, -26.2497978, 15.8739090, -34.0495682, 33.8665390
14: -56.0183067, -17.5603008, -55.8930244, -17.5519218, -37.9266052, 37.7294235
15: -14.4023991, 15.5117626, -14.3797379, 15.5298424, -27.9513626, 27.9094849
16: -14.1065836, 20.8073311, -14.0602875, 20.8681889, -31.1508026, 31.0376740
17: -57.9090805, -14.4239922, -57.8355560, -14.3882055, -41.7862930, 41.6048126
18: -21.6008530, 12.1922693, -21.7282887, 12.1700315, -29.5743027, 29.7909660
19: -22.2896404, 3.5648327, -22.3889503, 3.5624931, -22.7566376, 22.8461838
20: -23.2944183, 1.3610020, -23.4035835, 1.3657894, -19.1766357, 19.2529984
21: -26.8234062, 2.3981364, -26.9216003, 2.3951335, -25.5081253, 25.5859108
22: -28.4941998, 3.3474307, -28.6650791, 3.3229115, -24.7195892, 24.8015404
23: -22.3005390, 5.6974974, -22.4106884, 5.7000442, -21.9895020, 22.0988388
24: -18.3002968, 9.4436569, -18.4707756, 9.4407978, -22.7921295, 22.9594116
25: -23.8263969, 5.3981185, -23.9677143, 5.3882189, -24.4018097, 24.5012741
26: -41.0580597, -0.4766784, -41.1903687, -0.4782653, -30.5777664, 30.7036972
27: -21.5689907, 8.6377802, -21.7210922, 8.5745335, -26.3915100, 26.6302185
28: -24.1095276, 6.0923781, -24.2559280, 6.0606055, -21.9298248, 22.0928535
29: -27.8461609, -0.1909823, -27.9919205, -0.2169503, -23.9922218, 24.0470238
30: -28.1196060, 3.7583337, -28.2536278, 3.7513833, -26.1085587, 26.2126923
31: -22.7016506, 5.0535851, -22.7863808, 5.0552092, -25.0816345, 25.1497955
32: -23.9435616, 2.3555171, -23.9587898, 2.3684912, -21.4125595, 21.4017029
33: -36.4353447, 3.7412400, -36.4534874, 3.6732917, -33.3962097, 33.4245758
34: -37.8582916, -4.6773047, -37.8933105, -4.7546206, -27.7568130, 27.8297577
35: -32.9248810, 0.4074488, -32.9435463, 0.3142815, -28.1861801, 28.2751465
36: -36.8418961, -0.5486655, -36.8626404, -0.6561093, -29.0214920, 29.1469116
37: -44.5675659, -1.6068416, -44.6052856, -1.7077780, -38.7959747, 38.9330673
38: -43.9704590, 3.0179424, -43.9974213, 2.8951678, -40.7156982, 40.8653412
39: -43.6018753, 3.0709152, -43.6233597, 3.0818324, -41.4434509, 41.4142990
40: -32.7486267, 0.0726926, -32.7773476, 0.0769215, -31.1155853, 31.1384735
41: -20.7291069, 7.3589773, -20.7362041, 7.2956891, -26.4973450, 26.5759430
42: -22.9865131, -0.1983089, -22.9903622, -0.1845965, -18.5068626, 18.4725113

Time for backsubstitution: 2.27 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5318518, upper bound: 11.5588269
time: 42.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5459945, upper bound: 11.5588269
time: 32.06 seconds

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

Time for backsubstitution: 2.29 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5318518, upper bound: 11.5747583
time: 32.85 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5459945, upper bound: 11.5747583
time: 25.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.2671137, 19.0508137, -9.1805878, 18.9765778, -25.0776138, 25.0857315
1: -1.1559472, 22.8396492, -1.1042981, 22.7643394, -19.6316147, 19.6680641
2: -1.5688589, 20.9396019, -1.5415082, 20.8794289, -17.1606903, 17.1923141
3: -9.2836704, 16.5115433, -9.2721605, 16.4303055, -21.8614273, 21.9337578
4: -3.0680542, 22.2418480, -3.0333638, 22.2072124, -21.6195831, 21.6094284
5: -7.7762189, 20.6383839, -7.7340832, 20.5468941, -23.6141663, 23.6658134
6: -28.7947636, -1.4179821, -28.7315636, -1.4774704, -23.0769653, 23.0595818
7: -7.6281919, 21.6572838, -7.5891504, 21.5904903, -23.4561691, 23.4879341
8: -14.6970997, 14.7782173, -14.6445045, 14.6760292, -26.3048134, 26.3476181
9: -5.1371746, 21.2862453, -5.0981236, 21.2108917, -24.1196594, 24.1948662
10: -17.8314762, 17.5346298, -17.7657681, 17.3651924, -31.0323486, 31.1443100
11: -26.7998829, 3.5392833, -26.7134609, 3.4991026, -27.8345490, 27.7856369
12: -34.8754501, -2.3553205, -34.8609009, -2.4363379, -27.0850906, 27.1735229
13: -26.2052727, 15.7605610, -26.2123375, 15.6302576, -33.7284470, 33.8768005
14: -55.8917198, -17.5417500, -55.8118744, -17.7311249, -37.4764633, 37.6629105
15: -14.3138618, 15.4997406, -14.2360983, 15.4481163, -27.7798004, 27.7578506
16: -14.0162048, 20.8465786, -13.9672031, 20.7467499, -30.9405212, 31.0025177
17: -57.8410721, -14.3770962, -57.7684784, -14.5235996, -41.4239655, 41.5691910
18: -21.5997181, 12.1068668, -21.5026665, 12.0900259, -29.5144424, 29.4156952
19: -22.3333435, 3.5254171, -22.2017860, 3.5026364, -22.7356033, 22.6201363
20: -23.3503914, 1.3316145, -23.2160301, 1.2939258, -19.1663170, 19.0455627
21: -26.8570824, 2.3518019, -26.7137337, 2.3191617, -25.4561157, 25.3389511
22: -28.5773354, 3.2738955, -28.4189034, 3.2648683, -24.6955719, 24.5388870
23: -22.3445168, 5.6621089, -22.2362995, 5.6450043, -21.9967422, 21.8909760
24: -18.3792076, 9.3938198, -18.2212105, 9.3779087, -22.8274918, 22.6697159
25: -23.8658409, 5.3263555, -23.7672806, 5.3004951, -24.3223801, 24.2490273
26: -41.1122932, -0.5392399, -40.9681702, -0.5741892, -30.5651703, 30.4202423
27: -21.6342278, 8.5295839, -21.4770660, 8.5223446, -26.4260483, 26.2544479
28: -24.1781235, 6.0035000, -24.0474796, 5.9857001, -21.9433022, 21.8091278
29: -27.9189415, -0.2563241, -27.7751312, -0.2438536, -23.9562988, 23.8083572
30: -28.1757507, 3.7040443, -28.0565777, 3.6660342, -26.0625000, 25.9801941
31: -22.7026138, 4.9979434, -22.5877647, 4.9689021, -24.9998093, 24.9060555
32: -23.9233665, 2.2929487, -23.8582973, 2.2434545, -21.2822762, 21.2536697
33: -36.4342880, 3.6111345, -36.2717018, 3.5066733, -33.2177582, 33.1686783
34: -37.8758392, -4.7804065, -37.7331467, -4.8731985, -27.6564484, 27.5943069
35: -32.9304314, 0.2648115, -32.7750282, 0.1665173, -28.0520401, 27.9924469
36: -36.8567924, -0.6870861, -36.6813583, -0.7776256, -28.9316788, 28.8358231
37: -44.5676422, -1.7304435, -44.3765717, -1.7937045, -38.6956024, 38.5728683
38: -43.9625626, 2.8320518, -43.7554779, 2.7072511, -40.5552521, 40.4470978
39: -43.5690117, 2.9760451, -43.4224625, 2.8558216, -41.2037048, 41.1646271
40: -32.7096252, -0.0147967, -32.6154442, -0.0644794, -30.9379730, 30.8921013
41: -20.7258034, 7.2643785, -20.6109047, 7.2163391, -26.4347076, 26.3283310
42: -22.9759445, -0.2193193, -22.9572639, -0.2425292, -18.4062042, 18.4152451

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 974
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
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 523
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5102903, upper bound: 11.5363242
time: 35.22 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5244127, upper bound: 11.5363242
time: 38.54 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.2715645, 19.0512886, -9.2349262, 18.9996910, -25.1116867, 25.1443405
1: -1.1568789, 22.8403358, -1.1368423, 22.7954788, -19.6757240, 19.7027740
2: -1.5711920, 20.9428997, -1.5932307, 20.8943653, -17.2184753, 17.2209129
3: -9.2851830, 16.5125065, -9.2965384, 16.4452286, -21.9013519, 21.9538956
4: -3.0709476, 22.2428856, -3.0797119, 22.2149696, -21.6715622, 21.6532593
5: -7.7797108, 20.6412544, -7.7836313, 20.5646057, -23.6457634, 23.7012749
6: -28.8074360, -1.4160900, -28.7841263, -1.3992138, -23.1701431, 23.1005173
7: -7.6316490, 21.6594944, -7.6351156, 21.6058350, -23.4954033, 23.5324516
8: -14.6981316, 14.7830877, -14.7215214, 14.7061520, -26.3695755, 26.4138107
9: -5.1391444, 21.2959366, -5.1839695, 21.2575417, -24.1650009, 24.2915115
10: -17.8338985, 17.5771503, -17.9866447, 17.5279121, -31.1752014, 31.4049301
11: -26.8021717, 3.5426769, -26.7913322, 3.5256524, -27.8639908, 27.8774796
12: -34.8770294, -2.3501034, -34.8868828, -2.3973875, -27.1200333, 27.2320900
13: -26.2094421, 15.7636595, -26.2461605, 15.6641531, -33.7655182, 33.9183273
14: -55.8958588, -17.5088158, -55.9791336, -17.6050262, -37.5839767, 37.8636093
15: -14.3175249, 15.5025902, -14.2948284, 15.4651203, -27.8010025, 27.8334961
16: -14.0200214, 20.8561668, -14.0649652, 20.7898788, -30.9836197, 31.1035385
17: -57.8441429, -14.3662443, -57.8791389, -14.4671745, -41.4889908, 41.7020264
18: -21.6047668, 12.1086178, -21.5560608, 12.1268234, -29.5957642, 29.4687080
19: -22.3372612, 3.5264626, -22.2431412, 3.5152359, -22.7528229, 22.6748238
20: -23.3529434, 1.3327560, -23.2425232, 1.3077626, -19.1831818, 19.0810699
21: -26.8615608, 2.3537197, -26.7735691, 2.3394642, -25.4817810, 25.4127426
22: -28.5834484, 3.2755704, -28.4514294, 3.3022907, -24.7049255, 24.6321030
23: -22.3464680, 5.6629853, -22.2698822, 5.6621442, -22.0214958, 21.9267578
24: -18.3843040, 9.3946543, -18.2491188, 9.3880301, -22.8456116, 22.6979485
25: -23.8695984, 5.3284163, -23.7980766, 5.3282347, -24.3506050, 24.3213310
26: -41.1153069, -0.5311933, -41.0177994, -0.5336099, -30.6032486, 30.4830856
27: -21.6490898, 8.5320244, -21.5395432, 8.6001997, -26.5214233, 26.3095016
28: -24.1824303, 6.0056973, -24.0720425, 6.0272098, -21.9755859, 21.8420029
29: -27.9252090, -0.2549925, -27.8158798, -0.2075047, -23.9558640, 23.9199219
30: -28.1790142, 3.7070017, -28.0835762, 3.6973665, -26.1006546, 26.0338402
31: -22.7066975, 4.9996977, -22.6506386, 4.9869618, -25.0217896, 24.9746628
32: -23.9330711, 2.2949820, -23.8974266, 2.2937286, -21.3461838, 21.2895851
33: -36.4509048, 3.6123490, -36.3435211, 3.6102099, -33.3519974, 33.2385559
34: -37.8927002, -4.7775826, -37.7950668, -4.7680955, -27.7882233, 27.6525955
35: -32.9509811, 0.2660470, -32.8550034, 0.2886124, -28.1970978, 28.0641937
36: -36.8826447, -0.6855950, -36.7793770, -0.6443567, -29.0917435, 28.9186707
37: -44.5967140, -1.7297463, -44.4951096, -1.6719165, -38.8633118, 38.6771011
38: -43.9922676, 2.8348861, -43.8795280, 2.8772116, -40.7538757, 40.5607834
39: -43.5825310, 2.9778776, -43.4875946, 2.9274716, -41.3026733, 41.2332687
40: -32.7241211, -0.0139928, -32.6826401, 0.0246625, -31.0465240, 30.9593735
41: -20.7453136, 7.2660961, -20.6872940, 7.3136158, -26.5554276, 26.3961105
42: -22.9777069, -0.2169628, -22.9643898, -0.2146835, -18.4376106, 18.4474449

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 974
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
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5213971, upper bound: 11.5363242
time: 35.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5355202, upper bound: 11.5363242
time: 32.46 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.3280611, 19.0618515, -9.1967649, 18.9769707, -25.1431274, 25.1174316
1: -1.1991429, 22.8626366, -1.1169920, 22.7645645, -19.6697044, 19.7034302
2: -1.6051064, 20.9596901, -1.5521441, 20.8796921, -17.1908302, 17.2253113
3: -9.3288183, 16.5460377, -9.2856874, 16.4311619, -21.8961639, 21.9713936
4: -3.1285620, 22.2610397, -3.0507140, 22.2072983, -21.6703758, 21.6426163
5: -7.8193755, 20.6755238, -7.7467861, 20.5480576, -23.6465378, 23.7093735
6: -28.8142776, -1.3857450, -28.7374992, -1.4736328, -23.0938644, 23.0947571
7: -7.6817722, 21.6966572, -7.6057281, 21.5910301, -23.4985504, 23.5432053
8: -14.7575302, 14.8024731, -14.6620512, 14.6765776, -26.3574715, 26.3774033
9: -5.1790628, 21.3087978, -5.1084900, 21.2112122, -24.1642532, 24.2358932
10: -17.8601189, 17.5509071, -17.7726307, 17.3661499, -31.0594101, 31.1739807
11: -26.8059883, 3.5569496, -26.7137260, 3.5031433, -27.8501663, 27.8075409
12: -34.8861389, -2.3322763, -34.8636246, -2.4314179, -27.0993958, 27.2071037
13: -26.2556591, 15.8100920, -26.2265511, 15.6313210, -33.7735596, 33.9442673
14: -55.9093819, -17.5289116, -55.8143921, -17.7281609, -37.4943085, 37.7041473
15: -14.3430462, 15.5112648, -14.2411480, 15.4511919, -27.8124390, 27.7717819
16: -14.0525475, 20.8860989, -13.9766417, 20.7469692, -30.9691772, 31.0522079
17: -57.8454819, -14.3595028, -57.7694550, -14.5209026, -41.4262848, 41.6255417
18: -21.6532402, 12.1477633, -21.5038109, 12.1017437, -29.5839539, 29.4501572
19: -22.3696632, 3.5633612, -22.2026901, 3.5141959, -22.7833557, 22.6491051
20: -23.3824654, 1.3618479, -23.2164917, 1.3027134, -19.2105522, 19.0715294
21: -26.8879604, 2.3876221, -26.7143307, 2.3296704, -25.4944305, 25.3679962
22: -28.6340065, 3.3198526, -28.4195156, 3.2790308, -24.7672882, 24.5711212
23: -22.3883057, 5.7088313, -22.2367344, 5.6589589, -22.0563545, 21.9258728
24: -18.4293098, 9.4352160, -18.2222786, 9.3904657, -22.8925705, 22.7034454
25: -23.9138794, 5.3760204, -23.7682743, 5.3151760, -24.3871346, 24.2886581
26: -41.1777039, -0.4826007, -40.9691086, -0.5567865, -30.6539001, 30.4591522
27: -21.6774349, 8.5745077, -21.4779320, 8.5352802, -26.4831238, 26.2940369
28: -24.2290745, 6.0583634, -24.0480061, 6.0024290, -22.0127640, 21.8482895
29: -27.9683647, -0.2118636, -27.7754478, -0.2307925, -24.0187302, 23.8411865
30: -28.2033520, 3.7401407, -28.0569668, 3.6755862, -26.0987778, 26.0085297
31: -22.7409935, 5.0411320, -22.5890980, 4.9814825, -25.0526581, 24.9465027
32: -23.9478226, 2.3210251, -23.8644943, 2.2464364, -21.3040314, 21.2869644
33: -36.4523315, 3.6199050, -36.2745819, 3.5077591, -33.2369766, 33.1843185
34: -37.8851700, -4.7741666, -37.7338142, -4.8724380, -27.6695633, 27.6011124
35: -32.9437714, 0.2730703, -32.7765503, 0.1680937, -28.0691528, 28.0028458
36: -36.8667068, -0.6776080, -36.6824722, -0.7752914, -28.9491577, 28.8508911
37: -44.5888748, -1.7282639, -44.3797760, -1.7939181, -38.7136383, 38.5923920
38: -43.9828186, 2.8506083, -43.7576866, 2.7113628, -40.5929565, 40.4775009
39: -43.6062698, 2.9952674, -43.4305611, 2.8559370, -41.2368317, 41.1886292
40: -32.7464066, 0.0161228, -32.6227798, -0.0630684, -30.9710236, 30.9299850
41: -20.7369366, 7.2764215, -20.6135120, 7.2182722, -26.4470062, 26.3423386
42: -22.9844151, -0.2043500, -22.9596462, -0.2403891, -18.4143295, 18.4336891

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5334446, upper bound: 11.5375497
time: 33.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5475737, upper bound: 11.5375497
time: 39.54 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.3325338, 19.0623398, -9.2510967, 19.0001221, -25.1772079, 25.1760406
1: -1.2000837, 22.8633308, -1.1495132, 22.7957191, -19.7138176, 19.7381287
2: -1.6074657, 20.9629936, -1.6038766, 20.8946304, -17.2485962, 17.2538986
3: -9.3302965, 16.5469780, -9.3100767, 16.4461098, -21.9360542, 21.9915085
4: -3.1315107, 22.2620811, -3.0970702, 22.2150517, -21.7223663, 21.6864586
5: -7.8228769, 20.6784210, -7.7963433, 20.5657883, -23.6781235, 23.7448120
6: -28.8269882, -1.3837962, -28.7900486, -1.3953581, -23.1870117, 23.1356659
7: -7.6852608, 21.6988754, -7.6516876, 21.6063499, -23.5377693, 23.5877380
8: -14.7585754, 14.8073578, -14.7390490, 14.7066965, -26.4222717, 26.4435425
9: -5.1810665, 21.3184891, -5.1943064, 21.2578278, -24.2095947, 24.3325882
10: -17.8626156, 17.5934505, -17.9934845, 17.5289688, -31.2022934, 31.4346542
11: -26.8083458, 3.5603409, -26.7915573, 3.5296717, -27.8796158, 27.8993454
12: -34.8877182, -2.3270049, -34.8896217, -2.3925042, -27.1343536, 27.2656708
13: -26.2598190, 15.8131533, -26.2604046, 15.6652508, -33.8106918, 33.9858017
14: -55.9135590, -17.4959278, -55.9816933, -17.6020527, -37.6018600, 37.9049530
15: -14.3467159, 15.5140972, -14.2999039, 15.4682369, -27.8336792, 27.8474426
16: -14.0563545, 20.8956947, -14.0743284, 20.7901077, -31.0122986, 31.1531982
17: -57.8485603, -14.3486395, -57.8801346, -14.4644756, -41.4913483, 41.7583389
18: -21.6582413, 12.1495419, -21.5572319, 12.1385813, -29.6652832, 29.5031815
19: -22.3736324, 3.5644243, -22.2440376, 3.5267682, -22.8005829, 22.7037621
20: -23.3850155, 1.3630247, -23.2429695, 1.3165736, -19.2274551, 19.1070061
21: -26.8924789, 2.3895531, -26.7741852, 2.3499553, -25.5200958, 25.4417763
22: -28.6400585, 3.3215792, -28.4520874, 3.3164966, -24.7766571, 24.6643677
23: -22.3902512, 5.7097640, -22.2703285, 5.6761193, -22.0811272, 21.9616394
24: -18.4344063, 9.4360218, -18.2501945, 9.4005938, -22.9107208, 22.7316742
25: -23.9175777, 5.3780994, -23.7990875, 5.3429050, -24.4153671, 24.3609009
26: -41.1807976, -0.4745474, -41.0187531, -0.5162544, -30.6919327, 30.5219955
27: -21.6922512, 8.5769291, -21.5403900, 8.6131716, -26.5785141, 26.3490753
28: -24.2333965, 6.0605178, -24.0725708, 6.0439491, -22.0450516, 21.8811646
29: -27.9745903, -0.2105603, -27.8161831, -0.1944287, -24.0182571, 23.9528046
30: -28.2066116, 3.7431386, -28.0839329, 3.7068369, -26.1369209, 26.0621834
31: -22.7450466, 5.0428877, -22.6519508, 4.9995685, -25.0746460, 25.0151253
32: -23.9575005, 2.3230238, -23.9036293, 2.2967193, -21.3679199, 21.3228951
33: -36.4688759, 3.6211977, -36.3463631, 3.6113629, -33.3711700, 33.2542648
34: -37.9020576, -4.7713504, -37.7957077, -4.7673097, -27.8013229, 27.6593628
35: -32.9643478, 0.2742805, -32.8564758, 0.2901535, -28.2142410, 28.0745850
36: -36.8925552, -0.6760521, -36.7804756, -0.6419988, -29.1092300, 28.9337082
37: -44.6180038, -1.7275681, -44.4983559, -1.6721153, -38.8813171, 38.6966400
38: -44.0125122, 2.8534327, -43.8817291, 2.8813796, -40.7915955, 40.5912399
39: -43.6197739, 2.9971180, -43.4957161, 2.9276323, -41.3358307, 41.2572098
40: -32.7608795, 0.0168667, -32.6900024, 0.0260789, -31.0795441, 30.9972649
41: -20.7564621, 7.2781997, -20.6898804, 7.3155851, -26.5677490, 26.4101410
42: -22.9861984, -0.2019610, -22.9667797, -0.2125514, -18.4457550, 18.4658890

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1521
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5446359, upper bound: 11.5375497
time: 30.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5587628, upper bound: 11.5375497
time: 35.21 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.3091965, 19.0521946, -9.3109989, 19.0119553, -25.1604958, 25.1824188
1: -1.1851077, 22.8406563, -1.1913176, 22.7974472, -19.6984901, 19.7244339
2: -1.5886972, 20.9404907, -1.6019280, 20.8970375, -17.1989441, 17.2431259
3: -9.3033361, 16.5144310, -9.3293962, 16.4595909, -21.9208679, 21.9940872
4: -3.0955596, 22.2429352, -3.1186910, 22.2164574, -21.6545868, 21.6919594
5: -7.8049164, 20.6404819, -7.8194976, 20.5834389, -23.6794052, 23.7392807
6: -28.7981052, -1.3919172, -28.7694073, -1.3983927, -23.1534729, 23.1279907
7: -7.6556358, 21.6584435, -7.6727643, 21.6172562, -23.5114212, 23.5627823
8: -14.7402401, 14.7807293, -14.7720432, 14.7299786, -26.4015732, 26.4570007
9: -5.1622305, 21.2884502, -5.1733847, 21.2378674, -24.1713715, 24.2555962
10: -17.8637238, 17.5382996, -17.8632050, 17.3964977, -31.0987396, 31.2442474
11: -26.8053055, 3.5536323, -26.7371349, 3.5430355, -27.8835983, 27.8224487
12: -34.8801270, -2.3369136, -34.8768082, -2.3759732, -27.1570053, 27.2090683
13: -26.2094879, 15.7745352, -26.2289848, 15.6826897, -33.7963104, 33.9091339
14: -55.9205933, -17.5368233, -55.9040947, -17.6861687, -37.6576462, 37.7590790
15: -14.3482571, 15.5021486, -14.3428926, 15.4940300, -27.8604584, 27.8523560
16: -14.0349398, 20.8473415, -14.0246906, 20.7652054, -30.9670715, 31.0493927
17: -57.8637238, -14.3699913, -57.8390999, -14.4771547, -41.5763321, 41.6545029
18: -21.6056271, 12.1239033, -21.5536022, 12.1408968, -29.5773087, 29.4923096
19: -22.3384247, 3.5463839, -22.2513542, 3.5646882, -22.7969360, 22.6915436
20: -23.3526993, 1.3541327, -23.2692146, 1.3615060, -19.2008896, 19.1238747
21: -26.8627243, 2.3743696, -26.7667160, 2.3868024, -25.5200348, 25.4141922
22: -28.5805092, 3.2904992, -28.4643936, 3.3161657, -24.7243500, 24.6022720
23: -22.3471985, 5.6777291, -22.2676315, 5.6929674, -22.0325546, 21.9369125
24: -18.3815880, 9.4148178, -18.2738686, 9.4387760, -22.8638611, 22.7433319
25: -23.8685265, 5.3478041, -23.7967529, 5.3686771, -24.3775558, 24.2956467
26: -41.1148071, -0.5156131, -41.0105286, -0.5052509, -30.6066666, 30.4902496
27: -21.6381531, 8.5443478, -21.5093231, 8.5659046, -26.4719391, 26.3108978
28: -24.1802864, 6.0273166, -24.0858326, 6.0582113, -22.0046425, 21.8717575
29: -27.9238739, -0.2509480, -27.8079643, -0.2242088, -23.9888840, 23.8539886
30: -28.1777611, 3.7248671, -28.0934868, 3.7316873, -26.1062241, 26.0263824
31: -22.7079468, 5.0202117, -22.6417732, 5.0351734, -25.0555878, 24.9838638
32: -23.9262295, 2.3140957, -23.8985901, 2.3074925, -21.3281479, 21.3196640
33: -36.4384613, 3.6610885, -36.3635712, 3.6513057, -33.3058472, 33.3105164
34: -37.8778381, -4.7417932, -37.7977753, -4.7593055, -27.7263641, 27.7008209
35: -32.9334335, 0.3108711, -32.8472404, 0.2999239, -28.1553116, 28.1097870
36: -36.8597298, -0.6446767, -36.7467499, -0.6534872, -29.0350876, 28.9426422
37: -44.5741806, -1.7007666, -44.4510498, -1.7069755, -38.7886810, 38.6819534
38: -43.9684143, 2.8870826, -43.8502579, 2.8677392, -40.6828308, 40.5941391
39: -43.5755463, 3.0269337, -43.5305595, 3.0028834, -41.2982941, 41.3219376
40: -32.7160721, 0.0010407, -32.6732025, -0.0169225, -30.9945831, 30.9708862
41: -20.7300224, 7.2858100, -20.6520157, 7.2809010, -26.5034714, 26.4282913
42: -22.9809017, -0.2128170, -22.9741535, -0.2192957, -18.4423676, 18.4417915

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1491
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 974
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5109257, upper bound: 11.5582856
time: 28.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5250514, upper bound: 11.5582856
time: 36.95 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.3136978, 19.0526619, -9.3653011, 19.0350800, -25.1946182, 25.2410049
1: -1.1860523, 22.8413353, -1.2238359, 22.8286209, -19.7425766, 19.7591515
2: -1.5910633, 20.9438076, -1.6537032, 20.9119911, -17.2567329, 17.2717209
3: -9.3048115, 16.5154266, -9.3537159, 16.4745407, -21.9607468, 22.0142097
4: -3.0984535, 22.2439880, -3.1650910, 22.2242203, -21.7065430, 21.7358322
5: -7.8083954, 20.6433372, -7.8690543, 20.6011391, -23.7109909, 23.7747269
6: -28.8108120, -1.3899908, -28.8219376, -1.3201718, -23.2466507, 23.1688881
7: -7.6591558, 21.6606503, -7.7187529, 21.6325569, -23.5506287, 23.6072998
8: -14.7413063, 14.7856331, -14.8490534, 14.7601204, -26.4663506, 26.5231934
9: -5.1641960, 21.2981148, -5.2592072, 21.2844429, -24.2166672, 24.3522797
10: -17.8661194, 17.5808620, -18.0841217, 17.5592537, -31.2415771, 31.5049591
11: -26.8076077, 3.5569859, -26.8149796, 3.5695729, -27.9130402, 27.9141769
12: -34.8816833, -2.3316569, -34.9027710, -2.3370810, -27.1919861, 27.2676315
13: -26.2136345, 15.7776213, -26.2628517, 15.7166882, -33.8335114, 33.9506989
14: -55.9247665, -17.5038338, -56.0713692, -17.5601330, -37.7651520, 37.9597855
15: -14.3519239, 15.5049734, -14.4016285, 15.5111160, -27.8816986, 27.9280319
16: -14.0387783, 20.8568840, -14.1224241, 20.8083458, -31.0102005, 31.1504288
17: -57.8668175, -14.3591595, -57.9497986, -14.4207268, -41.6413116, 41.7872925
18: -21.6106758, 12.1256542, -21.6070633, 12.1777134, -29.6586304, 29.5454025
19: -22.3423862, 3.5474279, -22.2927017, 3.5772696, -22.8141708, 22.7462463
20: -23.3552475, 1.3553143, -23.2956657, 1.3753531, -19.2177773, 19.1593590
21: -26.8672733, 2.3762705, -26.8265419, 2.4070640, -25.5456848, 25.4880104
22: -28.5865936, 3.2922058, -28.4970093, 3.3536344, -24.7337341, 24.6955261
23: -22.3491211, 5.6786137, -22.3012638, 5.7101192, -22.0572929, 21.9726868
24: -18.3867188, 9.4156294, -18.3017960, 9.4489336, -22.8820343, 22.7715645
25: -23.8722610, 5.3498569, -23.8275509, 5.3964443, -24.4058228, 24.3679047
26: -41.1179352, -0.5076227, -41.0602264, -0.4646339, -30.6447144, 30.5530930
27: -21.6529999, 8.5467911, -21.5718021, 8.6438179, -26.5673447, 26.3659821
28: -24.1846066, 6.0295453, -24.1104469, 6.0997519, -22.0369301, 21.9046135
29: -27.9301758, -0.2496371, -27.8486481, -0.1878839, -23.9884453, 23.9655685
30: -28.1810322, 3.7278478, -28.1204872, 3.7629619, -26.1443405, 26.0800858
31: -22.7120094, 5.0219626, -22.7046623, 5.0532746, -25.0775604, 25.0525055
32: -23.9359207, 2.3161485, -23.9377213, 2.3577969, -21.3920898, 21.3555984
33: -36.4550743, 3.6623340, -36.4352684, 3.7548585, -33.4400787, 33.3804169
34: -37.8947563, -4.7389512, -37.8596458, -4.6542315, -27.8581009, 27.7590637
35: -32.9539986, 0.3120928, -32.9271507, 0.4219880, -28.3003769, 28.1815109
36: -36.8856125, -0.6432157, -36.8447609, -0.5201502, -29.1951523, 29.0254669
37: -44.6033173, -1.7001009, -44.5695992, -1.5851135, -38.9563751, 38.7862320
38: -43.9982071, 2.8899488, -43.9741669, 3.0376430, -40.8814697, 40.7078247
39: -43.5890961, 3.0288129, -43.5956268, 3.0745564, -41.3972778, 41.3905640
40: -32.7305603, 0.0018618, -32.7404556, 0.0721946, -31.1031647, 31.0381737
41: -20.7495461, 7.2875438, -20.7283936, 7.3781624, -26.6241913, 26.4960709
42: -22.9826851, -0.2104373, -22.9813061, -0.1914651, -18.4738083, 18.4739876

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1491
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 974
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
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 540
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
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1451

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5221250, upper bound: 11.5582856
time: 34.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5362537, upper bound: 11.5582856
time: 37.50 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.3701611, 19.0632210, -9.3271790, 19.0123634, -25.2260513, 25.2141151
1: -1.2283030, 22.8636208, -1.2040482, 22.7976799, -19.7365799, 19.7597809
2: -1.6249690, 20.9605980, -1.6125607, 20.8972874, -17.2290688, 17.2761345
3: -9.3484631, 16.5489235, -9.3429317, 16.4604492, -21.9555588, 22.0317192
4: -3.1561041, 22.2621460, -3.1360579, 22.2165337, -21.7053909, 21.7251358
5: -7.8481083, 20.6776085, -7.8322434, 20.5846062, -23.7117767, 23.7828369
6: -28.8176517, -1.3596854, -28.7753334, -1.3945789, -23.1703796, 23.1631050
7: -7.7092171, 21.6978550, -7.6893520, 21.6177750, -23.5537796, 23.6180954
8: -14.8007183, 14.8049736, -14.7895708, 14.7305136, -26.4542694, 26.4867859
9: -5.2041197, 21.3110390, -5.1837072, 21.2381916, -24.2160034, 24.2966690
10: -17.8923950, 17.5545845, -17.8700600, 17.3974438, -31.1257782, 31.2739868
11: -26.8114300, 3.5712895, -26.7373619, 3.5470638, -27.8992538, 27.8442993
12: -34.8908424, -2.3138413, -34.8795433, -2.3710780, -27.1713409, 27.2426872
13: -26.2598152, 15.8240185, -26.2432404, 15.6838102, -33.8415146, 33.9766464
14: -55.9382401, -17.5239201, -55.9066315, -17.6832504, -37.6755219, 37.8003769
15: -14.3774052, 15.5136642, -14.3479538, 15.4971333, -27.8930893, 27.8662643
16: -14.0712881, 20.8867722, -14.0340576, 20.7653923, -30.9957657, 31.0990372
17: -57.8681526, -14.3524122, -57.8400917, -14.4744358, -41.5787048, 41.7108231
18: -21.6591015, 12.1648436, -21.5547791, 12.1526356, -29.6468353, 29.5268021
19: -22.3747654, 3.5843284, -22.2522488, 3.5762300, -22.8446732, 22.7205009
20: -23.3847809, 1.3843904, -23.2696648, 1.3703279, -19.2451477, 19.1498032
21: -26.8936024, 2.4102170, -26.7672768, 2.3972437, -25.5583420, 25.4432449
22: -28.6371269, 3.3364968, -28.4649830, 3.3303907, -24.7960777, 24.6345139
23: -22.3909683, 5.7245154, -22.2680988, 5.7068663, -22.0921631, 21.9717789
24: -18.4316673, 9.4561958, -18.2749443, 9.4513140, -22.9289703, 22.7770767
25: -23.9165306, 5.3974748, -23.7977486, 5.3833094, -24.4423447, 24.3352432
26: -41.1802483, -0.4589777, -41.0114555, -0.4879155, -30.6953659, 30.5290909
27: -21.6813354, 8.5892591, -21.5102310, 8.5788660, -26.5290146, 26.3504715
28: -24.2312450, 6.0822287, -24.0863876, 6.0749207, -22.0741348, 21.9109001
29: -27.9733276, -0.2065196, -27.8082886, -0.2111664, -24.0512886, 23.8868484
30: -28.2053452, 3.7610099, -28.0938263, 3.7412074, -26.1425056, 26.0547409
31: -22.7463036, 5.0633807, -22.6430874, 5.0477896, -25.1084290, 25.0243378
32: -23.9506893, 2.3421679, -23.9047642, 2.3105156, -21.3499146, 21.3529549
33: -36.4565048, 3.6699042, -36.3663712, 3.6524291, -33.3250656, 33.3261795
34: -37.8871727, -4.7355447, -37.7984047, -4.7585821, -27.7395020, 27.7075577
35: -32.9468079, 0.3191128, -32.8487358, 0.3015175, -28.1724472, 28.1202240
36: -36.8696671, -0.6351223, -36.7478790, -0.6511512, -29.0525818, 28.9576874
37: -44.5954895, -1.6986542, -44.4542885, -1.7072101, -38.8067322, 38.7015457
38: -43.9886551, 2.9056463, -43.8523750, 2.8717985, -40.7205658, 40.6244812
39: -43.6127853, 3.0462208, -43.5386429, 3.0030622, -41.3314819, 41.3459473
40: -32.7528725, 0.0319343, -32.6805725, -0.0155337, -31.0276718, 31.0087814
41: -20.7411728, 7.2978425, -20.6546593, 7.2828302, -26.5158005, 26.4423065
42: -22.9893856, -0.1978581, -22.9765224, -0.2171586, -18.4505081, 18.4602394

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1491
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5340331, upper bound: 11.5594421
time: 35.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5481639, upper bound: 11.5594421
time: 34.82 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 72.61 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5077801, upper bound: 11.5104686
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5077801, upper bound: 11.5215026
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5077801, upper bound: 11.5270717
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5077801, upper bound: 11.5380848
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5309873, upper bound: 11.5116241
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5309873, upper bound: 11.5227192
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5309873, upper bound: 11.5282422
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5309873, upper bound: 11.5393120
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5084779, upper bound: 11.5323201
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5084779, upper bound: 11.5434971
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5084779, upper bound: 11.5488518
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5084779, upper bound: 11.5600182
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5316438, upper bound: 11.5334295
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5316438, upper bound: 11.5446582
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5316438, upper bound: 11.5499657
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5316438, upper bound: 11.5611768
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5240288, upper bound: 11.4887218
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5240288, upper bound: 11.4997714
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5240288, upper bound: 11.5064539
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5240288, upper bound: 11.5175091
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5472419, upper bound: 11.4898627
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5472419, upper bound: 11.5009722
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5472419, upper bound: 11.5075972
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5472419, upper bound: 11.5187029
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5246966, upper bound: 11.5104733
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5246966, upper bound: 11.5216577
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5246966, upper bound: 11.5281074
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5246966, upper bound: 11.5393035
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5478627, upper bound: 11.5115618
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5478627, upper bound: 11.5227922
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5478627, upper bound: 11.5291791
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5478627, upper bound: 11.5404359
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5237622
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5237622
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5397775
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5397775
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5348720
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5348720
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5509441
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5229372, upper bound: 11.5509441
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5099927, upper bound: 11.5469114
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5241337, upper bound: 11.5469114
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5099927, upper bound: 11.5628999
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5241337, upper bound: 11.5628999
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5581073
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5241337, upper bound: 11.5581073
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5088027, upper bound: 11.5741336
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5241337, upper bound: 11.5741336
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5307281, upper bound: 11.5244318
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5448637, upper bound: 11.5244318
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5307281, upper bound: 11.5403762
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5448637, upper bound: 11.5403762
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5307281, upper bound: 11.5356412
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5448637, upper bound: 11.5356412
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5307281, upper bound: 11.5516114
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5448637, upper bound: 11.5516114
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5318518, upper bound: 11.5475498
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5459945, upper bound: 11.5475498
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5318518, upper bound: 11.5634604
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5459945, upper bound: 11.5634604
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5318518, upper bound: 11.5588269
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5459945, upper bound: 11.5588269
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5318518, upper bound: 11.5747583
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5459945, upper bound: 11.5747583
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5102903, upper bound: 11.5363242
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5244127, upper bound: 11.5363242
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5213971, upper bound: 11.5363242
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5355202, upper bound: 11.5363242
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5334446, upper bound: 11.5375497
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5475737, upper bound: 11.5375497
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5446359, upper bound: 11.5375497
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5587628, upper bound: 11.5375497
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5109257, upper bound: 11.5582856
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5250514, upper bound: 11.5582856
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5221250, upper bound: 11.5582856
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5362537, upper bound: 11.5582856
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5340331, upper bound: 11.5594421
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 72.61
Output dim: 2, lower bound: -11.5481639, upper bound: 11.5594421
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5613048, upper bound: 11.5613050
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5417593, upper bound: 11.5175923
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5529021, upper bound: 11.5175923
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5648911, upper bound: 11.5187798
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5760876, upper bound: 11.5187798
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5423392, upper bound: 11.5393648
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5535543, upper bound: 11.5393648
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5654347, upper bound: 11.5404899
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5766891, upper bound: 11.5404899
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5518971, upper bound: 11.5257083
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5518971, upper bound: 11.5398423
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5518971, upper bound: 11.5368824
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5518971, upper bound: 11.5510219
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5488235
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5629606
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5600512
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5530577, upper bound: 11.5742005
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5263182
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5404558
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5375597
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5737276, upper bound: 11.5516993
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5493827
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5635265
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5606881
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 72.61
Output dim: 2, lower bound: -11.5748330, upper bound: 11.5748328

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 45.25 + 7190.48 = 7235.73 seconds
