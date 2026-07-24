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
execution time: IAR + RelationalAnalysis = 2.87 + 42.95 = 45.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -11.5844329, upper bound: 11.5844329

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5825690, upper bound: 11.5344348
time: 38.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5344348, upper bound: 11.5825690
time: 43.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 81.86 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 81.86
Output dim: 2, lower bound: -11.5825690, upper bound: 11.5344348
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 81.86
Output dim: 2, lower bound: -11.5344348, upper bound: 11.5825690

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2489243, 25.2447433
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7556992, 19.7503357
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2397270, 17.2244568
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0175171, 22.0127220
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7227287, 21.7117615
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7447205, 23.7335434
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2099304, 23.2107544
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5927582, 23.5869293
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5043793, 26.4956894
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2925873, 24.2917976
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2959061, 31.2874603
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8485947, 27.8573608
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2067490, 27.2136002
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8975296, 33.8877029
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8085938, 37.8086548
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8994675, 27.8932571
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0716171, 31.0703506
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6220474, 41.6265869
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5819092, 29.5910568
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7731552, 22.7800751
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1892128, 19.1998978
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4750061, 25.4901161
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7004013, 24.7118073
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9981003, 22.0084496
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8315353, 22.8368034
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3754845, 24.3868980
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5662231, 30.5849991
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4487076, 26.4597626
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9370346, 21.9541626
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9487267, 23.9568100
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0564270, 26.0737228
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0816727, 25.0845070
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3966217, 21.3997192
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4091187, 33.4126892
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7203903, 27.7373352
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1670380, 28.1740723
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0329895, 29.0436478
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7926025, 38.8040771
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7110138, 40.7210159
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4122314, 41.4122391
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0562057, 31.0588150
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5219650, 26.5246506
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4632111, 18.4660683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1653

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5800955, upper bound: 11.5206858
time: 31.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5688368, upper bound: 11.5319730
time: 37.86 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2447433, 25.2489243
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7503357, 19.7556992
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2244530, 17.2397308
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0127182, 22.0175171
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7117577, 21.7227249
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7335434, 23.7447205
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2107544, 23.2099266
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5869293, 23.5927582
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4956894, 26.5043793
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2917938, 24.2925873
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2874603, 31.2959061
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8573685, 27.8486023
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2136002, 27.2067490
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8877106, 33.8975296
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8086548, 37.8086014
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8932571, 27.8994751
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0703506, 31.0716171
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6265793, 41.6220474
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5910568, 29.5819092
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7800751, 22.7731552
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1999016, 19.1892128
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4901199, 25.4750023
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7118073, 24.7004013
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0084534, 21.9981041
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8367996, 22.8315392
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3868980, 24.3754883
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5850067, 30.5662308
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4597626, 26.4487076
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9541626, 21.9370346
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9568138, 23.9487267
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0737228, 26.0564270
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0845032, 25.0816727
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3997192, 21.3966179
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4126892, 33.4091110
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7373352, 27.7203903
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1740723, 28.1670380
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0436478, 29.0329895
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8040771, 38.7926025
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7210083, 40.7110138
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4122314, 41.4122391
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0588150, 31.0562057
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5246506, 26.5219650
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4660721, 18.4632111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1653

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5319730, upper bound: 11.5688368
time: 36.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5206858, upper bound: 11.5800955
time: 34.95 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 73.94 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 73.94
Output dim: 2, lower bound: -11.5800955, upper bound: 11.5206858
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 73.94
Output dim: 2, lower bound: -11.5688368, upper bound: 11.5319730
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 73.94
Output dim: 2, lower bound: -11.5319730, upper bound: 11.5688368
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 73.94
Output dim: 2, lower bound: -11.5206858, upper bound: 11.5800955

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2442780, 25.2414856
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7527733, 19.7479515
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2520218, 17.2183342
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0147209, 22.0023499
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7266006, 21.7106781
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7414551, 23.7216110
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2042084, 23.2080498
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5918350, 23.5795212
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5128098, 26.4886627
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2888184, 24.2868156
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2718430, 31.2370911
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8418045, 27.8497009
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2054214, 27.2160149
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8957748, 33.8865662
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7809525, 37.7535629
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8988953, 27.8927231
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0640106, 31.0532837
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6210938, 41.6256866
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5856171, 29.5902328
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7706528, 22.7801018
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1817474, 19.1915169
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4693146, 25.4839859
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6979065, 24.7269897
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9947243, 22.0028229
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8308182, 22.8356667
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3732033, 24.3903999
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5622406, 30.5743332
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4328995, 26.4492569
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9266434, 21.9547729
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9460716, 23.9725952
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0561066, 26.0747986
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0793076, 25.0816956
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3812714, 21.3908768
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3938904, 33.4033051
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6962280, 27.7243042
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1432190, 28.1626892
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9936066, 29.0248337
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7504425, 38.7838745
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6845703, 40.7082825
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4056091, 41.4074402
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0489731, 31.0555649
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.4962540, 26.5121765
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4631729, 18.4660873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1766

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5777087, upper bound: 11.5072917
time: 41.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5659397, upper bound: 11.5184693
time: 34.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2414856, 25.2442780
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7479515, 19.7527771
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2183304, 17.2520218
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0023460, 22.0147209
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7106781, 21.7266006
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7216110, 23.7414551
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2080536, 23.2042046
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5795212, 23.5918388
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4886551, 26.5128021
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2868195, 24.2888184
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2370911, 31.2718506
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8497009, 27.8418045
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2160110, 27.2054214
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8865662, 33.8957748
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7535782, 37.7809601
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8927231, 27.8988953
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0532913, 31.0640106
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6256866, 41.6210938
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5902328, 29.5856171
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7800980, 22.7706528
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1915131, 19.1817436
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4839859, 25.4693184
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7269897, 24.6979065
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0028267, 21.9947243
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8356628, 22.8308182
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3903999, 24.3731995
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5743332, 30.5622406
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4492569, 26.4328995
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9547729, 21.9266472
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9725914, 23.9460678
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0747986, 26.0561066
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0816879, 25.0793076
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3908768, 21.3812752
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4033051, 33.3938828
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7243042, 27.6962280
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1626892, 28.1432190
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0248337, 28.9936066
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7838745, 38.7504425
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7082825, 40.6845703
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4074402, 41.4056015
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0555649, 31.0489807
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5121765, 26.4962540
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4660873, 18.4631767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1766

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5184693, upper bound: 11.5659397
time: 29.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5072917, upper bound: 11.5777087
time: 34.20 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 66.73 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 66.73
Output dim: 2, lower bound: -11.5777087, upper bound: 11.5072917
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 66.73
Output dim: 2, lower bound: -11.5659397, upper bound: 11.5184693
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 66.73
Output dim: 2, lower bound: -11.5184693, upper bound: 11.5659397
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 66.73
Output dim: 2, lower bound: -11.5072917, upper bound: 11.5777087

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2450180, 25.2409477
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7523880, 19.7473602
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2497597, 17.2133484
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0152359, 22.0014114
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7250786, 21.7078819
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7412224, 23.7187653
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2035370, 23.2078438
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5913544, 23.5787125
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5111008, 26.4848633
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2887039, 24.2858086
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2716217, 31.2368164
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8400040, 27.8509674
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.1980515, 27.2135849
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8933334, 33.8850784
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7804642, 37.7531052
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8975601, 27.8887787
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0630798, 31.0519257
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6206970, 41.6252670
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5848007, 29.5896912
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7705841, 22.7800179
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1813889, 19.1918526
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4688568, 25.4844742
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6978683, 24.7269745
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9946098, 22.0026474
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8307800, 22.8356018
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3723488, 24.3898239
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5611267, 30.5734940
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4317245, 26.4476318
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9262390, 21.9542809
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9458389, 23.9721375
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0544205, 26.0735397
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0789337, 25.0812416
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3792572, 21.3913956
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3922424, 33.4022293
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6917343, 27.7225647
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1397552, 28.1610031
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9905624, 29.0243530
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7476501, 38.7849121
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6793823, 40.7083282
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4038391, 41.4077454
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0481491, 31.0561142
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.4956970, 26.5127487
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4623184, 18.4661713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5765069, upper bound: 11.4980562
time: 44.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5685762, upper bound: 11.5061030
time: 30.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2409439, 25.2450104
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7473602, 19.7523842
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2133522, 17.2497559
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0014114, 22.0152359
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7078819, 21.7250786
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7187614, 23.7412262
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2078400, 23.2035332
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5787048, 23.5913582
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4848633, 26.5111008
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2858124, 24.2887039
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2368164, 31.2716217
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8509674, 27.8400040
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2135849, 27.1980515
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8850784, 33.8933334
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7530899, 37.7804642
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8887787, 27.8975525
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0519257, 31.0630798
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6252594, 41.6206970
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5896912, 29.5848007
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7800140, 22.7705803
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1918488, 19.1813889
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4844742, 25.4688530
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7269745, 24.6978683
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0026512, 21.9946060
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8356018, 22.8307800
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3898277, 24.3723488
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5734940, 30.5611267
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4476318, 26.4317245
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9542847, 21.9262428
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9721375, 23.9458351
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0735397, 26.0544205
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0812378, 25.0789375
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3913956, 21.3792610
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4022217, 33.3922348
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7225647, 27.6917343
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1610031, 28.1397552
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0243530, 28.9905624
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7849121, 38.7476425
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7083282, 40.6793747
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4077454, 41.4038239
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0561142, 31.0481567
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5127563, 26.4956970
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4661713, 18.4623260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5061030, upper bound: 11.5685762
time: 32.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4980562, upper bound: 11.5765069
time: 35.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 70.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 70.62
Output dim: 2, lower bound: -11.5765069, upper bound: 11.4980562
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 70.62
Output dim: 2, lower bound: -11.5685762, upper bound: 11.5061030
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 70.62
Output dim: 2, lower bound: -11.5061030, upper bound: 11.5685762
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 70.62
Output dim: 2, lower bound: -11.4980562, upper bound: 11.5765069

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2441978, 25.2390099
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7521935, 19.7469139
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2490959, 17.2115440
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0144348, 21.9988251
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7250099, 21.7082863
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7407570, 23.7183685
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2014694, 23.2085495
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5904083, 23.5781479
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5107193, 26.4846268
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2880402, 24.2852516
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2713318, 31.2367401
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8394699, 27.8508606
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.1983032, 27.2133942
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8925400, 33.8778458
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7915955, 37.7520447
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8954010, 27.8874283
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0605774, 31.0509720
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6213837, 41.6238632
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5847931, 29.5896835
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7706528, 22.7789154
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1805573, 19.1914291
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4689713, 25.4839973
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6975708, 24.7262421
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9957466, 22.0005341
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8293839, 22.8335190
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3723869, 24.3894272
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5610580, 30.5734558
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4310760, 26.4475327
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9262009, 21.9542427
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9448700, 23.9709358
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0514755, 26.0725212
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0806732, 25.0799561
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3788757, 21.3904953
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3885040, 33.3978729
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6898880, 27.7204361
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1354828, 28.1557007
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9886169, 29.0210114
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7471619, 38.7860031
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6778564, 40.6989975
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4004211, 41.3978653
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0476303, 31.0616226
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.4954300, 26.5134125
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4608002, 18.4646912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5551421, upper bound: 11.4971471
time: 35.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5755844, upper bound: 11.4765816
time: 33.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2390099, 25.2441940
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7469139, 19.7521973
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2115440, 17.2490997
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -21.9988251, 22.0144348
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7082863, 21.7250099
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7183647, 23.7407608
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2085495, 23.2014694
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5781555, 23.5904045
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4846268, 26.5107193
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2852478, 24.2880440
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2367401, 31.2713318
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8508682, 27.8394775
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2133942, 27.1983032
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8778458, 33.8925400
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7520447, 37.7916031
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8874283, 27.8954010
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0509720, 31.0605774
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6238556, 41.6213837
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5896835, 29.5847931
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7789230, 22.7706566
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1914368, 19.1805573
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4839935, 25.4689713
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7262421, 24.6975746
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0005379, 21.9957466
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8335190, 22.8293800
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3894310, 24.3723907
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5734558, 30.5610580
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4475327, 26.4310760
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9542389, 21.9262009
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9709396, 23.9448700
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0725174, 26.0514793
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0799561, 25.0806732
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3904953, 21.3788757
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3978729, 33.3885040
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7204361, 27.6898880
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1557007, 28.1354828
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0210190, 28.9886169
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7860107, 38.7471619
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6990051, 40.6778488
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3978577, 41.4004288
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0616226, 31.0476379
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5134125, 26.4954300
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4646912, 18.4608002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4765816, upper bound: 11.5755844
time: 38.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.4971471, upper bound: 11.5551421
time: 26.67 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 67.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 67.37
Output dim: 2, lower bound: -11.5551421, upper bound: 11.4971471
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.37
Output dim: 2, lower bound: -11.5755844, upper bound: 11.4765816
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.37
Output dim: 2, lower bound: -11.4765816, upper bound: 11.5755844
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 67.37
Output dim: 2, lower bound: -11.4971471, upper bound: 11.5551421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2063408, 25.1905098
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7187843, 19.7039413
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2384338, 17.1976395
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0180550, 22.0029144
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7205772, 21.6998711
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7261963, 23.6991768
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1927338, 23.1995735
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5783920, 23.5623932
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4894791, 26.4566345
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2717667, 24.2654877
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2728729, 31.2381058
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8318787, 27.8471680
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.1956940, 27.2171135
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8840561, 33.8803940
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8239899, 37.8001251
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8803711, 27.8689270
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0497665, 31.0266571
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6479034, 41.6593933
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5867767, 29.5916252
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7640305, 22.7731247
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1338730, 19.1553421
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4510574, 25.4703979
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6625748, 24.6997681
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9759140, 21.9849510
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7944412, 22.8065643
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3483582, 24.3740044
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5218277, 30.5429382
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4353104, 26.4512711
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9101410, 21.9407692
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9456787, 23.9719772
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0212440, 26.0540504
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0590134, 25.0624161
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3483963, 21.3667068
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3047791, 33.3341751
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6219711, 27.6686401
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0897293, 28.1209869
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9560699, 28.9963226
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7499237, 38.7886658
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6255951, 40.6593399
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3185883, 41.3357849
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0466537, 31.0605087
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5129166, 26.5254822
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4631233, 18.4669342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5557994, upper bound: 11.4751709
time: 34.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5742306, upper bound: 11.4567760
time: 37.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1905174, 25.2063408
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7039452, 19.7187805
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.1976395, 17.2384300
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0029106, 22.0180511
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.6998711, 21.7205772
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.6991806, 23.7261963
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1995773, 23.1927299
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5623932, 23.5783920
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4566345, 26.4894714
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2654877, 24.2717667
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2381058, 31.2728729
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8471680, 27.8318787
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2171173, 27.1956940
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8803940, 33.8840561
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8001251, 37.8239899
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8689346, 27.8803711
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0266495, 31.0497589
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6593933, 41.6479034
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5916290, 29.5867767
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7731247, 22.7640305
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1553421, 19.1338692
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4703979, 25.4510536
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6997681, 24.6625748
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9849472, 21.9759102
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8065643, 22.7944374
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3740082, 24.3483543
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5429382, 30.5218353
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4512711, 26.4353104
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9407654, 21.9101410
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9719772, 23.9456825
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0540504, 26.0212440
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0624161, 25.0590172
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3667068, 21.3483963
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3341675, 33.3047714
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6686401, 27.6219788
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1209869, 28.0897293
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9963226, 28.9560699
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7886658, 38.7499237
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6593323, 40.6255951
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3357849, 41.3185959
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0605087, 31.0466537
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5254822, 26.5129166
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4669304, 18.4631271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4567760, upper bound: 11.5742306
time: 35.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.4751709, upper bound: 11.5557994
time: 35.40 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 73.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 73.36
Output dim: 2, lower bound: -11.5557994, upper bound: 11.4751709
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 73.36
Output dim: 2, lower bound: -11.5742306, upper bound: 11.4567760
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 73.36
Output dim: 2, lower bound: -11.4567760, upper bound: 11.5742306
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 73.36
Output dim: 2, lower bound: -11.4751709, upper bound: 11.5557994

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2062454, 25.1903915
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7215347, 19.7051926
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2390900, 17.1979980
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0151520, 21.9986458
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7230988, 21.7012749
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7261429, 23.6989899
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1905594, 23.1960373
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5799255, 23.5613022
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4904900, 26.4563828
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2727127, 24.2658348
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2695389, 31.2335892
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8297348, 27.8452682
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.1967010, 27.2191124
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8539124, 33.8465958
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8243332, 37.8008728
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8780899, 27.8665619
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0504990, 31.0253983
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6507568, 41.6647491
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5613785, 29.5709610
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7641144, 22.7739868
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1180496, 19.1420174
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4505844, 25.4700241
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6641617, 24.7040634
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9749451, 21.9841652
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7888794, 22.8020134
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3455009, 24.3717346
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5158920, 30.5380859
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4173660, 26.4360886
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9085541, 21.9394646
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9480209, 23.9788246
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0150223, 26.0486336
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0539703, 25.0584106
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3485184, 21.3667412
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2817535, 33.3087769
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6078873, 27.6558685
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0513687, 28.0830917
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9365311, 28.9809341
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7347107, 38.7763443
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6184235, 40.6536331
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3003998, 41.3156281
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0507965, 31.0652771
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5106354, 26.5223160
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4629822, 18.4658470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5730937, upper bound: 11.4522206
time: 30.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5701938, upper bound: 11.4552565
time: 30.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1903915, 25.2062378
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7051926, 19.7215309
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.1979980, 17.2390938
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -21.9986420, 22.0151520
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7012787, 21.7231026
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.6989899, 23.7261429
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1960373, 23.1905632
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5613022, 23.5799255
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4563789, 26.4904861
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2658310, 24.2727165
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2335892, 31.2695389
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8452682, 27.8297272
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2191162, 27.1966972
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8466034, 33.8539047
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8008652, 37.8243332
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8665619, 27.8780899
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0253983, 31.0504990
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6647491, 41.6507568
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5709610, 29.5613785
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7739868, 22.7641106
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1420212, 19.1180496
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4700241, 25.4505844
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7040634, 24.6641617
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9841614, 21.9749413
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8020096, 22.7888794
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3717384, 24.3454971
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5380859, 30.5158920
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4360886, 26.4173660
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9394684, 21.9085617
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9788208, 23.9480209
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0486374, 26.0150261
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0584106, 25.0539703
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3667450, 21.3485184
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3087769, 33.2817535
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6558685, 27.6078873
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0830917, 28.0513687
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9809341, 28.9365311
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7763519, 38.7347031
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6536407, 40.6184235
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3156281, 41.3003998
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0652771, 31.0507965
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5223236, 26.5106354
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4658508, 18.4629784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.4552565, upper bound: 11.5701938
time: 44.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4522206, upper bound: 11.5730937
time: 39.47 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 86.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 86.34
Output dim: 2, lower bound: -11.5730937, upper bound: 11.4522206
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 86.34
Output dim: 2, lower bound: -11.5701938, upper bound: 11.4552565
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 86.34
Output dim: 2, lower bound: -11.4552565, upper bound: 11.5701938
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 86.34
Output dim: 2, lower bound: -11.4522206, upper bound: 11.5730937

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2061691, 25.1898918
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7211266, 19.7045135
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2390251, 17.1981316
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0166397, 21.9961967
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7230225, 21.7025604
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7264481, 23.6988754
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1914215, 23.1934738
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5802765, 23.5610046
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4895287, 26.4558105
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2714615, 24.2657661
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2669678, 31.2338791
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8302994, 27.8451538
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.1973267, 27.2179794
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8554764, 33.8403168
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8250198, 37.8008118
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8746338, 27.8671494
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0479355, 31.0260315
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6524506, 41.6646423
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5594025, 29.5725784
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7640839, 22.7739639
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1174927, 19.1418495
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4513779, 25.4697609
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6641235, 24.7039948
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9747353, 21.9840050
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7887497, 22.8020477
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3468094, 24.3715744
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5138855, 30.5387650
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4164658, 26.4364471
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9077377, 21.9399567
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9466400, 23.9783707
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0147476, 26.0486069
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0555344, 25.0575600
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3495140, 21.3627434
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2822418, 33.3020554
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6067276, 27.6522141
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0518188, 28.0787048
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9376984, 28.9762421
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7346954, 38.7763062
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6204224, 40.6454926
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3026886, 41.3064575
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0507660, 31.0653152
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5106201, 26.5220718
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4636688, 18.4630966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 547

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5690691, upper bound: 11.4512366
time: 34.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5721553, upper bound: 11.4475608
time: 33.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1903915, 25.2061729
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7051926, 19.7211227
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.1979980, 17.2390289
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -21.9961929, 22.0151520
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7012787, 21.7230263
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.6988831, 23.7261429
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1934738, 23.1905632
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5610046, 23.5799255
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4563789, 26.4895248
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2658310, 24.2714577
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2335892, 31.2669678
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8451538, 27.8297272
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2179794, 27.1966972
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8403168, 33.8539047
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8008194, 37.8243332
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8665619, 27.8746338
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0253983, 31.0479355
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6646423, 41.6507568
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5709610, 29.5594025
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7739868, 22.7640800
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1418457, 19.1180496
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4697571, 25.4505844
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7039948, 24.6641617
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9840126, 21.9749413
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8020096, 22.7887497
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3715744, 24.3454971
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5380859, 30.5138855
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4360886, 26.4164658
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9394684, 21.9077415
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9788208, 23.9466400
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0486069, 26.0150261
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0575638, 25.0539703
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3627434, 21.3485184
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3020477, 33.2817535
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6522141, 27.6078873
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0787048, 28.0513687
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9762421, 28.9365311
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7763519, 38.7346878
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6454926, 40.6184235
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3064575, 41.3003998
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0652771, 31.0507660
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5220718, 26.5106354
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4630966, 18.4629784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1350

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 547

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.4475608, upper bound: 11.5721553
time: 28.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.4512366, upper bound: 11.5690691
time: 34.12 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 64.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 64.66
Output dim: 2, lower bound: -11.5690691, upper bound: 11.4512366
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 64.66
Output dim: 2, lower bound: -11.5721553, upper bound: 11.4475608
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 64.66
Output dim: 2, lower bound: -11.4475608, upper bound: 11.5721553
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 64.66
Output dim: 2, lower bound: -11.4512366, upper bound: 11.5690691

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 45.82 + 1092.14 = 1137.96 seconds
