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
execution time: IAR + RelationalAnalysis = 2.75 + 43.37 = 46.12 seconds
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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5825690, upper bound: 11.5344348
time: 37.46 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5344348, upper bound: 11.5825690
time: 41.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 79.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 79.38
Output dim: 2, lower bound: -11.5825690, upper bound: 11.5344348
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 79.38
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

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5800955, upper bound: 11.5206858
time: 31.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
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

Time for backsubstitution: 2.23 seconds

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
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5319730, upper bound: 11.5688368
time: 36.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5206858, upper bound: 11.5800955
time: 35.23 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 73.63 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 73.63
Output dim: 2, lower bound: -11.5800955, upper bound: 11.5206858
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 73.63
Output dim: 2, lower bound: -11.5688368, upper bound: 11.5319730
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 73.63
Output dim: 2, lower bound: -11.5319730, upper bound: 11.5688368
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 73.63
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

Time for backsubstitution: 2.29 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1766

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5777087, upper bound: 11.5072917
time: 41.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5659397, upper bound: 11.5184693
time: 34.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2456741, 25.2400970
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7533226, 19.7474098
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2336044, 17.2367477
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0071449, 22.0099258
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7216415, 21.7156372
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7327881, 23.7302742
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2072296, 23.2050323
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5853500, 23.5860138
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4973450, 26.5041199
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2876053, 24.2880249
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2455368, 31.2634048
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8409348, 27.8505707
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2091751, 27.2122688
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8963852, 33.8859558
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7535172, 37.7810059
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8989410, 27.8926773
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0545578, 31.0627441
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6211548, 41.6256332
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5810852, 29.5947647
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7731705, 22.7775764
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1808319, 19.1924286
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4688721, 25.4844284
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7155838, 24.7093124
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9924736, 22.0050735
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8303986, 22.8360786
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3789864, 24.3846130
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5555496, 30.5810165
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4382019, 26.4439545
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9376450, 21.9437752
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9645042, 23.9541550
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0575027, 26.0734024
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0788574, 25.0821381
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3877716, 21.3843803
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3997192, 33.3974686
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7073593, 27.7131805
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1556549, 28.1502533
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0141830, 29.0042648
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7723999, 38.7619171
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6982727, 40.6945801
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4074402, 41.4055939
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0529556, 31.0515900
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5094910, 26.4989471
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4632339, 18.4660301

Time for backsubstitution: 2.26 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5664386, upper bound: 11.5185885
time: 52.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5546468, upper bound: 11.5297894
time: 29.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2400970, 25.2456741
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7474098, 19.7533188
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2367477, 17.2336082
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0099297, 22.0071411
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7156372, 21.7216415
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7302704, 23.7327881
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2050323, 23.2072220
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5860138, 23.5853500
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5041122, 26.4973450
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2880249, 24.2876053
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2634048, 31.2455368
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8505707, 27.8409348
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2122726, 27.2091675
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8859558, 33.8963852
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7810135, 37.7535172
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8926773, 27.8989410
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0627441, 31.0545502
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6256256, 41.6211548
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5947647, 29.5810852
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7775803, 22.7731781
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1924286, 19.1808319
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4844284, 25.4688759
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7093124, 24.7155838
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0050774, 21.9924736
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8360825, 22.8304024
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3846092, 24.3789902
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5810165, 30.5555496
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4439545, 26.4382019
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9437714, 21.9376450
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9541588, 23.9645081
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0734024, 26.0575066
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0821381, 25.0788612
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3843765, 21.3877754
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3974609, 33.3997192
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7131805, 27.7073593
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1502533, 28.1556549
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0042648, 29.0141830
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7619171, 38.7723999
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6945801, 40.6982803
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4055939, 41.4074478
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0515976, 31.0529480
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.4989471, 26.5094910
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4660263, 18.4632339

Time for backsubstitution: 2.20 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5297894, upper bound: 11.5546468
time: 27.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5185885, upper bound: 11.5664386
time: 29.22 seconds

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

Time for backsubstitution: 2.24 seconds

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
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5184693, upper bound: 11.5659397
time: 29.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5072917, upper bound: 11.5777087
time: 33.97 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 65.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 65.97
Output dim: 2, lower bound: -11.5777087, upper bound: 11.5072917
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 65.97
Output dim: 2, lower bound: -11.5659397, upper bound: 11.5184693
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 65.97
Output dim: 2, lower bound: -11.5664386, upper bound: 11.5185885
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 65.97
Output dim: 2, lower bound: -11.5546468, upper bound: 11.5297894
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 65.97
Output dim: 2, lower bound: -11.5297894, upper bound: 11.5546468
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 65.97
Output dim: 2, lower bound: -11.5185885, upper bound: 11.5664386
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 65.97
Output dim: 2, lower bound: -11.5184693, upper bound: 11.5659397
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 65.97
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

Time for backsubstitution: 2.23 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5765069, upper bound: 11.4980562
time: 44.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5685762, upper bound: 11.5061030
time: 30.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2437363, 25.2422180
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7521820, 19.7475624
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2470360, 17.2162018
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0137863, 22.0028610
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7238045, 21.7091827
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7386131, 23.7215729
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2038422, 23.2073784
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5910187, 23.5790558
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5090103, 26.4871216
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2878113, 24.2867508
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2715759, 31.2368317
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8431549, 27.8479004
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2029877, 27.2086487
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8942947, 33.8841248
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7804947, 37.7530746
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8949509, 27.8911591
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0626450, 31.0523605
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6206665, 41.6252899
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5850677, 29.5894165
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7705688, 22.7800331
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1820831, 19.1911583
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4699402, 25.4835205
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6979065, 24.7269516
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9945564, 22.0027084
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8307495, 22.8356323
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3726692, 24.3895493
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5615158, 30.5732193
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4312820, 26.4481277
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9261932, 21.9543686
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9456100, 23.9723625
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0549088, 26.0731125
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0788498, 25.0812607
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3815613, 21.3888626
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3928223, 33.4016571
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6947556, 27.7198105
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1416016, 28.1592255
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9932022, 29.0217896
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7514801, 38.7810669
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6846008, 40.7030869
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4059143, 41.4056625
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0494003, 31.0547409
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.4968338, 26.5116196
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4632034, 18.4652367

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5647421, upper bound: 11.5092532
time: 33.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5567834, upper bound: 11.5172778
time: 44.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2464066, 25.2395554
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7529297, 19.7468185
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2313423, 17.2317619
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0076599, 22.0089912
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7201195, 21.7128410
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7325554, 23.7274284
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2065582, 23.2048264
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5848694, 23.5852013
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4956360, 26.5003204
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2874908, 24.2870216
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2453079, 31.2631302
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8391342, 27.8518372
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2017975, 27.2098351
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8939438, 33.8844681
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7530289, 37.7805405
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8975983, 27.8887329
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0536270, 31.0613785
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6207581, 41.6252060
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5802689, 29.5942154
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7731018, 22.7774925
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1804810, 19.1927681
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4684143, 25.4849167
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7155457, 24.7092972
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9923592, 22.0048981
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8303680, 22.8360176
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3781395, 24.3840370
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5544357, 30.5801773
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4370270, 26.4423370
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9372406, 21.9432831
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9642715, 23.9536972
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0558243, 26.0721359
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0784912, 25.0816879
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3857574, 21.3848953
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3980713, 33.3963928
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7028656, 27.7114334
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1521912, 28.1485596
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0111389, 29.0037842
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7695923, 38.7629547
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6930847, 40.6946182
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4056702, 41.4058990
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0521164, 31.0521469
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5089340, 26.4995193
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4623795, 18.4661102

Time for backsubstitution: 2.25 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5652361, upper bound: 11.5094059
time: 36.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5572535, upper bound: 11.5174036
time: 36.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2451401, 25.2408257
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7527237, 19.7470169
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2286263, 17.2346153
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0062103, 22.0104446
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7188530, 21.7141418
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7299461, 23.7302399
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2068634, 23.2043571
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5845337, 23.5855484
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4935455, 26.5025787
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2865982, 24.2879601
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2452621, 31.2631378
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8422928, 27.8487701
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2067337, 27.2049026
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8949051, 33.8835144
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7530594, 37.7805099
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8949966, 27.8911133
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0531921, 31.0618134
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6207275, 41.6252289
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5805359, 29.5939484
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7731018, 22.7775078
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1811676, 19.1920738
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4694977, 25.4839630
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7155762, 24.7092743
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9923058, 22.0049553
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8303375, 22.8360443
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3784599, 24.3837585
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5548325, 30.5799026
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4365845, 26.4428253
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9371948, 21.9433708
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9640503, 23.9539261
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0563126, 26.0717163
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0784073, 25.0817070
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3880615, 21.3823624
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3986511, 33.3958206
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7058868, 27.7086868
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1540451, 28.1467896
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0137711, 29.0012207
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7734375, 38.7591171
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6983185, 40.6893845
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4077454, 41.4038162
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0533676, 31.0507736
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5100632, 26.4983826
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4632645, 18.4651794

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5534543, upper bound: 11.5206036
time: 37.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5454312, upper bound: 11.5285946
time: 33.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2408218, 25.2451324
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7470169, 19.7527275
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2346153, 17.2286263
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0104446, 22.0062065
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7141380, 21.7188492
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7302361, 23.7299423
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2043610, 23.2068634
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5855408, 23.5845375
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5025787, 26.4935455
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2879639, 24.2865982
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2631378, 31.2452621
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8487701, 27.8422852
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2049026, 27.2067337
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8835144, 33.8949051
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7805252, 37.7530518
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8911209, 27.8949966
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0618134, 31.0531921
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6252289, 41.6207275
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5939484, 29.5805397
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7775116, 22.7730980
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1920776, 19.1811714
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4839630, 25.4694977
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7092743, 24.7155800
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0049553, 21.9923058
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8360443, 22.8303413
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3837547, 24.3784561
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5799026, 30.5548325
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4428253, 26.4365845
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9433746, 21.9371986
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9539261, 23.9640503
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0717163, 26.0563087
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0817032, 25.0784073
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3823624, 21.3880653
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3958130, 33.3986435
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7086868, 27.7058868
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1467896, 28.1540451
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0012207, 29.0137787
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7591248, 38.7734375
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6893921, 40.6983185
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4038086, 41.4077454
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0507736, 31.0533600
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.4983826, 26.5100632
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4651794, 18.4632568

Time for backsubstitution: 2.21 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5285946, upper bound: 11.5454312
time: 30.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5206035, upper bound: 11.5534543
time: 37.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2395554, 25.2463989
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7468185, 19.7529297
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2317619, 17.2313461
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0089951, 22.0076561
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7128410, 21.7201233
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7274284, 23.7325592
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2048187, 23.2065544
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5852051, 23.5848694
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5003204, 26.4956360
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2870178, 24.2874908
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2631302, 31.2453079
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8518295, 27.8391418
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2098389, 27.2017975
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8844681, 33.8939438
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7805557, 37.7530212
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8887329, 27.8975983
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0613785, 31.0536270
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6252136, 41.6207504
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5942154, 29.5802689
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7774963, 22.7731056
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1927643, 19.1804733
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4849167, 25.4684105
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7092972, 24.7155457
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0049019, 21.9923553
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8360138, 22.8303680
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3840370, 24.3781357
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5801773, 30.5544357
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4423370, 26.4370270
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9432831, 21.9372368
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9536972, 23.9642754
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0721359, 26.0558205
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0816879, 25.0784912
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3848953, 21.3857613
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3963928, 33.3980789
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7114334, 27.7028656
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1485596, 28.1521912
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0037842, 29.0111389
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7629547, 38.7695923
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6946106, 40.6930847
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4059143, 41.4056702
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0521469, 31.0521317
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.4995193, 26.5089340
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4661179, 18.4623833

Time for backsubstitution: 2.14 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5174036, upper bound: 11.5572535
time: 39.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5094059, upper bound: 11.5652361
time: 30.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2422256, 25.2437439
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7475662, 19.7521820
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2162056, 17.2470360
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0028610, 22.0137863
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7091866, 21.7238083
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7215767, 23.7386093
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2073822, 23.2038460
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5790558, 23.5910263
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4871216, 26.5090103
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2867508, 24.2878113
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2368317, 31.2715759
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8479004, 27.8431549
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2086487, 27.2029877
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8841248, 33.8942947
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7530594, 37.7804947
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8911591, 27.8949509
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0523605, 31.0626450
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6252899, 41.6206741
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5894165, 29.5850677
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7800293, 22.7705727
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1911621, 19.1820869
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4835205, 25.4699402
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7269516, 24.6978989
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0027046, 21.9945564
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8356323, 22.8307533
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3895454, 24.3726692
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5732193, 30.5615158
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4481277, 26.4312820
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9543610, 21.9262009
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9723587, 23.9456100
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0731125, 26.0549088
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0812607, 25.0788536
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3888626, 21.3815651
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4016571, 33.3928146
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7198105, 27.6947556
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1592255, 28.1416016
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0217896, 28.9932022
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7810669, 38.7514801
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7030945, 40.6846085
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4056702, 41.4058990
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0547409, 31.0493851
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5116196, 26.4968338
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4652328, 18.4631996

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5172778, upper bound: 11.5567834
time: 37.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5092532, upper bound: 11.5647421
time: 29.86 seconds

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

Time for backsubstitution: 2.21 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5061030, upper bound: 11.5685762
time: 32.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4980562, upper bound: 11.5765069
time: 35.55 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 70.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 70.76
Output dim: 2, lower bound: -11.5765069, upper bound: 11.4980562
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 70.76
Output dim: 2, lower bound: -11.5685762, upper bound: 11.5061030
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 70.76
Output dim: 2, lower bound: -11.5647421, upper bound: 11.5092532
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 70.76
Output dim: 2, lower bound: -11.5567834, upper bound: 11.5172778
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 70.76
Output dim: 2, lower bound: -11.5652361, upper bound: 11.5094059
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 70.76
Output dim: 2, lower bound: -11.5572535, upper bound: 11.5174036
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 70.76
Output dim: 2, lower bound: -11.5534543, upper bound: 11.5206036
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 70.76
Output dim: 2, lower bound: -11.5454312, upper bound: 11.5285946
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 70.76
Output dim: 2, lower bound: -11.5285946, upper bound: 11.5454312
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 70.76
Output dim: 2, lower bound: -11.5206035, upper bound: 11.5534543
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 70.76
Output dim: 2, lower bound: -11.5174036, upper bound: 11.5572535
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 70.76
Output dim: 2, lower bound: -11.5094059, upper bound: 11.5652361
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 70.76
Output dim: 2, lower bound: -11.5172778, upper bound: 11.5567834
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 70.76
Output dim: 2, lower bound: -11.5092532, upper bound: 11.5647421
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 70.76
Output dim: 2, lower bound: -11.5061030, upper bound: 11.5685762
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 70.76
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

Time for backsubstitution: 2.23 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5551421, upper bound: 11.4971471
time: 35.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5755844, upper bound: 11.4765816
time: 33.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2430687, 25.2401314
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7519341, 19.7471695
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2479515, 17.2126923
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0126495, 22.0006142
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7254829, 21.7078133
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7408257, 23.7182999
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2042389, 23.2057800
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5907898, 23.5777588
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5108643, 26.4844818
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2881470, 24.2851486
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2715378, 31.2365417
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8398972, 27.8504333
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.1978683, 27.2138367
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8861008, 33.8842850
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7794037, 37.7642365
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8962097, 27.8866272
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0621262, 31.0494232
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6192932, 41.6259460
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5847931, 29.5896835
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7694778, 22.7800941
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1809692, 19.1910248
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4683762, 25.4845886
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6971359, 24.7266808
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9924889, 22.0037880
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8286972, 22.8342018
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3719521, 24.3898659
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5610886, 30.5734253
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4316254, 26.4469833
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9262009, 21.9542389
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9446335, 23.9711685
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0534058, 26.0705986
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0776520, 25.0829811
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3783569, 21.3910103
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3878784, 33.3984985
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6896057, 27.7207184
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1344528, 28.1567383
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9872208, 29.0224075
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7487183, 38.7844315
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6700439, 40.7068100
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3939514, 41.4043427
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0536728, 31.0555954
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.4963608, 26.5124817
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4608383, 18.4646492

Time for backsubstitution: 2.20 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5472165, upper bound: 11.5051904
time: 25.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5676571, upper bound: 11.4846306
time: 31.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2429314, 25.2402802
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7519951, 19.7471161
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2463799, 17.2143936
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0129852, 22.0002747
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7237434, 21.7095871
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7381477, 23.7211761
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2017822, 23.2080803
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5900726, 23.5784912
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5086288, 26.4868851
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2871475, 24.2861938
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2712860, 31.2367554
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8426285, 27.8478012
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2032471, 27.2084579
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8935013, 33.8768845
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7916260, 37.7520142
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8927994, 27.8898087
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0601425, 31.0514069
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6213531, 41.6238861
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5850677, 29.5894165
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7706528, 22.7789268
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1812592, 19.1907349
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4700546, 25.4830475
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6976089, 24.7262192
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9956932, 22.0005913
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8293533, 22.8335457
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3727074, 24.3891525
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5614471, 30.5731812
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4306335, 26.4480286
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9261551, 21.9543304
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9446411, 23.9711609
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0519638, 26.0720978
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0805969, 25.0799751
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3811798, 21.3879623
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3890839, 33.3973007
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6929016, 27.7176819
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1373367, 28.1539230
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9912567, 29.0184479
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7510071, 38.7821579
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6830750, 40.6937637
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4024963, 41.3957901
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0488815, 31.0602493
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.4965591, 26.5122757
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4616776, 18.4637566

Time for backsubstitution: 2.21 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5433532, upper bound: 11.5083375
time: 33.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5638170, upper bound: 11.4877983
time: 30.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2418022, 25.2414017
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7517357, 19.7473717
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2452278, 17.2155457
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0112000, 22.0020638
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7242012, 21.7091179
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7382164, 23.7211075
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2045517, 23.2053146
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5904694, 23.5781021
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5087738, 26.4867401
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2872543, 24.2860870
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2714844, 31.2365494
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8430557, 27.8473740
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2028046, 27.2089005
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8870621, 33.8833313
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7794342, 37.7642059
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8936005, 27.8890076
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0616913, 31.0498581
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6192627, 41.6259766
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5850677, 29.5894165
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7694626, 22.7801094
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1816635, 19.1903267
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4694595, 25.4836388
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6971664, 24.7266541
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9924431, 22.0038452
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8286667, 22.8342285
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3722725, 24.3895874
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5614777, 30.5731506
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4311829, 26.4474792
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9261627, 21.9543228
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9444122, 23.9713974
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0538940, 26.0701752
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0775681, 25.0830002
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3806610, 21.3884773
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3884583, 33.3979263
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6926270, 27.7179642
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1363068, 28.1549530
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9898605, 29.0198441
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7525635, 38.7805862
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6752777, 40.7015686
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3960266, 41.4022675
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0548935, 31.0542221
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.4974899, 26.5113449
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4617233, 18.4637146

Time for backsubstitution: 2.23 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5353786, upper bound: 11.5163587
time: 43.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5558588, upper bound: 11.4958266
time: 30.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2455864, 25.2376175
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7527351, 19.7463684
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2306862, 17.2299538
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0068588, 22.0064011
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7200508, 21.7132416
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7320900, 23.7270317
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2044907, 23.2055283
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5839081, 23.5846367
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4952545, 26.5000839
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2868347, 24.2864647
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2450256, 31.2630463
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8386078, 27.8517303
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2020569, 27.2096481
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8931503, 33.8772354
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7641602, 37.7794876
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8954468, 27.8873825
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0511246, 31.0604248
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6214447, 41.6238022
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5802612, 29.5942154
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7731857, 22.7763901
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1796417, 19.1923447
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4685287, 25.4844398
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7152557, 24.7085648
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9934959, 22.0027847
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8289642, 22.8339310
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3781776, 24.3836403
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5543747, 30.5801392
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4363708, 26.4422379
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9371948, 21.9432449
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9633102, 23.9524956
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0528793, 26.0711212
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0802307, 25.0804024
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3853760, 21.3839951
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3943481, 33.3920364
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7010193, 27.7093048
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1479263, 28.1432571
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0091858, 29.0004425
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7691040, 38.7640457
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6915588, 40.6852875
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4022827, 41.3960190
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0516129, 31.0576553
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5086594, 26.5001755
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4608612, 18.4646301

Time for backsubstitution: 2.22 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5438461, upper bound: 11.5084842
time: 42.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5643198, upper bound: 11.4879590
time: 32.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2444572, 25.2387390
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7524834, 19.7466278
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2295341, 17.2311020
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0050659, 22.0081940
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7205238, 21.7127724
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7321663, 23.7269630
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2072601, 23.2027626
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5843048, 23.5842476
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4954071, 26.4999390
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2869339, 24.2863617
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2452240, 31.2628479
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8390350, 27.8513031
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2016144, 27.2100906
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8867111, 33.8836746
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7519684, 37.7916794
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8962479, 27.8865814
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0526657, 31.0588760
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6193390, 41.6258926
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5802612, 29.5942116
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7720108, 22.7775688
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1800537, 19.1919403
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4679337, 25.4850311
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7148132, 24.7090034
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9902382, 22.0060387
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8282852, 22.8346176
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3777428, 24.3840790
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5543976, 30.5801086
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4369278, 26.4416885
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9372025, 21.9432411
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9630737, 23.9527283
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0548019, 26.0691986
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0772095, 25.0834274
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3848572, 21.3845100
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3937225, 33.3926620
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7007370, 27.7095871
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1468887, 28.1442947
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0077972, 29.0018311
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7706909, 38.7624741
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6837463, 40.6930923
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3957825, 41.4024963
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0576401, 31.0516281
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5095901, 26.4992447
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4608994, 18.4645882

Time for backsubstitution: 2.25 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5358584, upper bound: 11.5164804
time: 36.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5563395, upper bound: 11.4959630
time: 45.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2443199, 25.2388878
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7525368, 19.7465706
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2279625, 17.2328072
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0054092, 22.0078545
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7187843, 21.7145424
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7294807, 23.7298431
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2048035, 23.2050629
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5835876, 23.5849838
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4931641, 26.5023422
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2859344, 24.2874031
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2449799, 31.2630615
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8417587, 27.8486633
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2069855, 27.2047119
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8941116, 33.8762741
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7641907, 37.7794571
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8928375, 27.8897705
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0506897, 31.0608597
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6214142, 41.6238251
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5805359, 29.5939445
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7731705, 22.7764053
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1803436, 19.1916504
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4696121, 25.4834900
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7152863, 24.7085419
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9934425, 22.0028419
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8289413, 22.8339577
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3784981, 24.3833618
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5547638, 30.5798645
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4359283, 26.4427338
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9371567, 21.9433327
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9630814, 23.9527245
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0533676, 26.0706978
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0801468, 25.0804214
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3876801, 21.3814621
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3949127, 33.3914642
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7040329, 27.7065582
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1497803, 28.1414795
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0118256, 28.9978790
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7729492, 38.7602005
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6968079, 40.6800461
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4043579, 41.3939438
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0528488, 31.0562820
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5097961, 26.4990463
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4617386, 18.4636993

Time for backsubstitution: 2.13 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5320113, upper bound: 11.5196834
time: 32.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5525306, upper bound: 11.4991921
time: 32.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2431908, 25.2400093
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7522774, 19.7468300
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2268181, 17.2339554
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0036163, 22.0096436
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7192574, 21.7140732
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7295494, 23.7297745
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2075653, 23.2022934
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5839691, 23.5845947
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4933167, 26.5021973
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2860413, 24.2873001
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2451782, 31.2628632
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8421860, 27.8482361
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2065506, 27.2051544
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8876724, 33.8827209
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7519989, 37.7916489
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8936462, 27.8889618
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0522385, 31.0593109
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6193237, 41.6259155
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5805359, 29.5939407
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7719955, 22.7775841
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1807480, 19.1912422
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4690170, 25.4840813
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7148438, 24.7089767
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9901924, 22.0060959
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8282547, 22.8346443
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3780632, 24.3838005
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5547943, 30.5798340
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4364777, 26.4421768
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9371567, 21.9433289
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9628525, 23.9529572
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0552902, 26.0687752
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0771255, 25.0834427
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3871613, 21.3819771
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3942871, 33.3920898
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7037506, 27.7068329
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1487427, 28.1425171
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0104370, 28.9992676
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7745361, 38.7586365
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6889954, 40.6878586
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3978577, 41.4004211
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0588760, 31.0502548
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5107269, 26.4981155
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4617767, 18.4636536

Time for backsubstitution: 2.27 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5239838, upper bound: 11.5276706
time: 36.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5445107, upper bound: 11.5071868
time: 30.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2400017, 25.2431946
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7468300, 19.7522812
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2339592, 17.2268181
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0096436, 22.0036163
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7140694, 21.7192497
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7297707, 23.7295456
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2022934, 23.2075691
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5845947, 23.5839729
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5021973, 26.4933167
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2873001, 24.2860413
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2628555, 31.2451782
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8482361, 27.8421860
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2051544, 27.2065468
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8827209, 33.8876724
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7916565, 37.7519989
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8889618, 27.8936462
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0593109, 31.0522308
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6259155, 41.6193237
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5939407, 29.5805359
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7775803, 22.7719917
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1912384, 19.1807480
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4840851, 25.4690170
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7089767, 24.7148476
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0060921, 21.9901886
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8346481, 22.8282547
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3838005, 24.3780594
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5798340, 30.5547943
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4421768, 26.4364777
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9433289, 21.9371605
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9529572, 23.9628487
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0687790, 26.0552902
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0834427, 25.0771255
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3819733, 21.3871651
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3920898, 33.3942871
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7068329, 27.7037506
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1425171, 28.1487427
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9992676, 29.0104370
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7586365, 38.7745285
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6878662, 40.6889877
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4004211, 41.3978653
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0502548, 31.0588760
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.4981155, 26.5107193
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4636536, 18.4617767

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5071868, upper bound: 11.5445107
time: 32.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5276706, upper bound: 11.5239838
time: 36.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2388878, 25.2443161
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7465706, 19.7525368
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2328072, 17.2279663
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0078506, 22.0054092
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7145424, 21.7187805
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7298470, 23.7294769
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2050629, 23.2047997
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5849762, 23.5835838
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5023422, 26.4931641
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2873993, 24.2859383
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2630615, 31.2449799
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8486633, 27.8417587
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2047119, 27.2069893
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8762741, 33.8941116
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7794495, 37.7641907
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8897629, 27.8928375
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0608597, 31.0506897
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6238251, 41.6214142
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5939484, 29.5805359
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7764053, 22.7731743
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1916504, 19.1803398
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4834900, 25.4696121
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7085419, 24.7152863
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0028419, 21.9934464
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8339615, 22.8289413
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3833656, 24.3784981
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5798645, 30.5547638
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4427338, 26.4359283
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9433365, 21.9371567
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9527206, 23.9630814
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0707016, 26.0533676
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0804214, 25.0801468
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3814621, 21.3876801
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3914642, 33.3949127
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7065582, 27.7040329
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1414795, 28.1497803
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9978790, 29.0118256
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7601929, 38.7729568
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6800537, 40.6968002
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3939514, 41.4043503
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0562820, 31.0528488
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.4990463, 26.5097961
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4636993, 18.4617348

Time for backsubstitution: 2.23 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4991921, upper bound: 11.5525306
time: 42.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5196834, upper bound: 11.5320113
time: 28.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2387352, 25.2444611
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7466316, 19.7524796
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2311058, 17.2295380
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0081940, 22.0050659
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7127724, 21.7205238
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7269630, 23.7321625
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2027588, 23.2072563
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5842438, 23.5843048
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4999390, 26.4954071
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2863617, 24.2869339
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2628479, 31.2452240
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8513031, 27.8390350
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2100906, 27.2016106
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8836746, 33.8867111
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7916870, 37.7519684
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8865814, 27.8962479
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0588760, 31.0526733
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6259003, 41.6193466
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5942078, 29.5802650
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7775650, 22.7720032
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1919403, 19.1800499
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4850311, 25.4679337
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7089996, 24.7148132
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0060387, 21.9902420
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8346176, 22.8282814
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3840752, 24.3777390
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5801086, 30.5543976
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4416885, 26.4369278
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9432373, 21.9372025
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9527283, 23.9630737
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0691986, 26.0548019
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0834274, 25.0772057
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3845062, 21.3848610
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3926697, 33.3937225
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7095871, 27.7007370
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1442947, 28.1468887
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0018387, 29.0077972
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7624817, 38.7706833
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6930847, 40.6837540
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4024963, 41.3957901
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0516281, 31.0576401
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.4992523, 26.5095901
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4645920, 18.4609032

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4959630, upper bound: 11.5563395
time: 33.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5164804, upper bound: 11.5358584
time: 32.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2376213, 25.2455826
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7463722, 19.7527390
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2299538, 17.2306862
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0064011, 22.0068588
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7132454, 21.7200508
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7270317, 23.7320938
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2055283, 23.2044868
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5846405, 23.5839157
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5000839, 26.4952545
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2864609, 24.2868309
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2630463, 31.2450256
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8517303, 27.8386078
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2096558, 27.2020531
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8772354, 33.8931503
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7794800, 37.7641602
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8873825, 27.8954468
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0604248, 31.0511246
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6238098, 41.6214371
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5942154, 29.5802650
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7763901, 22.7731819
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1923523, 19.1796417
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4844360, 25.4685287
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7085648, 24.7152519
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0027809, 21.9934959
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8339310, 22.8289680
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3836403, 24.3781776
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5801392, 30.5543747
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4422379, 26.4363708
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9432449, 21.9371948
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9524918, 23.9633102
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0711212, 26.0528793
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0804062, 25.0802307
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3839951, 21.3853760
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3920288, 33.3943481
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7093048, 27.7010193
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1432571, 28.1479263
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0004425, 29.0091858
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7640381, 38.7691116
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6852875, 40.6915588
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3960266, 41.4022751
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0576553, 31.0516129
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5001755, 26.5086594
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4646301, 18.4608612

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4879590, upper bound: 11.5643198
time: 38.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5084843, upper bound: 11.5438461
time: 34.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2414055, 25.2418060
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7473717, 19.7517357
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2155418, 17.2452316
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0020676, 22.0111961
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7091103, 21.7242088
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7211113, 23.7382126
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2053146, 23.2045517
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5781097, 23.5904617
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4867401, 26.5087738
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2860870, 24.2872543
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2365494, 31.2714920
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8473740, 27.8430557
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2089081, 27.2028008
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8833313, 33.8870621
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7642059, 37.7794342
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8890076, 27.8936005
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0498581, 31.0616913
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6259766, 41.6192703
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5894165, 29.5850639
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7801132, 22.7694664
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1903229, 19.1816635
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4836426, 25.4694595
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7266541, 24.6971664
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0038490, 21.9924393
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8342285, 22.8286667
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3895912, 24.3722725
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5731506, 30.5614777
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4474792, 26.4311829
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9543228, 21.9261627
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9713974, 23.9444122
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0701752, 26.0538940
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0830002, 25.0775719
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3884735, 21.3806648
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3979187, 33.3884583
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7179642, 27.6926270
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1549530, 28.1363068
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0198441, 28.9898605
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7805786, 38.7525711
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7015686, 40.6752777
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4022522, 41.3960190
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0542221, 31.0549011
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5113525, 26.4974899
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4637146, 18.4617195

Time for backsubstitution: 2.24 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4958266, upper bound: 11.5558588
time: 31.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5163587, upper bound: 11.5353786
time: 27.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2402763, 25.2429276
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7471123, 19.7519951
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2143974, 17.2463760
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0002747, 22.0129852
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7095833, 21.7237358
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7211800, 23.7381439
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2080841, 23.2017822
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5784912, 23.5900726
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4868851, 26.5086288
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2861938, 24.2871475
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2367477, 31.2712860
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8478012, 27.8426285
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2084656, 27.2032394
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8768845, 33.8935013
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7520142, 37.7916336
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8898087, 27.8927994
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0513992, 31.0601425
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6238861, 41.6213531
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5894165, 29.5850639
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7789230, 22.7706490
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1907349, 19.1812553
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4830475, 25.4700546
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7262192, 24.6976089
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0005913, 21.9956970
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8335419, 22.8293533
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3891487, 24.3727112
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5731812, 30.5614471
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4480286, 26.4306335
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9543304, 21.9261589
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9711609, 23.9446411
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0720978, 26.0519676
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0799789, 25.0805931
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3879623, 21.3811798
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3973083, 33.3890839
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7176819, 27.6929016
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1539230, 28.1373367
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0184479, 28.9912567
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7821655, 38.7509995
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6937561, 40.6830826
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3957825, 41.4025040
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0602493, 31.0488739
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5122757, 26.4965591
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4637527, 18.4616776

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4877983, upper bound: 11.5638171
time: 32.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5083375, upper bound: 11.5433532
time: 44.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2401237, 25.2430725
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7471733, 19.7519379
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2126884, 17.2479477
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0006104, 22.0126457
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7078133, 21.7254829
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7182961, 23.7408295
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2057800, 23.2042389
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5777588, 23.5907936
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4844818, 26.5108643
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2851486, 24.2881470
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2365341, 31.2715378
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8504410, 27.8399048
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2138367, 27.1978645
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8842850, 33.8861008
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7642365, 37.7794037
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8866272, 27.8962097
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0494232, 31.0621262
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6259460, 41.6192932
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5896835, 29.5847931
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7800980, 22.7694778
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1910248, 19.1809654
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4845886, 25.4683762
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7266846, 24.6971359
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0037880, 21.9924927
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8342056, 22.8286934
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3898659, 24.3719521
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5734253, 30.5610886
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4469833, 26.4316254
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9542389, 21.9262047
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9711685, 23.9446335
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0705948, 26.0534058
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0829773, 25.0776520
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3910065, 21.3783607
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3984985, 33.3878784
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7207184, 27.6896057
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1567383, 28.1344528
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0224075, 28.9872208
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7844238, 38.7487259
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7068176, 40.6700439
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4043579, 41.3939438
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0555954, 31.0536652
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5124817, 26.4963531
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4646530, 18.4608421

Time for backsubstitution: 2.27 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4846306, upper bound: 11.5676570
time: 36.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5051904, upper bound: 11.5472165
time: 34.97 seconds

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

Time for backsubstitution: 2.17 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4765816, upper bound: 11.5755844
time: 38.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4971471, upper bound: 11.5551421
time: 26.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 67.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5551421, upper bound: 11.4971471
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5755844, upper bound: 11.4765816
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5472165, upper bound: 11.5051904
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5676571, upper bound: 11.4846306
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5433532, upper bound: 11.5083375
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5638170, upper bound: 11.4877983
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5353786, upper bound: 11.5163587
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5558588, upper bound: 11.4958266
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5438461, upper bound: 11.5084842
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5643198, upper bound: 11.4879590
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5358584, upper bound: 11.5164804
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5563395, upper bound: 11.4959630
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5320113, upper bound: 11.5196834
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5525306, upper bound: 11.4991921
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5239838, upper bound: 11.5276706
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5445107, upper bound: 11.5071868
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5071868, upper bound: 11.5445107
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5276706, upper bound: 11.5239838
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.4991921, upper bound: 11.5525306
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5196834, upper bound: 11.5320113
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.4959630, upper bound: 11.5563395
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5164804, upper bound: 11.5358584
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.4879590, upper bound: 11.5643198
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5084843, upper bound: 11.5438461
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.4958266, upper bound: 11.5558588
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5163587, upper bound: 11.5353786
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.4877983, upper bound: 11.5638171
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5083375, upper bound: 11.5433532
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.4846306, upper bound: 11.5676570
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.5051904, upper bound: 11.5472165
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.4765816, upper bound: 11.5755844
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.08
Output dim: 2, lower bound: -11.4971471, upper bound: 11.5551421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1956902, 25.2011566
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7092247, 19.7135010
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2351913, 17.2008743
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0185280, 22.0024414
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7165947, 21.7038498
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7215652, 23.7038040
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1924973, 23.1998100
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5746460, 23.5661316
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4827271, 26.4633789
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2682724, 24.2689743
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2726974, 31.2382812
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8357773, 27.8432693
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2020264, 27.2107849
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8950882, 33.8693619
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8396912, 37.7844315
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8769073, 27.8723907
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0362625, 31.0401535
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6569061, 41.6503906
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5867310, 29.5916672
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7648697, 22.7722855
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1444702, 19.1447411
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4553757, 25.4660797
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6711044, 24.6912422
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9801559, 21.9806976
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8024292, 22.7985764
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3569641, 24.3653946
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5305405, 30.5342255
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4348145, 26.4517670
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9127350, 21.9381828
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9459076, 23.9717484
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0330086, 26.0422859
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0631332, 25.0583000
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3550797, 21.3600159
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3247986, 33.3141403
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6380920, 27.6525192
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1007767, 28.1099472
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9639206, 28.9884720
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7498322, 38.7887650
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6381836, 40.6467438
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3383331, 41.3160400
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0465164, 31.0606461
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5074997, 26.5308990
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4630394, 18.4670181

Time for backsubstitution: 2.21 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5353595, upper bound: 11.4957360
time: 34.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5537897, upper bound: 11.4773529
time: 29.06 seconds

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

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5557994, upper bound: 11.4751709
time: 34.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5742306, upper bound: 11.4567760
time: 37.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1945763, 25.2022781
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7089653, 19.7137604
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2340469, 17.2020226
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0167351, 22.0042305
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7170677, 21.7033806
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7216415, 23.7037354
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1952667, 23.1970406
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5750351, 23.5657425
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4828720, 26.4632416
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2683792, 24.2688713
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2728958, 31.2380829
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8362045, 27.8428421
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2015839, 27.2112236
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8886490, 33.8758011
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8274841, 37.7966309
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8777084, 27.8715973
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0378036, 31.0386047
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6548157, 41.6524734
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5867310, 29.5916672
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7636871, 22.7734680
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1448746, 19.1443329
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4547806, 25.4666748
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6706696, 24.6916809
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9769058, 21.9839516
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8017426, 22.7992630
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3565292, 24.3658295
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5305710, 30.5342026
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4353638, 26.4512177
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9127350, 21.9381790
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9456787, 23.9719849
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0349312, 26.0403633
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0601120, 25.0613213
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3545761, 21.3605309
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3241882, 33.3147659
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6378098, 27.6528015
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0997391, 28.1109848
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9625320, 28.9898605
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7513885, 38.7872009
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6303711, 40.6545486
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3318634, 41.3225174
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0525436, 31.0546188
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5084229, 26.5299759
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4630852, 18.4669762

Time for backsubstitution: 2.28 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5273836, upper bound: 11.5038092
time: 32.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5458522, upper bound: 11.4854407
time: 32.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2052269, 25.1916313
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7185249, 19.7042007
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2372818, 17.1987877
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0162621, 22.0047035
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7210503, 21.6994019
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7262650, 23.6991081
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1955032, 23.1968040
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5787811, 23.5620041
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4896240, 26.4564819
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2718658, 24.2653809
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2730789, 31.2378998
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8323059, 27.8467407
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.1952515, 27.2175560
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8776093, 33.8868332
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8117981, 37.8123245
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8811722, 27.8681259
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0513077, 31.0251083
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6458130, 41.6614838
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5867767, 29.5916214
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7628479, 22.7743034
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1342773, 19.1549339
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4504623, 25.4709930
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6621399, 24.7002106
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9726486, 21.9882050
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7937546, 22.8072510
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3479156, 24.3744431
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5218582, 30.5429077
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4358597, 26.4507217
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9101410, 21.9407654
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9454498, 23.9722137
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0231667, 26.0521278
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0559921, 25.0654373
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3478775, 21.3672218
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3041382, 33.3348007
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6216965, 27.6689224
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0886993, 28.1220245
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9546814, 28.9977112
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7514954, 38.7871017
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6177979, 40.6671448
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3121185, 41.3422623
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0526962, 31.0544739
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5138474, 26.5245514
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4631691, 18.4668884

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5478359, upper bound: 11.4832473
time: 39.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5662923, upper bound: 11.4648639
time: 31.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1944237, 25.2024269
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7090263, 19.7136993
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2324753, 17.2037277
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0170784, 22.0038910
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7153282, 21.7051544
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7189560, 23.7066154
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1928101, 23.1993446
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5743179, 23.5664787
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4806366, 26.4656372
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2673798, 24.2699127
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2726517, 31.2382965
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8389282, 27.8402100
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2069550, 27.2058487
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8960495, 33.8684006
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8397217, 37.7844086
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8742981, 27.8747711
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0358276, 31.0405884
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6568909, 41.6504059
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5870056, 29.5913963
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7648621, 22.7723007
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1451645, 19.1440468
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4564590, 25.4651337
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6711349, 24.6912155
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9801102, 21.9807549
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8023987, 22.7986031
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3572845, 24.3651161
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5309296, 30.5339508
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4343643, 26.4522629
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9126892, 21.9382706
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9456863, 23.9719772
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0334969, 26.0418625
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0630493, 25.0583191
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3573837, 21.3574829
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3253784, 33.3135605
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6411133, 27.6497726
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1026306, 28.1081696
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9665604, 28.9859085
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7536774, 38.7849197
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6434326, 40.6415024
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3404236, 41.3139572
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0477524, 31.0592728
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5086288, 26.5297699
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4639168, 18.4660835

Time for backsubstitution: 2.25 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5235333, upper bound: 11.5069242
time: 34.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5419954, upper bound: 11.4885599
time: 39.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2050743, 25.1917801
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7185783, 19.7041435
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2357101, 17.2004929
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0166054, 22.0043640
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7193031, 21.7011719
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7235794, 23.7019882
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1930466, 23.1991081
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5780640, 23.5627365
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4873886, 26.4588928
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2708740, 24.2664261
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2728271, 31.2381134
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8350296, 27.8441010
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2006226, 27.2121773
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8850174, 33.8794327
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8240204, 37.8000946
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8777618, 27.8713074
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0493317, 31.0270844
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6478882, 41.6594086
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5870438, 29.5913544
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7640228, 22.7731400
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1345673, 19.1546440
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4521408, 25.4694481
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6626053, 24.6997452
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9758530, 21.9850044
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7944107, 22.8065948
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3486710, 24.3737259
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5222244, 30.5426559
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4348602, 26.4517670
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9100952, 21.9408607
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9454575, 23.9722061
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0217323, 26.0536270
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0589371, 25.0624352
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3507004, 21.3641739
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3053436, 33.3336029
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6249924, 27.6658936
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0915909, 28.1192093
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9587097, 28.9937592
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7537689, 38.7848206
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6308289, 40.6540985
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3206787, 41.3337021
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0478897, 31.0591278
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5140533, 26.5243454
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4640007, 18.4659958

Time for backsubstitution: 2.23 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5440122, upper bound: 11.4863923
time: 33.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5624568, upper bound: 11.4680173
time: 31.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1933098, 25.2035484
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7087669, 19.7139587
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2313232, 17.2048759
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0152855, 22.0056801
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7157936, 21.7046814
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7190247, 23.7065430
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1955719, 23.1965752
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5747070, 23.5660896
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4807816, 26.4654999
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2674866, 24.2698097
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2728500, 31.2380905
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8393555, 27.8397827
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2065125, 27.2062874
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8896103, 33.8748398
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8275146, 37.7966003
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8750992, 27.8739777
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0373764, 31.0390396
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6548004, 41.6524963
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5870056, 29.5913963
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7636795, 22.7734795
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1455765, 19.1436386
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4558640, 25.4657249
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6707001, 24.6916542
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9768600, 21.9840088
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8017197, 22.7992897
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3568497, 24.3655548
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5309601, 30.5339203
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4349213, 26.4517059
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9126892, 21.9382668
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9454498, 23.9722099
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0354195, 26.0399399
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0600281, 25.0613403
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3568802, 21.3579979
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3247528, 33.3141861
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6408310, 27.6500549
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1015930, 28.1092072
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9651718, 28.9872971
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7552338, 38.7833557
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6356201, 40.6493149
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3339386, 41.3204346
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0537796, 31.0532455
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5095596, 26.5288391
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4639626, 18.4660416

Time for backsubstitution: 2.25 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5155082, upper bound: 11.5149789
time: 39.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5339934, upper bound: 11.4966204
time: 44.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2039452, 25.1929016
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7183189, 19.7044029
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2345581, 17.2016373
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0148125, 22.0061531
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7197762, 21.7007027
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7236481, 23.7019196
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1958160, 23.1963387
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5784454, 23.5623474
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4875336, 26.4587402
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2709732, 24.2663231
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2730331, 31.2379150
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8354568, 27.8436737
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2001801, 27.2126198
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8785706, 33.8858795
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8118286, 37.8122940
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8785629, 27.8705139
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0508728, 31.0255432
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6457977, 41.6614990
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5870438, 29.5913544
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7628403, 22.7743187
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1349716, 19.1542397
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4515457, 25.4700432
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6621704, 24.7001839
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9726028, 21.9882622
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7937241, 22.8072815
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3482361, 24.3741608
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5222549, 30.5426331
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4354172, 26.4512100
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9101105, 21.9408531
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9452209, 23.9724388
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0236549, 26.0517044
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0559158, 25.0654564
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3501816, 21.3646889
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3047180, 33.3342285
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6247101, 27.6661682
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0905533, 28.1202469
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9573135, 28.9951477
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7553406, 38.7832565
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6230164, 40.6619034
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3141937, 41.3401794
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0539169, 31.0531006
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5149841, 26.5234222
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4640465, 18.4659576

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5359954, upper bound: 11.4944473
time: 32.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5544762, upper bound: 11.4760811
time: 39.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1970787, 25.1997681
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7097664, 19.7129555
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2167816, 17.2192841
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0109444, 22.0100174
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7116356, 21.7088089
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7128983, 23.7124710
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1955109, 23.1967926
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5681610, 23.5726242
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4672623, 26.4788437
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2670670, 24.2701874
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2463913, 31.2645874
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8349152, 27.8441315
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2057648, 27.2070351
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8956985, 33.8687515
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8122406, 37.8118744
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8769455, 27.8723526
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0268021, 31.0496063
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6569672, 41.6503296
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5822067, 29.5961952
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7673950, 22.7697639
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1435547, 19.1456566
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4549332, 25.4665222
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6887817, 24.6735649
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9779129, 21.9829483
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8020172, 22.7989883
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3627548, 24.3596039
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5238571, 30.5409088
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4401093, 26.4464722
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9237213, 21.9271889
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9643555, 23.9533081
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0344124, 26.0408859
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0626831, 25.0587463
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3615799, 21.3535156
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3306427, 33.3082962
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6492233, 27.6413879
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1132126, 28.0975037
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9844971, 28.9678955
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7717743, 38.7668076
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6519012, 40.6330338
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3401947, 41.3141937
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0504990, 31.0566711
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5207291, 26.5176697
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4631004, 18.4669571

Time for backsubstitution: 2.25 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5240997, upper bound: 11.5070979
time: 31.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5424691, upper bound: 11.4886120
time: 26.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2077293, 25.1891174
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7193260, 19.7033997
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2200165, 17.2160492
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0104713, 22.0104904
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7156181, 21.7048302
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7175293, 23.7078400
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1957550, 23.1965561
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5718994, 23.5688820
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4740143, 26.4720917
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2705536, 24.2666969
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2465668, 31.2644119
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8310089, 27.8480377
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.1994324, 27.2133675
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8846664, 33.8797836
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7965546, 37.8275681
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8804092, 27.8688889
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0403061, 31.0361099
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6479645, 41.6593323
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5822449, 29.5961533
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7665558, 22.7705994
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1329575, 19.1562576
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4506149, 25.4708405
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6802521, 24.6820908
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9736557, 21.9872017
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7940292, 22.8069763
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3541412, 24.3682175
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5151443, 30.5496216
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4406052, 26.4459763
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9211426, 21.9297752
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9641266, 23.9535370
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0226479, 26.0526505
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0585709, 25.0628624
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3548965, 21.3602066
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3106079, 33.3283386
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6331024, 27.6575089
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1021729, 28.1085434
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9766388, 28.9757538
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7718811, 38.7667084
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6393127, 40.6456223
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3204498, 41.3339386
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0506210, 31.0565338
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5261536, 26.5122452
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4631844, 18.4668732

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5445921, upper bound: 11.4865729
time: 27.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5629334, upper bound: 11.4680840
time: 32.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1959648, 25.2008896
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7095070, 19.7132149
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2156372, 17.2204361
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0091591, 22.0118103
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7121086, 21.7083359
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7129745, 23.7124023
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1982803, 23.1940269
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5685501, 23.5722351
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4674149, 26.4786987
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2671661, 24.2700806
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2465897, 31.2643890
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8353424, 27.8437042
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2053223, 27.2074776
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8892593, 33.8751907
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8000488, 37.8240662
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8777542, 27.8715515
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0283508, 31.0480652
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6548767, 41.6524200
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5822067, 29.5961952
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7662125, 22.7709427
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1439590, 19.1452484
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4543381, 25.4671173
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6883392, 24.6740036
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9746628, 21.9862022
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8013306, 22.7996750
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3623199, 24.3600426
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5238876, 30.5408859
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4406586, 26.4459229
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9237366, 21.9271812
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9641190, 23.9535446
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0363350, 26.0389633
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0596619, 25.0617676
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3610687, 21.3540306
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3300171, 33.3089218
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6489410, 27.6416702
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1121750, 28.0985413
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9831009, 28.9692917
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7733459, 38.7652435
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6441040, 40.6408386
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3337097, 41.3206711
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0565109, 31.0506439
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5216599, 26.5167389
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4631386, 18.4669151

Time for backsubstitution: 2.25 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5161036, upper bound: 11.5151113
time: 38.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5344640, upper bound: 11.4966484
time: 42.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2066154, 25.1902390
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7190666, 19.7036591
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2188721, 17.2171974
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0086861, 22.0122833
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7160912, 21.7043571
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7175980, 23.7077713
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1985168, 23.1937866
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5722885, 23.5684929
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4741592, 26.4719467
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2706604, 24.2665939
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2467651, 31.2642136
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8314362, 27.8476105
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.1990051, 27.2138062
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8782196, 33.8862228
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7843628, 37.8397675
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8812180, 27.8680878
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0418549, 31.0345612
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6458740, 41.6614227
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5822449, 29.5961533
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7653809, 22.7717819
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1333618, 19.1558495
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4500198, 25.4714355
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6798096, 24.6825294
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9704056, 21.9904518
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7933426, 22.8076630
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3537064, 24.3686523
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5151749, 30.5495911
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4411621, 26.4454193
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9211426, 21.9297676
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9638901, 23.9537735
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0245705, 26.0507278
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0555496, 25.0658836
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3543777, 21.3607216
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3099823, 33.3289642
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6328201, 27.6577911
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1011353, 28.1095810
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9752502, 28.9771423
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7734528, 38.7651443
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6315002, 40.6534348
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3139648, 41.3404160
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0566635, 31.0505066
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5270844, 26.5113220
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4632301, 18.4668274

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5366062, upper bound: 11.4945926
time: 30.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5549411, upper bound: 11.4761288
time: 30.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1958122, 25.2010345
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7095680, 19.7131577
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2140656, 17.2221375
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0094948, 22.0114670
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7103691, 21.7101097
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7102890, 23.7152786
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1958237, 23.1963234
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5678253, 23.5729675
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4651718, 26.4811020
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2661743, 24.2711258
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2463455, 31.2646027
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8380661, 27.8410721
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2107086, 27.2021027
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8966599, 33.8677902
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8122711, 37.8118439
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8743439, 27.8747330
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0263748, 31.0500412
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6569519, 41.6503525
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5824738, 29.5959244
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7673798, 22.7697754
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1442490, 19.1449623
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4560165, 25.4655762
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6888123, 24.6735382
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9778671, 21.9830055
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8019867, 22.7990150
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3630753, 24.3593292
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5242462, 30.5406342
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4396667, 26.4469604
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9236908, 21.9272728
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9641266, 23.9535370
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0349007, 26.0404625
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0626068, 25.0587654
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3638840, 21.3509827
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3312073, 33.3077316
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6522446, 27.6386414
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1150665, 28.0957260
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9871368, 28.9653320
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7756195, 38.7629700
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6571350, 40.6278000
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3422699, 41.3121109
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0517197, 31.0552979
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5218658, 26.5165405
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4639778, 18.4660263

Time for backsubstitution: 2.27 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5122689, upper bound: 11.5182981
time: 32.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5306337, upper bound: 11.4998019
time: 30.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2064629, 25.1903877
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7191277, 19.7035980
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2173004, 17.2189026
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0090218, 22.0119400
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7143440, 21.7061310
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7149124, 23.7106514
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1960602, 23.1960869
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5715714, 23.5692253
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4719315, 26.4743500
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2696609, 24.2676392
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2465210, 31.2644196
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8341675, 27.8449707
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2043762, 27.2084312
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8856277, 33.8788223
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7965851, 37.8275375
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8778076, 27.8712692
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0398788, 31.0365448
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6479340, 41.6593552
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5825195, 29.5958862
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7665482, 22.7706146
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1336517, 19.1555595
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4516983, 25.4698906
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6802826, 24.6820679
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9736099, 21.9872551
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7939987, 22.8070068
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3544617, 24.3679352
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5155411, 30.5493469
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4401627, 26.4464645
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9210968, 21.9298630
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9638977, 23.9537659
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0231361, 26.0522270
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0584869, 25.0628815
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3572006, 21.3576736
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3111877, 33.3277664
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6361237, 27.6547623
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1040268, 28.1067734
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9792786, 28.9731827
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7757263, 38.7628632
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6445312, 40.6403885
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3225250, 41.3318558
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0518723, 31.0551529
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5272827, 26.5111160
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4640617, 18.4659386

Time for backsubstitution: 2.35 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5328016, upper bound: 11.4978084
time: 34.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5511496, upper bound: 11.4793163
time: 33.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1946983, 25.2021561
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7093086, 19.7134171
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2129135, 17.2232857
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0077095, 22.0132599
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7108345, 21.7096405
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7103577, 23.7152100
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1985931, 23.1935577
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5682220, 23.5725784
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4653244, 26.4809570
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2662735, 24.2710228
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2465439, 31.2644043
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8384933, 27.8406448
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2102661, 27.2025414
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8902206, 33.8742294
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8000793, 37.8240356
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8751450, 27.8739319
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0279160, 31.0484924
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6548462, 41.6524429
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5824738, 29.5959244
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7662048, 22.7709579
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1446609, 19.1445503
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4554214, 25.4661674
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6883698, 24.6739769
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9746017, 21.9862595
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8013000, 22.7997017
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3626328, 24.3597641
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5242691, 30.5406036
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4402161, 26.4464111
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9236908, 21.9272690
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9638901, 23.9537735
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0368233, 26.0385399
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0595779, 25.0617867
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3633804, 21.3514977
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3305969, 33.3083572
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6519623, 27.6389236
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1140366, 28.0967636
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9857407, 28.9667206
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7771912, 38.7613983
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6493225, 40.6356049
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3357849, 41.3185883
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0577469, 31.0492706
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5227966, 26.5156097
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4640236, 18.4659805

Time for backsubstitution: 2.29 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5042083, upper bound: 11.5263032
time: 32.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5225762, upper bound: 11.5078388
time: 36.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2053490, 25.1915092
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7188683, 19.7038574
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2161484, 17.2200508
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0072365, 22.0137329
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7148170, 21.7056580
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7149887, 23.7105827
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1988297, 23.1933174
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5719604, 23.5688362
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4720688, 26.4742050
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2697678, 24.2675323
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2467194, 31.2642212
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8345947, 27.8445435
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2039337, 27.2088699
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8791809, 33.8852692
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7843933, 37.8397369
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8786087, 27.8704681
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0414200, 31.0349960
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6458435, 41.6614456
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5825195, 29.5958824
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7653656, 22.7717934
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1340561, 19.1551552
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4511032, 25.4704857
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6798477, 24.6825066
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9703598, 21.9905090
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7933121, 22.8076935
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3540268, 24.3683739
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5155640, 30.5493164
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4407120, 26.4459152
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9210968, 21.9298592
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9636612, 23.9540024
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0250587, 26.0503044
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0554657, 25.0659027
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3566818, 21.3581886
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3105469, 33.3283920
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6358414, 27.6550446
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1029892, 28.1078110
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9778900, 28.9745789
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7772980, 38.7612991
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6367340, 40.6482010
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3160400, 41.3383331
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0578842, 31.0491257
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5282135, 26.5101852
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4641075, 18.4658966

Time for backsubstitution: 2.30 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5247509, upper bound: 11.5058174
time: 35.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5430983, upper bound: 11.4873541
time: 36.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1915092, 25.2053452
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7038612, 19.7188644
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2200470, 17.2161484
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0137367, 22.0072327
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7056618, 21.7148170
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7105789, 23.7149849
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1933212, 23.1988297
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5688324, 23.5719604
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4742050, 26.4720764
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2675323, 24.2697639
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2642212, 31.2467194
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8445435, 27.8345947
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2088776, 27.2039337
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8852692, 33.8791885
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8397369, 37.7843857
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8704681, 27.8786087
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0349960, 31.0414200
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6614380, 41.6458511
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5958862, 29.5825195
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7717972, 22.7653656
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1551514, 19.1340599
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4704895, 25.4511032
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6825104, 24.6798477
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9905167, 21.9703560
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8076935, 22.7933121
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3683777, 24.3540268
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5493164, 30.5155640
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4459152, 26.4407120
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9298553, 21.9211006
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9540024, 23.9636612
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0503044, 26.0250549
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0659027, 25.0554657
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3581924, 21.3566856
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3283844, 33.3105545
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6550369, 27.6358414
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1078110, 28.1029892
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9745789, 28.9778900
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7613068, 38.7772903
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6481934, 40.6367340
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3383331, 41.3160400
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0491257, 31.0578918
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5101852, 26.5282135
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4659004, 18.4641037

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4873541, upper bound: 11.5430983
time: 28.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5058173, upper bound: 11.5247509
time: 33.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2021599, 25.1946945
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7134132, 19.7093086
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2232895, 17.2129135
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0132561, 22.0077057
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7096367, 21.7108383
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7152100, 23.7103577
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1935577, 23.1985931
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5725784, 23.5682182
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4809570, 26.4653168
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2710190, 24.2662773
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2643967, 31.2465439
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8406448, 27.8384933
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2025452, 27.2102661
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8742294, 33.8902206
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8240356, 37.8000793
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8739319, 27.8751450
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0485001, 31.0279236
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6524353, 41.6548538
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5959244, 29.5824738
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7709579, 22.7662048
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1445541, 19.1446571
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4661713, 25.4554214
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6739807, 24.6883736
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9862595, 21.9746056
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7997055, 22.8013000
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3597641, 24.3626366
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5406113, 30.5242767
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4464111, 26.4402161
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9272766, 21.9236908
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9537735, 23.9638901
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0385399, 26.0368195
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0617905, 25.0595818
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3514938, 21.3633728
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3083496, 33.3305893
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6389236, 27.6519623
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0967636, 28.1140366
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9667206, 28.9857407
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7613983, 38.7771912
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6356049, 40.6493301
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3185883, 41.3357849
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0492783, 31.0577469
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5156097, 26.5227966
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4659843, 18.4640198

Time for backsubstitution: 2.25 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5078388, upper bound: 11.5225762
time: 38.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5263032, upper bound: 11.5042083
time: 31.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1903954, 25.2064667
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7036018, 19.7191238
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2189026, 17.2173004
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0119438, 22.0090256
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7061272, 21.7143478
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7106552, 23.7149162
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1960907, 23.1960640
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5692291, 23.5715714
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4743500, 26.4719238
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2676392, 24.2696609
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2644196, 31.2465210
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8449707, 27.8341675
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2084351, 27.2043762
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8788223, 33.8856277
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8275452, 37.7965775
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8712692, 27.8778076
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0365372, 31.0398712
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6593628, 41.6479416
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5958862, 29.5825157
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7706146, 22.7665443
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1555634, 19.1336517
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4698944, 25.4516983
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6820679, 24.6802826
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9872513, 21.9736061
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8070068, 22.7939987
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3679352, 24.3544617
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5493393, 30.5155334
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4464645, 26.4401627
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9298553, 21.9210968
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9537659, 23.9638977
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0522270, 26.0231323
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0628815, 25.0584908
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3576736, 21.3572006
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3277588, 33.3111801
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6547623, 27.6361237
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1067734, 28.1040268
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9731827, 28.9792786
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7628632, 38.7757263
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6403809, 40.6445389
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3318634, 41.3225174
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0551682, 31.0518646
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5111160, 26.5272827
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4659386, 18.4640617

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4793163, upper bound: 11.5511496
time: 32.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4978084, upper bound: 11.5328016
time: 42.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2010307, 25.1958160
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7131538, 19.7095680
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2221375, 17.2140617
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0114708, 22.0094986
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7101097, 21.7103653
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7152786, 23.7102890
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1963272, 23.1958237
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5729675, 23.5678291
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4811020, 26.4651718
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2711258, 24.2661743
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2645950, 31.2463379
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8410721, 27.8380661
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2021027, 27.2107086
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8677902, 33.8966599
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8118439, 37.8122711
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8747330, 27.8743439
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0500412, 31.0263748
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6503601, 41.6569443
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5959244, 29.5824738
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7697754, 22.7673798
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1449661, 19.1442528
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4655762, 25.4560165
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6735382, 24.6888123
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9830093, 21.9778595
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7990189, 22.8019867
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3593292, 24.3630753
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5406342, 30.5242462
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4469604, 26.4396667
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9272766, 21.9236870
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9535370, 23.9641266
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0404625, 26.0348969
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0587616, 25.0626068
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3509827, 21.3638916
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3077240, 33.3312149
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6386414, 27.6522369
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0957260, 28.1150665
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9653320, 28.9871368
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7629700, 38.7756271
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6278076, 40.6571350
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3121185, 41.3422623
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0552902, 31.0517197
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5165405, 26.5218658
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4660225, 18.4639740

Time for backsubstitution: 2.23 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4998019, upper bound: 11.5306337
time: 31.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5182981, upper bound: 11.5122689
time: 33.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1902428, 25.2066116
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7036552, 19.7190666
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2172012, 17.2188721
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0122795, 22.0086823
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7043571, 21.7160873
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7077713, 23.7175980
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1937866, 23.1985207
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5684967, 23.5722885
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4719467, 26.4741669
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2665939, 24.2706604
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2642136, 31.2467651
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8476105, 27.8314362
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2138062, 27.1989975
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8862228, 33.8782196
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8397675, 37.7843552
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8680801, 27.8812103
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0345612, 31.0418549
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6614227, 41.6458740
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5961533, 29.5822449
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7717819, 22.7653732
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1558456, 19.1333618
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4714355, 25.4500198
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6825256, 24.6798134
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9904556, 21.9704056
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8076630, 22.7933388
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3686523, 24.3537064
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5495987, 30.5151749
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4454193, 26.4411621
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9297638, 21.9211426
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9537735, 23.9638901
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0507240, 26.0245667
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0658798, 25.0555496
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3607254, 21.3543816
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3289642, 33.3099823
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6577911, 27.6328278
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1095810, 28.1011353
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9771423, 28.9752502
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7651520, 38.7734528
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6534424, 40.6315002
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3404083, 41.3139648
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0504990, 31.0566559
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5113220, 26.5270844
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4668312, 18.4632301

Time for backsubstitution: 2.15 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4761288, upper bound: 11.5549411
time: 31.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4945926, upper bound: 11.5366062
time: 33.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2008934, 25.1959610
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7132149, 19.7095108
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2204361, 17.2156334
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0118065, 22.0091553
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7083397, 21.7121124
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7124023, 23.7129707
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1940231, 23.1982841
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5722351, 23.5685501
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4786987, 26.4674072
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2700806, 24.2671700
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2643890, 31.2465897
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8437042, 27.8353424
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2074738, 27.2053299
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8751907, 33.8892593
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8240662, 37.8000488
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8715439, 27.8777466
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0480652, 31.0283508
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6524200, 41.6548767
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5961914, 29.5822067
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7709427, 22.7662125
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1452484, 19.1439629
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4671173, 25.4543381
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6740036, 24.6883430
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9861984, 21.9746590
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7996750, 22.8013306
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3600388, 24.3623161
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5408859, 30.5238800
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4459229, 26.4406586
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9271851, 21.9237328
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9535446, 23.9641190
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0389595, 26.0363312
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0617676, 25.0596657
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3540268, 21.3610687
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3089294, 33.3300171
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6416702, 27.6489410
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0985413, 28.1121750
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9692917, 28.9831009
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7652435, 38.7733459
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6408386, 40.6440964
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3206635, 41.3337097
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0506516, 31.0565186
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5167389, 26.5216599
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4669151, 18.4631424

Time for backsubstitution: 2.20 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4966484, upper bound: 11.5344640
time: 42.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5151113, upper bound: 11.5161036
time: 42.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1891136, 25.2077332
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7033958, 19.7193260
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2160492, 17.2200165
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0104942, 22.0104752
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7048302, 21.7156181
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7078400, 23.7175293
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1965561, 23.1957512
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5688782, 23.5718994
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4720917, 26.4740143
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2667007, 24.2705536
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2644119, 31.2465668
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8480377, 27.8310089
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2133636, 27.1994400
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8797836, 33.8846664
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8275757, 37.7965469
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8688889, 27.8804169
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0361099, 31.0403061
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6593323, 41.6479645
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5961533, 29.5822449
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7705994, 22.7665558
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1562576, 19.1329536
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4708405, 25.4506111
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6820908, 24.6802521
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9872055, 21.9736595
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8069763, 22.7940254
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3682175, 24.3541412
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5496216, 30.5151520
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4459763, 26.4406052
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9297791, 21.9211388
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9535370, 23.9641228
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0526466, 26.0226440
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0628586, 25.0585709
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3602066, 21.3548965
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3283386, 33.3106079
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6575089, 27.6331024
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1085434, 28.1021729
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9757538, 28.9766388
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7667084, 38.7718811
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6456299, 40.6393051
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3339386, 41.3204422
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0565414, 31.0506287
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5122452, 26.5261536
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4668770, 18.4631882

Time for backsubstitution: 2.14 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4680840, upper bound: 11.5629334
time: 31.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4865729, upper bound: 11.5445921
time: 29.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1997643, 25.1970825
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7129555, 19.7097702
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2192841, 17.2167816
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0100212, 22.0109482
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7088127, 21.7116394
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7124710, 23.7129021
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1967926, 23.1955147
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5726242, 23.5681610
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4788437, 26.4672623
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2701874, 24.2670670
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2645874, 31.2463913
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8441315, 27.8349152
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2070312, 27.2057724
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8687439, 33.8956985
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8118744, 37.8122406
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8723526, 27.8769531
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0496063, 31.0268097
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6503296, 41.6569672
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5961914, 29.5822067
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7697601, 22.7673912
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1456604, 19.1435547
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4665222, 25.4549294
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6735687, 24.6887817
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9829483, 21.9779091
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7989883, 22.8020172
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3596039, 24.3627548
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5409164, 30.5238571
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4464722, 26.4401093
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9271851, 21.9237251
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9533081, 23.9643517
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0408821, 26.0344086
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0587463, 25.0626869
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3535156, 21.3615875
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3083038, 33.3306427
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6413956, 27.6492233
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0975037, 28.1132126
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9678955, 28.9844971
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7668152, 38.7717819
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6330261, 40.6519012
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3141937, 41.3401871
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0566635, 31.0504913
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5176697, 26.5207291
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4669609, 18.4631004

Time for backsubstitution: 2.14 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4886120, upper bound: 11.5424691
time: 36.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5070979, upper bound: 11.5240997
time: 31.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1928978, 25.2039490
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7044029, 19.7183228
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2016373, 17.2345619
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0061531, 22.0148125
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7007027, 21.7197723
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7019196, 23.7236481
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1963348, 23.1958122
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5623474, 23.5784492
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4587479, 26.4875336
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2663193, 24.2709770
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2379074, 31.2730331
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8436737, 27.8354568
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2126160, 27.2001877
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8858795, 33.8785782
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8122864, 37.8118286
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8705063, 27.8785706
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0255356, 31.0508728
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6614990, 41.6457977
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5913544, 29.5870438
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7743225, 22.7628403
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1542435, 19.1349754
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4700470, 25.4515457
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7001801, 24.6621704
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9882584, 21.9726067
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8072815, 22.7937241
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3741608, 24.3482361
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5426331, 30.5222473
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4512100, 26.4354172
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9408569, 21.9101067
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9724426, 23.9452209
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0517006, 26.0236588
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0654526, 25.0559120
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3646927, 21.3501854
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3342285, 33.3047180
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6661682, 27.6247101
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1202469, 28.0905533
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9951477, 28.9573135
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7832489, 38.7553329
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6618958, 40.6230240
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3401794, 41.3141937
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0530930, 31.0539169
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5234222, 26.5149841
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4659538, 18.4640465

Time for backsubstitution: 2.16 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4760811, upper bound: 11.5544762
time: 42.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4944473, upper bound: 11.5359954
time: 33.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2035484, 25.1933060
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7139626, 19.7087631
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2048798, 17.2313232
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0056801, 22.0152855
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7046776, 21.7157974
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7065430, 23.7190247
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1965790, 23.1955757
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5660858, 23.5747070
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4654922, 26.4807816
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2698135, 24.2674866
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2380905, 31.2728577
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8397827, 27.8393555
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2062836, 27.2065201
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8748398, 33.8896103
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7966003, 37.8275223
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8739700, 27.8751068
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0390396, 31.0373688
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6524963, 41.6548004
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5914001, 29.5870056
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7734833, 22.7636795
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1436386, 19.1455727
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4657288, 25.4558640
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6916504, 24.6706963
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9840164, 21.9768562
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7992859, 22.8017159
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3655548, 24.3568497
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5339203, 30.5309601
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4517059, 26.4349213
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9382629, 21.9126930
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9722137, 23.9454536
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0399361, 26.0354195
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0613403, 25.0600281
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3579941, 21.3568726
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3141785, 33.3247528
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6500549, 27.6408310
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1092072, 28.1015930
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9872971, 28.9651718
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7833557, 38.7552338
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6493225, 40.6356201
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3204346, 41.3339386
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0532455, 31.0537796
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5288391, 26.5095596
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4660378, 18.4639587

Time for backsubstitution: 2.15 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4966204, upper bound: 11.5339934
time: 31.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5149789, upper bound: 11.5155083
time: 32.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1917839, 25.2050705
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7041435, 19.7185822
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2004929, 17.2357101
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0043678, 22.0166016
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7011757, 21.7193031
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7019882, 23.7235794
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1991119, 23.1930428
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5627365, 23.5780602
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4588928, 26.4873810
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2664261, 24.2708740
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2381134, 31.2728271
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8441010, 27.8350296
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2121735, 27.2006302
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8794327, 33.8850174
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8000946, 37.8240204
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8713150, 27.8777618
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0270844, 31.0493317
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6594086, 41.6478882
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5913544, 29.5870438
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7731400, 22.7640228
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1546478, 19.1345634
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4694519, 25.4521408
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6997452, 24.6626053
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9850082, 21.9758568
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8065948, 22.7944107
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3737259, 24.3486748
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5426559, 30.5222244
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4517670, 26.4348602
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9408569, 21.9100990
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9722061, 23.9454575
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0536232, 26.0217323
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0624313, 25.0589371
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3641739, 21.3507004
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3336029, 33.3053436
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6658859, 27.6249924
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1192093, 28.0915909
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9937592, 28.9587097
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7848206, 38.7537689
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6540985, 40.6308365
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3337097, 41.3206711
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0591354, 31.0478897
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5243454, 26.5140533
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4659996, 18.4640045

Time for backsubstitution: 2.19 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4680173, upper bound: 11.5624568
time: 36.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4863923, upper bound: 11.5440122
time: 33.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2024345, 25.1944275
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7137032, 19.7090225
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2037277, 17.2324715
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0038872, 22.0170746
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7051506, 21.7153244
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7066116, 23.7189560
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1993408, 23.1928062
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5664749, 23.5743179
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4656448, 26.4806366
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2699127, 24.2673836
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2382889, 31.2726517
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8402100, 27.8389282
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2058411, 27.2069588
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8684006, 33.8960495
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7844086, 37.8397141
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8747787, 27.8742981
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0405884, 31.0358276
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6504059, 41.6568909
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5914001, 29.5870056
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7723007, 22.7648582
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1440506, 19.1451683
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4651337, 25.4564590
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6912155, 24.6711349
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9807510, 21.9801102
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7986069, 22.8024025
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3651123, 24.3572845
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5339508, 30.5309296
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4522629, 26.4343643
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9382782, 21.9126892
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9719772, 23.9456863
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0418587, 26.0334969
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0583191, 25.0630493
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3574829, 21.3573914
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3135681, 33.3253784
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6497726, 27.6411133
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1081696, 28.1026306
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9859085, 28.9665604
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7849274, 38.7536697
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6415100, 40.6434250
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3139648, 41.3404160
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0592575, 31.0477524
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5297699, 26.5086288
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4660835, 18.4639168

Time for backsubstitution: 2.17 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4885599, upper bound: 11.5419954
time: 27.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5069242, upper bound: 11.5235333
time: 26.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1916313, 25.2052193
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7042046, 19.7185211
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.1987839, 17.2372818
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0047035, 22.0162621
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.6993980, 21.7210464
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.6991043, 23.7262650
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1968079, 23.1954994
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5620041, 23.5787811
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4564896, 26.4896240
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2653809, 24.2718697
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2378998, 31.2730789
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8467407, 27.8323059
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2175598, 27.1952515
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8868332, 33.8776093
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8123169, 37.8117981
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8681259, 27.8811722
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0251083, 31.0513077
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6614838, 41.6458130
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5916214, 29.5867767
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7743073, 22.7628517
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1549377, 19.1342773
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4709930, 25.4504623
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7002106, 24.6621361
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9881973, 21.9726562
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8072510, 22.7937546
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3744431, 24.3479156
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5429077, 30.5218582
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4507217, 26.4358597
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9407654, 21.9101448
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9722137, 23.9454498
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0521278, 26.0231705
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0654373, 25.0559959
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3672256, 21.3478813
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3347931, 33.3041458
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6689224, 27.6216888
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1220245, 28.0886993
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9977112, 28.9546814
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7870941, 38.7514954
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6671448, 40.6177902
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3422699, 41.3121185
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0544662, 31.0526810
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5245514, 26.5138474
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4668922, 18.4631691

Time for backsubstitution: 2.15 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4648639, upper bound: 11.5662923
time: 30.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4832472, upper bound: 11.5478359
time: 33.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2022820, 25.1945724
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7137566, 19.7089653
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2020264, 17.2340469
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0042305, 22.0167351
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7033806, 21.7170677
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7037354, 23.7216377
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1970444, 23.1952629
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5657425, 23.5750389
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4632339, 26.4828720
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2688675, 24.2683792
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2380753, 31.2729034
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8428421, 27.8362045
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2112274, 27.2015839
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8758011, 33.8886490
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7966309, 37.8274918
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8715897, 27.8777084
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0386124, 31.0378113
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6524811, 41.6548157
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5916672, 29.5867348
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7734680, 22.7636871
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1443329, 19.1448784
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4666748, 25.4547806
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6916809, 24.6706657
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9839554, 21.9769096
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7992630, 22.8017426
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3658295, 24.3565292
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5342026, 30.5305634
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4512177, 26.4353638
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9381714, 21.9127350
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9719849, 23.9456787
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0403633, 26.0349312
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0613174, 25.0601120
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3605270, 21.3545685
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3147583, 33.3241806
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6528015, 27.6378098
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1109848, 28.0997391
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9898605, 28.9625320
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7872009, 38.7513885
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6545410, 40.6303787
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3225250, 41.3318634
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0546188, 31.0525436
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5299759, 26.5084229
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4669762, 18.4630814

Time for backsubstitution: 2.16 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4854407, upper bound: 11.5458522
time: 32.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5038092, upper bound: 11.5273836
time: 29.88 seconds

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

Time for backsubstitution: 2.16 seconds

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
time: 35.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4751709, upper bound: 11.5557994
time: 35.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2011528, 25.1956940
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7134972, 19.7092247
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2008743, 17.2351913
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0024376, 22.0185242
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7038536, 21.7165985
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7038040, 23.7215691
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1998138, 23.1924934
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5661316, 23.5746498
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4633865, 26.4827271
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2689743, 24.2682762
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2382812, 31.2726974
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8432693, 27.8357773
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2107849, 27.2020226
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8693542, 33.8950882
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7844391, 37.8396835
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8723984, 27.8769073
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0401535, 31.0362625
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6503906, 41.6569061
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5916672, 29.5867310
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7722931, 22.7648697
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1447449, 19.1444702
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4660797, 25.4553719
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6912384, 24.6711044
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9807053, 21.9801598
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7985764, 22.8024292
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3653946, 24.3569641
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5342331, 30.5305405
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4517670, 26.4348145
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9381866, 21.9127312
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9717484, 23.9459114
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0422859, 26.0330086
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0582962, 25.0631332
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3600159, 21.3550873
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3141327, 33.3248062
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6525192, 27.6380920
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1099472, 28.1007767
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9884720, 28.9639206
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7887726, 38.7498245
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6467438, 40.6381912
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3160400, 41.3383408
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0606461, 31.0465164
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5308990, 26.5074997
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4670143, 18.4630432

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4773529, upper bound: 11.5537897
time: 32.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4957359, upper bound: 11.5353595
time: 29.63 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 64.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5353595, upper bound: 11.4957360
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5537897, upper bound: 11.4773529
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5557994, upper bound: 11.4751709
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5742306, upper bound: 11.4567760
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5273836, upper bound: 11.5038092
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5458522, upper bound: 11.4854407
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5478359, upper bound: 11.4832473
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5662923, upper bound: 11.4648639
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5235333, upper bound: 11.5069242
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5419954, upper bound: 11.4885599
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5440122, upper bound: 11.4863923
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5624568, upper bound: 11.4680173
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5155082, upper bound: 11.5149789
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5339934, upper bound: 11.4966204
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5359954, upper bound: 11.4944473
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5544762, upper bound: 11.4760811
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5240997, upper bound: 11.5070979
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5424691, upper bound: 11.4886120
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5445921, upper bound: 11.4865729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5629334, upper bound: 11.4680840
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5161036, upper bound: 11.5151113
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5344640, upper bound: 11.4966484
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5366062, upper bound: 11.4945926
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5549411, upper bound: 11.4761288
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5122689, upper bound: 11.5182981
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5306337, upper bound: 11.4998019
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5328016, upper bound: 11.4978084
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5511496, upper bound: 11.4793163
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5042083, upper bound: 11.5263032
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5225762, upper bound: 11.5078388
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5247509, upper bound: 11.5058174
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5430983, upper bound: 11.4873541
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4873541, upper bound: 11.5430983
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5058173, upper bound: 11.5247509
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5078388, upper bound: 11.5225762
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5263032, upper bound: 11.5042083
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4793163, upper bound: 11.5511496
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4978084, upper bound: 11.5328016
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4998019, upper bound: 11.5306337
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5182981, upper bound: 11.5122689
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4761288, upper bound: 11.5549411
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4945926, upper bound: 11.5366062
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4966484, upper bound: 11.5344640
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5151113, upper bound: 11.5161036
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4680840, upper bound: 11.5629334
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4865729, upper bound: 11.5445921
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4886120, upper bound: 11.5424691
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5070979, upper bound: 11.5240997
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4760811, upper bound: 11.5544762
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4944473, upper bound: 11.5359954
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4966204, upper bound: 11.5339934
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5149789, upper bound: 11.5155083
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4680173, upper bound: 11.5624568
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4863923, upper bound: 11.5440122
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4885599, upper bound: 11.5419954
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5069242, upper bound: 11.5235333
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4648639, upper bound: 11.5662923
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4832472, upper bound: 11.5478359
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4854407, upper bound: 11.5458522
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.5038092, upper bound: 11.5273836
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4567760, upper bound: 11.5742306
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4751709, upper bound: 11.5557994
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4773529, upper bound: 11.5537897
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 64.02
Output dim: 2, lower bound: -11.4957359, upper bound: 11.5353595

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1955795, 25.2010574
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7104721, 19.7162476
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2355499, 17.2015381
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0142593, 21.9995384
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7180023, 21.7063789
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7213821, 23.7037506
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1889572, 23.1976395
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5735550, 23.5676689
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4824715, 26.4643936
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2686234, 24.2699242
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2681808, 31.2349472
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8338776, 27.8411255
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2040253, 27.2117882
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8612976, 33.8392105
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8404160, 37.7847824
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8745346, 27.8701096
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0350113, 31.0408859
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6622620, 41.6532364
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5660706, 29.5662689
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7657318, 22.7723694
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1311417, 19.1289215
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4550018, 25.4656105
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6753998, 24.6928253
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9793701, 21.9797287
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7978745, 22.7930145
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3546944, 24.3625412
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5256882, 30.5282898
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4196243, 26.4338303
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9114227, 21.9366035
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9527588, 23.9740868
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0275955, 26.0360680
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0591278, 25.0532532
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3551178, 21.3601379
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2994080, 33.2911148
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6253204, 27.6384354
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0628815, 28.0715790
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9485321, 28.9689331
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7375031, 38.7735519
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6324921, 40.6395721
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3181915, 41.2978439
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0513000, 31.0647812
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5043411, 26.5286179
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4619598, 18.4668694

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5342280, upper bound: 11.4911668
time: 28.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5313311, upper bound: 11.4942059
time: 30.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1955948, 25.2010384
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7119751, 19.7147484
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2358551, 17.2012329
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0156250, 21.9981689
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7191315, 21.7052536
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7215118, 23.7036171
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1903229, 23.1962738
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5761871, 23.5650406
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4837379, 26.4631271
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2692261, 24.2693253
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2693558, 31.2337646
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8336258, 27.8413696
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2030334, 27.2127838
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8649445, 33.8355637
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8400192, 37.7851791
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8746185, 27.8700256
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0369949, 31.0389023
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6597595, 41.6557465
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5613403, 29.5709991
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7649460, 22.7731476
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1286545, 19.1314163
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4549026, 25.4657059
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6726837, 24.6955338
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9791870, 21.9799118
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7968674, 22.7940216
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3541069, 24.3631248
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5245972, 30.5293732
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4168701, 26.4365845
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9111481, 21.9368782
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9482498, 23.9785919
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0267868, 26.0368690
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0580902, 25.0542946
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3552094, 21.3600502
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3017883, 33.2887421
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6240005, 27.6397476
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0624084, 28.0720444
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9443817, 28.9730759
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7346039, 38.7764511
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6310120, 40.6410370
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3201447, 41.2958832
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0506592, 31.0654221
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5052109, 26.5277405
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4628906, 18.4659348

Time for backsubstitution: 2.16 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5526537, upper bound: 11.4727836
time: 37.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5497591, upper bound: 11.4758325
time: 32.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2062149, 25.1904068
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7200317, 19.7066917
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2387848, 17.1983032
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0137863, 22.0000153
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7219849, 21.7023964
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7260056, 23.6991234
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1891937, 23.1974030
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5773010, 23.5639267
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4892235, 26.4576416
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2721176, 24.2664375
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2683563, 31.2347641
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8299789, 27.8450165
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.1976929, 27.2181168
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8502579, 33.8502502
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8247299, 37.8004761
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8779984, 27.8666458
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0485077, 31.0273895
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6532593, 41.6622391
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5661087, 29.5662308
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7648926, 22.7732048
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1205444, 19.1395226
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4506836, 25.4699287
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6668701, 24.7013550
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9751282, 21.9839821
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7898865, 22.8010025
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3460884, 24.3711472
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5169830, 30.5369949
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4201279, 26.4333344
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9088287, 21.9391899
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9525223, 23.9743156
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0158310, 26.0478325
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0550156, 25.0573692
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3484344, 21.3668251
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2793732, 33.3111496
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6091995, 27.6545563
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0518417, 28.0826187
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9406815, 28.9767838
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7375946, 38.7734528
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6198883, 40.6521606
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.2984467, 41.3175888
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0514221, 31.0646439
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5097580, 26.5231934
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4620438, 18.4667854

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5546692, upper bound: 11.4706181
time: 28.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5517688, upper bound: 11.4736526
time: 35.02 seconds

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

Time for backsubstitution: 2.19 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5730937, upper bound: 11.4522206
time: 30.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5701938, upper bound: 11.4552565
time: 30.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1944504, 25.2021790
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7102127, 19.7165070
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2344055, 17.2026863
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0124664, 22.0013275
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7184753, 21.7059059
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7214508, 23.7036819
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1917267, 23.1948738
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5739441, 23.5672798
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4826164, 26.4642487
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2687302, 24.2698212
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2683792, 31.2347412
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8343048, 27.8406982
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2035828, 27.2122269
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8548584, 33.8456573
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8282394, 37.7969742
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8753433, 27.8693085
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0365601, 31.0393448
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6601715, 41.6553268
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5660706, 29.5662689
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7645493, 22.7735481
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1315536, 19.1285133
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4544067, 25.4662018
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6749573, 24.6932640
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9761200, 21.9829826
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7971878, 22.7937012
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3542595, 24.3629761
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5257111, 30.5282593
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4201813, 26.4332733
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9114227, 21.9365959
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9525223, 23.9743195
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0295181, 26.0341454
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0561066, 25.0562782
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3545990, 21.3606529
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2987976, 33.2917404
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6250381, 27.6387177
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0618439, 28.0726166
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9471436, 28.9703293
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7390747, 38.7719803
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6246796, 40.6473770
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3117065, 41.3043213
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0573120, 31.0587540
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5052643, 26.5276871
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4620056, 18.4668274

Time for backsubstitution: 2.19 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5257999, upper bound: 11.4997980
time: 48.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5227819, upper bound: 11.5027025
time: 32.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1944656, 25.2021599
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7117157, 19.7150078
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2347107, 17.2023811
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0138321, 21.9999619
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7195892, 21.7047844
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7215881, 23.7035446
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1930923, 23.1935043
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5765762, 23.5646515
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4838829, 26.4629822
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2693253, 24.2692184
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2695618, 31.2335587
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8340530, 27.8409424
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2025909, 27.2132225
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8585052, 33.8420029
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8278427, 37.7973709
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8754196, 27.8692245
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0385437, 31.0373535
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6576691, 41.6578369
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5613403, 29.5709991
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7637711, 22.7743301
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1290588, 19.1310081
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4543076, 25.4663010
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6722488, 24.6959724
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9759369, 21.9831657
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7961807, 22.7947083
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3536720, 24.3635635
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5246277, 30.5293503
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4174271, 26.4360352
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9111481, 21.9368744
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9480133, 23.9788284
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0287094, 26.0349464
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0550690, 25.0573196
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3546906, 21.3605690
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3011627, 33.2893677
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6237259, 27.6400299
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0613785, 28.0730820
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9429932, 28.9744720
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7361755, 38.7748795
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6232147, 40.6488495
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3136597, 41.3023605
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0566864, 31.0593948
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5061417, 26.5268173
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4629364, 18.4658928

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5442609, upper bound: 11.4814279
time: 35.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5412623, upper bound: 11.4843270
time: 29.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2051010, 25.1915283
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7197723, 19.7069473
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2376404, 17.1994514
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0119934, 22.0018044
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7224426, 21.7019272
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7260742, 23.6990547
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1919632, 23.1946373
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5776901, 23.5635376
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4893684, 26.4574966
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2722168, 24.2663345
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2685623, 31.2345657
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8304062, 27.8445892
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.1972504, 27.2185593
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8438187, 33.8566895
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8125229, 37.8126678
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8788071, 27.8658447
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0500565, 31.0258408
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6511688, 41.6643295
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5661087, 29.5662270
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7637100, 22.7743835
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1209488, 19.1391144
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4500885, 25.4705200
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6664276, 24.7017937
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9718628, 21.9872360
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7891998, 22.8016891
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3456459, 24.3715858
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5170059, 30.5369720
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4206772, 26.4327774
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9088440, 21.9391861
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9522934, 23.9745483
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0177536, 26.0459099
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0519943, 25.0603943
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3479156, 21.3673439
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2787476, 33.3117752
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6089249, 27.6548309
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0508041, 28.0836563
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9392853, 28.9781799
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7391815, 38.7718811
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6120758, 40.6599731
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.2919617, 41.3240662
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0574646, 31.0586166
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5106888, 26.5222626
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4620819, 18.4667397

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5462489, upper bound: 11.4792444
time: 31.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5432238, upper bound: 11.4821500
time: 30.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2051163, 25.1915131
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7212753, 19.7054520
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2379456, 17.1991425
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0133591, 22.0004349
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7235718, 21.7008018
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7262115, 23.6989212
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1933289, 23.1932678
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5803146, 23.5609131
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4906349, 26.4562302
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2728195, 24.2657318
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2697372, 31.2333832
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8301620, 27.8448410
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.1962585, 27.2195549
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8474731, 33.8530426
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8121262, 37.8130646
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8788910, 27.8657608
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0520477, 31.0238571
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6486664, 41.6668396
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5613785, 29.5709610
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7629318, 22.7751656
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1184616, 19.1416092
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4499893, 25.4706192
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6637268, 24.7045021
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9716797, 21.9874153
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7881927, 22.8027000
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3450661, 24.3721733
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5159149, 30.5380554
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4179230, 26.4355316
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9085693, 21.9394608
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9477844, 23.9790573
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0169449, 26.0467110
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0509491, 25.0614319
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3480072, 21.3672562
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2811279, 33.3094025
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6076050, 27.6561432
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0503311, 28.0841293
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9351425, 28.9823227
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7362823, 38.7747803
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6106110, 40.6614380
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.2939148, 41.3221054
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0568237, 31.0592499
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5115662, 26.5213928
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4630203, 18.4658089

Time for backsubstitution: 2.21 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5646977, upper bound: 11.4608619
time: 40.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5616931, upper bound: 11.4637569
time: 35.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1942978, 25.2023239
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7102737, 19.7164497
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2328339, 17.2043915
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0128098, 22.0009880
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7167206, 21.7076797
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7187653, 23.7065620
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1892700, 23.1971741
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5732269, 23.5680122
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4803810, 26.4666519
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2677307, 24.2708664
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2681351, 31.2349548
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8370285, 27.8380585
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2089615, 27.2068520
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8622589, 33.8382568
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8404465, 37.7847519
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8719330, 27.8724899
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0345764, 31.0413208
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6622467, 41.6532593
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5663376, 29.5660019
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7657166, 22.7723846
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1318436, 19.1282272
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4560852, 25.4646606
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6754303, 24.6928024
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9793243, 21.9797859
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7978516, 22.7930412
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3550148, 24.3622589
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5260773, 30.5280075
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4191818, 26.4343185
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9113770, 21.9366913
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9525299, 23.9743156
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0280838, 26.0356445
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0590515, 25.0532722
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3574219, 21.3576050
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2999878, 33.2905426
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6283340, 27.6356812
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0647354, 28.0698013
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9511719, 28.9663696
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7413483, 38.7697067
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6377106, 40.6343307
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3202667, 41.2957611
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0525208, 31.0634079
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5054703, 26.5274811
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4628296, 18.4659348

Time for backsubstitution: 2.19 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5224196, upper bound: 11.5023539
time: 26.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5195180, upper bound: 11.5053848
time: 32.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1943283, 25.2023087
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7117691, 19.7149506
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2331390, 17.2040863
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0141754, 21.9996185
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7178497, 21.7065544
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7189026, 23.7064285
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1906357, 23.1958084
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5758514, 23.5653877
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4816475, 26.4653854
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2683334, 24.2702637
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2693100, 31.2337723
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8367844, 27.8383026
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2079620, 27.2078476
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8659058, 33.8346024
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8400497, 37.7851486
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8720169, 27.8724060
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0365601, 31.0393372
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6597443, 41.6557617
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5616074, 29.5707321
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7649384, 22.7731628
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1293488, 19.1307182
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4559860, 25.4647598
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6727219, 24.6955109
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9791412, 21.9799690
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7968369, 22.7940521
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3544273, 24.3628464
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5249939, 30.5290985
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4164276, 26.4370804
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9111023, 21.9369659
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9480209, 23.9788208
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0272751, 26.0364494
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0580063, 25.0543137
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3575134, 21.3575172
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3023682, 33.2881699
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6270218, 27.6370010
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0642624, 28.0702744
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9470215, 28.9705124
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7384491, 38.7726059
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6362457, 40.6358032
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3222198, 41.2938080
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0518799, 31.0640411
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5063477, 26.5266113
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4637680, 18.4650002

Time for backsubstitution: 2.17 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5408710, upper bound: 11.4839767
time: 31.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5379796, upper bound: 11.4870194
time: 30.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2049484, 25.1916809
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7198334, 19.7068939
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2360687, 17.2011566
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0123367, 22.0014648
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7207031, 21.7037010
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7233887, 23.7019348
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1895065, 23.1969376
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5769730, 23.5642700
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4871330, 26.4598999
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2712173, 24.2673759
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2683105, 31.2347794
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8331375, 27.8419571
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2026291, 27.2131805
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8512268, 33.8492966
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8247604, 37.8004456
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8753967, 27.8690262
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0480804, 31.0278168
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6532440, 41.6622620
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5663757, 29.5659561
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7648849, 22.7732201
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1212387, 19.1388245
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4517670, 25.4689789
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6669006, 24.7013321
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9750671, 21.9840355
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7898560, 22.8010330
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3464088, 24.3708725
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5173645, 30.5367126
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4196777, 26.4338226
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9087982, 21.9392776
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9523010, 23.9745445
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0163193, 26.0474091
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0549316, 25.0573883
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3507385, 21.3642921
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2799530, 33.3105774
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6122208, 27.6518021
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0536880, 28.0808487
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9433212, 28.9742203
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7414398, 38.7696075
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6251221, 40.6469269
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3005219, 41.3155060
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0526733, 31.0632706
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5108948, 26.5220642
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4629211, 18.4658470

Time for backsubstitution: 2.21 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5428955, upper bound: 11.4818401
time: 34.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5399855, upper bound: 11.4848588
time: 35.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2049637, 25.1916580
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7213287, 19.7053947
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2363739, 17.2008476
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0137024, 22.0000954
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7218323, 21.7025757
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7235260, 23.7017975
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1908722, 23.1955681
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5795975, 23.5616455
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4883995, 26.4586411
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2718201, 24.2667770
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2694931, 31.2335968
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8328857, 27.8422089
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2016296, 27.2141762
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8548737, 33.8456421
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8243637, 37.8008423
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8754807, 27.8689423
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0500641, 31.0258331
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6507416, 41.6647720
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5616455, 29.5706902
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7641068, 22.7739983
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1187515, 19.1413193
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4516678, 25.4690781
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6641922, 24.7040405
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9748840, 21.9842186
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7888489, 22.8020401
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3458214, 24.3714561
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5162811, 30.5378036
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4169235, 26.4365768
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9085236, 21.9395561
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9477921, 23.9790497
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0155106, 26.0482140
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0538940, 25.0584297
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3508224, 21.3642082
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2823181, 33.3082047
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6109085, 27.6531219
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0532227, 28.0813141
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9391708, 28.9783707
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7385559, 38.7724991
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6236572, 40.6483917
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3024750, 41.3135529
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0520325, 31.0639038
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5117645, 26.5211868
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4638519, 18.4649162

Time for backsubstitution: 2.19 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5613325, upper bound: 11.4634534
time: 37.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5584303, upper bound: 11.4664862
time: 33.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1931839, 25.2034454
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7100143, 19.7167091
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2316818, 17.2055397
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0110168, 22.0027771
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7171936, 21.7072105
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7188339, 23.7064934
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1920395, 23.1944046
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5736160, 23.5676231
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4805260, 26.4665070
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2678375, 24.2707596
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2683334, 31.2347488
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8374557, 27.8376312
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2085190, 27.2072906
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8558197, 33.8447037
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8282700, 37.7969437
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8727341, 27.8716888
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0361176, 31.0397797
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6601562, 41.6553497
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5663376, 29.5659981
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7645416, 22.7735596
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1322479, 19.1278191
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4554901, 25.4652519
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6749878, 24.6932411
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9760742, 21.9830399
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7971649, 22.7937279
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3545799, 24.3626976
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5261078, 30.5279846
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4197311, 26.4337692
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9113922, 21.9366875
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9522934, 23.9745483
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0300064, 26.0337219
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0560303, 25.0562973
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3569031, 21.3581200
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2993622, 33.2911682
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6280594, 27.6359634
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0636978, 28.0708389
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9497757, 28.9677582
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7429199, 38.7681351
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6299133, 40.6421356
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3137817, 41.3022461
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0585632, 31.0573807
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5064011, 26.5265579
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4628754, 18.4658928

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5139760, upper bound: 11.5109634
time: 27.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5109435, upper bound: 11.5138589
time: 30.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1931992, 25.2034302
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7115097, 19.7152100
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2319870, 17.2052307
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0123825, 22.0014114
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7183228, 21.7060852
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7189713, 23.7063560
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1934052, 23.1930389
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5762405, 23.5649986
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4817924, 26.4652405
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2684402, 24.2701607
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2695160, 31.2335739
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8372116, 27.8378754
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2075195, 27.2082863
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8594666, 33.8410492
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8278732, 37.7973404
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8728180, 27.8716049
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0381088, 31.0377884
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6576538, 41.6578522
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5616074, 29.5707321
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7637558, 22.7743416
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1297531, 19.1303101
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4553909, 25.4653511
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6722870, 24.6959496
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9758911, 21.9832230
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7961502, 22.7947350
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3539925, 24.3632812
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5250168, 30.5290680
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4169769, 26.4365234
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9111176, 21.9369621
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9477921, 23.9790573
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0291977, 26.0345268
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0549850, 25.0573387
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3569946, 21.3580360
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3017273, 33.2887955
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6267471, 27.6372833
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0632248, 28.0713120
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9456329, 28.9719086
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7400208, 38.7710342
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6284332, 40.6436081
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3157501, 41.3002853
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0579224, 31.0580139
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5072784, 26.5256805
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4638062, 18.4649582

Time for backsubstitution: 2.19 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5324480, upper bound: 11.4926077
time: 30.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5294373, upper bound: 11.4954837
time: 33.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2038345, 25.1928024
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7195740, 19.7071495
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2349167, 17.2023010
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0105438, 22.0032539
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7211761, 21.7032280
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7234650, 23.7018661
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1922760, 23.1941681
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5773544, 23.5638809
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4872780, 26.4597549
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2713242, 24.2672729
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2685165, 31.2345734
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8335648, 27.8415298
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2021866, 27.2136230
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8447800, 33.8557358
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8125534, 37.8126373
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8761978, 27.8682251
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0496216, 31.0262756
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6511536, 41.6643524
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5663834, 29.5659561
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7637024, 22.7743988
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1216507, 19.1384201
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4511719, 25.4695702
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6664581, 24.7017670
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9718170, 21.9872932
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7891693, 22.8017197
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3459663, 24.3713074
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5173950, 30.5366898
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4202347, 26.4332733
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9087982, 21.9392738
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9520645, 23.9747772
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0182419, 26.0454826
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0519104, 25.0604095
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3502197, 21.3648109
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2793274, 33.3112030
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6119385, 27.6520844
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0526505, 28.0818787
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9419250, 28.9756088
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7430115, 38.7680435
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6173248, 40.6547318
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.2940369, 41.3219910
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0586853, 31.0572433
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5118256, 26.5211334
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4629669, 18.4658089

Time for backsubstitution: 2.24 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5344523, upper bound: 11.4904397
time: 31.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5314110, upper bound: 11.4933396
time: 36.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2038498, 25.1927795
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7210693, 19.7056503
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2352219, 17.2019958
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0119095, 22.0018845
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7223053, 21.7021065
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7236023, 23.7017288
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1936417, 23.1928024
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5799866, 23.5612564
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4885445, 26.4584885
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2719269, 24.2666702
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2696915, 31.2333908
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8333130, 27.8417816
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2011948, 27.2146187
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8484344, 33.8520889
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8121567, 37.8130341
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8762817, 27.8681412
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0516052, 31.0242920
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6486511, 41.6668549
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5616455, 29.5706863
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7629242, 22.7751770
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1191559, 19.1409111
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4510727, 25.4696693
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6637573, 24.7044754
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9716339, 21.9874763
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7881622, 22.8027267
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3453865, 24.3718948
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5163116, 30.5377808
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4174728, 26.4360275
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9085236, 21.9395485
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9475632, 23.9792862
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0174332, 26.0462914
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0508728, 25.0614548
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3503113, 21.3647232
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2816925, 33.3088303
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6106262, 27.6533966
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0521851, 28.0823517
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9377823, 28.9797592
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7401276, 38.7709351
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6158447, 40.6562042
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.2960052, 41.3200302
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0580597, 31.0578766
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5126953, 26.5202560
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4638977, 18.4648705

Time for backsubstitution: 2.24 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5529220, upper bound: 11.4720736
time: 21.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5498923, upper bound: 11.4749607
time: 34.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1969681, 25.1996651
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7110214, 19.7157059
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2171402, 17.2199478
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0066757, 22.0071182
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7130432, 21.7113380
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7127151, 23.7124176
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1919785, 23.1946220
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5670700, 23.5741577
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4670143, 26.4798508
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2674179, 24.2711372
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2418747, 31.2612534
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8330154, 27.8419876
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2077713, 27.2080383
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8619080, 33.8386002
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8129807, 37.8122177
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8745804, 27.8700638
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0255585, 31.0503387
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6623230, 41.6531830
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5615387, 29.5708008
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7682571, 22.7698441
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1302261, 19.1298370
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4545593, 25.4660530
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6930695, 24.6751480
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9771271, 21.9819794
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7974625, 22.7934265
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3604851, 24.3567505
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5190048, 30.5349731
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4249268, 26.4285278
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9224243, 21.9256058
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9711990, 23.9556465
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0289993, 26.0346680
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0586853, 25.0536995
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3616180, 21.3536377
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3052521, 33.2852783
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6364517, 27.6273041
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0753250, 28.0591354
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9691010, 28.9483566
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7594604, 38.7515945
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6461945, 40.6258545
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3200378, 41.2959976
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0552673, 31.0608139
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5175705, 26.5153809
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4620209, 18.4668083

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5229504, upper bound: 11.5025289
time: 34.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5200905, upper bound: 11.5055606
time: 32.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1969833, 25.1996460
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7125168, 19.7142067
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2174454, 17.2196426
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0080490, 22.0057487
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7141724, 21.7102089
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7128525, 23.7122803
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1933441, 23.1932564
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5696945, 23.5715332
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4682732, 26.4785919
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2680130, 24.2705345
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2430496, 31.2600708
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8327637, 27.8422394
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2067719, 27.2090340
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8655548, 33.8349533
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8125839, 37.8126144
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8746643, 27.8699875
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0275421, 31.0483551
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6598206, 41.6556854
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5568085, 29.5755310
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7674713, 22.7706223
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1277390, 19.1323318
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4544601, 25.4661484
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6903687, 24.6778564
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9769440, 21.9821625
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7964554, 22.7944336
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3598976, 24.3573380
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5179138, 30.5360565
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4221725, 26.4312897
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9221497, 21.9258804
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9666901, 23.9601555
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0281906, 26.0354729
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0576401, 25.0547409
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3617096, 21.3535500
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3076172, 33.2829056
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6351395, 27.6286163
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0748520, 28.0596085
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9649582, 28.9525070
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7565613, 38.7544937
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6447296, 40.6273270
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3219910, 41.2940369
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0546265, 31.0614471
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5184479, 26.5145111
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4629517, 18.4658775

Time for backsubstitution: 2.19 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5413255, upper bound: 11.4840327
time: 31.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5384577, upper bound: 11.4870817
time: 36.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2076187, 25.1890182
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7205734, 19.7061462
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2203751, 17.2167130
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0062027, 22.0075912
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7170258, 21.7073555
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7173462, 23.7077904
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1922150, 23.1943855
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5708084, 23.5704193
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4737663, 26.4731064
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2709045, 24.2676468
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2420502, 31.2610779
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8291168, 27.8458862
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2014389, 27.2143707
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8508682, 33.8496399
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7972946, 37.8279190
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8780441, 27.8666000
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0390549, 31.0368423
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6533203, 41.6621857
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5615845, 29.5707550
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7674179, 22.7706833
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1196289, 19.1404381
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4502411, 25.4703712
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6845474, 24.6836777
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9728699, 21.9862328
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7894745, 22.8014183
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3518715, 24.3653603
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5102921, 30.5436783
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4254227, 26.4280319
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9198303, 21.9281960
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9709702, 23.9558792
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0172348, 26.0464287
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0545654, 25.0578156
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3549347, 21.3603249
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2852173, 33.3053131
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6203308, 27.6434174
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0642776, 28.0701828
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9612503, 28.9562149
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7595673, 38.7514954
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6335907, 40.6384506
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3002930, 41.3157349
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0554047, 31.0606689
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5229950, 26.5099640
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4621048, 18.4667244

Time for backsubstitution: 2.19 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5434393, upper bound: 11.4820256
time: 31.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5405756, upper bound: 11.4850447
time: 32.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2076340, 25.1889992
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7220764, 19.7046509
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2206802, 17.2164078
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0075760, 22.0062256
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7181396, 21.7062302
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7174759, 23.7076530
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1935806, 23.1930161
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5734329, 23.5677910
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4750328, 26.4718399
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2715073, 24.2670479
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2432327, 31.2598953
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8288651, 27.8461380
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2004471, 27.2153664
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8545227, 33.8459854
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7968979, 37.8283157
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8781281, 27.8665237
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0410461, 31.0348511
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6508179, 41.6646881
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5568542, 29.5754890
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7666397, 22.7714615
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1171341, 19.1429291
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4501419, 25.4704666
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6818390, 24.6863861
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9726868, 21.9864159
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7884674, 22.8024254
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3512917, 24.3659477
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5092010, 30.5447693
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4226685, 26.4307861
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9195557, 21.9284706
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9664612, 23.9603844
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0164261, 26.0472374
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0535278, 25.0588570
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3550186, 21.3602409
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2875824, 33.3029404
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6190186, 27.6447372
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0638046, 28.0706558
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9571075, 28.9603577
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7566681, 38.7543869
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6321259, 40.6399231
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3022461, 41.3137817
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0547791, 31.0613022
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5238647, 26.5090866
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4630432, 18.4657898

Time for backsubstitution: 2.20 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5617952, upper bound: 11.4635257
time: 32.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5589091, upper bound: 11.4665640
time: 32.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1958389, 25.2007866
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7107620, 19.7159653
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2159882, 17.2210999
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0048904, 22.0089073
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7135162, 21.7108650
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7127838, 23.7123489
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1947403, 23.1918564
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5674591, 23.5737686
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4671593, 26.4797134
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2675171, 24.2710342
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2420731, 31.2610550
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8334427, 27.8415604
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2073288, 27.2084808
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8554688, 33.8450470
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8007736, 37.8244171
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8753815, 27.8692703
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0270996, 31.0487976
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6602325, 41.6552734
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5615387, 29.5707970
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7670746, 22.7710228
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1306381, 19.1294289
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4539642, 25.4666443
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6926346, 24.6755867
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9738770, 21.9852333
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7967758, 22.7941132
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3600502, 24.3571854
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5190277, 30.5349426
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4254761, 26.4279785
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9224243, 21.9256020
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9709625, 23.9558792
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0309219, 26.0327415
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0556641, 25.0567245
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3610992, 21.3541527
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3046265, 33.2859039
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6361694, 27.6275864
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0742874, 28.0601730
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9677124, 28.9497528
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7610321, 38.7500229
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6383820, 40.6336670
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3135529, 41.3024750
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0612793, 31.0547867
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5185013, 26.5144577
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4620590, 18.4667664

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5145336, upper bound: 11.5110918
time: 28.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5115359, upper bound: 11.5140037
time: 30.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1958542, 25.2007675
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7122574, 19.7144623
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2162933, 17.2207909
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0062561, 22.0075378
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7146454, 21.7097397
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7129211, 23.7122116
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1961136, 23.1904869
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5700836, 23.5711441
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4684258, 26.4784470
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2681198, 24.2704315
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2432480, 31.2598724
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8331909, 27.8418121
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2063370, 27.2094765
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8591156, 33.8413925
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8004074, 37.8248062
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8754654, 27.8691788
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0290833, 31.0468140
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6577301, 41.6577759
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5568085, 29.5755310
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7662964, 22.7718048
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1281433, 19.1319237
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4538651, 25.4667435
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6899261, 24.6782951
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9736938, 21.9854164
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7957687, 22.7951202
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3594627, 24.3577728
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5179443, 30.5360336
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4227219, 26.4307327
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9221497, 21.9258766
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9664536, 23.9603882
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0301132, 26.0335503
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0546188, 25.0577621
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3611908, 21.3540688
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3070068, 33.2835312
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6348495, 27.6288986
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0738144, 28.0606461
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9635620, 28.9538956
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7581329, 38.7529221
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6369171, 40.6351318
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3155060, 41.3005142
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0606537, 31.0554199
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5193710, 26.5135803
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4629974, 18.4658318

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5329024, upper bound: 11.4926266
time: 34.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5298987, upper bound: 11.4955490
time: 28.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2064896, 25.1901398
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7203140, 19.7064056
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2192307, 17.2178612
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0044174, 22.0093842
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7174988, 21.7068863
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7174072, 23.7077217
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1949844, 23.1916199
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5711975, 23.5700302
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4739113, 26.4729538
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2710037, 24.2675438
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2422485, 31.2608719
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8295441, 27.8454590
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2010040, 27.2148132
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8444290, 33.8560791
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7850876, 37.8401108
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8788452, 27.8658066
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0406036, 31.0352936
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6512299, 41.6642761
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5615845, 29.5707550
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7662354, 22.7718582
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1200333, 19.1400299
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4496460, 25.4709625
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6841049, 24.6841164
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9696198, 21.9894829
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7887878, 22.8021049
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3514366, 24.3657990
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5103226, 30.5436554
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4259720, 26.4274826
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9198303, 21.9281883
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9707336, 23.9561119
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0191574, 26.0445061
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0515442, 25.0608368
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3544159, 21.3608437
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2845917, 33.3059387
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6200485, 27.6437073
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0632401, 28.0712204
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9598618, 28.9576035
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7611237, 38.7499313
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6257935, 40.6462555
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.2938080, 41.3222198
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0614319, 31.0546417
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5239258, 26.5090332
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4621429, 18.4666786

Time for backsubstitution: 2.85 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5350215, upper bound: 11.4905828
time: 27.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5320190, upper bound: 11.4934888
time: 36.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2065048, 25.1901207
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7218170, 19.7049065
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2195358, 17.2175560
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0057831, 22.0080147
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7186127, 21.7057610
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7175446, 23.7075844
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1963501, 23.1902504
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5738297, 23.5674019
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4751778, 26.4716949
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2716064, 24.2669449
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2434311, 31.2596970
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8292923, 27.8457031
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2000046, 27.2158051
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8480835, 33.8524323
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7846909, 37.8405075
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8789291, 27.8657150
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0425873, 31.0333099
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6487274, 41.6667786
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5568542, 29.5754852
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7654572, 22.7726402
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1175461, 19.1425247
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4495468, 25.4710617
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6813965, 24.6868248
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9694366, 21.9896660
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7877808, 22.8031120
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3508492, 24.3663826
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5092316, 30.5447388
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4232178, 26.4302368
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9195557, 21.9284630
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9662247, 23.9606171
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0183487, 26.0453148
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0505066, 25.0618820
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3545074, 21.3607559
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2869568, 33.3035660
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6187363, 27.6450195
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0627747, 28.0716858
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9557114, 28.9617538
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7582245, 38.7528229
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6243134, 40.6477280
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.2957611, 41.3202591
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0607910, 31.0552750
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5247955, 26.5081558
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4630737, 18.4657478

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5533614, upper bound: 11.4721163
time: 34.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5503538, upper bound: 11.4750312
time: 31.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1957016, 25.2009354
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7108154, 19.7159042
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2144165, 17.2228012
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0052261, 22.0085678
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7117767, 21.7126389
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7100983, 23.7152290
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1922836, 23.1941566
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5667343, 23.5745010
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4649239, 26.4821091
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2665176, 24.2720757
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2418289, 31.2612686
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8361664, 27.8389282
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2127075, 27.2031021
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8628693, 33.8376465
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8130112, 37.8121872
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8719711, 27.8724442
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0251160, 31.0507812
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6623077, 41.6531982
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5618134, 29.5705299
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7682419, 22.7698593
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1309280, 19.1291428
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4556427, 25.4651031
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6931076, 24.6751251
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9770813, 21.9820366
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7974319, 22.7934570
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3608055, 24.3564720
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5193939, 30.5346985
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4244766, 26.4290237
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9223785, 21.9256935
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9709702, 23.9558754
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0294876, 26.0342445
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0586014, 25.0537186
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3639221, 21.3511047
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3058167, 33.2847061
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6394730, 27.6245575
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0771713, 28.0573654
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9717407, 28.9457932
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7633057, 38.7477493
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6514282, 40.6206207
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3221130, 41.2939148
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0564880, 31.0594406
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5187073, 26.5142517
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4628906, 18.4658775

Time for backsubstitution: 2.21 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5111482, upper bound: 11.5137076
time: 29.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5082608, upper bound: 11.5167358
time: 29.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1957169, 25.2009163
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7123184, 19.7144089
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2147217, 17.2224960
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0065994, 22.0072021
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7128906, 21.7115135
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7102356, 23.7150917
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1936569, 23.1927872
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5693665, 23.5718765
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4661827, 26.4808502
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2671204, 24.2714767
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2430038, 31.2600861
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8359146, 27.8391724
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2117157, 27.2041016
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8665161, 33.8339920
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8126144, 37.8125839
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8720627, 27.8723679
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0271072, 31.0487900
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6597900, 41.6557083
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5570755, 29.5752602
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7674637, 22.7706375
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1284332, 19.1316338
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4555435, 25.4652023
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6903992, 24.6778336
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9768982, 21.9822197
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7964249, 22.7944641
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3602180, 24.3570557
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5183029, 30.5357819
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4217224, 26.4317780
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9221039, 21.9259682
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9664612, 23.9603806
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0286789, 26.0350494
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0575638, 25.0547600
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3640137, 21.3510170
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3081970, 33.2823334
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6381531, 27.6258698
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0766983, 28.0578308
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9675980, 28.9499435
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7604065, 38.7506485
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6499634, 40.6220856
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3240662, 41.2919617
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0558624, 31.0600739
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5195770, 26.5133743
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4638214, 18.4649429

Time for backsubstitution: 2.32 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5295234, upper bound: 11.4952060
time: 32.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5266258, upper bound: 11.4982499
time: 39.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2063370, 25.1902847
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7203751, 19.7063484
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2176514, 17.2195663
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0047531, 22.0090408
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7157440, 21.7086601
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7147293, 23.7106018
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1925278, 23.1939163
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5704803, 23.5707626
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4716759, 26.4753647
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2700119, 24.2685852
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2420044, 31.2610855
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8322678, 27.8428192
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2063751, 27.2094345
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8518372, 33.8486862
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7973251, 37.8278885
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8754425, 27.8689804
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0386200, 31.0372772
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6533051, 41.6622086
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5618515, 29.5704880
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7674103, 22.7706947
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1203232, 19.1397400
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4513245, 25.4694214
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6845779, 24.6836548
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9728241, 21.9862862
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7894440, 22.8014450
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3521919, 24.3650818
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5106812, 30.5434036
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4249802, 26.4285278
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9197845, 21.9282799
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9707413, 23.9561043
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0177231, 26.0460091
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0544891, 25.0578346
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3572388, 21.3577919
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2857819, 33.3047485
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6233521, 27.6406708
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0661316, 28.0684052
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9638901, 28.9536514
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7634125, 38.7476501
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6388397, 40.6332092
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3023682, 41.3136597
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0566406, 31.0592957
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5241241, 26.5088272
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4629822, 18.4657898

Time for backsubstitution: 2.31 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5316657, upper bound: 11.4932404
time: 33.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5287873, upper bound: 11.4962589
time: 35.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2063675, 25.1902695
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7218704, 19.7048492
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2179642, 17.2192612
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0061264, 22.0076752
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7168732, 21.7075348
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7148666, 23.7104645
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1938934, 23.1925507
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5731049, 23.5681343
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4729424, 26.4740982
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2706070, 24.2679863
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2431870, 31.2599106
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8320160, 27.8430710
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2053833, 27.2104301
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8554840, 33.8450317
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7969284, 37.8282852
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8755264, 27.8689041
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0406036, 31.0352936
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6507874, 41.6647186
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5571213, 29.5752182
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7666245, 22.7714767
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1178360, 19.1422348
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4512253, 25.4695206
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6818695, 24.6863632
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9726410, 21.9864693
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7884369, 22.8024521
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3516121, 24.3656693
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5095978, 30.5444946
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4222183, 26.4312820
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9195099, 21.9285545
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9662323, 23.9606133
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0169144, 26.0468102
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0534439, 25.0588760
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3573227, 21.3577080
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2881622, 33.3023682
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6220322, 27.6419830
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0656586, 28.0688705
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9597473, 28.9577942
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7605133, 38.7505493
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6373749, 40.6346893
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3043213, 41.3117065
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0559998, 31.0599289
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5250015, 26.5079575
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4639130, 18.4648552

Time for backsubstitution: 2.28 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5500308, upper bound: 11.4747331
time: 30.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5471338, upper bound: 11.4777736
time: 36.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1945724, 25.2020569
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7105560, 19.7161636
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2132721, 17.2239494
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0034409, 22.0103607
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7122345, 21.7121658
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7101669, 23.7151566
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1950531, 23.1913872
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5671310, 23.5741119
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4650688, 26.4819717
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2666245, 24.2719727
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2420273, 31.2610626
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8365936, 27.8384933
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2122650, 27.2035446
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8564301, 33.8440933
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8008041, 37.8243866
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8727798, 27.8716507
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0266647, 31.0492325
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6602173, 41.6552887
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5618134, 29.5705261
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7670670, 22.7710381
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1313324, 19.1287308
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4550476, 25.4656944
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6926651, 24.6755638
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9738159, 21.9852905
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7967453, 22.7941437
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3603706, 24.3569107
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5194244, 30.5346680
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4250336, 26.4284668
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9223785, 21.9256897
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9707413, 23.9561081
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0314102, 26.0323219
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0555801, 25.0567398
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3634033, 21.3516197
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3051910, 33.2853317
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6391830, 27.6248322
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0761337, 28.0584030
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9703522, 28.9471893
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7648773, 38.7461777
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6436310, 40.6284256
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3156281, 41.3003998
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0625305, 31.0534134
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5196304, 26.5133209
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4629364, 18.4658318

Time for backsubstitution: 2.32 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5026778, upper bound: 11.5222778
time: 30.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4996451, upper bound: 11.5251835
time: 36.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1945877, 25.2020378
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7120590, 19.7146645
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2135773, 17.2236443
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0048065, 22.0089912
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7133636, 21.7110405
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7103043, 23.7150230
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1964264, 23.1900177
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5697556, 23.5714874
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4663353, 26.4807053
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2672272, 24.2713699
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2432022, 31.2598801
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8363419, 27.8387451
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2112732, 27.2045403
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8600769, 33.8404388
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8004379, 37.8247757
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8728638, 27.8715591
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0286560, 31.0472412
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6576996, 41.6577988
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5570831, 29.5752602
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7662888, 22.7718163
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1288376, 19.1312256
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4549484, 25.4657936
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6899567, 24.6782684
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9736328, 21.9854698
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7957382, 22.7951508
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3597832, 24.3574944
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5183334, 30.5357513
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4222794, 26.4312286
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9221039, 21.9259644
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9662247, 23.9606171
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0306015, 26.0331230
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0545349, 25.0577850
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3634949, 21.3515358
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3075714, 33.2829590
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6378708, 27.6261520
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0756683, 28.0588684
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9662018, 28.9513321
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7619781, 38.7490768
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6421509, 40.6298981
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3175812, 41.2984390
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0618896, 31.0540390
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5205078, 26.5124435
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4638672, 18.4649010

Time for backsubstitution: 2.28 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5210440, upper bound: 11.5038096
time: 30.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5180207, upper bound: 11.5067257
time: 29.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2052231, 25.1914062
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7201157, 19.7066078
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2165070, 17.2207146
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0029678, 22.0108337
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7162170, 21.7081871
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7147980, 23.7105331
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1952896, 23.1911507
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5708694, 23.5703735
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4718208, 26.4752121
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2701111, 24.2684822
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2422028, 31.2608871
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8326950, 27.8423920
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2059326, 27.2098770
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8453903, 33.8551254
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7851181, 37.8400803
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8762436, 27.8681870
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0401688, 31.0357285
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6511993, 41.6642914
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5618515, 29.5704880
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7662277, 22.7718735
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1207352, 19.1393356
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4507294, 25.4700127
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6841431, 24.6840897
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9695740, 21.9895401
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7887573, 22.8021317
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3517570, 24.3655205
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5107117, 30.5433731
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4255295, 26.4279709
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9197998, 21.9282761
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9705048, 23.9563370
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0196457, 26.0440865
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0514679, 25.0608597
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3567200, 21.3583107
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2851562, 33.3053741
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6230698, 27.6409531
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0650940, 28.0694427
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9625015, 28.9550400
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7649689, 38.7460861
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6310272, 40.6410217
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.2958832, 41.3201447
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0626526, 31.0532684
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5250549, 26.5078964
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4630203, 18.4657478

Time for backsubstitution: 2.32 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5232143, upper bound: 11.5018064
time: 37.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5201720, upper bound: 11.5047042
time: 31.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2052383, 25.1913910
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7216110, 19.7051086
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2168121, 17.2204056
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0043335, 22.0094643
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7173462, 21.7070618
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7149353, 23.7103958
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1966629, 23.1897812
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5734940, 23.5677452
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4730873, 26.4739532
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2707138, 24.2678833
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2433853, 31.2597046
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8324432, 27.8426437
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2049408, 27.2108688
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8490448, 33.8514786
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7847214, 37.8404770
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8763275, 27.8680954
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0421524, 31.0337448
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6486969, 41.6668015
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5571213, 29.5752182
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7654495, 22.7726555
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1182404, 19.1418266
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4506302, 25.4701118
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6814270, 24.6867981
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9693909, 21.9897232
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7877502, 22.8031387
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3511696, 24.3661041
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5096283, 30.5444641
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4227753, 26.4307251
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9195251, 21.9285545
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9659958, 23.9608459
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0188370, 26.0448875
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0504227, 25.0618973
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3568115, 21.3582230
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2875366, 33.3029938
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6217575, 27.6422729
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0646286, 28.0699081
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9583511, 28.9591904
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7620697, 38.7489777
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6295624, 40.6424942
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.2978363, 41.3181839
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0620270, 31.0539017
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5259247, 26.5070267
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4639587, 18.4648132

Time for backsubstitution: 2.35 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5415618, upper bound: 11.4833383
time: 35.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5385278, upper bound: 11.4862426
time: 28.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1913834, 25.2052422
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7051086, 19.7216148
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2204056, 17.2168121
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0094681, 22.0043335
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7070618, 21.7173424
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7103958, 23.7149315
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1897812, 23.1966629
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5677414, 23.5734940
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4739494, 26.4730835
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2678833, 24.2707138
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2597046, 31.2433853
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8426437, 27.8324432
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2108765, 27.2049408
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8514709, 33.8490372
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8404770, 37.7847366
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8680954, 27.8763275
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0337448, 31.0421524
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6668091, 41.6486969
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5752182, 29.5571213
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7726517, 22.7654457
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1418228, 19.1182404
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4701080, 25.4506340
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6867981, 24.6814308
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9897308, 21.9693871
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8031387, 22.7877502
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3661079, 24.3511734
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5444641, 30.5096283
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4307327, 26.4227753
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9285583, 21.9195213
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9608459, 23.9659996
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0448837, 26.0188370
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0618973, 25.0504227
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3582230, 21.3568077
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3029938, 33.2875366
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6422653, 27.6217499
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0699081, 28.0646286
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9591827, 28.9583511
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7489777, 38.7620773
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6425018, 40.6295624
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3181763, 41.2978439
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0538940, 31.0620270
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5070267, 26.5259247
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4648132, 18.4639549

Time for backsubstitution: 2.31 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4862427, upper bound: 11.5385278
time: 32.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4833383, upper bound: 11.5415618
time: 39.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1914139, 25.2052231
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7066040, 19.7201157
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2207184, 17.2165070
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0108337, 22.0029640
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7081909, 21.7162209
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7105331, 23.7147980
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1911469, 23.1952934
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5703735, 23.5708694
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4752159, 26.4718170
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2684860, 24.2701149
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2608871, 31.2422028
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8423920, 27.8326950
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2098770, 27.2059326
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8551254, 33.8453903
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8400803, 37.7851257
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8681793, 27.8762436
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0357285, 31.0401688
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6642914, 41.6512070
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5704880, 29.5618515
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7718735, 22.7662277
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1393356, 19.1207314
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4700165, 25.4507294
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6840897, 24.6841393
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9895477, 21.9695702
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8021317, 22.7887573
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3655205, 24.3517570
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5433731, 30.5107117
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4279709, 26.4255295
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9282684, 21.9197960
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9563370, 23.9705086
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0440903, 26.0196419
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0608597, 25.0514641
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3583069, 21.3567200
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3053741, 33.2851562
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6409531, 27.6230698
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0694427, 28.0650940
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9550400, 28.9625015
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7460785, 38.7649765
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6410217, 40.6310272
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3201447, 41.2958832
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0532684, 31.0626602
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5079041, 26.5250549
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4657440, 18.4630241

Time for backsubstitution: 2.35 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5047042, upper bound: 11.5201719
time: 33.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5018065, upper bound: 11.5232143
time: 32.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2020340, 25.1945953
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7146683, 19.7120590
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2236481, 17.2135773
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0089874, 22.0048065
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7110443, 21.7133636
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7150269, 23.7103043
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1900177, 23.1964264
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5714874, 23.5697556
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4807014, 26.4663315
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2713699, 24.2672272
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2598801, 31.2432022
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8387451, 27.8363419
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2045441, 27.2112694
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8404388, 33.8600769
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8247910, 37.8004227
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8715591, 27.8728638
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0472412, 31.0286560
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6577911, 41.6577072
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5752563, 29.5570793
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7718201, 22.7662811
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1312256, 19.1288376
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4657898, 25.4549522
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6782684, 24.6899605
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9854736, 21.9736366
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7951508, 22.7957420
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3574944, 24.3597794
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5357513, 30.5183334
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4312286, 26.4222794
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9259644, 21.9221077
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9606171, 23.9662323
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0331192, 26.0306015
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0577850, 25.0545387
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3515320, 21.3634987
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2829590, 33.3075714
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6261520, 27.6378708
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0588684, 28.0756683
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9513321, 28.9662094
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7490692, 38.7619781
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6298981, 40.6421509
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.2984314, 41.3175888
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0540466, 31.0618896
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5124512, 26.5205078
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4649048, 18.4638710

Time for backsubstitution: 2.33 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5067257, upper bound: 11.5180207
time: 32.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5038096, upper bound: 11.5210440
time: 37.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2020493, 25.1945763
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7161636, 19.7105598
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2239532, 17.2132721
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0103607, 22.0034409
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7121735, 21.7122383
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7151566, 23.7101669
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1913834, 23.1950569
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5741119, 23.5671272
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4819679, 26.4650650
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2719727, 24.2666245
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2610626, 31.2420273
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8384933, 27.8365936
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2035446, 27.2122650
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8440933, 33.8564224
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8243942, 37.8008194
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8716431, 27.8727798
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0492325, 31.0266647
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6552887, 41.6602097
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5705261, 29.5618095
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7710342, 22.7670631
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1287308, 19.1313324
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4656982, 25.4550476
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6755676, 24.6926651
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9852905, 21.9738197
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7941437, 22.7967491
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3569069, 24.3603668
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5346680, 30.5194244
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4284744, 26.4250336
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9256897, 21.9223824
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9561081, 23.9707375
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0323257, 26.0314064
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0567474, 25.0555801
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3516235, 21.3634071
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2853394, 33.3051910
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6248398, 27.6391907
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0584030, 28.0761337
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9471893, 28.9703522
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7461853, 38.7648697
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6284332, 40.6436234
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3003998, 41.3156281
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0534058, 31.0625229
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5133209, 26.5196304
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4658356, 18.4629364

Time for backsubstitution: 2.29 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5251835, upper bound: 11.4996451
time: 35.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5222778, upper bound: 11.5026778
time: 31.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1902695, 25.2063637
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7048492, 19.7218742
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2192612, 17.2179642
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0076752, 22.0061226
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7075348, 21.7168732
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7104645, 23.7148628
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1925507, 23.1938934
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5681381, 23.5731049
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4741020, 26.4729385
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2679825, 24.2706108
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2599106, 31.2431870
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8430710, 27.8320160
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2104340, 27.2053795
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8450317, 33.8554840
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8282700, 37.7969284
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8688965, 27.8755264
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0352936, 31.0406036
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6647186, 41.6507874
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5752182, 29.5571213
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7714767, 22.7666245
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1422348, 19.1178322
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4695206, 25.4512253
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6863632, 24.6818695
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9864655, 21.9726372
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8024521, 22.7884369
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3656654, 24.3516083
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5444946, 30.5095978
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4312820, 26.4222260
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9285583, 21.9195137
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9606094, 23.9662323
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0468063, 26.0169144
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0588760, 25.0534439
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3577118, 21.3573227
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3023682, 33.2881622
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6419907, 27.6220322
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0688782, 28.0656586
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9577942, 28.9597473
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7505493, 38.7605057
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6346893, 40.6373672
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3117065, 41.3043213
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0599365, 31.0559998
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5079575, 26.5250015
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4648590, 18.4639130

Time for backsubstitution: 2.31 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4777736, upper bound: 11.5471338
time: 39.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4747331, upper bound: 11.5500308
time: 35.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.1902847, 25.2063446
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7063522, 19.7203751
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2195663, 17.2176552
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0090408, 22.0047531
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7086639, 21.7157478
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7106018, 23.7147293
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1939163, 23.1925240
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5707626, 23.5704803
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4753609, 26.4716721
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2685852, 24.2700119
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2610855, 31.2420044
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8428192, 27.8322678
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2094345, 27.2063751
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8486862, 33.8518295
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8278732, 37.7973251
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8689804, 27.8754425
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0372772, 31.0386200
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6622009, 41.6532974
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5704880, 29.5618515
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7706985, 22.7674065
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1397400, 19.1203232
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4694214, 25.4513245
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6836548, 24.6845779
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9862823, 21.9728203
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8014450, 22.7894440
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3650856, 24.3521957
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5434036, 30.5106812
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4285202, 26.4249802
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9282837, 21.9197922
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9561081, 23.9707413
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0460129, 26.0177193
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0578384, 25.0544853
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3577881, 21.3572350
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3047485, 33.2857819
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6406708, 27.6233521
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0684052, 28.0661316
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9536514, 28.9638901
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7476501, 38.7634048
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6332092, 40.6388397
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3136597, 41.3023682
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0592957, 31.0566330
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5088272, 26.5241241
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4657898, 18.4629784

Time for backsubstitution: 2.37 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4962589, upper bound: 11.5287873
time: 35.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4932404, upper bound: 11.5316657
time: 28.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2009201, 25.1957169
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7144089, 19.7123146
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2224960, 17.2147255
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0072021, 22.0065994
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7115173, 21.7128944
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7150879, 23.7102356
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1927872, 23.1936569
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5718765, 23.5693626
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4808464, 26.4661865
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2714767, 24.2671204
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2600861, 31.2430038
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8391724, 27.8359146
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2041016, 27.2117081
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8339996, 33.8665161
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8125839, 37.8126221
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8723679, 27.8720627
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0487900, 31.0271072
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6557159, 41.6597900
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5752563, 29.5570755
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7706375, 22.7674637
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1316299, 19.1284332
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4652023, 25.4555435
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6778336, 24.6903992
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9822235, 21.9768906
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7944641, 22.7964287
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3570595, 24.3602180
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5357819, 30.5183029
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4317780, 26.4217224
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9259644, 21.9221039
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9603806, 23.9664612
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0350571, 26.0286789
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0547638, 25.0575600
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3510132, 21.3640099
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2823334, 33.3081970
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6258698, 27.6381531
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0578384, 28.0766983
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9499435, 28.9675980
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7506561, 38.7604065
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6220856, 40.6499634
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.2919617, 41.3240662
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0600739, 31.0558624
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5133743, 26.5195770
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4649353, 18.4638252

Time for backsubstitution: 2.32 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4982499, upper bound: 11.5266258
time: 31.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4952060, upper bound: 11.5295234
time: 38.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2009354, 25.1956978
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7159042, 19.7108154
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2228012, 17.2144203
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0085678, 22.0052299
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7126312, 21.7117691
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7152252, 23.7100983
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1941528, 23.1922874
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5745010, 23.5667381
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.4821129, 26.4649200
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2720718, 24.2665215
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2612686, 31.2418289
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8389206, 27.8361664
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2031097, 27.2127075
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8376465, 33.8628693
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8121872, 37.8130188
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8724442, 27.8719788
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0507812, 31.0251160
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6531982, 41.6623001
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5705261, 29.5618095
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7698593, 22.7682419
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1291428, 19.1309242
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.4651031, 25.4556427
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6751251, 24.6931076
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -21.9820404, 21.9770737
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.7934570, 22.7974358
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3564720, 24.3608017
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5346985, 30.5193939
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4290237, 26.4244843
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9256897, 21.9223785
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9558716, 23.9709702
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0342484, 26.0294838
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0537186, 25.0586014
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3511047, 21.3639259
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.2846985, 33.3058167
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.6245499, 27.6394653
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.0573654, 28.0771713
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -28.9457932, 28.9717407
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7477570, 38.7633057
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.6206207, 40.6514282
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.2939148, 41.3221130
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0594482, 31.0564957
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5142517, 26.5186996
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4658737, 18.4628944

Time for backsubstitution: 2.36 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 46.12 + 7155.20 = 7201.32 seconds
