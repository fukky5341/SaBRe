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
execution time: IAR + RelationalAnalysis = 2.80 + 43.31 = 46.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -11.5844329, upper bound: 11.5844329

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 534

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 877

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5491362, upper bound: 11.5841564
time: 31.98 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5841564, upper bound: 11.5491362
time: 34.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 66.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 66.78
Output dim: 2, lower bound: -11.5491362, upper bound: 11.5841564
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 66.78
Output dim: 2, lower bound: -11.5841564, upper bound: 11.5491362

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2645950, 25.2674065
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7711143, 19.7740250
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2882233, 17.2905960
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0245819, 22.0285835
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7564125, 21.7600555
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7726173, 23.7778625
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2220840, 23.2221909
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6058922, 23.6093025
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5310936, 26.5325470
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2896423, 24.2942123
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3097610, 31.3105011
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8935928, 27.8916245
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2298889, 27.2275085
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9115143, 33.9157867
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8080292, 37.8027878
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9280243, 27.9305038
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0670547, 31.0712662
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6463470, 41.6379547
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6182251, 29.6149025
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7947006, 22.7911911
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2251740, 19.2206764
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5330505, 25.5304604
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7341461, 24.7302017
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0404663, 22.0353889
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8565369, 22.8508224
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4210434, 24.4146652
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6449738, 30.6396255
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4643250, 26.4639816
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -22.0075417, 22.0004883
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9715385, 23.9692383
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1356087, 26.1290207
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0921173, 25.0886612
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3945847, 21.3957062
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4144897, 33.4086304
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7934036, 27.7891159
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2118530, 28.2102432
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0760880, 29.0764618
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8287048, 38.8253250
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7540588, 40.7540665
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4119720, 41.4119492
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0530472, 31.0515976
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5298462, 26.5319977
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4745789, 18.4744339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1554

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 672

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5454162, upper bound: 11.5808416
time: 32.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5458584, upper bound: 11.5804065
time: 38.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2674103, 25.2645950
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7740288, 19.7711143
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2905960, 17.2882271
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0285797, 22.0245781
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7600594, 21.7564125
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7778587, 23.7726212
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2221909, 23.2220879
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6093102, 23.6058960
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5325584, 26.5310974
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2942123, 24.2896423
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3105011, 31.3097610
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8916245, 27.8935852
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2275085, 27.2298889
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9157867, 33.9115143
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8027802, 37.8080215
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9305038, 27.9280243
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0712662, 31.0670624
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6379547, 41.6463470
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6148987, 29.6182213
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7911911, 22.7947044
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2206802, 19.2251740
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5304565, 25.5330505
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7302017, 24.7341423
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0353928, 22.0404701
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8508224, 22.8565369
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4146652, 24.4210434
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6396332, 30.6449738
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4639816, 26.4643250
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -22.0004921, 22.0075417
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9692345, 23.9715424
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1290245, 26.1356087
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0886612, 25.0921173
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3957062, 21.3945808
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4086304, 33.4144821
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7891235, 27.7934036
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2102432, 28.2118530
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0764618, 29.0760880
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8253174, 38.8287125
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7540741, 40.7540588
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4119568, 41.4119720
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0515976, 31.0530548
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5319977, 26.5298462
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4744339, 18.4745789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 618

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1407

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5835553, upper bound: 11.5422596
time: 39.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5772841, upper bound: 11.5485415
time: 34.81 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 76.66 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 76.66
Output dim: 2, lower bound: -11.5454162, upper bound: 11.5808416
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 76.66
Output dim: 2, lower bound: -11.5458584, upper bound: 11.5804065
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 76.66
Output dim: 2, lower bound: -11.5835553, upper bound: 11.5422596
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 76.66
Output dim: 2, lower bound: -11.5772841, upper bound: 11.5485415

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2644958, 25.2679482
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7705803, 19.7754517
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2879524, 17.2921257
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0244446, 22.0285683
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7560768, 21.7606049
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7725716, 23.7779350
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2224960, 23.2221680
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6055107, 23.6108894
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5312881, 26.5324249
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2895813, 24.2940826
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3094788, 31.3120041
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8933563, 27.8915634
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2297745, 27.2283211
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9110641, 33.9197159
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8055954, 37.8018341
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9284058, 27.9299316
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0669403, 31.0714111
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6473541, 41.6367188
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6194763, 29.6147614
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7971344, 22.7905846
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2269974, 19.2204666
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5350647, 25.5298805
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7369385, 24.7283401
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0420609, 22.0344734
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8569031, 22.8505859
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4212914, 24.4144745
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6506653, 30.6382751
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4685059, 26.4627838
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -22.0110779, 21.9994087
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9737282, 23.9671211
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1358643, 26.1286201
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0933304, 25.0882607
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3952713, 21.3956642
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4130554, 33.4106293
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7921524, 27.7890167
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2118073, 28.2103348
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0772858, 29.0762329
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8284454, 38.8257675
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7537994, 40.7544937
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4113159, 41.4139099
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0529556, 31.0513077
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5308762, 26.5319748
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4746094, 18.4743767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1308

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1527

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5444290, upper bound: 11.5724749
time: 44.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5370491, upper bound: 11.5798500
time: 38.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2645950, 25.2673035
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7711143, 19.7734909
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2882233, 17.2903175
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0245590, 22.0285835
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7564125, 21.7597237
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7726173, 23.7778091
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2220688, 23.2221909
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6058922, 23.6089172
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5309677, 26.5325470
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2895126, 24.2942123
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3097610, 31.3102264
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8935242, 27.8916245
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2298889, 27.2273941
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9115143, 33.9153519
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8070755, 37.8027878
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9274445, 27.9305038
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0670547, 31.0711441
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6451111, 41.6379547
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6180878, 29.6149025
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7940979, 22.7911911
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2249680, 19.2206764
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5324707, 25.5304604
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7322845, 24.7302017
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0395508, 22.0353889
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8563080, 22.8508224
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4208565, 24.4146652
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6436234, 30.6396255
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4631271, 26.4639816
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -22.0064697, 22.0004883
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9694252, 23.9692383
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1352081, 26.1290207
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0917130, 25.0886612
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3945389, 21.3957062
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4144897, 33.4072113
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7932968, 27.7891159
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2118530, 28.2101974
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0758591, 29.0764618
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8287048, 38.8250580
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7540588, 40.7537994
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4119720, 41.4113007
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0530472, 31.0514908
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5298233, 26.5319977
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4745178, 18.4744339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1683

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1001

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.4954041, upper bound: 11.5376777
time: 47.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5030415, upper bound: 11.5300557
time: 33.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2632637, 25.2596817
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7687149, 19.7641182
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2867012, 17.2834167
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0227203, 22.0167961
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7585983, 21.7545815
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7723122, 23.7650223
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2269669, 23.2256279
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6023636, 23.5966721
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5265274, 26.5234451
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2923355, 24.2868805
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3123322, 31.3109665
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8936234, 27.8953476
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2201614, 27.2243500
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9199524, 33.9162750
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7939606, 37.8022308
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9336777, 27.9315948
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0794983, 31.0720749
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6264343, 41.6390839
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6102600, 29.6147423
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7865295, 22.7910576
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2178726, 19.2226906
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5295792, 25.5322685
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7170181, 24.7240829
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0315132, 22.0374222
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8443451, 22.8516235
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4068527, 24.4150963
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6271057, 30.6352386
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4642563, 26.4645844
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9940872, 22.0023689
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9602814, 23.9644737
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1259155, 26.1328201
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0855865, 25.0896988
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3966484, 21.3953972
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4097061, 33.4173889
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7816772, 27.7877808
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2120895, 28.2155762
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0682602, 29.0699692
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8174438, 38.8227081
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7466431, 40.7486191
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4057007, 41.4073486
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0546036, 31.0555878
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5350876, 26.5322266
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4790268, 18.4777985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1710

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 592

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5819345, upper bound: 11.5385830
time: 31.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5798469, upper bound: 11.5406906
time: 34.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2624931, 25.2604523
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7670288, 19.7658043
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2857857, 17.2843323
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0208130, 22.0187073
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7582321, 21.7549553
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7702675, 23.7670670
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2257309, 23.2268600
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6000748, 23.5989532
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5248947, 26.5250702
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2914505, 24.2877693
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3117065, 31.3115921
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8933868, 27.8955841
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2219696, 27.2225418
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9205551, 33.9156723
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7969971, 37.7991943
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9340744, 27.9311905
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0762787, 31.0752945
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6306915, 41.6348267
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6114197, 29.6135788
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7875366, 22.7900429
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2181931, 19.2223701
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5296707, 25.5321732
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7201385, 24.7209625
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0323372, 22.0365868
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8459015, 22.8500633
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4087143, 24.4132309
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6298981, 30.6324463
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4642334, 26.4645996
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9953079, 22.0011406
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9621735, 23.9625816
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1262360, 26.1325073
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0862427, 25.0890465
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3965263, 21.3955231
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4115372, 33.4155579
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7834930, 27.7859497
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2139664, 28.2136993
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0703430, 29.0678864
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8193207, 38.8208313
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7486267, 40.7466202
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4073334, 41.4057159
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0541306, 31.0560455
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5343781, 26.5329285
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4776535, 18.4791679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 974

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1318

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5767786, upper bound: 11.5461288
time: 30.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5748712, upper bound: 11.5480339
time: 30.11 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 63.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 63.25
Output dim: 2, lower bound: -11.5444290, upper bound: 11.5724749
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 63.25
Output dim: 2, lower bound: -11.5370491, upper bound: 11.5798500
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 63.25
Output dim: 2, lower bound: -11.4954041, upper bound: 11.5376777
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 63.25
Output dim: 2, lower bound: -11.5030415, upper bound: 11.5300557
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 63.25
Output dim: 2, lower bound: -11.5819345, upper bound: 11.5385830
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 63.25
Output dim: 2, lower bound: -11.5798469, upper bound: 11.5406906
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 63.25
Output dim: 2, lower bound: -11.5767786, upper bound: 11.5461288
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 63.25
Output dim: 2, lower bound: -11.5748712, upper bound: 11.5480339

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2638512, 25.2672691
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7695312, 19.7746239
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2805367, 17.2859573
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0227280, 22.0273857
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7497559, 21.7555275
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7707138, 23.7769394
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2220840, 23.2222366
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6006012, 23.6068993
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5278130, 26.5296555
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2875977, 24.2926636
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3016739, 31.3066177
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8898468, 27.8867950
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2301636, 27.2285538
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9116440, 33.9212265
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8050842, 37.8010330
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9265823, 27.9283218
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0679169, 31.0721588
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6399765, 41.6273651
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6150360, 29.6091843
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7968597, 22.7892036
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2270584, 19.2205124
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5323486, 25.5254593
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7307472, 24.7208862
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0364532, 22.0275040
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8559494, 22.8488770
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4185944, 24.4103317
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6437912, 30.6297226
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4710159, 26.4642181
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -22.0048180, 21.9911385
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9645462, 23.9563408
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1302261, 26.1205254
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0931625, 25.0882339
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3866310, 21.3877754
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4141388, 33.4116592
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7885284, 27.7840500
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2044830, 28.2008667
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0758209, 29.0736389
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8249969, 38.8199615
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7537384, 40.7540207
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4081726, 41.4110794
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0504150, 31.0490723
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5308990, 26.5319672
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4717941, 18.4718552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1318

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1770

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5361579, upper bound: 11.5518428
time: 36.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5089520, upper bound: 11.5789580
time: 34.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2632637, 25.2595673
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7686844, 19.7634010
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2865601, 17.2834358
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0224609, 22.0174103
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7585144, 21.7541771
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7721596, 23.7666168
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2222290, 23.2230186
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6023254, 23.5961533
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5264053, 26.5243835
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2915192, 24.2846489
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3093033, 31.3059921
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8915253, 27.8935699
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2192459, 27.2241478
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9198914, 33.9162369
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7869415, 37.7983932
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9335251, 27.9321976
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0792694, 31.0703506
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6263428, 41.6403656
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6110153, 29.6131897
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7863998, 22.7918243
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2171860, 19.2225838
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5283356, 25.5315895
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7160492, 24.7240753
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0295486, 22.0365486
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8435287, 22.8504753
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4062881, 24.4144592
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6254883, 30.6351547
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4629440, 26.4638214
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9938698, 22.0023537
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9579620, 23.9644966
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1230316, 26.1300964
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0855255, 25.0898170
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3945274, 21.3942795
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4090652, 33.4167557
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7813187, 27.7877045
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2120514, 28.2155762
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0658493, 29.0689697
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8174286, 38.8226929
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7462921, 40.7482376
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4050293, 41.4063187
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0525284, 31.0542908
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5322342, 26.5303879
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4763870, 18.4769440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1475

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 545

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5793712, upper bound: 11.5372473
time: 39.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5805937, upper bound: 11.5360231
time: 29.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2631493, 25.2596817
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7679977, 19.7640839
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2867203, 17.2832756
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0233307, 22.0165367
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7581940, 21.7545013
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7739067, 23.7648773
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2243500, 23.2209015
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6018448, 23.5966377
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5274582, 26.5233307
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2901077, 24.2860603
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3073578, 31.3079453
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8918457, 27.8932495
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2199554, 27.2234421
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9199142, 33.9162064
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7901306, 37.7952118
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9342804, 27.9314423
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0777740, 31.0718460
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6277161, 41.6389923
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6087112, 29.6154938
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7873001, 22.7909241
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2177658, 19.2220001
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5289078, 25.5310211
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7170105, 24.7231140
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0306320, 22.0354691
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8432007, 22.8508072
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4062195, 24.4145279
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6270218, 30.6336212
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4635010, 26.4632721
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9940758, 22.0021553
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9603043, 23.9621506
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1231918, 26.1299324
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0857086, 25.0896301
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3955345, 21.3932762
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4090805, 33.4167480
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7815933, 27.7874222
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2120819, 28.2155457
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0672531, 29.0675583
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8174438, 38.8226852
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7462463, 40.7482758
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4046631, 41.4066772
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0532913, 31.0535278
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5332489, 26.5293732
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4781723, 18.4751663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1613

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5652614, upper bound: 11.5399834
time: 40.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5791385, upper bound: 11.5260922
time: 34.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2627029, 25.2607002
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7668571, 19.7656364
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2853241, 17.2838745
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0191002, 22.0175400
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7590942, 21.7560730
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7679520, 23.7652702
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2266464, 23.2276764
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6007080, 23.5995865
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5231857, 26.5228882
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2912979, 24.2884254
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3106689, 31.3108444
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8955536, 27.8971176
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2207489, 27.2214088
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9192963, 33.9160614
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7956390, 37.7972794
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9372253, 27.9354782
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0765381, 31.0755157
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6297531, 41.6330414
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6093521, 29.6104279
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7893295, 22.7913132
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2212410, 19.2245216
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5314865, 25.5335159
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7197342, 24.7205734
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0326767, 22.0360870
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8429413, 22.8460617
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4076767, 24.4113083
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6305008, 30.6325073
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4634171, 26.4631500
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9970741, 22.0019646
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9609108, 23.9612503
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1284866, 26.1339722
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0873947, 25.0892258
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3955536, 21.3949203
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4116821, 33.4157028
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7846527, 27.7873154
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2141495, 28.2149429
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0671463, 29.0659409
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8201447, 38.8217316
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7486572, 40.7471008
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4089203, 41.4081268
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0526733, 31.0543861
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5334473, 26.5324554
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4790649, 18.4802399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1355

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 974

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5765109, upper bound: 11.5241870
time: 26.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5548711, upper bound: 11.5458648
time: 38.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2627335, 25.2606659
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7668648, 19.7656288
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2853241, 17.2838745
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0196342, 22.0170021
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7593384, 21.7558174
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7684708, 23.7647591
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2265549, 23.2277756
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6007080, 23.5995827
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5227127, 26.5233612
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2921066, 24.2876167
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3109665, 31.3105469
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8949127, 27.8977585
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2208405, 27.2213173
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9209442, 33.9144058
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7950897, 37.7978516
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9383545, 27.9343414
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0764923, 31.0755539
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6289139, 41.6338882
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6082687, 29.6115112
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7888107, 22.7918320
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2203484, 19.2254105
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5310211, 25.5339890
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7197571, 24.7205582
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0318375, 22.0369225
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8419037, 22.8470955
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4067841, 24.4121971
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6299515, 30.6330643
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4627914, 26.4637833
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9961281, 22.0029144
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9608345, 23.9613266
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1277008, 26.1347542
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0864258, 25.0902023
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3959198, 21.3945503
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4116821, 33.4157028
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7848587, 27.7871094
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2152100, 28.2138824
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0684052, 29.0646820
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8202209, 38.8216553
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7490997, 40.7466507
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4097290, 41.4073105
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0524750, 31.0545845
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5338974, 26.5320053
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4787216, 18.4805794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 827

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5595728, upper bound: 11.5477728
time: 34.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5746151, upper bound: 11.5327096
time: 30.44 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 66.73 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 66.73
Output dim: 2, lower bound: -11.5361579, upper bound: 11.5518428
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 66.73
Output dim: 2, lower bound: -11.5089520, upper bound: 11.5789580
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 66.73
Output dim: 2, lower bound: -11.5793712, upper bound: 11.5372473
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 66.73
Output dim: 2, lower bound: -11.5805937, upper bound: 11.5360231
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 66.73
Output dim: 2, lower bound: -11.5652614, upper bound: 11.5399834
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 66.73
Output dim: 2, lower bound: -11.5791385, upper bound: 11.5260922
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 66.73
Output dim: 2, lower bound: -11.5765109, upper bound: 11.5241870
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 66.73
Output dim: 2, lower bound: -11.5548711, upper bound: 11.5458648
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 66.73
Output dim: 2, lower bound: -11.5595728, upper bound: 11.5477728
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 66.73
Output dim: 2, lower bound: -11.5746151, upper bound: 11.5327096

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2425690, 25.2510414
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7556534, 19.7639351
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2519684, 17.2639771
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0223045, 22.0270119
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7286301, 21.7390213
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7579651, 23.7681732
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2079620, 23.2043533
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5958176, 23.6028976
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5213280, 26.5244293
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2787628, 24.2843475
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2822037, 31.2925491
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8688354, 27.8592911
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2223129, 27.2185974
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8933258, 33.9081421
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8031540, 37.7993469
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8977356, 27.9062729
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0673599, 31.0718918
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6263046, 41.6125488
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6055984, 29.5968475
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7887421, 22.7762489
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2257004, 19.2123566
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5113831, 25.4947662
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7178726, 24.7035980
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0203781, 22.0064545
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8470306, 22.8396912
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4250717, 24.4113235
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6144257, 30.5912857
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4661331, 26.4555435
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9808884, 21.9595222
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9559212, 23.9454918
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0999222, 26.0814705
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.1017685, 25.0927429
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3845367, 21.3825188
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4205933, 33.4168777
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7667160, 27.7549591
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1913605, 28.1838074
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0666199, 29.0610046
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8178101, 38.8081665
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7431946, 40.7393875
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4116669, 41.4145050
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0534363, 31.0490685
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5265274, 26.5259705
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4645653, 18.4597282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1474

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 679

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5037867, upper bound: 11.5751996
time: 34.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5051875, upper bound: 11.5738153
time: 28.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2604256, 25.2584267
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7682648, 19.7627449
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2850533, 17.2818108
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0157318, 22.0168686
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7560806, 21.7501526
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7717476, 23.7696915
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2154160, 23.2194748
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6020203, 23.5951195
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5259323, 26.5244370
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2914047, 24.2843971
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3063354, 31.3006897
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8874969, 27.8877869
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2163086, 27.2224350
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9109116, 33.9136810
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7754517, 37.7905426
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9332581, 27.9319458
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0783768, 31.0685959
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6248093, 41.6393890
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6155701, 29.6099243
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7851944, 22.7912292
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2143173, 19.2159538
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5236969, 25.5268173
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7159119, 24.7240219
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0270424, 22.0339661
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8447571, 22.8481369
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4026718, 24.4101028
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6252136, 30.6304626
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4620056, 26.4632797
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9945297, 22.0015602
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9578629, 23.9656639
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1170959, 26.1200638
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0850906, 25.0899620
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3855629, 21.3893509
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4039993, 33.4142990
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7784119, 27.7862854
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2070694, 28.2138214
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0573349, 29.0655823
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8172607, 38.8224792
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7390137, 40.7456894
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3930054, 41.4018631
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0529785, 31.0542603
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5261536, 26.5271683
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4737167, 18.4763184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1683

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1653

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5768929, upper bound: 11.5234596
time: 28.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5656643, upper bound: 11.5348206
time: 26.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2621269, 25.2567291
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7680283, 19.7629852
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2849388, 17.2819252
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0219116, 22.0106850
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7544861, 21.7517433
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7752342, 23.7662010
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2186890, 23.2162018
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6012955, 23.5958443
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5264511, 26.5239182
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2912674, 24.2845383
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3040009, 31.3030243
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8857346, 27.8895416
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2175369, 27.2212067
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9173279, 33.9072647
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7790985, 37.7868881
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9332809, 27.9319305
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0775146, 31.0694580
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6253738, 41.6388245
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6077423, 29.6177444
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7857971, 22.7906189
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2105484, 19.2197151
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5235672, 25.5269508
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7159958, 24.7239380
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0269661, 22.0340385
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8411865, 22.8517075
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4019318, 24.4108429
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6207809, 30.6348877
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4623947, 26.4628830
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9930801, 22.0030098
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9591293, 23.9643974
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1129990, 26.1241608
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0856705, 25.0893784
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3896065, 21.3853111
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4066238, 33.4116898
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7799072, 27.7848053
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2102890, 28.2105942
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0624619, 29.0604553
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8172150, 38.8225327
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7437439, 40.7409668
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4005585, 41.3942947
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0524902, 31.0547485
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5290146, 26.5243073
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4757614, 18.4742775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 525

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1478

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5732644, upper bound: 11.5282433
time: 30.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5725530, upper bound: 11.5289789
time: 30.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2617683, 25.2545013
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7664490, 19.7582321
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2856483, 17.2792130
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0230713, 22.0162506
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7578354, 21.7535286
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7729988, 23.7616005
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2240753, 23.2206459
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6004715, 23.5917816
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5263901, 26.5192719
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2893753, 24.2847404
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3073349, 31.3079147
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8904572, 27.8909760
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2192078, 27.2260017
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9163589, 33.9157181
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7897491, 37.7969208
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9342422, 27.9313736
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0754318, 31.0630264
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6271362, 41.6425629
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6086578, 29.6150360
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7844925, 22.7901993
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2137070, 19.2209244
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5266876, 25.5304184
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7079163, 24.7207184
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0288239, 22.0349426
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8409119, 22.8502350
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4011612, 24.4133377
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6216202, 30.6321564
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4633560, 26.4631577
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9908447, 22.0014076
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9572678, 23.9613113
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1219406, 26.1299667
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0815277, 25.0886040
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3890572, 21.3915596
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3999939, 33.4143906
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7722778, 27.7849579
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2061310, 28.2138443
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0613174, 29.0659943
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8172607, 38.8225250
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7392883, 40.7464218
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3943176, 41.4039536
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0533676, 31.0534706
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5336304, 26.5293274
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4777718, 18.4748192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1605

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1436

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5771857, upper bound: 11.5200355
time: 28.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5730576, upper bound: 11.5241848
time: 39.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2577591, 25.2548981
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7553673, 19.7500343
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2722778, 17.2665634
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0046501, 21.9972038
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7496262, 21.7403297
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7492256, 23.7399368
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2291794, 23.2280579
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5831375, 23.5759125
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5147400, 26.5102844
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2866211, 24.2830276
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3004608, 31.2982330
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8862686, 27.8900223
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2265701, 27.2294235
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9050674, 33.8970795
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8146133, 37.8241882
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9377136, 27.9358215
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0625153, 31.0569458
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6526337, 41.6624298
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.5847549, 29.5919724
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7710876, 22.7775116
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.1966324, 19.2060966
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5175552, 25.5246887
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.6904907, 24.6984634
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0006447, 22.0119820
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8201752, 22.8290100
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3804855, 24.3906288
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.5788498, 30.5938034
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4468536, 26.4507599
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9593353, 21.9736023
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9381523, 23.9447060
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1200409, 26.1320000
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0793152, 25.0830421
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3934822, 21.3909531
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4106445, 33.4151382
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7766113, 27.7805557
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2089767, 28.2108459
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0600891, 29.0598068
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8155518, 38.8199692
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7392731, 40.7389450
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4116211, 41.4074554
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0522308, 31.0525208
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5368500, 26.5344162
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4791145, 18.4807739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1383

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1605

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5648892, upper bound: 11.4890702
time: 28.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5412066, upper bound: 11.5128578
time: 36.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2602005, 25.2551041
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7659645, 19.7638435
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2848282, 17.2832184
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0186348, 22.0152664
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7580185, 21.7542648
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7676163, 23.7631454
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2229309, 23.2241249
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6002808, 23.5989990
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5221252, 26.5226898
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2912827, 24.2858200
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3045883, 31.3071289
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8945465, 27.8976898
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2123032, 27.2155647
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9189224, 33.9116821
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7933655, 37.7965393
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9383240, 27.9337845
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0770798, 31.0744324
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6286087, 41.6338501
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6075516, 29.6109123
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7877121, 22.7916031
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2199173, 19.2252159
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5296860, 25.5335655
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7195587, 24.7204704
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0295639, 22.0356331
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8403397, 22.8467064
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4052658, 24.4120026
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6282959, 30.6318436
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4630661, 26.4633255
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9924812, 22.0002937
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9620857, 23.9608269
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1262207, 26.1345673
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0856476, 25.0900955
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3878937, 21.3893280
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4111633, 33.4160690
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7845764, 27.7896423
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2137527, 28.2124634
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0673065, 29.0630646
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8198853, 38.8211517
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7453918, 40.7418442
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4047852, 41.4030762
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0503693, 31.0558586
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5327759, 26.5302124
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4782104, 18.4802208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1563

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5733166, upper bound: 11.5282306
time: 31.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5701399, upper bound: 11.5314413
time: 29.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 62.95 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 62.95
Output dim: 2, lower bound: -11.5037867, upper bound: 11.5751996
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 62.95
Output dim: 2, lower bound: -11.5051875, upper bound: 11.5738153
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 62.95
Output dim: 2, lower bound: -11.5768929, upper bound: 11.5234596
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 62.95
Output dim: 2, lower bound: -11.5656643, upper bound: 11.5348206
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 62.95
Output dim: 2, lower bound: -11.5732644, upper bound: 11.5282433
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 62.95
Output dim: 2, lower bound: -11.5725530, upper bound: 11.5289789
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 62.95
Output dim: 2, lower bound: -11.5771857, upper bound: 11.5200355
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 62.95
Output dim: 2, lower bound: -11.5730576, upper bound: 11.5241848
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 62.95
Output dim: 2, lower bound: -11.5648892, upper bound: 11.4890702
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 62.95
Output dim: 2, lower bound: -11.5412066, upper bound: 11.5128578
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 62.95
Output dim: 2, lower bound: -11.5733166, upper bound: 11.5282306
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 62.95
Output dim: 2, lower bound: -11.5701399, upper bound: 11.5314413

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2436676, 25.2489433
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7545967, 19.7632294
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2437210, 17.2582321
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0213432, 22.0270767
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7137451, 21.7290039
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7514648, 23.7624931
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2030869, 23.1976242
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5898361, 23.6021118
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5166740, 26.5249939
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2767487, 24.2829018
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2709732, 31.2893295
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8613358, 27.8559570
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2180328, 27.2148438
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8885574, 33.9047165
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8018799, 37.7990417
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8963394, 27.9018936
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0640640, 31.0709305
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6267166, 41.6101303
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6054306, 29.5959625
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7860336, 22.7750168
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2241287, 19.2121964
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5105743, 25.4945374
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7183304, 24.7023735
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0137329, 22.0008240
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8464508, 22.8387222
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4246140, 24.4083557
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6144257, 30.5912857
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4624557, 26.4491653
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9717026, 21.9468956
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9583855, 23.9370499
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0914154, 26.0680389
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0922852, 25.0878143
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3844986, 21.3825302
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4147949, 33.4059219
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7485580, 27.7289886
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1815948, 28.1660461
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0590363, 29.0495071
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8120422, 38.7923508
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7324066, 40.7313232
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4060364, 41.4119568
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0467529, 31.0395012
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5203705, 26.5166931
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4547272, 18.4485321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1304

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1387

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5037867, upper bound: 11.5727447
time: 35.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5012333, upper bound: 11.5751996
time: 36.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2404709, 25.2521400
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7549400, 19.7628822
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2462234, 17.2557259
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0223656, 22.0260544
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7186127, 21.7241287
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7522812, 23.7616730
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2012329, 23.1994781
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5950394, 23.5969162
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5218773, 26.5197830
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2773209, 24.2823372
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2789841, 31.2813187
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8654938, 27.8517914
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2185669, 27.2143059
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8899002, 33.9033737
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8028564, 37.7980728
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8933563, 27.9048843
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0663986, 31.0685959
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6238785, 41.6129608
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6047134, 29.5966721
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7875061, 22.7735405
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2255478, 19.2107811
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5111618, 25.4939575
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7166443, 24.7040558
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0147476, 21.9998055
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8460617, 22.8391113
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4221039, 24.4108620
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6144257, 30.5912857
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4597549, 26.4518661
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9682617, 21.9503326
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9474754, 23.9479637
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0864868, 26.0729637
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0968323, 25.0832634
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3845444, 21.3824806
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4096375, 33.4110794
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7407532, 27.7368011
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1735992, 28.1740417
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0551224, 29.0534210
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8019867, 38.8024063
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7351379, 40.7285919
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4091187, 41.4088745
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0438690, 31.0423851
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5172424, 26.5198212
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4533691, 18.4498940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 930

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1317

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5049721, upper bound: 11.5700733
time: 34.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5014500, upper bound: 11.5735982
time: 47.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2557869, 25.2551765
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7653542, 19.7603722
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2973404, 17.2756920
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0129280, 22.0064964
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7599602, 21.7490768
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7684746, 23.7577553
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2096863, 23.2167664
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6010933, 23.5877075
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5343552, 26.5174026
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2876358, 24.2794189
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2822571, 31.2503052
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8806992, 27.8801193
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2149811, 27.2248459
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9091492, 33.9125214
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7478180, 37.7354736
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9326782, 27.9314117
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0707703, 31.0515289
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6238556, 41.6384964
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6192780, 29.6091118
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7826920, 22.7912445
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2068481, 19.2075729
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5180130, 25.5206909
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7134247, 24.7392120
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0236740, 22.0283394
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8440323, 22.8469963
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4003830, 24.4136009
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6212311, 30.6197968
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4462128, 26.4527740
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9841309, 22.0021591
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9552231, 23.9814682
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1167755, 26.1211472
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0827179, 25.0871429
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3702126, 21.3805008
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3887863, 33.4049149
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7542572, 27.7732544
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1832504, 28.2024460
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0179443, 29.0467682
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7751160, 38.8022766
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7125854, 40.7329559
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3863831, 41.3970795
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0457764, 31.0510101
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5004425, 26.5146942
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4736786, 18.4763374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 898

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1444

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5763342, upper bound: 11.5228098
time: 28.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5762439, upper bound: 11.5229009
time: 37.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2620583, 25.2564545
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7677727, 19.7621841
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2846260, 17.2814369
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0218048, 22.0101738
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7540207, 21.7518425
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7749825, 23.7659454
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2186127, 23.2161217
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6009598, 23.5951653
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5260849, 26.5231018
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2912292, 24.2844849
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3029938, 31.3026810
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8858719, 27.8892593
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2172241, 27.2212219
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9171524, 33.9072800
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7786713, 37.7861176
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9322968, 27.9313660
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0772400, 31.0688400
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6250381, 41.6378479
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6076126, 29.6175537
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7858658, 22.7905235
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2100601, 19.2195129
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5236435, 25.5267639
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7145309, 24.7225533
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0268860, 22.0335350
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8410568, 22.8509750
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4015961, 24.4104538
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6191711, 30.6335983
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4619446, 26.4613800
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9928207, 22.0019226
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9580650, 23.9627533
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1125488, 26.1224518
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0853577, 25.0890961
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3891945, 21.3851891
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4065933, 33.4116135
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7791977, 27.7840347
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2106705, 28.2102051
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0619125, 29.0600967
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8163757, 38.8222885
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7436981, 40.7409592
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4001160, 41.3941116
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0523376, 31.0546570
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5289536, 26.5242844
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4753342, 18.4739532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1756

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1387

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5732644, upper bound: 11.5257105
time: 33.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5708207, upper bound: 11.5282433
time: 33.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2615662, 25.2540436
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7661171, 19.7572899
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2853394, 17.2777786
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0229187, 22.0159760
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7577095, 21.7524910
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7728271, 23.7609749
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2225189, 23.2206612
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6002045, 23.5903358
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5262909, 26.5182800
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2888184, 24.2835846
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3061676, 31.3059616
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8904648, 27.8901749
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2175217, 27.2261887
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9151382, 33.9153519
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7894592, 37.7965012
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9341660, 27.9313583
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0743942, 31.0610199
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6253357, 41.6426315
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6086349, 29.6148834
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7844696, 22.7901611
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2136612, 19.2208214
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5267715, 25.5300941
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7062378, 24.7201920
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0287704, 22.0346451
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8410034, 22.8496933
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4004974, 24.4132233
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6210556, 30.6315002
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4631805, 26.4628372
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9903870, 22.0009918
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9569321, 23.9620667
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1217995, 26.1296082
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0814590, 25.0885468
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3871193, 21.3912735
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3979187, 33.4137726
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7694244, 27.7837448
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2043304, 28.2135086
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0578918, 29.0646362
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8150635, 38.8219528
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7353668, 40.7441254
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3921204, 41.4031525
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0521622, 31.0532455
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5325394, 26.5292740
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4772682, 18.4755058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1347

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5734527, upper bound: 11.5093662
time: 30.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5665314, upper bound: 11.5164271
time: 32.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2613068, 25.2545013
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7655067, 19.7582321
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2842102, 17.2792130
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0227966, 22.0162506
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7567940, 21.7535286
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7723846, 23.7616005
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2240753, 23.2191010
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5990295, 23.5917816
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5253983, 26.5192719
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2882156, 24.2847404
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3053741, 31.3079147
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8896561, 27.8909760
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2192078, 27.2243195
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9163589, 33.9144974
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7897491, 37.7966461
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9342422, 27.9312973
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0734329, 31.0630264
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6271362, 41.6407547
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6084976, 29.6150360
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7844543, 22.7901993
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2137070, 19.2208824
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5263672, 25.5304184
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7079163, 24.7190399
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0285263, 22.0349426
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8403778, 22.8502350
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4011612, 24.4126740
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6216202, 30.6315994
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4633560, 26.4629822
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9908447, 22.0009499
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9572678, 23.9609680
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1215858, 26.1299667
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0815277, 25.0885391
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3890572, 21.3896217
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3999939, 33.4123077
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7722778, 27.7821045
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2061310, 28.2120438
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0613174, 29.0625687
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8172607, 38.8203354
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7392883, 40.7424927
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3943176, 41.4017258
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0533676, 31.0522614
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5336304, 26.5282288
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4777718, 18.4743156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1527

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5727331, upper bound: 11.5241290
time: 45.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.5730022, upper bound: 11.5238584
time: 31.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2618408, 25.2548828
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7645607, 19.7586250
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2849579, 17.2824478
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0185661, 22.0160637
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7578354, 21.7529716
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7675705, 23.7627869
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2230225, 23.2239494
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5989838, 23.5947495
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5214691, 26.5200119
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2909698, 24.2833481
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3041687, 31.3054047
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8929825, 27.8941269
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2116699, 27.2162437
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9110641, 33.9112015
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7849731, 37.7914276
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9379654, 27.9337845
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0751572, 31.0633621
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6283798, 41.6357422
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6073990, 29.6087875
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7851181, 22.7906113
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2187729, 19.2250023
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5295639, 25.5339394
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7168121, 24.7241249
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0290070, 22.0356026
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8400040, 22.8465805
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4024200, 24.4140663
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6271210, 30.6316528
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4593430, 26.4569626
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9920197, 22.0001335
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9619102, 23.9620743
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1218033, 26.1307907
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0830383, 25.0914841
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3840179, 21.3876915
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3949585, 33.4121170
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7809677, 27.7887573
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2068481, 28.2107849
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0643539, 29.0623398
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8173676, 38.8204880
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7409668, 40.7407532
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3897705, 41.3994217
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0516357, 31.0557671
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5342712, 26.5301132
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4773941, 18.4794006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 964

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1416

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5587237, upper bound: 11.5266215
time: 35.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5717040, upper bound: 11.5136487
time: 32.08 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 69.76 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 69.76
Output dim: 2, lower bound: -11.5037867, upper bound: 11.5727447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 69.76
Output dim: 2, lower bound: -11.5012333, upper bound: 11.5751996
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 69.76
Output dim: 2, lower bound: -11.5049721, upper bound: 11.5700733
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 69.76
Output dim: 2, lower bound: -11.5014500, upper bound: 11.5735982
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 69.76
Output dim: 2, lower bound: -11.5763342, upper bound: 11.5228098
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 69.76
Output dim: 2, lower bound: -11.5762439, upper bound: 11.5229009
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 69.76
Output dim: 2, lower bound: -11.5732644, upper bound: 11.5257105
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 69.76
Output dim: 2, lower bound: -11.5708207, upper bound: 11.5282433
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 69.76
Output dim: 2, lower bound: -11.5734527, upper bound: 11.5093662
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 69.76
Output dim: 2, lower bound: -11.5665314, upper bound: 11.5164271
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 69.76
Output dim: 2, lower bound: -11.5727331, upper bound: 11.5241290
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 69.76
Output dim: 2, lower bound: -11.5730022, upper bound: 11.5238584
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 69.76
Output dim: 2, lower bound: -11.5587237, upper bound: 11.5266215
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 69.76
Output dim: 2, lower bound: -11.5717040, upper bound: 11.5136487

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2424049, 25.2477951
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7519417, 19.7605896
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2429733, 17.2576027
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0195541, 22.0257454
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7128906, 21.7280769
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7504578, 23.7617683
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2028961, 23.1974487
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5876694, 23.6000519
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5152664, 26.5239410
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2766418, 24.2823334
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2710037, 31.2893448
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8615036, 27.8566666
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2181702, 27.2145462
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8889008, 33.9049911
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7989655, 37.7963409
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8971100, 27.9026642
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0649414, 31.0721664
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6261520, 41.6096497
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6056366, 29.5963097
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7849197, 22.7741432
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2221603, 19.2100525
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5105896, 25.4945602
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7190514, 24.7031097
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0132942, 22.0004616
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8458099, 22.8381271
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4219131, 24.4059677
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6140289, 30.5910263
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4630356, 26.4500885
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9713631, 21.9466133
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9590454, 23.9378586
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0913811, 26.0680008
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0913391, 25.0870781
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3841476, 21.3820114
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4107895, 33.4014740
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7489777, 27.7293167
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1813507, 28.1658173
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0594406, 29.0499954
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8093872, 38.7899323
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7324219, 40.7313461
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4043121, 41.4101715
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0459442, 31.0387650
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5208206, 26.5172043
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4527092, 18.4461823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1525

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1306

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4979593, upper bound: 11.5746177
time: 28.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5006558, upper bound: 11.5719187
time: 30.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2400551, 25.2517662
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7544479, 19.7624092
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2462578, 17.2557678
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0221024, 22.0251770
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7191162, 21.7245369
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7503510, 23.7596092
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1997986, 23.1981544
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5946312, 23.5964890
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5220413, 26.5199661
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2785416, 24.2829666
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2776489, 31.2795868
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8656616, 27.8520432
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2187042, 27.2144814
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8903198, 33.9036560
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8021240, 37.7978363
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8939056, 27.9050903
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0663452, 31.0685425
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6235809, 41.6133499
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6061478, 29.5988197
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7874222, 22.7734756
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2258263, 19.2111664
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5112076, 25.4940186
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7160263, 24.7031555
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0145302, 21.9997902
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8458099, 22.8392906
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4226761, 24.4119186
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6147614, 30.5916519
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4590912, 26.4512939
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9684830, 21.9506645
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9463768, 23.9466324
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0870819, 26.0736618
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0970688, 25.0836258
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3841705, 21.3810081
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4083786, 33.4101105
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7414398, 27.7373581
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1736374, 28.1740799
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0557022, 29.0536575
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7972107, 38.7982254
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7343445, 40.7279129
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4097900, 41.4094391
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0428467, 31.0414505
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5179749, 26.5201797
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4534912, 18.4500656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1710

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 528

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.4915200, upper bound: 11.5723407
time: 29.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5001876, upper bound: 11.5636706
time: 28.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2554665, 25.2541199
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7654991, 19.7603416
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2956886, 17.2741661
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0110474, 22.0041237
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7573090, 21.7466087
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7685051, 23.7577782
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2091141, 23.2164612
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6022797, 23.5888901
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5342674, 26.5171890
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2817154, 24.2716370
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2781677, 31.2452774
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8792343, 27.8784332
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2072525, 27.2186050
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8933029, 33.8992004
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7466660, 37.7341461
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9329453, 27.9312515
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0618591, 31.0406342
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6224518, 41.6380539
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6171341, 29.6066818
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7823944, 22.7909317
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2086639, 19.2093582
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5154266, 25.5174332
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7123909, 24.7382050
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0190773, 22.0227127
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8435440, 22.8464508
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4024162, 24.4158173
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6190872, 30.6164322
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4359665, 26.4406128
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9780846, 21.9947014
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9491425, 23.9749413
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1100922, 26.1129990
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0862350, 25.0909958
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3679123, 21.3785248
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3805542, 33.3982239
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7606430, 27.7789154
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1833344, 28.2025909
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0162201, 29.0454102
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7696228, 38.7980347
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7018738, 40.7238464
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3753815, 41.3884964
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0437393, 31.0494995
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5011368, 26.5157471
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4734650, 18.4761925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1710

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1000

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.4812895, upper bound: 11.4277777
time: 58.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.4814408, upper bound: 11.4276256
time: 31.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2547340, 25.2548599
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7653236, 19.7605209
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2958183, 17.2740364
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0105591, 22.0046082
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7574921, 21.7464294
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7685051, 23.7577820
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2093887, 23.2161865
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6022797, 23.5888901
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5341454, 26.5173111
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2798538, 24.2734985
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2772293, 31.2462234
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8790131, 27.8786621
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2087326, 27.2171288
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8958282, 33.8966751
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7464828, 37.7343216
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9325180, 27.9316788
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0598755, 31.0426178
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6234131, 41.6370850
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6168518, 29.6069641
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7823715, 22.7909546
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2086334, 19.2093964
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5147552, 25.5181084
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7124214, 24.7381821
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0180397, 22.0237503
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8434830, 22.8465042
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4025993, 24.4156303
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6178741, 30.6176453
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4340515, 26.4425354
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9766731, 21.9961128
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9487000, 23.9753876
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1086273, 26.1144638
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0865707, 25.0906601
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3682327, 21.3782043
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3820953, 33.3966827
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7599106, 27.7796478
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1834030, 28.2025299
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0165939, 29.0450363
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.7708588, 38.7967911
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7034607, 40.7222366
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3777924, 41.3860931
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0442581, 31.0489807
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5015030, 26.5153885
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4735336, 18.4761276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1729

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5594178, upper bound: 11.5111006
time: 37.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5645147, upper bound: 11.5059970
time: 32.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2609024, 25.2551804
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7651253, 19.7595215
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2840042, 17.2807045
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0204849, 22.0083923
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7531204, 21.7510109
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7742538, 23.7649384
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2184219, 23.2159195
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5988770, 23.5929794
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5250244, 26.5216904
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2906494, 24.2843666
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3029938, 31.3027039
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8865814, 27.8894272
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2169113, 27.2213440
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9174194, 33.9076080
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7759705, 37.7831955
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9330597, 27.9321365
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0784912, 31.0697327
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6245575, 41.6372833
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6079712, 29.6177673
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7849960, 22.7894173
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2079163, 19.2175484
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5236511, 25.5267715
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7152824, 24.7232971
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0265198, 22.0331001
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8404770, 22.8503494
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3992157, 24.4077606
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6189117, 30.6332016
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4628906, 26.4619827
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9925461, 22.0015907
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9588699, 23.9633980
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1125259, 26.1224289
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0845947, 25.0881233
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3886719, 21.3848419
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4021301, 33.4075928
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7795334, 27.7844620
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2104416, 28.2099609
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0623932, 29.0604935
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8139801, 38.8196640
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7437134, 40.7409744
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3983307, 41.3924026
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0515900, 31.0538559
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5294571, 26.5247345
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4729767, 18.4719200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1653

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1528

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5716747, upper bound: 11.5137000
time: 35.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5613047, upper bound: 11.5241112
time: 29.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2597504, 25.2509995
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7650719, 19.7564240
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2848129, 17.2766647
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0229263, 22.0159721
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7577972, 21.7519951
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7728615, 23.7604408
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2224731, 23.2206192
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5991821, 23.5897446
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5252609, 26.5173187
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2859268, 24.2809677
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.3037949, 31.3032150
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8846893, 27.8865891
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2152634, 27.2222366
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9104462, 33.9074173
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7885284, 37.7951813
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9343414, 27.9293823
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0732956, 31.0611115
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6237335, 41.6415558
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6035080, 29.6119003
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7810974, 22.7881432
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2086411, 19.2178497
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5202255, 25.5258751
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7052002, 24.7194481
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0245361, 22.0321732
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8361511, 22.8469582
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.3963776, 24.4107971
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6147537, 30.6276398
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4586639, 26.4601593
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9848633, 21.9979172
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9568253, 23.9619827
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1121407, 26.1237335
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0778198, 25.0861206
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3858719, 21.3883057
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3974152, 33.4130936
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7683640, 27.7835388
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2039413, 28.2128372
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0566559, 29.0624695
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8147430, 38.8215637
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7351227, 40.7420044
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3906250, 41.4002609
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0515366, 31.0532494
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5313339, 26.5276031
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4770470, 18.4753227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 529

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1449

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5720705, upper bound: 11.5082672
time: 30.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5723542, upper bound: 11.5080091
time: 46.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2620926, 25.2553635
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7659950, 19.7587051
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2857666, 17.2812195
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0231056, 22.0164719
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7588806, 21.7557220
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7745819, 23.7644691
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2213745, 23.2173882
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.6005402, 23.5932388
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5270424, 26.5212860
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2801056, 24.2739334
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2928925, 31.2912903
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8885803, 27.8890762
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2204666, 27.2256393
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.9061508, 33.9066696
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7906570, 37.7953110
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.9349823, 27.9323883
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0663757, 31.0537262
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6217804, 41.6375427
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6036072, 29.6074066
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7847443, 22.7906189
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2135620, 19.2193108
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5224228, 25.5253334
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7079582, 24.7191124
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0303001, 22.0353851
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8409424, 22.8499985
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4018707, 24.4134674
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6244659, 30.6327133
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4625168, 26.4619675
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9929810, 22.0026054
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9532318, 23.9581947
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.1198730, 26.1258430
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0823288, 25.0895767
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3886986, 21.3892975
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.3983154, 33.4110107
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7809448, 27.7884903
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.2044220, 28.2107620
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0537643, 29.0567780
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8132935, 38.8179932
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7344971, 40.7386475
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3899384, 41.3984528
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0532303, 31.0525513
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5309067, 26.5261612
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4765854, 18.4732285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 532

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 852

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5371752, upper bound: 11.5235047
time: 32.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.5726469, upper bound: 11.4880075
time: 32.19 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 66.56 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 66.56
Output dim: 2, lower bound: -11.4979593, upper bound: 11.5746177
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 66.56
Output dim: 2, lower bound: -11.5006558, upper bound: 11.5719187
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 66.56
Output dim: 2, lower bound: -11.4915200, upper bound: 11.5723407
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 66.56
Output dim: 2, lower bound: -11.5001876, upper bound: 11.5636706
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 66.56
Output dim: 2, lower bound: -11.4812895, upper bound: 11.4277777
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 66.56
Output dim: 2, lower bound: -11.4814408, upper bound: 11.4276256
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 66.56
Output dim: 2, lower bound: -11.5594178, upper bound: 11.5111006
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 66.56
Output dim: 2, lower bound: -11.5645147, upper bound: 11.5059970
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 66.56
Output dim: 2, lower bound: -11.5716747, upper bound: 11.5137000
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 66.56
Output dim: 2, lower bound: -11.5613047, upper bound: 11.5241112
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 66.56
Output dim: 2, lower bound: -11.5720705, upper bound: 11.5082672
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 66.56
Output dim: 2, lower bound: -11.5723542, upper bound: 11.5080091
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 66.56
Output dim: 2, lower bound: -11.5371752, upper bound: 11.5235047
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 66.56
Output dim: 2, lower bound: -11.5726469, upper bound: 11.4880075

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2416115, 25.2472687
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7518654, 19.7605476
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2423744, 17.2573776
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0192642, 22.0257072
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7119446, 21.7278023
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7500381, 23.7614441
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.2029800, 23.1966934
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5874023, 23.5999298
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5150833, 26.5238190
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2765198, 24.2821045
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2705307, 31.2893753
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8613510, 27.8565216
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2180939, 27.2144585
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8861923, 33.9033890
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.7988739, 37.7961578
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8967743, 27.9020462
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0643997, 31.0718002
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6260910, 41.6095428
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6056137, 29.5959396
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7847366, 22.7738037
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2222366, 19.2095680
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5103989, 25.4940338
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7189331, 24.7027321
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0127335, 21.9992294
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8454361, 22.8377533
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4218597, 24.4054413
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6139679, 30.5902939
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4626160, 26.4485931
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9708290, 21.9451904
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9587708, 23.9372482
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0911598, 26.0673637
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0908813, 25.0867958
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3839111, 21.3814926
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4104843, 33.4011078
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7489014, 27.7291870
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1804810, 28.1651840
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0592651, 29.0497665
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8092194, 38.7895813
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7323608, 40.7312241
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.4029694, 41.4097900
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0457764, 31.0380554
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5207062, 26.5165253
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4518890, 18.4442596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1404

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -11.4786568, upper bound: 11.5735919
time: 35.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.4969157, upper bound: 11.5553630
time: 33.78 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 72.03 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 72.03
Output dim: 2, lower bound: -11.4786568, upper bound: 11.5735919
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 72.03
Output dim: 2, lower bound: -11.4969157, upper bound: 11.5553630

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.3994713, 19.0156326, -9.3994713, 19.0156326, -25.2404671, 25.2460289
1: -1.2535338, 22.8033047, -1.2535338, 22.8033047, -19.7468834, 19.7625961
2: -1.6569021, 20.9150467, -1.6569021, 20.9150467, -17.2366180, 17.2583771
3: -9.3836384, 16.4814606, -9.3836384, 16.4814606, -22.0095901, 22.0269623
4: -3.1860042, 22.2249680, -3.1860042, 22.2249680, -21.7011414, 21.7302704
5: -7.8757224, 20.6075287, -7.8757224, 20.6075287, -23.7384338, 23.7629623
6: -28.8302269, -1.3732400, -28.8302269, -1.3732400, -23.1955566, 23.1980438
7: -7.7362618, 21.6280556, -7.7362618, 21.6280556, -23.5780029, 23.6012917
8: -14.8421936, 14.7563114, -14.8421936, 14.7563114, -26.5108032, 26.5251083
9: -5.2202339, 21.2809486, -5.2202339, 21.2809486, -24.2776489, 24.2817993
10: -17.9087944, 17.5610924, -17.9087944, 17.5610924, -31.2705154, 31.2892914
11: -26.7545547, 3.5995460, -26.7545547, 3.5995460, -27.8645172, 27.8498764
12: -34.8962288, -2.3362265, -34.8962288, -2.3362265, -27.2175598, 27.2142410
13: -26.3062592, 15.7039289, -26.3062592, 15.7039289, -33.8688354, 33.9062805
14: -55.9367104, -17.5484047, -55.9367104, -17.5484047, -37.8120193, 37.7896271
15: -14.3917484, 15.5168705, -14.3917484, 15.5168705, -27.8963089, 27.9040756
16: -14.0787201, 20.8080311, -14.0787201, 20.8080311, -31.0630798, 31.0708237
17: -57.8655167, -14.4164047, -57.8655167, -14.4164047, -41.6324158, 41.6087265
18: -21.5803757, 12.2107496, -21.5803757, 12.2107496, -29.6079788, 29.5808372
19: -22.2763252, 3.6176648, -22.2763252, 3.6176648, -22.7861481, 22.7642746
20: -23.2826729, 1.4265695, -23.2826729, 1.4265695, -19.2243576, 19.1968079
21: -26.7924690, 2.4563313, -26.7924690, 2.4563313, -25.5125809, 25.4808731
22: -28.4929962, 3.3823078, -28.4929962, 3.3823078, -24.7208748, 24.6932793
23: -22.2814445, 5.7620778, -22.2814445, 5.7620778, -22.0157051, 21.9806366
24: -18.2995319, 9.4943590, -18.2995319, 9.4943590, -22.8473969, 22.8259354
25: -23.8173294, 5.4395990, -23.8173294, 5.4395990, -24.4235992, 24.3949280
26: -41.0261421, -0.3901229, -41.0261421, -0.3901229, -30.6185226, 30.5628738
27: -21.5695000, 8.6333017, -21.5695000, 8.6333017, -26.4641266, 26.4387817
28: -24.1100121, 6.1282749, -24.1100121, 6.1282749, -21.9732971, 21.9254341
29: -27.8398094, -0.1768060, -27.8398094, -0.1768060, -23.9602661, 23.9282608
30: -28.1108170, 3.8069706, -28.1108170, 3.8069706, -26.0953827, 26.0560951
31: -22.6675529, 5.1007361, -22.6675529, 5.1007361, -25.0914612, 25.0830307
32: -23.9515438, 2.3312340, -23.9515438, 2.3312340, -21.3759384, 21.3822021
33: -36.4425774, 3.6759300, -36.4425774, 3.6759300, -33.4039841, 33.4010620
34: -37.8637314, -4.7164421, -37.8637314, -4.7164421, -27.7487183, 27.7293777
35: -32.9313774, 0.3271227, -32.9313774, 0.3271227, -28.1792145, 28.1657562
36: -36.8488464, -0.6216030, -36.8488464, -0.6216030, -29.0576477, 29.0521240
37: -44.5799294, -1.6877723, -44.5799294, -1.6877723, -38.8096313, 38.7894287
38: -43.9702988, 2.9170003, -43.9702988, 2.9170003, -40.7315521, 40.7333527
39: -43.6241150, 3.0258360, -43.6241150, 3.0258360, -41.3868103, 41.4124985
40: -32.7614670, -0.0047204, -32.7614670, -0.0047204, -31.0399017, 31.0390396
41: -20.7425346, 7.3007545, -20.7425346, 7.3007545, -26.5162582, 26.5176163
42: -22.9948425, -0.1967542, -22.9948425, -0.1967542, -18.4516373, 18.4438400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=174, inp2_unstable=174, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=197, inp2_unstable=197, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1473

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1432

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.4588098, upper bound: 11.5714875
time: 42.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -11.4765742, upper bound: 11.5537916
time: 35.11 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 79.70 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 79.70
Output dim: 2, lower bound: -11.4588098, upper bound: 11.5714875
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 79.70
Output dim: 2, lower bound: -11.4765742, upper bound: 11.5537916

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 46.11 + 2417.80 = 2463.91 seconds
