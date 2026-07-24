## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 7)
Time budget: 1800 seconds
Split limit: 100
Threshold: 10.6492795605


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924)
1: (-11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544)
2: (-13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.5111237, 19.5111237)
3: (-18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3367386, 24.3367386)
4: (-20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3111115, 21.3111115)
5: (-18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2404251, 25.2404251)
6: (-36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7976227, 20.7976265)
7: (-24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0694427, 24.0694427)
8: (-27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4327621, 25.4327621)
9: (-11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.8047028, 20.8047028)
10: (-17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6827240, 29.6827240)
11: (-16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8160324, 23.8160324)
12: (-24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3984375, 33.3984375)
13: (-22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7179260, 32.7179260)
14: (-34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8146362, 36.8146439)
15: (-8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3423691, 23.3423691)
16: (-22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630)
17: (-28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362)
18: (-12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4271851, 29.4271851)
19: (-8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5475731, 16.5475769)
20: (-9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7397041, 17.7397079)
21: (-12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2488022, 20.2487984)
22: (-2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7594414, 18.7594414)
23: (-3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5418129, 17.5418129)
24: (-5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9421463, 19.9421425)
25: (2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4483719, 19.4483681)
26: (-11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286)
27: (-15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8619308, 23.8619270)
28: (-3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5932961, 19.5932961)
29: (-3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1883316, 15.1883316)
30: (-13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7468872, 24.7468910)
31: (-9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718)
32: (-30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8545761, 22.8545761)
33: (-41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0324020, 31.0324020)
34: (-36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.4049072, 25.4049110)
35: (-24.9038906, 5.5323420, -24.9038906, 5.5323420, -26.0001907, 26.0001907)
36: (-24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1228638, 28.1228638)
37: (-42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1177673, 32.1177673)
38: (-34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7828369, 35.7828217)
39: (-47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1182175, 37.1182251)
40: (-45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1954041, 21.1954079)
41: (-33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2726974, 22.2726974)
42: (-24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7295189, 19.7295189)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.46 + 44.12 = 46.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 25, lower bound: -10.6599395, upper bound: 10.6599395

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6512251, upper bound: 10.6495461
time: 24.79 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6571650, upper bound: 10.6571647
time: 26.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 51.06 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 51.06
Output dim: 25, lower bound: -10.6512251, upper bound: 10.6495461
IS_B2, status: Status.UNKNOWN, split count: 1, time: 51.06
Output dim: 25, lower bound: -10.6571650, upper bound: 10.6571647

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -22.5215492, 8.6257858, -22.4893284, 8.5923634, -31.1139126, 31.1151142
1: -11.8727808, 6.3623581, -11.8629456, 6.3375659, -18.2103462, 18.2253036
2: -13.1288118, 7.4190187, -13.0733528, 7.3829689, -19.4251328, 19.4052124
3: -18.6535168, 6.2480755, -18.5931244, 6.2050095, -24.2254791, 24.2034836
4: -20.2803879, 3.2721190, -20.2516060, 3.2413979, -21.2461014, 21.2467041
5: -18.1254539, 7.5422425, -18.0389652, 7.4896646, -25.1124878, 25.0759583
6: -36.7386780, -11.2631598, -36.7160072, -11.2928476, -20.7370415, 20.7562447
7: -24.1114407, 1.3136659, -24.0690193, 1.2623646, -23.9441757, 23.9441910
8: -27.3748722, 1.0857534, -27.3451977, 1.0502224, -25.3516541, 25.3576126
9: -11.5317001, 11.8605347, -11.4905195, 11.8072948, -20.6995544, 20.7106934
10: -17.6165848, 12.3423281, -17.5394821, 12.2440634, -29.4761047, 29.4991226
11: -16.7615013, 10.4878654, -16.7316628, 10.4708853, -23.7557831, 23.7446060
12: -24.1111164, 11.2831192, -24.0142326, 11.1379719, -33.1213226, 33.1703796
13: -22.3387871, 12.3462553, -22.2974110, 12.3158607, -32.6348038, 32.6169281
14: -34.8111076, 6.3790083, -34.6945877, 6.2380247, -36.5542603, 36.5819092
15: -8.7085075, 16.7678795, -8.6802731, 16.7537270, -23.2979202, 23.2754059
16: -22.7501698, 3.1268532, -22.7144299, 3.1025116, -25.8526821, 25.8412838
17: -27.9706268, 8.0601749, -27.8846931, 7.9776068, -35.9482346, 35.9448700
18: -12.5767937, 18.6084766, -12.5416899, 18.5843391, -29.3285751, 29.3350067
19: -8.9385662, 8.0364475, -8.9189882, 8.0279980, -16.5097885, 16.5010643
20: -9.9174376, 8.7735939, -9.8872023, 8.7621593, -17.6977806, 17.6812248
21: -12.5363331, 9.1557884, -12.5055904, 9.1447897, -20.1942863, 20.1812286
22: -2.7661839, 18.5074425, -2.7236085, 18.5105400, -18.7033386, 18.6667557
23: -3.9238982, 15.2512035, -3.9016924, 15.2432995, -17.5062943, 17.4999352
24: -5.5209627, 17.3426743, -5.4981241, 17.3252773, -19.8952103, 19.8917923
25: 2.0831947, 24.2871017, 2.1109352, 24.2798386, -19.4010315, 19.3870163
26: -11.6453419, 21.5799122, -11.5705872, 21.5392570, -33.1846008, 33.1504974
27: -14.9942360, 9.9685040, -14.9464912, 9.9429169, -23.7926865, 23.7684784
28: -2.9943705, 18.0184650, -2.9661031, 18.0075741, -19.5548782, 19.5288048
29: -3.3380437, 15.5080614, -3.3053594, 15.4995174, -15.1360664, 15.1034698
30: -13.5805225, 13.7901697, -13.5552578, 13.7743473, -24.6950912, 24.6825523
31: -9.6842537, 11.3003292, -9.6624756, 11.2840290, -20.9682827, 20.9628048
32: -30.6959801, -3.7906542, -30.6706123, -3.8119359, -22.7820396, 22.8005600
33: -41.5087280, -3.2643766, -41.4715843, -3.3078623, -30.9334717, 30.9248505
34: -36.7843666, -3.8751578, -36.7629623, -3.9014163, -25.3399506, 25.3403778
35: -24.8955669, 5.4970026, -24.8679371, 5.4645333, -25.9219742, 25.8886719
36: -24.5464478, 6.3965583, -24.5176582, 6.3726993, -28.0418396, 28.0327530
37: -42.8658981, -6.5825424, -42.8403587, -6.6115780, -32.0534210, 32.0605621
38: -34.3667755, 3.4732833, -34.3328285, 3.4482708, -35.6825104, 35.6838760
39: -47.7190323, -7.0987043, -47.6918030, -7.1259232, -37.0487595, 37.0505371
40: -45.9546204, -18.9952316, -45.9242325, -19.0258102, -21.1263199, 21.1520271
41: -33.4822159, -4.6816721, -33.4575577, -4.7122908, -22.2000961, 22.2258148
42: -24.0329437, -0.4414439, -24.0158730, -0.4638455, -19.6604462, 19.6825161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=162, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1629

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6499590, upper bound: 10.6343760
time: 31.65 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6499590, upper bound: 10.6482929
time: 28.46 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -22.5306702, 8.6581221, -22.5300369, 8.6567993, -31.1874695, 31.1881599
1: -11.8803167, 6.3851671, -11.8792076, 6.3840399, -18.2643566, 18.2643738
2: -13.1329432, 7.4597759, -13.1327257, 7.4579506, -19.4662323, 19.5074577
3: -18.6600380, 6.2936792, -18.6595364, 6.2916546, -24.3147202, 24.3325424
4: -20.2850342, 3.2993541, -20.2847366, 3.2979300, -21.2915649, 21.3082199
5: -18.1319237, 7.6015167, -18.1315002, 7.5993204, -25.1780319, 25.2359085
6: -36.7442627, -11.2461872, -36.7434807, -11.2532473, -20.7731400, 20.7846107
7: -24.1220589, 1.3690536, -24.1215916, 1.3668845, -24.0287170, 24.0647659
8: -27.3802338, 1.1203165, -27.3799057, 1.1190386, -25.4082794, 25.4297791
9: -11.5763931, 11.8646889, -11.5746136, 11.8641052, -20.8004913, 20.7313690
10: -17.7056274, 12.3522835, -17.7024765, 12.3510103, -29.6742554, 29.6300201
11: -16.7896385, 10.4889050, -16.7883625, 10.4856033, -23.8132553, 23.8077393
12: -24.2233677, 11.2927933, -24.2196732, 11.2922516, -33.3911591, 33.3317184
13: -22.3793850, 12.3554974, -22.3779793, 12.3546753, -32.7058868, 32.6877594
14: -34.9177704, 6.3823247, -34.9141312, 6.3817840, -36.8073196, 36.7290192
15: -8.7074404, 16.7785645, -8.6970720, 16.7778759, -23.3241653, 23.3421211
16: -22.7865028, 3.1331518, -22.7849770, 3.1270049, -25.9135075, 25.9181290
17: -28.0601959, 8.0696192, -28.0569439, 8.0691872, -36.1293831, 36.1265640
18: -12.5984440, 18.6190472, -12.5962811, 18.6183929, -29.4223938, 29.3965836
19: -8.9544420, 8.0365143, -8.9532909, 8.0324612, -16.5415802, 16.5409698
20: -9.9369812, 8.7733994, -9.9359999, 8.7704659, -17.7355118, 17.7344704
21: -12.5633345, 9.1565275, -12.5618496, 9.1546879, -20.2343330, 20.2279701
22: -2.7921228, 18.5127239, -2.7840610, 18.5124550, -18.7463531, 18.7310371
23: -3.9373703, 15.2544537, -3.9364696, 15.2520208, -17.5524139, 17.5327873
24: -5.5280542, 17.3590832, -5.5275445, 17.3583031, -19.9289322, 19.9351883
25: 2.0683680, 24.2902565, 2.0723977, 24.2899399, -19.4433937, 19.4202538
26: -11.7096891, 21.5880089, -11.7042208, 21.5875301, -33.2972183, 33.2922287
27: -15.0035648, 9.9992733, -15.0024796, 9.9981775, -23.8031464, 23.8523140
28: -3.0070901, 18.0290642, -3.0032234, 18.0285320, -19.5820999, 19.6052971
29: -3.3593616, 15.5120869, -3.3529639, 15.5119276, -15.1819229, 15.1813297
30: -13.5982523, 13.7948217, -13.5973721, 13.7927322, -24.7353058, 24.7477798
31: -9.7040625, 11.3153229, -9.7027731, 11.3130074, -21.0170708, 21.0180969
32: -30.7136574, -3.7858167, -30.7124996, -3.7861352, -22.8668747, 22.8412514
33: -41.5144234, -3.2247558, -41.5138474, -3.2263832, -31.0198517, 31.0350952
34: -36.7852135, -3.8505182, -36.7793427, -3.8516541, -25.3937225, 25.3991356
35: -24.9028358, 5.5304484, -24.9021626, 5.5291643, -25.9802704, 26.0223618
36: -24.5638332, 6.4261465, -24.5613708, 6.4249721, -28.1046066, 28.1085739
37: -42.8781204, -6.5635729, -42.8774071, -6.5647964, -32.1126099, 32.1099167
38: -34.3823395, 3.4960599, -34.3807526, 3.4940324, -35.7779312, 35.7534103
39: -47.7326164, -7.0812373, -47.7305069, -7.0828156, -37.1155777, 37.1104279
40: -45.9581947, -18.9645138, -45.9578590, -18.9655800, -21.1811218, 21.1913795
41: -33.4874420, -4.6639929, -33.4865913, -4.6674552, -22.2566452, 22.2617531
42: -24.0439262, -0.4334321, -24.0427113, -0.4351308, -19.7429771, 19.7153664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=162, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1629

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6419278, upper bound: 10.6559057
time: 27.47 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6559059, upper bound: 10.6559057
time: 27.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 56.97 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 56.97
Output dim: 25, lower bound: -10.6499590, upper bound: 10.6343760
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 56.97
Output dim: 25, lower bound: -10.6499590, upper bound: 10.6482929
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 56.97
Output dim: 25, lower bound: -10.6419278, upper bound: 10.6559057
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 56.97
Output dim: 25, lower bound: -10.6559059, upper bound: 10.6559057

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -22.5183239, 8.5865688, -22.4838104, 8.5250959, -31.0434189, 31.0703793
1: -11.8715897, 6.3391056, -11.8609219, 6.2977881, -18.1693783, 18.2000275
2: -13.1278162, 7.3974032, -13.0717087, 7.3460159, -19.3586349, 19.3531914
3: -18.6515503, 6.2214074, -18.5898361, 6.1592798, -24.1320801, 24.1280212
4: -20.2779751, 3.2546673, -20.2475033, 3.2118692, -21.1531067, 21.1646347
5: -18.1232624, 7.5118551, -18.0352440, 7.4376116, -25.0396500, 25.0236130
6: -36.7319946, -11.2794437, -36.7049332, -11.3207493, -20.7127151, 20.7374763
7: -24.1093826, 1.2873716, -24.0654812, 1.2173617, -23.8283844, 23.8458023
8: -27.3731499, 1.0628581, -27.3422375, 1.0115018, -25.1794662, 25.1984406
9: -11.5288811, 11.8323116, -11.4857597, 11.7589331, -20.4903259, 20.5191879
10: -17.5982189, 12.3335743, -17.5084496, 12.2291050, -29.4430084, 29.4593048
11: -16.7402039, 10.4854727, -16.6951771, 10.4669285, -23.6632080, 23.6392021
12: -24.0996723, 11.2780972, -23.9946938, 11.1293783, -33.1503906, 33.1932602
13: -22.3336372, 12.2972364, -22.2888126, 12.2317238, -32.4888000, 32.5028000
14: -34.8018837, 6.3742003, -34.6790123, 6.2298250, -36.5873260, 36.6105804
15: -8.7028828, 16.7573509, -8.6707726, 16.7358303, -23.2355270, 23.2170296
16: -22.7452679, 3.1010537, -22.7061653, 3.0584459, -25.8037148, 25.8072186
17: -27.9637871, 8.0535984, -27.8730984, 7.9665565, -35.9303436, 35.9266968
18: -12.5381689, 18.6057053, -12.4754190, 18.5797691, -29.1971130, 29.1772385
19: -8.9166565, 8.0353785, -8.8814945, 8.0262547, -16.4579086, 16.4333267
20: -9.8929214, 8.7722311, -9.8456802, 8.7599373, -17.6479034, 17.6151810
21: -12.5145636, 9.1540089, -12.4686565, 9.1417952, -20.2022552, 20.1735458
22: -2.7425289, 18.5061512, -2.6831522, 18.5083618, -18.6444206, 18.5911636
23: -3.8948631, 15.2491989, -3.8519034, 15.2399254, -17.3919449, 17.3658295
24: -5.4884520, 17.3413296, -5.4423280, 17.3230495, -19.7345505, 19.7083397
25: 2.1109457, 24.2855721, 2.1584158, 24.2772789, -19.3148918, 19.2816277
26: -11.6015472, 21.5765839, -11.4955406, 21.5336037, -33.1351509, 33.0721245
27: -14.9679852, 9.9660358, -14.9015293, 9.9387016, -23.7517090, 23.7096405
28: -2.9654379, 18.0161819, -2.9164786, 18.0037651, -19.4569550, 19.4116173
29: -3.3248358, 15.5062180, -3.2827663, 15.4965115, -15.0997047, 15.0592728
30: -13.5537043, 13.7867317, -13.5093060, 13.7686310, -24.6308517, 24.6023636
31: -9.6545935, 11.2983418, -9.6119003, 11.2806759, -20.9352684, 20.9102421
32: -30.6886578, -3.8014202, -30.6584320, -3.8302288, -22.8135910, 22.8324623
33: -41.4972153, -3.2871146, -41.4523239, -3.3460608, -31.0582809, 31.0520782
34: -36.7625809, -3.8773694, -36.7258072, -3.9050646, -25.2425766, 25.2286987
35: -24.8894558, 5.4933472, -24.8574982, 5.4585161, -25.9011765, 25.8659286
36: -24.5403690, 6.3912873, -24.5074368, 6.3638229, -27.9978561, 27.9880371
37: -42.8563919, -6.5928288, -42.8244171, -6.6289520, -32.1873627, 32.1913834
38: -34.3540115, 3.4705067, -34.3110962, 3.4436831, -35.5672607, 35.5619354
39: -47.7099991, -7.1227360, -47.6766815, -7.1667304, -37.2773743, 37.2841797
40: -45.9463348, -19.0139256, -45.9106407, -19.0571156, -21.3437500, 21.3651009
41: -33.4775162, -4.6889124, -33.4496460, -4.7243829, -22.1928139, 22.2184029
42: -24.0268555, -0.4571311, -24.0058517, -0.4907098, -19.6064873, 19.6359253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1645

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6298060, upper bound: 10.6332053
time: 30.07 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6493776, upper bound: 10.6332086
time: 29.11 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -22.5207424, 8.6205635, -22.5599632, 8.5896597, -31.1104012, 31.1805267
1: -11.8722477, 6.3589816, -11.9012318, 6.3374424, -18.2096901, 18.2602139
2: -13.1285324, 7.4159560, -13.1057291, 7.3824420, -19.4206772, 19.4341965
3: -18.6529961, 6.2443638, -18.6415558, 6.2052541, -24.2214813, 24.2477112
4: -20.2799263, 3.2694538, -20.2842903, 3.2420835, -21.2398605, 21.2759399
5: -18.1245651, 7.5383725, -18.0911522, 7.4904118, -25.1097412, 25.1231995
6: -36.7364159, -11.2684374, -36.7397919, -11.2943840, -20.7307396, 20.7736740
7: -24.1106873, 1.3103077, -24.1208477, 1.2634287, -23.9412003, 23.9918442
8: -27.3744736, 1.0826912, -27.3866959, 1.0540657, -25.3482361, 25.3911667
9: -11.5310116, 11.8567104, -11.5370960, 11.8042259, -20.6849823, 20.7546921
10: -17.6134453, 12.3405361, -17.5461197, 12.2742414, -29.5042114, 29.5015945
11: -16.7580700, 10.4872742, -16.7399330, 10.5136604, -23.7953415, 23.7415352
12: -24.1076279, 11.2815838, -24.0188599, 11.1628017, -33.1449280, 33.1678848
13: -22.3371315, 12.3393240, -22.3827305, 12.3132582, -32.6232300, 32.6960220
14: -34.8085899, 6.3780403, -34.7107887, 6.2443633, -36.5609589, 36.5945358
15: -8.7071981, 16.7646790, -8.6923313, 16.7539978, -23.2956924, 23.2861824
16: -22.7482147, 3.1235895, -22.7603188, 3.1028605, -25.8510742, 25.8839073
17: -27.9684563, 8.0587521, -27.9095001, 7.9847097, -35.9531670, 35.9682541
18: -12.5713243, 18.6078072, -12.5443068, 18.6567192, -29.3954315, 29.3304520
19: -8.9351826, 8.0361729, -8.9248257, 8.0508804, -16.5276680, 16.5037689
20: -9.9134045, 8.7729855, -9.8905754, 8.7860556, -17.7177887, 17.6814194
21: -12.5327415, 9.1553230, -12.5126467, 9.1682606, -20.2151375, 20.1851883
22: -2.7630167, 18.5069904, -2.7300763, 18.5468903, -18.7360039, 18.6662083
23: -3.9199867, 15.2508259, -3.9036379, 15.2849417, -17.5431824, 17.4945450
24: -5.5162849, 17.3422699, -5.4986954, 17.3819408, -19.9468842, 19.8820953
25: 2.0874958, 24.2865353, 2.1085815, 24.3127899, -19.4296341, 19.3817635
26: -11.6391926, 21.5790176, -11.5744991, 21.6172714, -33.2564621, 33.1535187
27: -14.9899406, 9.9679012, -14.9471378, 9.9915209, -23.8375320, 23.7634125
28: -2.9900322, 18.0180588, -2.9693017, 18.0436192, -19.5855865, 19.5266228
29: -3.3359179, 15.5077581, -3.3143172, 15.5292110, -15.1638699, 15.1042366
30: -13.5752354, 13.7894764, -13.5536327, 13.8175621, -24.7345276, 24.6706390
31: -9.6799393, 11.2998705, -9.6675873, 11.3152370, -20.9951763, 20.9674568
32: -30.6942120, -3.7950020, -30.6855221, -3.8133993, -22.7761879, 22.8056412
33: -41.5065498, -3.2660198, -41.5175018, -3.3008022, -30.9230347, 30.9616013
34: -36.7808762, -3.8758917, -36.7656746, -3.8613839, -25.3753128, 25.3382454
35: -24.8940506, 5.4949832, -24.8802795, 5.4701200, -25.9297791, 25.8955154
36: -24.5451279, 6.3938932, -24.5306034, 6.3747516, -28.0426941, 28.0417175
37: -42.8631897, -6.5873280, -42.8619194, -6.6108494, -32.0525436, 32.0600739
38: -34.3638153, 3.4716492, -34.3427391, 3.4589796, -35.6851196, 35.6938858
39: -47.7168770, -7.1037540, -47.7445297, -7.1221724, -37.0367584, 37.0886078
40: -45.9522247, -18.9976616, -45.9590988, -19.0212555, -21.1225433, 21.1528702
41: -33.4808655, -4.6856046, -33.4645805, -4.7117209, -22.2041550, 22.2222824
42: -24.0313187, -0.4439106, -24.0382633, -0.4601760, -19.6578751, 19.7014923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1645

## Relational analysis of IS_B1_B2_A1

### Relational analysis result of IS_B1_B2_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6298221, upper bound: 10.6476961
time: 32.89 seconds

## Relational analysis of IS_B1_B2_A2

### Relational analysis result of IS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6493776, upper bound: 10.6477266
time: 29.14 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -22.5251274, 8.5908031, -22.5268326, 8.6175823, -31.1427097, 31.1176357
1: -11.8782854, 6.3453565, -11.8780136, 6.3608007, -18.2390862, 18.2233696
2: -13.1312981, 7.4227734, -13.1317616, 7.4363298, -19.4142456, 19.4409714
3: -18.6567574, 6.2479725, -18.6575813, 6.2649708, -24.2392883, 24.2391129
4: -20.2809372, 3.2698641, -20.2823200, 3.2805066, -21.2094803, 21.2152252
5: -18.1282120, 7.5495281, -18.1293259, 7.5690060, -25.1256943, 25.1630783
6: -36.7331924, -11.2741098, -36.7368317, -11.2695332, -20.7543793, 20.7603340
7: -24.1185570, 1.3240013, -24.1195297, 1.3405707, -23.9303360, 23.9490280
8: -27.3773155, 1.0816364, -27.3782005, 1.0961742, -25.2491760, 25.2575531
9: -11.5716114, 11.8163366, -11.5717974, 11.8359003, -20.6089783, 20.5220947
10: -17.6746082, 12.3373547, -17.6841927, 12.3422222, -29.6344452, 29.5968323
11: -16.7531700, 10.4849443, -16.7670555, 10.4832344, -23.7079468, 23.7151871
12: -24.2038345, 11.2841406, -24.2081985, 11.2871895, -33.4140701, 33.3608627
13: -22.3707829, 12.2714167, -22.3728466, 12.3056669, -32.5917969, 32.5418167
14: -34.9020386, 6.3741384, -34.9048538, 6.3770065, -36.8355408, 36.7623596
15: -8.6979055, 16.7606468, -8.6914139, 16.7673626, -23.2657776, 23.2796974
16: -22.7782555, 3.0890789, -22.7801247, 3.1011877, -25.8794441, 25.8692036
17: -28.0486107, 8.0585537, -28.0500793, 8.0625687, -36.1111794, 36.1086349
18: -12.5322189, 18.6144600, -12.5576687, 18.6155968, -29.2646561, 29.2651596
19: -8.9169540, 8.0347509, -8.9313974, 8.0313931, -16.4738235, 16.4891167
20: -9.8954315, 8.7711668, -9.9114866, 8.7691059, -17.6694527, 17.6845741
21: -12.5263844, 9.1535358, -12.5400772, 9.1528940, -20.2266388, 20.2359772
22: -2.7517028, 18.5105438, -2.7604342, 18.5111694, -18.6708107, 18.6721058
23: -3.8875680, 15.2510376, -3.9074392, 15.2499971, -17.4182854, 17.4184418
24: -5.4722438, 17.3568344, -5.4950285, 17.3569984, -19.7454987, 19.7744904
25: 2.1159163, 24.2877216, 2.1001658, 24.2884235, -19.3380127, 19.3341141
26: -11.6346760, 21.5823879, -11.6604147, 21.5841751, -33.2188492, 33.2428017
27: -14.9586163, 9.9950476, -14.9762554, 9.9956760, -23.7443619, 23.8113861
28: -2.9574671, 18.0252686, -2.9742527, 18.0262146, -19.4649048, 19.5073700
29: -3.3368235, 15.5090513, -3.3397532, 15.5101042, -15.1378136, 15.1449966
30: -13.5522852, 13.7891455, -13.5705585, 13.7893171, -24.6551208, 24.6834679
31: -9.6535006, 11.3119793, -9.6731377, 11.3109999, -20.9645004, 20.9851170
32: -30.7014503, -3.8040943, -30.7051582, -3.7968874, -22.8987656, 22.8728218
33: -41.4951210, -3.2629523, -41.5022659, -3.2491469, -31.1469727, 31.1599655
34: -36.7480392, -3.8541937, -36.7575417, -3.8538284, -25.2820892, 25.3017998
35: -24.8923912, 5.5243874, -24.8960075, 5.5256176, -25.9574509, 26.0015411
36: -24.5536156, 6.4172711, -24.5552902, 6.4196730, -28.0599060, 28.0645981
37: -42.8621979, -6.5810614, -42.8679504, -6.5750999, -32.2434921, 32.2438126
38: -34.3606148, 3.4914260, -34.3679581, 3.4912353, -35.6560059, 35.6382141
39: -47.7174797, -7.1220341, -47.7214470, -7.1068745, -37.3491821, 37.3390503
40: -45.9446297, -18.9957771, -45.9495583, -18.9843025, -21.3941956, 21.4088936
41: -33.4795494, -4.6761360, -33.4818726, -4.6747098, -22.2492332, 22.2543488
42: -24.0338936, -0.4602790, -24.0366001, -0.4508133, -19.6963539, 19.6614666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=162, inp2_unstable=162, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1645

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6407621, upper bound: 10.6357092
time: 30.08 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6407650, upper bound: 10.6553288
time: 27.05 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -22.6013145, 8.6553812, -22.5292320, 8.6515732, -31.2528877, 31.1846123
1: -11.9186039, 6.3850164, -11.8786774, 6.3806915, -18.2992954, 18.2636948
2: -13.1653223, 7.4592218, -13.1324959, 7.4548607, -19.4952621, 19.5029602
3: -18.7084751, 6.2938824, -18.6590271, 6.2879486, -24.3589401, 24.3285217
4: -20.3177357, 3.3000779, -20.2842712, 3.2952640, -21.3207855, 21.3019791
5: -18.1841087, 7.6023264, -18.1306229, 7.5954666, -25.2252274, 25.2331238
6: -36.7680473, -11.2477570, -36.7412491, -11.2585526, -20.7905884, 20.7782822
7: -24.1738968, 1.3700764, -24.1208496, 1.3635199, -24.0763779, 24.0618134
8: -27.4217567, 1.1241884, -27.3795090, 1.1159925, -25.4418869, 25.4262924
9: -11.6229382, 11.8616362, -11.5739660, 11.8602686, -20.8444519, 20.7167664
10: -17.7122383, 12.3824883, -17.6993446, 12.3491774, -29.6767273, 29.6580811
11: -16.7979279, 10.5316877, -16.7849541, 10.4850187, -23.8101959, 23.8472748
12: -24.2279892, 11.3175735, -24.2161312, 11.2907028, -33.3886566, 33.3553619
13: -22.4647408, 12.3529415, -22.3763809, 12.3477030, -32.7850494, 32.6762314
14: -34.9338226, 6.3886356, -34.9115982, 6.3808146, -36.8199158, 36.7357330
15: -8.7194881, 16.7788067, -8.6957541, 16.7746639, -23.3349991, 23.3398438
16: -22.8323898, 3.1334667, -22.7830410, 3.1237185, -25.9561081, 25.9165077
17: -28.0850887, 8.0767422, -28.0547810, 8.0677032, -36.1527939, 36.1315231
18: -12.6010723, 18.6914406, -12.5908356, 18.6177368, -29.4178925, 29.4634018
19: -8.9602890, 8.0594025, -8.9498978, 8.0322094, -16.5442657, 16.5588646
20: -9.9403381, 8.7973013, -9.9319839, 8.7698612, -17.7357101, 17.7544899
21: -12.5703773, 9.1800194, -12.5582457, 9.1541929, -20.2382965, 20.2488480
22: -2.7986064, 18.5490627, -2.7808924, 18.5119991, -18.7458344, 18.7636604
23: -3.9393392, 15.2960596, -3.9325380, 15.2516050, -17.5470009, 17.5696602
24: -5.5286455, 17.4157448, -5.5228562, 17.3579311, -19.9192505, 19.9868355
25: 2.0660701, 24.3231907, 2.0767097, 24.2893639, -19.4381332, 19.4488525
26: -11.7136211, 21.6660213, -11.6980972, 21.5866833, -33.3003044, 33.3641205
27: -15.0042114, 10.0478783, -14.9982071, 9.9975185, -23.7981262, 23.8971405
28: -3.0103083, 18.0651474, -2.9988742, 18.0281029, -19.5799332, 19.6359940
29: -3.3683529, 15.5417480, -3.3508372, 15.5116301, -15.1827469, 15.2091370
30: -13.5966568, 13.8380842, -13.5920620, 13.7920399, -24.7234268, 24.7872009
31: -9.7091846, 11.3465490, -9.6984863, 11.3125200, -21.0217056, 21.0450363
32: -30.7285538, -3.7873001, -30.7107048, -3.7904472, -22.8719177, 22.8353386
33: -41.5603294, -3.2177820, -41.5116463, -3.2280583, -31.0566101, 31.0245895
34: -36.7878914, -3.8105083, -36.7758484, -3.8523593, -25.3916473, 25.4345360
35: -24.9151821, 5.5360208, -24.9006176, 5.5271616, -25.9870834, 26.0301132
36: -24.5767937, 6.4282084, -24.5600224, 6.4223080, -28.1135864, 28.1094131
37: -42.8996887, -6.5630074, -42.8747025, -6.5695829, -32.1121292, 32.1089859
38: -34.3922882, 3.5067477, -34.3777351, 3.4923801, -35.7879486, 35.7560349
39: -47.7853470, -7.0774894, -47.7282906, -7.0878797, -37.1536713, 37.0984192
40: -45.9930954, -18.9599686, -45.9554520, -18.9680195, -21.1819839, 21.1876221
41: -33.4944916, -4.6634464, -33.4852600, -4.6713843, -22.2531242, 22.2656670
42: -24.0663147, -0.4297571, -24.0410728, -0.4376161, -19.7619667, 19.7128029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=162, inp2_unstable=162, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1645

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6553041, upper bound: 10.6357218
time: 26.81 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6553290, upper bound: 10.6553288
time: 29.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 58.76 seconds
IS_B1_B1_A1, status: Status.VERIFIED, split count: 3, time: 58.76
Output dim: 25, lower bound: -10.6298060, upper bound: 10.6332053
IS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 58.76
Output dim: 25, lower bound: -10.6493776, upper bound: 10.6332086
IS_B1_B2_A1, status: Status.VERIFIED, split count: 3, time: 58.76
Output dim: 25, lower bound: -10.6298221, upper bound: 10.6476961
IS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 58.76
Output dim: 25, lower bound: -10.6493776, upper bound: 10.6477266
IS_B2_A1_B1, status: Status.VERIFIED, split count: 3, time: 58.76
Output dim: 25, lower bound: -10.6407621, upper bound: 10.6357092
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 58.76
Output dim: 25, lower bound: -10.6407650, upper bound: 10.6553288
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 58.76
Output dim: 25, lower bound: -10.6553041, upper bound: 10.6357218
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 58.76
Output dim: 25, lower bound: -10.6553290, upper bound: 10.6553288

## BFS IS instance: IS_B1_B1_A2

### Backsubstitution after applying IS history:
0: -22.5179825, 8.5820484, -22.4835968, 8.5223703, -31.0403519, 31.0656452
1: -11.8714237, 6.3365092, -11.8608322, 6.2960315, -18.1674557, 18.1973419
2: -13.1277056, 7.3946509, -13.0716515, 7.3443518, -19.3570938, 19.3436012
3: -18.6512909, 6.2177353, -18.5896854, 6.1570959, -24.1296005, 24.1162491
4: -20.2777596, 3.2528200, -20.2473450, 3.2107606, -21.1513443, 21.1461945
5: -18.1230106, 7.5081568, -18.0350742, 7.4353952, -25.0370789, 25.0165787
6: -36.7310181, -11.2830925, -36.7043076, -11.3230572, -20.7089233, 20.7315407
7: -24.1090164, 1.2843359, -24.0652351, 1.2152853, -23.8267822, 23.8334579
8: -27.3729286, 1.0592117, -27.3421059, 1.0092793, -25.1776581, 25.1710663
9: -11.5284262, 11.8285732, -11.4855099, 11.7566786, -20.4884949, 20.4852676
10: -17.5962791, 12.3323355, -17.5072441, 12.2283316, -29.4395752, 29.4575348
11: -16.7376366, 10.4851370, -16.6936378, 10.4667044, -23.6419296, 23.6372681
12: -24.0979424, 11.2774191, -23.9935360, 11.1289568, -33.1465759, 33.1948776
13: -22.3327808, 12.2941656, -22.2882843, 12.2296867, -32.4864502, 32.4822083
14: -34.8006973, 6.3735914, -34.6783218, 6.2294874, -36.5744476, 36.6246948
15: -8.7019567, 16.7550678, -8.6701860, 16.7344131, -23.2333145, 23.2054291
16: -22.7443237, 3.0985839, -22.7055435, 3.0568266, -25.8011513, 25.8041267
17: -27.9611492, 8.0529823, -27.8715324, 7.9661803, -35.9273300, 35.9245148
18: -12.5343170, 18.6052246, -12.4730911, 18.5794411, -29.1821899, 29.1750870
19: -8.9136600, 8.0352192, -8.8796740, 8.0261354, -16.4529533, 16.4312248
20: -9.8905048, 8.7720413, -9.8442268, 8.7598228, -17.6390991, 17.6133919
21: -12.5124302, 9.1537838, -12.4673653, 9.1416435, -20.1963654, 20.1770706
22: -2.7399025, 18.5060062, -2.6815567, 18.5082779, -18.6351852, 18.5895824
23: -3.8914165, 15.2489405, -3.8498077, 15.2397413, -17.3768463, 17.3631592
24: -5.4845057, 17.3411350, -5.4399567, 17.3229446, -19.7113190, 19.7057381
25: 2.1148520, 24.2852898, 2.1607561, 24.2770939, -19.3009949, 19.2788353
26: -11.5970192, 21.5759583, -11.4928303, 21.5332127, -33.1302338, 33.0687866
27: -14.9654846, 9.9656935, -14.8999243, 9.9384708, -23.7447433, 23.7081146
28: -2.9616699, 18.0158501, -2.9142203, 18.0035400, -19.4406967, 19.4088364
29: -3.3229017, 15.5060444, -3.2816076, 15.4963903, -15.0827751, 15.0575848
30: -13.5503721, 13.7863045, -13.5072756, 13.7683516, -24.6095886, 24.6001663
31: -9.6507320, 11.2980127, -9.6095610, 11.2804642, -20.9311962, 20.9075737
32: -30.6878128, -3.8041515, -30.6578941, -3.8319173, -22.8097954, 22.8298340
33: -41.4955215, -3.2922406, -41.4513016, -3.3492522, -31.0499496, 31.0506668
34: -36.7599411, -3.8777399, -36.7241859, -3.9052725, -25.2258759, 25.2269363
35: -24.8881836, 5.4930644, -24.8567352, 5.4582825, -25.8987427, 25.8652191
36: -24.5375881, 6.3908572, -24.5057106, 6.3635440, -27.9923553, 27.9859695
37: -42.8548737, -6.5930481, -42.8234596, -6.6291142, -32.1770554, 32.2025299
38: -34.3503189, 3.4697223, -34.3087959, 3.4431844, -35.5512390, 35.5580597
39: -47.7086830, -7.1280036, -47.6758728, -7.1698332, -37.2609940, 37.2935791
40: -45.9451714, -19.0184307, -45.9099121, -19.0598907, -21.3317261, 21.3747063
41: -33.4767647, -4.6894622, -33.4491730, -4.7247052, -22.1910286, 22.2179680
42: -24.0261078, -0.4585087, -24.0053806, -0.4915428, -19.6046219, 19.6264915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=162, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 569

## Relational analysis of IS_B1_B1_A2_B1

### Relational analysis result of IS_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6289291, upper bound: 10.6177124
time: 28.03 seconds

## Relational analysis of IS_B1_B1_A2_B2

### Relational analysis result of IS_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6484985, upper bound: 10.6323119
time: 29.93 seconds

## BFS IS instance: IS_B1_B2_A2

### Backsubstitution after applying IS history:
0: -22.5204144, 8.6160593, -22.5597458, 8.5869970, -31.1074104, 31.1758041
1: -11.8720646, 6.3564825, -11.9011211, 6.3357944, -18.2078590, 18.2576027
2: -13.1284294, 7.4132552, -13.1056614, 7.3808441, -19.4189529, 19.4262924
3: -18.6527481, 6.2407341, -18.6413879, 6.2030921, -24.2190933, 24.2386322
4: -20.2796268, 3.2676182, -20.2841167, 3.2409964, -21.2382355, 21.2644272
5: -18.1242943, 7.5347056, -18.0910110, 7.4882388, -25.1072159, 25.1170349
6: -36.7354126, -11.2718582, -36.7391930, -11.2965269, -20.7279587, 20.7694893
7: -24.1103649, 1.3072810, -24.1206512, 1.2613721, -23.9389877, 23.9832306
8: -27.3742695, 1.0790749, -27.3865700, 1.0519247, -25.3464508, 25.3749313
9: -11.5305815, 11.8529682, -11.5368271, 11.8020153, -20.6832123, 20.7335434
10: -17.6115932, 12.3392811, -17.5450230, 12.2735281, -29.5009155, 29.5000763
11: -16.7555313, 10.4868822, -16.7384396, 10.5134258, -23.7821808, 23.7393303
12: -24.1058998, 11.2809658, -24.0177536, 11.1624813, -33.1421661, 33.1672897
13: -22.3362942, 12.3363132, -22.3822460, 12.3112774, -32.6209869, 32.6817856
14: -34.8074493, 6.3774920, -34.7101059, 6.2439904, -36.5508728, 36.5951920
15: -8.7062473, 16.7624397, -8.6917686, 16.7526913, -23.2935867, 23.2788048
16: -22.7473373, 3.1211479, -22.7597961, 3.1012866, -25.8486233, 25.8809433
17: -27.9658585, 8.0580540, -27.9078999, 7.9842787, -35.9501381, 35.9659538
18: -12.5674953, 18.6073341, -12.5419979, 18.6564255, -29.3859100, 29.3277054
19: -8.9322624, 8.0360107, -8.9230728, 8.0507793, -16.5244598, 16.5017929
20: -9.9110594, 8.7727680, -9.8891802, 8.7859135, -17.7103920, 17.6796875
21: -12.5306587, 9.1550703, -12.5114202, 9.1681004, -20.2101517, 20.1870232
22: -2.7603993, 18.5068092, -2.7285252, 18.5467815, -18.7287064, 18.6647186
23: -3.9165621, 15.2505732, -3.9016056, 15.2847843, -17.5346451, 17.4919853
24: -5.5123425, 17.3420525, -5.4963617, 17.3818550, -19.9317017, 19.8795433
25: 2.0913377, 24.2862473, 2.1108589, 24.3126335, -19.4189911, 19.3790665
26: -11.6347532, 21.5784721, -11.5718269, 21.6169205, -33.2516747, 33.1502991
27: -14.9874840, 9.9675264, -14.9455643, 9.9913282, -23.8311768, 23.7619324
28: -2.9863281, 18.0176773, -2.9671316, 18.0434113, -19.5740623, 19.5239563
29: -3.3340168, 15.5075464, -3.3132095, 15.5290804, -15.1519394, 15.1024971
30: -13.5719776, 13.7889595, -13.5517244, 13.8172588, -24.7182617, 24.6681099
31: -9.6761894, 11.2995224, -9.6653309, 11.3150215, -20.9912109, 20.9648533
32: -30.6933193, -3.7975893, -30.6850281, -3.8148971, -22.7730026, 22.8009377
33: -41.5047760, -3.2705984, -41.5164948, -3.3036823, -30.9173584, 30.9499283
34: -36.7783432, -3.8763046, -36.7640991, -3.8616228, -25.3636093, 25.3365974
35: -24.8928452, 5.4945922, -24.8795547, 5.4699330, -25.9271393, 25.8946838
36: -24.5423412, 6.3934937, -24.5289268, 6.3745065, -28.0386810, 28.0396729
37: -42.8616714, -6.5875654, -42.8610420, -6.6110020, -32.0453186, 32.0635452
38: -34.3602066, 3.4708834, -34.3405495, 3.4585319, -35.6734390, 35.6902008
39: -47.7155342, -7.1083322, -47.7437706, -7.1246643, -37.0296860, 37.0841217
40: -45.9508820, -19.0021305, -45.9583054, -19.0240307, -21.1157684, 21.1447372
41: -33.4801598, -4.6861000, -33.4641571, -4.7120271, -22.2026367, 22.2215729
42: -24.0303898, -0.4452465, -24.0377178, -0.4609861, -19.6561775, 19.6962757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=162, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 569

## Relational analysis of IS_B1_B2_A2_B1

### Relational analysis result of IS_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6484985, upper bound: 10.6322568
time: 28.40 seconds

## Relational analysis of IS_B1_B2_A2_B2

### Relational analysis result of IS_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6484985, upper bound: 10.6468573
time: 27.36 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -22.5249348, 8.5881138, -22.5264702, 8.6130514, -31.1379852, 31.1145840
1: -11.8781605, 6.3436389, -11.8778477, 6.3582268, -18.2363873, 18.2214870
2: -13.1312284, 7.4211135, -13.1316376, 7.4335442, -19.4046326, 19.4394379
3: -18.6565933, 6.2457457, -18.6573296, 6.2612939, -24.2275085, 24.2366257
4: -20.2807732, 3.2687621, -20.2820683, 3.2786167, -21.1910706, 21.2134171
5: -18.1280289, 7.5473237, -18.1290512, 7.5652895, -25.1185684, 25.1605225
6: -36.7325821, -11.2764149, -36.7358437, -11.2731762, -20.7484169, 20.7565384
7: -24.1183071, 1.3219578, -24.1191845, 1.3375363, -23.9180222, 23.9474182
8: -27.3771725, 1.0793939, -27.3780060, 1.0924664, -25.2217331, 25.2557755
9: -11.5713482, 11.8141060, -11.5713673, 11.8321247, -20.5750198, 20.5202713
10: -17.6734314, 12.3365726, -17.6822357, 12.3409748, -29.6327057, 29.5934448
11: -16.7516289, 10.4847469, -16.7644920, 10.4828882, -23.7059937, 23.6938591
12: -24.2026749, 11.2837086, -24.2063980, 11.2865467, -33.4156265, 33.3570099
13: -22.3702545, 12.2693691, -22.3719826, 12.3025703, -32.5712128, 32.5394058
14: -34.9013519, 6.3737745, -34.9037552, 6.3764300, -36.8495483, 36.7495041
15: -8.6973305, 16.7592659, -8.6904945, 16.7650261, -23.2541809, 23.2775040
16: -22.7776299, 3.0874500, -22.7791252, 3.0987160, -25.8763466, 25.8665752
17: -28.0470352, 8.0581818, -28.0475063, 8.0619097, -36.1089439, 36.1056900
18: -12.5299263, 18.6141548, -12.5537920, 18.6151009, -29.2625275, 29.2502213
19: -8.9151411, 8.0346413, -8.9283714, 8.0312386, -16.4717140, 16.4841499
20: -9.8939762, 8.7710457, -9.9090824, 8.7689095, -17.6676636, 17.6757698
21: -12.5251007, 9.1533794, -12.5379143, 9.1526814, -20.2301865, 20.2300873
22: -2.7501125, 18.5104389, -2.7577796, 18.5110168, -18.6692276, 18.6628456
23: -3.8855162, 15.2508678, -3.9039898, 15.2497482, -17.4156342, 17.4033279
24: -5.4698982, 17.3567219, -5.4910851, 17.3568077, -19.7428894, 19.7512932
25: 2.1182499, 24.2875137, 2.1040888, 24.2881279, -19.3352051, 19.3202057
26: -11.6319914, 21.5819988, -11.6559124, 21.5835400, -33.2155304, 33.2379112
27: -14.9570007, 9.9948177, -14.9737787, 9.9953346, -23.7428284, 23.8043823
28: -2.9552341, 18.0250626, -2.9705124, 18.0259056, -19.4621544, 19.4911041
29: -3.3356419, 15.5089331, -3.3378353, 15.5098934, -15.1361160, 15.1280727
30: -13.5502605, 13.7888803, -13.5671949, 13.7888880, -24.6528854, 24.6621971
31: -9.6511612, 11.3117638, -9.6692638, 11.3106613, -20.9618225, 20.9810276
32: -30.7009258, -3.8057833, -30.7043056, -3.7996168, -22.8960915, 22.8690147
33: -41.4941330, -3.2661982, -41.5006409, -3.2542839, -31.1455688, 31.1516418
34: -36.7464409, -3.8544321, -36.7548866, -3.8542304, -25.2803497, 25.2850800
35: -24.8916435, 5.5241742, -24.8947792, 5.5252547, -25.9567642, 25.9990845
36: -24.5518894, 6.4170241, -24.5525284, 6.4193029, -28.0579071, 28.0590897
37: -42.8612366, -6.5812092, -42.8664169, -6.5753975, -32.2545242, 32.2334747
38: -34.3583679, 3.4908829, -34.3642578, 3.4904189, -35.6521301, 35.6221695
39: -47.7166634, -7.1251421, -47.7201195, -7.1121445, -37.3585205, 37.3226700
40: -45.9438896, -18.9985600, -45.9484024, -18.9888248, -21.4037704, 21.3968735
41: -33.4790649, -4.6764593, -33.4811516, -4.6752605, -22.2488060, 22.2525673
42: -24.0334320, -0.4611421, -24.0358410, -0.4521861, -19.6869125, 19.6595993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=162, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 569

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6252827, upper bound: 10.6544994
time: 27.91 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6399149, upper bound: 10.6544994
time: 37.21 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -22.5988960, 8.6168518, -22.4762764, 8.5848160, -31.1837120, 31.0931282
1: -11.9174461, 6.3603210, -11.8455048, 6.3376317, -18.2550774, 18.2058258
2: -13.1643391, 7.4379206, -13.1054306, 7.4174433, -19.4281693, 19.4251633
3: -18.7068481, 6.2636814, -18.6171188, 6.2340565, -24.2590103, 24.2107468
4: -20.3155174, 3.2752788, -20.2491074, 3.2507620, -21.2114029, 21.1771774
5: -18.1823959, 7.5710087, -18.0887260, 7.5392447, -25.1498795, 25.1422882
6: -36.7624550, -11.2541361, -36.7274857, -11.2731028, -20.7583618, 20.7474632
7: -24.1718216, 1.3414004, -24.0748405, 1.3132317, -23.9554672, 23.9170074
8: -27.4199677, 1.0956154, -27.3371944, 1.0647140, -25.2536697, 25.2208099
9: -11.6202736, 11.8294621, -11.5334873, 11.8038330, -20.6276016, 20.4786453
10: -17.7032394, 12.3743248, -17.6752090, 12.3245745, -29.6397629, 29.6245880
11: -16.7776566, 10.5294781, -16.7371788, 10.4537983, -23.6834030, 23.7257996
12: -24.2114010, 11.3128805, -24.1819801, 11.2543106, -33.3794250, 33.3711014
13: -22.4616718, 12.3214455, -22.3327541, 12.2905617, -32.6656876, 32.5392075
14: -34.9259453, 6.3845663, -34.8832397, 6.3666396, -36.8202591, 36.7652512
15: -8.7151442, 16.7605667, -8.6692028, 16.7412720, -23.2563629, 23.2541351
16: -22.8277893, 3.1129918, -22.7433701, 3.0872097, -25.9149990, 25.8563614
17: -28.0758362, 8.0722198, -28.0136261, 8.0411491, -36.1169853, 36.0858459
18: -12.5691900, 18.6891747, -12.5298281, 18.5689659, -29.2455063, 29.3117142
19: -8.9368725, 8.0584755, -8.9024353, 8.0131779, -16.4744873, 16.4832306
20: -9.9203625, 8.7962484, -9.8930597, 8.7533007, -17.6734810, 17.6889915
21: -12.5537186, 9.1782207, -12.5197515, 9.1408024, -20.2309418, 20.2385521
22: -2.7797832, 18.5479927, -2.7423759, 18.4902992, -18.6714134, 18.6909447
23: -3.9097710, 15.2945375, -3.8770638, 15.2217674, -17.3977127, 17.4261360
24: -5.4958577, 17.4147282, -5.4637518, 17.3189678, -19.7184372, 19.8007126
25: 2.0968509, 24.3220444, 2.1334901, 24.2588997, -19.3151855, 19.3305550
26: -11.6771431, 21.6634483, -11.6303234, 21.5392170, -33.2163620, 33.2937698
27: -14.9840279, 10.0457993, -14.9602842, 9.9672556, -23.7367401, 23.8463058
28: -2.9793859, 18.0635967, -2.9410753, 17.9995251, -19.4522400, 19.5102272
29: -3.3554487, 15.5401955, -3.3182516, 15.4901781, -15.1291428, 15.1562233
30: -13.5718346, 13.8354721, -13.5459394, 13.7606077, -24.6353989, 24.7071457
31: -9.6789474, 11.3443327, -9.6400604, 11.2854462, -20.9643936, 20.9843941
32: -30.7238770, -3.7939358, -30.6965523, -3.8053098, -22.9019852, 22.8730011
33: -41.5493622, -3.2294893, -41.4826279, -3.2550292, -31.1540070, 31.1323853
34: -36.7687416, -3.8124523, -36.7400093, -3.8864708, -25.2501907, 25.3089447
35: -24.9093399, 5.5334544, -24.8871098, 5.5097446, -25.9466095, 25.9993896
36: -24.5683041, 6.4258385, -24.5426025, 6.4069710, -28.0612869, 28.0609894
37: -42.8907890, -6.5649958, -42.8551064, -6.5790176, -32.2140961, 32.2375336
38: -34.3729858, 3.5036645, -34.3385696, 3.4587965, -35.6486816, 35.6142349
39: -47.7779922, -7.0849743, -47.7011528, -7.1092720, -37.3712234, 37.3399963
40: -45.9874458, -18.9693127, -45.9325066, -18.9918060, -21.3611069, 21.4187164
41: -33.4900970, -4.6670928, -33.4756699, -4.6822195, -22.2376747, 22.2566414
42: -24.0605927, -0.4391103, -24.0197659, -0.4571264, -19.7155952, 19.6612988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=162, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 569

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6398537, upper bound: 10.6348966
time: 29.42 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6544653, upper bound: 10.6348966
time: 81.91 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.6011124, 8.6527138, -22.5288811, 8.6471176, -31.2482300, 31.1815948
1: -11.9184895, 6.3833756, -11.8785334, 6.3781581, -18.2966480, 18.2619095
2: -13.1652622, 7.4576025, -13.1323481, 7.4521513, -19.4873657, 19.5012589
3: -18.7082977, 6.2917533, -18.6587582, 6.2843156, -24.3498459, 24.3260803
4: -20.3175583, 3.2989955, -20.2840004, 3.2934716, -21.3093262, 21.3003082
5: -18.1839638, 7.6001692, -18.1303387, 7.5917974, -25.2190552, 25.2306366
6: -36.7674637, -11.2499256, -36.7402115, -11.2619505, -20.7863808, 20.7755089
7: -24.1736698, 1.3680284, -24.1205292, 1.3604655, -24.0677948, 24.0595627
8: -27.4216423, 1.1219873, -27.3792877, 1.1123362, -25.4255676, 25.4245682
9: -11.6227112, 11.8594084, -11.5735130, 11.8565254, -20.8233185, 20.7149734
10: -17.7111092, 12.3817568, -17.6975307, 12.3479233, -29.6752090, 29.6547699
11: -16.7964325, 10.5314379, -16.7824135, 10.4846210, -23.8079987, 23.8341103
12: -24.2269096, 11.3172369, -24.2144241, 11.2900543, -33.3880234, 33.3525925
13: -22.4642220, 12.3509712, -22.3755112, 12.3446999, -32.7707596, 32.6738968
14: -34.9331322, 6.3883076, -34.9104347, 6.3802691, -36.8204193, 36.7256775
15: -8.7189121, 16.7775116, -8.6948175, 16.7724075, -23.3276062, 23.3377533
16: -22.8318901, 3.1319046, -22.7821922, 3.1212904, -25.9531803, 25.9140968
17: -28.0835476, 8.0763092, -28.0521431, 8.0670414, -36.1505890, 36.1284523
18: -12.5987968, 18.6911392, -12.5870152, 18.6172028, -29.4151306, 29.4539642
19: -8.9585266, 8.0592899, -8.9469643, 8.0320187, -16.5422821, 16.5556374
20: -9.9389515, 8.7971592, -9.9296436, 8.7696276, -17.7339783, 17.7470932
21: -12.5691624, 9.1798763, -12.5561781, 9.1539497, -20.2401772, 20.2438583
22: -2.7970715, 18.5489807, -2.7782946, 18.5118446, -18.7443428, 18.7563744
23: -3.9373012, 15.2959127, -3.9291167, 15.2513657, -17.5444374, 17.5611382
24: -5.5263181, 17.4156208, -5.5189562, 17.3577232, -19.9166565, 19.9716606
25: 2.0683489, 24.3230324, 2.0805426, 24.2890759, -19.4354362, 19.4382172
26: -11.7109489, 21.6657028, -11.6936340, 21.5860977, -33.2970467, 33.3593369
27: -15.0026665, 10.0476761, -14.9957714, 9.9971991, -23.7966309, 23.8907852
28: -3.0081100, 18.0649071, -2.9951620, 18.0277519, -19.5772781, 19.6245003
29: -3.3672552, 15.5416079, -3.3489313, 15.5114079, -15.1809959, 15.1972160
30: -13.5947285, 13.8377733, -13.5888176, 13.7915401, -24.7208939, 24.7709503
31: -9.7069340, 11.3463402, -9.6947231, 11.3121824, -21.0191154, 21.0410633
32: -30.7280159, -3.7888026, -30.7098026, -3.7930560, -22.8672104, 22.8321724
33: -41.5592804, -3.2206416, -41.5098648, -3.2326136, -31.0449295, 31.0189667
34: -36.7864113, -3.8106985, -36.7732925, -3.8527517, -25.3900375, 25.4228439
35: -24.9144745, 5.5357985, -24.8994083, 5.5268254, -25.9862366, 26.0274811
36: -24.5751286, 6.4279861, -24.5572395, 6.4219294, -28.1115646, 28.1053543
37: -42.8987808, -6.5631151, -42.8732452, -6.5698032, -32.1156006, 32.1017227
38: -34.3900986, 3.5063038, -34.3741150, 3.4916205, -35.7842178, 35.7443619
39: -47.7845726, -7.0799522, -47.7269821, -7.0924249, -37.1491699, 37.0913391
40: -45.9923210, -18.9627132, -45.9541397, -18.9724922, -21.1738358, 21.1808357
41: -33.4940529, -4.6637921, -33.4845276, -4.6719260, -22.2524185, 22.2641182
42: -24.0657845, -0.4305792, -24.0401592, -0.4389493, -19.7567596, 19.7111282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=162, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 569

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6398792, upper bound: 10.6544994
time: 27.65 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6544995, upper bound: 10.6544994
time: 29.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 59.10 seconds
IS_B1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 59.10
Output dim: 25, lower bound: -10.6289291, upper bound: 10.6177124
IS_B1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 59.10
Output dim: 25, lower bound: -10.6484985, upper bound: 10.6323119
IS_B1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 59.10
Output dim: 25, lower bound: -10.6484985, upper bound: 10.6322568
IS_B1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 59.10
Output dim: 25, lower bound: -10.6484985, upper bound: 10.6468573
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 59.10
Output dim: 25, lower bound: -10.6252827, upper bound: 10.6544994
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 59.10
Output dim: 25, lower bound: -10.6399149, upper bound: 10.6544994
IS_B2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 59.10
Output dim: 25, lower bound: -10.6398537, upper bound: 10.6348966
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 59.10
Output dim: 25, lower bound: -10.6544653, upper bound: 10.6348966
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 59.10
Output dim: 25, lower bound: -10.6398792, upper bound: 10.6544994
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 59.10
Output dim: 25, lower bound: -10.6544995, upper bound: 10.6544994

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -22.5135155, 8.5856647, -22.5213757, 8.6119642, -31.1254807, 31.1070404
1: -11.8741875, 6.3392553, -11.8760834, 6.3562698, -18.2304573, 18.2153397
2: -13.1286640, 7.4095588, -13.1305218, 7.4283447, -19.3966064, 19.4267540
3: -18.6529045, 6.2257695, -18.6557198, 6.2521319, -24.2134323, 24.2146378
4: -20.2732544, 3.2645018, -20.2787628, 3.2766619, -21.1806107, 21.2046204
5: -18.1249962, 7.5287104, -18.1277370, 7.5566578, -25.1064911, 25.1401596
6: -36.7286148, -11.3083782, -36.7340622, -11.2873745, -20.7296371, 20.7220345
7: -24.1145172, 1.3011105, -24.1175194, 1.3281946, -23.9052429, 23.9259262
8: -27.3693962, 1.0689106, -27.3746109, 1.0878239, -25.2075424, 25.2407150
9: -11.5628843, 11.8113365, -11.5675812, 11.8308868, -20.5622253, 20.5108185
10: -17.6670532, 12.3224812, -17.6793995, 12.3346872, -29.6193390, 29.5757370
11: -16.7450829, 10.4677286, -16.7615757, 10.4754400, -23.6917267, 23.6733551
12: -24.1978416, 11.2710381, -24.2042751, 11.2809238, -33.4046326, 33.3420105
13: -22.3629589, 12.2425556, -22.3687801, 12.2908039, -32.5511475, 32.5073395
14: -34.8718414, 6.3678389, -34.8906441, 6.3737912, -36.8138504, 36.7278214
15: -8.6646013, 16.7554893, -8.6759977, 16.7633705, -23.2187805, 23.2586746
16: -22.7697372, 3.0780344, -22.7756577, 3.0945325, -25.8642693, 25.8536911
17: -28.0281563, 8.0519934, -28.0390816, 8.0591984, -36.0873566, 36.0910759
18: -12.4928331, 18.6068230, -12.5373650, 18.6118698, -29.2217102, 29.2260895
19: -8.9061356, 8.0327578, -8.9242859, 8.0303993, -16.4611320, 16.4769630
20: -9.8896074, 8.7670097, -9.9071159, 8.7670937, -17.6610756, 17.6693077
21: -12.5162611, 9.1508026, -12.5339632, 9.1515188, -20.2193947, 20.2228317
22: -2.7218390, 18.5083332, -2.7452078, 18.5100632, -18.6390419, 18.6475048
23: -3.8733311, 15.2477684, -3.8984232, 15.2483845, -17.4007950, 17.3938675
24: -5.4492264, 17.3547401, -5.4818316, 17.3559189, -19.7212372, 19.7398071
25: 2.1432657, 24.2850285, 2.1153059, 24.2870026, -19.3087196, 19.3060112
26: -11.6015186, 21.5759201, -11.6424713, 21.5808792, -33.1823959, 33.2183914
27: -14.9389324, 9.9902611, -14.9656944, 9.9933205, -23.7217865, 23.7911377
28: -2.9344711, 18.0214081, -2.9611130, 18.0242786, -19.4390907, 19.4775276
29: -3.3174686, 15.5075140, -3.3296213, 15.5092793, -15.1167297, 15.1181107
30: -13.5331650, 13.7834721, -13.5595398, 13.7865334, -24.6323471, 24.6484451
31: -9.6389809, 11.3084679, -9.6638517, 11.3092375, -20.9482193, 20.9723206
32: -30.6966400, -3.8410559, -30.7024078, -3.8152719, -22.8759575, 22.8320732
33: -41.4851151, -3.2688470, -41.4965668, -3.2554536, -31.1340408, 31.1436768
34: -36.7408218, -3.8592339, -36.7523918, -3.8563476, -25.2655716, 25.2722168
35: -24.8849297, 5.5214648, -24.8917961, 5.5240588, -25.9469604, 25.9908829
36: -24.5466309, 6.3999104, -24.5501480, 6.4117346, -28.0446625, 28.0387573
37: -42.8472023, -6.5826931, -42.8599930, -6.5759888, -32.2340698, 32.2207336
38: -34.3502197, 3.4615474, -34.3606644, 3.4774156, -35.6311340, 35.5884247
39: -47.7083740, -7.1437836, -47.7164497, -7.1203299, -37.3398666, 37.2959137
40: -45.9351044, -19.0106583, -45.9445267, -18.9941692, -21.3838158, 21.3770027
41: -33.4746704, -4.7007527, -33.4791756, -4.6860147, -22.2332687, 22.2252998
42: -24.0296459, -0.4866786, -24.0341530, -0.4634955, -19.6718636, 19.6320419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1726

## Relational analysis of IS_B2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6146755, upper bound: 10.6538375
time: 27.42 seconds

## Relational analysis of IS_B2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6246236, upper bound: 10.6538416
time: 24.79 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -22.5328064, 8.5947313, -22.5231380, 8.6112766, -31.1440830, 31.1178703
1: -11.8891239, 6.3475709, -11.8766804, 6.3565540, -18.2456779, 18.2242508
2: -13.1412258, 7.4215693, -13.1307631, 7.4294176, -19.4096451, 19.4391518
3: -18.6905994, 6.2474051, -18.6561337, 6.2579260, -24.2530212, 24.2351837
4: -20.2837181, 3.2795217, -20.2802467, 3.2757592, -21.1920929, 21.2210541
5: -18.1585503, 7.5486078, -18.1280937, 7.5612822, -25.1426773, 25.1591339
6: -36.7937851, -11.2741623, -36.7336426, -11.2776546, -20.8061981, 20.7538872
7: -24.1661072, 1.3221598, -24.1177883, 1.3343911, -23.9660797, 23.9455795
8: -27.3781319, 1.0939054, -27.3766212, 1.0896864, -25.2167740, 25.2739868
9: -11.5737991, 11.8158522, -11.5687847, 11.8302650, -20.5716324, 20.5170670
10: -17.7106342, 12.3410311, -17.6794434, 12.3381901, -29.6665421, 29.5941010
11: -16.8263187, 10.4838791, -16.7607479, 10.4809284, -23.7784195, 23.6850357
12: -24.2225151, 11.2911358, -24.2029343, 11.2838287, -33.4355011, 33.3608398
13: -22.4378529, 12.2712870, -22.3688049, 12.2996473, -32.6359406, 32.5366745
14: -34.9047394, 6.4401984, -34.8971939, 6.3738828, -36.8444519, 36.8003159
15: -8.6988964, 16.8247318, -8.6864605, 16.7626686, -23.2496796, 23.3391495
16: -22.8188419, 3.0881553, -22.7767563, 3.0958047, -25.9146461, 25.8649120
17: -28.0747604, 8.1108942, -28.0407524, 8.0595589, -36.1343193, 36.1516457
18: -12.5369873, 18.6758060, -12.5489321, 18.6125755, -29.2651215, 29.3050613
19: -8.9282932, 8.0380497, -8.9259796, 8.0303307, -16.4823341, 16.4822655
20: -9.8990250, 8.7724676, -9.9071503, 8.7667608, -17.6705437, 17.6766319
21: -12.5426788, 9.1644011, -12.5356073, 9.1513281, -20.2449875, 20.2383308
22: -2.7594295, 18.5697422, -2.7541604, 18.5090637, -18.6714439, 18.7174416
23: -3.8925247, 15.2658663, -3.9010277, 15.2489061, -17.4197006, 17.4151726
24: -5.4730234, 17.4050064, -5.4876642, 17.3551064, -19.7433243, 19.7967834
25: 2.1129289, 24.3385506, 2.1077776, 24.2869301, -19.3363152, 19.3677368
26: -11.6373062, 21.6525974, -11.6511135, 21.5812149, -33.2185211, 33.3037109
27: -14.9602184, 10.0429344, -14.9712181, 9.9933929, -23.7423859, 23.8499680
28: -2.9590216, 18.0628242, -2.9666858, 18.0243244, -19.4630432, 19.5251160
29: -3.3533392, 15.5465221, -3.3347359, 15.5088844, -15.1505241, 15.1625710
30: -13.5587311, 13.8306637, -13.5644321, 13.7872887, -24.6575241, 24.7001610
31: -9.6611195, 11.3124275, -9.6664829, 11.3090172, -20.9701366, 20.9789104
32: -30.7784462, -3.8018079, -30.7015915, -3.8044806, -22.9692497, 22.8680649
33: -41.4993782, -3.2573619, -41.4967957, -3.2562075, -31.1510315, 31.1573563
34: -36.7547722, -3.8426361, -36.7527924, -3.8556495, -25.2754059, 25.3049164
35: -24.9031219, 5.5340581, -24.8927250, 5.5243654, -25.9655914, 26.0030136
36: -24.5882111, 6.4229102, -24.5498695, 6.4169164, -28.0917587, 28.0606689
37: -42.8513184, -6.5741158, -42.8526077, -6.5763474, -32.2489243, 32.2320557
38: -34.4167023, 3.5018592, -34.3603592, 3.4865737, -35.7069550, 35.6265411
39: -47.7679520, -7.1140661, -47.7172318, -7.1149287, -37.4080658, 37.3305283
40: -45.9561920, -18.9815331, -45.9460220, -18.9911022, -21.4035797, 21.4073410
41: -33.5385742, -4.6744924, -33.4787331, -4.6784286, -22.3063660, 22.2490692
42: -24.0855217, -0.4586899, -24.0328712, -0.4560118, -19.7368622, 19.6565609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1726

## Relational analysis of IS_B2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6293081, upper bound: 10.6538375
time: 29.37 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6392565, upper bound: 10.6538416
time: 43.27 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.6067619, 8.6233902, -22.4729156, 8.5830841, -31.1898460, 31.0963058
1: -11.9284134, 6.3642554, -11.8443537, 6.3359761, -18.2643890, 18.2086086
2: -13.1743641, 7.4384079, -13.1045303, 7.4132981, -19.4331894, 19.4248695
3: -18.7408638, 6.2653084, -18.6159592, 6.2306948, -24.2845688, 24.2093430
4: -20.3184357, 3.2860432, -20.2472763, 3.2478836, -21.2124481, 21.1847916
5: -18.2129002, 7.5723262, -18.0877323, 7.5352669, -25.1739578, 25.1409378
6: -36.8236389, -11.2519207, -36.7252922, -11.2775822, -20.8161469, 20.7448196
7: -24.2196045, 1.3415809, -24.0734749, 1.3100839, -24.0035324, 23.9151993
8: -27.4209175, 1.1100912, -27.3357964, 1.0618892, -25.2486801, 25.2390213
9: -11.6227016, 11.8312292, -11.5308895, 11.8019962, -20.6241608, 20.4754562
10: -17.7404747, 12.3788109, -17.6724052, 12.3217869, -29.6736221, 29.6252594
11: -16.8522987, 10.5286322, -16.7334671, 10.4518375, -23.7558060, 23.7169991
12: -24.2312164, 11.3202820, -24.1784935, 11.2516108, -33.3993378, 33.3749390
13: -22.5292816, 12.3233433, -22.3295898, 12.2876415, -32.7304535, 32.5365067
14: -34.9294052, 6.4510775, -34.8766785, 6.3640490, -36.8152847, 36.8160782
15: -8.7167015, 16.8260040, -8.6651688, 16.7388763, -23.2518616, 23.3157578
16: -22.8690243, 3.1136854, -22.7409935, 3.0843158, -25.9533405, 25.8546791
17: -28.1035938, 8.1249809, -28.0068626, 8.0387220, -36.1423149, 36.1318436
18: -12.5762653, 18.7508373, -12.5249548, 18.5664368, -29.2481079, 29.3665543
19: -8.9500141, 8.0618916, -8.9000626, 8.0122757, -16.4850922, 16.4813309
20: -9.9253750, 8.7976513, -9.8911295, 8.7511387, -17.6763344, 17.6898384
21: -12.5712967, 9.1892281, -12.5174274, 9.1394768, -20.2457199, 20.2468071
22: -2.7890959, 18.6072502, -2.7387767, 18.4883423, -18.6736069, 18.7455330
23: -3.9167771, 15.3095217, -3.8740973, 15.2209129, -17.4017715, 17.4379463
24: -5.4989605, 17.4629784, -5.4603305, 17.3173065, -19.7188416, 19.8461876
25: 2.0915647, 24.3730774, 2.1371961, 24.2576942, -19.3162270, 19.3780251
26: -11.6824493, 21.7340260, -11.6254768, 21.5369034, -33.2193527, 33.3595047
27: -14.9872379, 10.0938768, -14.9577694, 9.9653339, -23.7362595, 23.8918915
28: -2.9831734, 18.1013641, -2.9372530, 17.9979534, -19.4531174, 19.5442619
29: -3.3731089, 15.5777950, -3.3151531, 15.4891605, -15.1435375, 15.1907120
30: -13.5802650, 13.8772736, -13.5431499, 13.7589788, -24.6400223, 24.7450714
31: -9.6888781, 11.3450041, -9.6372929, 11.2837973, -20.9726753, 20.9822960
32: -30.8014088, -3.7899904, -30.6938515, -3.8102021, -22.9751434, 22.8721199
33: -41.5546494, -3.2206721, -41.4787827, -3.2569523, -31.1595306, 31.1381531
34: -36.7770844, -3.8006516, -36.7378845, -3.8879046, -25.2452545, 25.3288078
35: -24.9208565, 5.5432930, -24.8850574, 5.5087962, -25.9554672, 26.0033646
36: -24.6046505, 6.4317541, -24.5399532, 6.4045997, -28.0951309, 28.0626144
37: -42.8808861, -6.5579472, -42.8413239, -6.5800142, -32.2084503, 32.2360764
38: -34.4313736, 3.5146098, -34.3346596, 3.4549274, -35.7035751, 35.6186142
39: -47.8292618, -7.0738978, -47.6982193, -7.1120768, -37.4208679, 37.3478165
40: -45.9997787, -18.9523735, -45.9301567, -18.9941177, -21.3609848, 21.4292145
41: -33.5496101, -4.6651282, -33.4732704, -4.6853743, -22.2952347, 22.2531433
42: -24.1126442, -0.4366579, -24.0168037, -0.4609530, -19.7655334, 19.6582718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1726

## Relational analysis of IS_B2_A2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6538046, upper bound: 10.6242978
time: 31.26 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2

### Relational analysis result of IS_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6538081, upper bound: 10.6342381
time: 32.44 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -22.5897255, 8.6502781, -22.5238228, 8.6460333, -31.2357597, 31.1741009
1: -11.9145145, 6.3789778, -11.8767643, 6.3762274, -18.2907410, 18.2557411
2: -13.1626987, 7.4460521, -13.1312504, 7.4469595, -19.4793549, 19.4885254
3: -18.7046185, 6.2717285, -18.6571503, 6.2751760, -24.3358536, 24.3040924
4: -20.3100357, 3.2947567, -20.2806988, 3.2915044, -21.2988739, 21.2915115
5: -18.1809196, 7.5815458, -18.1290359, 7.5831633, -25.2069778, 25.2102737
6: -36.7634811, -11.2818565, -36.7384377, -11.2761345, -20.7676315, 20.7409706
7: -24.1698914, 1.3471999, -24.1188622, 1.3511395, -24.0549850, 24.0381088
8: -27.4138527, 1.1114769, -27.3759193, 1.1077023, -25.4113693, 25.4095154
9: -11.6142406, 11.8566637, -11.5697269, 11.8553123, -20.8105087, 20.7055511
10: -17.7047253, 12.3676357, -17.6947098, 12.3416386, -29.6618881, 29.6371002
11: -16.7898598, 10.5144501, -16.7795067, 10.4771824, -23.7937164, 23.8135834
12: -24.2220516, 11.3045158, -24.2122746, 11.2844753, -33.3770065, 33.3375778
13: -22.4569435, 12.3241320, -22.3723507, 12.3329268, -32.7506409, 32.6417694
14: -34.9035835, 6.3823338, -34.8973541, 6.3776522, -36.7848206, 36.7042007
15: -8.6862106, 16.7737465, -8.6803455, 16.7707653, -23.2921829, 23.3189545
16: -22.8240013, 3.1224709, -22.7786865, 3.1170793, -25.9410801, 25.9011574
17: -28.0646362, 8.0701485, -28.0438194, 8.0642834, -36.1289215, 36.1139679
18: -12.5617075, 18.6837902, -12.5705843, 18.6139755, -29.3743210, 29.4298172
19: -8.9495173, 8.0573988, -8.9428949, 8.0311794, -16.5317001, 16.5484619
20: -9.9345589, 8.7931156, -9.9276886, 8.7678070, -17.7273636, 17.7406235
21: -12.5603189, 9.1772718, -12.5522137, 9.1527948, -20.2293625, 20.2366524
22: -2.7687931, 18.5468483, -2.7657304, 18.5108967, -18.7141495, 18.7410126
23: -3.9251289, 15.2928104, -3.9235559, 15.2500219, -17.5296021, 17.5516624
24: -5.5056243, 17.4136066, -5.5096617, 17.3568344, -19.8949890, 19.9601402
25: 2.0933828, 24.3205414, 2.0917850, 24.2879677, -19.4089584, 19.4240150
26: -11.6804676, 21.6596241, -11.6801224, 21.5834408, -33.2639084, 33.3397446
27: -14.9845533, 10.0431118, -14.9877110, 9.9951954, -23.7755890, 23.8775635
28: -2.9873633, 18.0612907, -2.9857826, 18.0261631, -19.5541954, 19.6109543
29: -3.3490491, 15.5401974, -3.3407087, 15.5107861, -15.1616249, 15.1872406
30: -13.5776033, 13.8324013, -13.5811501, 13.7892075, -24.7003174, 24.7571602
31: -9.6947613, 11.3430548, -9.6893082, 11.3107281, -21.0054893, 21.0323639
32: -30.7237701, -3.8240519, -30.7079124, -3.8087416, -22.8470688, 22.7952232
33: -41.5502853, -3.2233315, -41.5058441, -3.2338238, -31.0333481, 31.0109863
34: -36.7806969, -3.8155661, -36.7707901, -3.8548779, -25.3751984, 25.4099541
35: -24.9077644, 5.5331039, -24.8964577, 5.5256362, -25.9764786, 26.0192947
36: -24.5698776, 6.4109058, -24.5549107, 6.4143414, -28.0982895, 28.0850525
37: -42.8847847, -6.5646381, -42.8668404, -6.5704842, -32.0950623, 32.0889740
38: -34.3819885, 3.4769788, -34.3705635, 3.4786067, -35.7632599, 35.7106323
39: -47.7762527, -7.0985708, -47.7232971, -7.1006398, -37.1305237, 37.0645752
40: -45.9835663, -18.9748554, -45.9502563, -18.9778709, -21.1539192, 21.1609573
41: -33.4896202, -4.6880856, -33.4825745, -4.6826510, -22.2368889, 22.2368813
42: -24.0619755, -0.4561293, -24.0384750, -0.4502397, -19.7416992, 19.6835823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1726

## Relational analysis of IS_B2_A2_B2_A1_A1

### Relational analysis result of IS_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6392170, upper bound: 10.6439022
time: 31.63 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2

### Relational analysis result of IS_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6392206, upper bound: 10.6538415
time: 24.84 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.6090393, 8.6592922, -22.5255470, 8.6453533, -31.2543926, 31.1848392
1: -11.9294529, 6.3873081, -11.8773460, 6.3765006, -18.3059540, 18.2646542
2: -13.1752748, 7.4580665, -13.1314793, 7.4480057, -19.4923706, 19.5009613
3: -18.7423077, 6.2933989, -18.6575890, 6.2809334, -24.3754120, 24.3246613
4: -20.3204994, 3.3097441, -20.2821541, 3.2905817, -21.3103256, 21.3079147
5: -18.2144718, 7.6014299, -18.1293640, 7.5878205, -25.2431946, 25.2291870
6: -36.8286819, -11.2476892, -36.7380066, -11.2664127, -20.8441582, 20.7727928
7: -24.2214851, 1.3681967, -24.1191635, 1.3573208, -24.1158066, 24.0578003
8: -27.4226170, 1.1364918, -27.3779545, 1.1095037, -25.4206085, 25.4427414
9: -11.6251488, 11.8611746, -11.5709267, 11.8546734, -20.8199005, 20.7117920
10: -17.7483044, 12.3862448, -17.6947632, 12.3451519, -29.7090454, 29.6554947
11: -16.8710594, 10.5306301, -16.7786961, 10.4826822, -23.8803558, 23.8253365
12: -24.2466812, 11.3246002, -24.2109451, 11.2873793, -33.4079437, 33.3564682
13: -22.5318584, 12.3528748, -22.3723640, 12.3417549, -32.8356247, 32.6711884
14: -34.9365845, 6.4547620, -34.9039688, 6.3777409, -36.8154907, 36.7766800
15: -8.7204962, 16.8429489, -8.6907587, 16.7700462, -23.3231125, 23.3993912
16: -22.8731003, 3.1326110, -22.7797966, 3.1183991, -25.9914989, 25.9124069
17: -28.1113167, 8.1290741, -28.0454197, 8.0646086, -36.1759262, 36.1744919
18: -12.6058731, 18.7527943, -12.5821056, 18.6146927, -29.4176941, 29.5087814
19: -8.9716606, 8.0627041, -8.9445877, 8.0311298, -16.5528755, 16.5537643
20: -9.9439697, 8.7985668, -9.9277039, 8.7674656, -17.7368011, 17.7479553
21: -12.5866976, 9.1908998, -12.5538588, 9.1526203, -20.2549171, 20.2521248
22: -2.8063731, 18.6082554, -2.7746806, 18.5098915, -18.7465324, 18.8109703
23: -3.9443226, 15.3108892, -3.9261727, 15.2505302, -17.5484962, 17.5729713
24: -5.5294037, 17.4639015, -5.5155048, 17.3560524, -19.9170914, 20.0171280
25: 2.0630503, 24.3740635, 2.0842562, 24.2878952, -19.4365234, 19.4857254
26: -11.7162418, 21.7362900, -11.6887665, 21.5837708, -33.3000107, 33.4250565
27: -15.0058441, 10.0957527, -14.9932327, 9.9952822, -23.7961731, 23.9364128
28: -3.0118999, 18.1026821, -2.9913325, 18.0261936, -19.5781288, 19.6585121
29: -3.3848753, 15.5792046, -3.3458319, 15.5103893, -15.1953049, 15.2317104
30: -13.6031704, 13.8795567, -13.5860329, 13.7899246, -24.7255402, 24.8088799
31: -9.7168579, 11.3470058, -9.6919498, 11.3105316, -21.0273895, 21.0389557
32: -30.8055840, -3.7848668, -30.7071037, -3.7979279, -22.9403381, 22.8312149
33: -41.5645638, -3.2118368, -41.5060043, -3.2345653, -31.0502853, 31.0245361
34: -36.7946701, -3.7989202, -36.7711716, -3.8541679, -25.3850632, 25.4426994
35: -24.9259796, 5.5456867, -24.8973389, 5.5259171, -25.9951401, 26.0314407
36: -24.6115170, 6.4338942, -24.5546417, 6.4195557, -28.1454544, 28.1069946
37: -42.8889275, -6.5560265, -42.8594742, -6.5708456, -32.1099396, 32.1003113
38: -34.4484291, 3.5172544, -34.3702545, 3.4877810, -35.8391037, 35.7487564
39: -47.8358459, -7.0689154, -47.7240906, -7.0952792, -37.1987915, 37.0991821
40: -46.0046387, -18.9457321, -45.9517517, -18.9748116, -21.1737137, 21.1912155
41: -33.5535812, -4.6618247, -33.4821167, -4.6750450, -22.3099442, 22.2606430
42: -24.1178627, -0.4281187, -24.0371971, -0.4427829, -19.8066826, 19.7081032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1726

## Relational analysis of IS_B2_A2_B2_A2_A1

### Relational analysis result of IS_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6538376, upper bound: 10.6439022
time: 26.77 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2

### Relational analysis result of IS_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6538417, upper bound: 10.6538415
time: 30.45 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 59.43 seconds
IS_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 59.43
Output dim: 25, lower bound: -10.6146755, upper bound: 10.6538375
IS_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 59.43
Output dim: 25, lower bound: -10.6246236, upper bound: 10.6538416
IS_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 59.43
Output dim: 25, lower bound: -10.6293081, upper bound: 10.6538375
IS_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 59.43
Output dim: 25, lower bound: -10.6392565, upper bound: 10.6538416
IS_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 59.43
Output dim: 25, lower bound: -10.6538046, upper bound: 10.6242978
IS_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 59.43
Output dim: 25, lower bound: -10.6538081, upper bound: 10.6342381
IS_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 59.43
Output dim: 25, lower bound: -10.6392170, upper bound: 10.6439022
IS_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 59.43
Output dim: 25, lower bound: -10.6392206, upper bound: 10.6538415
IS_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 59.43
Output dim: 25, lower bound: -10.6538376, upper bound: 10.6439022
IS_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 59.43
Output dim: 25, lower bound: -10.6538417, upper bound: 10.6538415

## BFS IS instance: IS_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -22.4903088, 8.5836697, -22.4787598, 8.5735588, -31.0638676, 31.0624294
1: -11.8635502, 6.3379002, -11.8578453, 6.3324275, -18.1959782, 18.1957455
2: -13.1152916, 7.4080782, -13.1078911, 7.4011049, -19.3573151, 19.4035263
3: -18.6357841, 6.2236886, -18.6251411, 6.2226105, -24.1654205, 24.1817017
4: -20.2509155, 3.2628026, -20.2381001, 3.2438755, -21.1262665, 21.1630173
5: -18.1056328, 7.5256429, -18.0932331, 7.5229521, -25.0521240, 25.1028137
6: -36.7265244, -11.3131027, -36.7268219, -11.3002329, -20.7159348, 20.7091827
7: -24.0956917, 1.2987781, -24.0849648, 1.2945271, -23.8522720, 23.8904114
8: -27.3524933, 1.0670195, -27.3442535, 1.0581408, -25.1606445, 25.2086411
9: -11.5497417, 11.8100567, -11.5422659, 11.8070803, -20.5218811, 20.4828491
10: -17.6629028, 12.3193884, -17.6649437, 12.3200788, -29.5998840, 29.5576172
11: -16.7425995, 10.4535017, -16.7362423, 10.4489908, -23.6631012, 23.6325226
12: -24.1954174, 11.2580175, -24.1923561, 11.2529125, -33.3737869, 33.3171234
13: -22.3240013, 12.2385693, -22.2990723, 12.2219429, -32.4428329, 32.4329224
14: -34.8654404, 6.3614001, -34.8707275, 6.3567953, -36.7895432, 36.7010193
15: -8.6594009, 16.7532959, -8.6603289, 16.7556648, -23.2061462, 23.2415237
16: -22.7595901, 3.0761976, -22.7553291, 3.0720756, -25.8316650, 25.8315277
17: -28.0080376, 8.0496864, -27.9997654, 8.0346928, -36.0427322, 36.0494537
18: -12.4885693, 18.5757046, -12.4798145, 18.5570335, -29.1630402, 29.1376724
19: -8.9022388, 8.0259686, -8.8990221, 8.0178528, -16.4443283, 16.4417801
20: -9.8862247, 8.7516565, -9.8751526, 8.7396603, -17.6306000, 17.6222076
21: -12.5126181, 9.1348600, -12.5002403, 9.1224422, -20.1879120, 20.1735802
22: -2.7185507, 18.4974937, -2.7169127, 18.4909935, -18.6171799, 18.6083870
23: -3.8703413, 15.2269192, -3.8583927, 15.2104816, -17.3601074, 17.3331528
24: -5.4451976, 17.3336086, -5.4441757, 17.3191013, -19.6805115, 19.6811752
25: 2.1467819, 24.2686615, 2.1468163, 24.2577744, -19.2771835, 19.2581940
26: -11.5960369, 21.5405159, -11.5800438, 21.5196114, -33.1156464, 33.1205597
27: -14.9352150, 9.9606800, -14.9136934, 9.9411335, -23.6652374, 23.7090149
28: -2.9304504, 18.0029640, -2.9194980, 17.9903297, -19.4009552, 19.4175072
29: -3.3147416, 15.4966640, -3.3056507, 15.4893951, -15.0946560, 15.0828514
30: -13.5313015, 13.7666702, -13.5316525, 13.7533092, -24.5972977, 24.6040115
31: -9.6346445, 11.2976379, -9.6293163, 11.2887135, -20.9233589, 20.9269543
32: -30.6912441, -3.8437529, -30.6883926, -3.8267903, -22.8598366, 22.8101425
33: -41.4688683, -3.2714295, -41.4607697, -3.2807603, -31.1039658, 31.1102448
34: -36.7370377, -3.8659754, -36.7333374, -3.8684855, -25.2462921, 25.2359467
35: -24.8759422, 5.5199881, -24.8701782, 5.5170541, -25.9301300, 25.9675598
36: -24.5432854, 6.3988161, -24.5366707, 6.4077692, -28.0356064, 28.0224838
37: -42.8385925, -6.5850000, -42.8381920, -6.5858593, -32.2179413, 32.1975250
38: -34.3445244, 3.4535408, -34.3356209, 3.4625249, -35.6113892, 35.5614548
39: -47.6801109, -7.1454892, -47.6625519, -7.1547298, -37.2922363, 37.2465820
40: -45.9273987, -19.0120258, -45.9240952, -19.0064945, -21.3713112, 21.3444595
41: -33.4720459, -4.7049637, -33.4697762, -4.6977000, -22.2177124, 22.2081909
42: -24.0286026, -0.4947259, -24.0221424, -0.4824247, -19.6523819, 19.6104317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=160, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6116048, upper bound: 10.6324997
time: 25.31 seconds

## Relational analysis of IS_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6116048, upper bound: 10.6508600
time: 31.14 seconds

## BFS IS instance: IS_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -22.5119553, 8.5854912, -22.5191021, 8.6117029, -31.1236572, 31.1045933
1: -11.8734617, 6.3391228, -11.8749638, 6.3560524, -18.2295151, 18.2140865
2: -13.1280766, 7.4094357, -13.1295424, 7.4281335, -19.3956833, 19.4133224
3: -18.6516266, 6.2255325, -18.6536083, 6.2518005, -24.2123566, 24.2049484
4: -20.2718315, 3.2643416, -20.2767029, 3.2763717, -21.1791916, 21.1722794
5: -18.1235828, 7.5284729, -18.1253967, 7.5562439, -25.1054993, 25.1369858
6: -36.7284470, -11.3100739, -36.7337570, -11.2902565, -20.7261543, 20.7199707
7: -24.1135406, 1.3009379, -24.1158905, 1.3278315, -23.9034042, 23.9160461
8: -27.3679504, 1.0687084, -27.3722038, 1.0874381, -25.2060852, 25.2250061
9: -11.5614834, 11.8112116, -11.5653419, 11.8307009, -20.5611343, 20.4934349
10: -17.6657906, 12.3222799, -17.6773777, 12.3342810, -29.6174393, 29.5730362
11: -16.7445335, 10.4666681, -16.7606201, 10.4736834, -23.6671753, 23.6717644
12: -24.1975651, 11.2698193, -24.2038174, 11.2788668, -33.3997192, 33.3400345
13: -22.3600960, 12.2423019, -22.3639069, 12.2902985, -32.5477448, 32.4800797
14: -34.8712807, 6.3655453, -34.8898506, 6.3699708, -36.8107681, 36.7259216
15: -8.6632290, 16.7552910, -8.6737576, 16.7630138, -23.2170563, 23.2560272
16: -22.7685909, 3.0778389, -22.7737885, 3.0942144, -25.8628044, 25.8516273
17: -28.0261478, 8.0518074, -28.0357666, 8.0588989, -36.0850449, 36.0875740
18: -12.4924698, 18.6045418, -12.5367575, 18.6080647, -29.2030716, 29.2232285
19: -8.9057140, 8.0321665, -8.9235954, 8.0294151, -16.4588242, 16.4772568
20: -9.8893414, 8.7658787, -9.9066715, 8.7651920, -17.6502457, 17.6677475
21: -12.5158186, 9.1496439, -12.5331974, 9.1495667, -20.2073021, 20.2210197
22: -2.7214522, 18.5075150, -2.7445483, 18.5087261, -18.6369019, 18.6461334
23: -3.8730674, 15.2462721, -3.8979425, 15.2458811, -17.3753052, 17.3918953
24: -5.4489231, 17.3531799, -5.4813223, 17.3533249, -19.7055130, 19.7377777
25: 2.1435537, 24.2838345, 2.1158109, 24.2850266, -19.2901726, 19.3048668
26: -11.6010313, 21.5733795, -11.6416225, 21.5766106, -33.1776428, 33.2150040
27: -14.9386435, 9.9881210, -14.9652328, 9.9897413, -23.7038116, 23.7885513
28: -2.9341755, 18.0200844, -2.9605923, 18.0220509, -19.4283371, 19.4757042
29: -3.3169661, 15.5067387, -3.3287816, 15.5079670, -15.0976601, 15.1168518
30: -13.5329504, 13.7822866, -13.5591717, 13.7844944, -24.6157761, 24.6469231
31: -9.6385918, 11.3075256, -9.6631966, 11.3076277, -20.9462204, 20.9707222
32: -30.6958961, -3.8412676, -30.7011395, -3.8156505, -22.8706512, 22.8364601
33: -41.4839401, -3.2691879, -41.4946289, -3.2560320, -31.1260605, 31.1287384
34: -36.7405243, -3.8606806, -36.7519302, -3.8587594, -25.2613831, 25.2742386
35: -24.8830986, 5.5212216, -24.8887005, 5.5236535, -25.9406662, 25.9844131
36: -24.5463238, 6.3997421, -24.5496311, 6.4113965, -28.0423355, 28.0362320
37: -42.8457947, -6.5829306, -42.8576355, -6.5764122, -32.2314072, 32.2147293
38: -34.3497849, 3.4594688, -34.3598938, 3.4740696, -35.6332016, 35.5805664
39: -47.7062721, -7.1440840, -47.7130127, -7.1208820, -37.3347626, 37.2576752
40: -45.9344749, -19.0109940, -45.9434738, -18.9947491, -21.3701172, 21.3887939
41: -33.4744415, -4.7021160, -33.4787827, -4.6882434, -22.2313690, 22.2247353
42: -24.0295181, -0.4874246, -24.0339222, -0.4647517, -19.6571121, 19.6301384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=160, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6215255, upper bound: 10.6324997
time: 29.41 seconds

## Relational analysis of IS_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6215255, upper bound: 10.6508600
time: 31.04 seconds

## BFS IS instance: IS_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -22.5095711, 8.5926809, -22.4805107, 8.5728607, -31.0824318, 31.0731926
1: -11.8784714, 6.3462362, -11.8584299, 6.3327274, -18.2111988, 18.2046661
2: -13.1278496, 7.4201226, -13.1081257, 7.4021788, -19.3703537, 19.4159470
3: -18.6735001, 6.2453308, -18.6255779, 6.2283654, -24.2049942, 24.2022400
4: -20.2613564, 3.2778237, -20.2395725, 3.2429755, -21.1377182, 21.1793976
5: -18.1392117, 7.5455322, -18.0935898, 7.5275707, -25.0882645, 25.1217270
6: -36.7917023, -11.2789631, -36.7263908, -11.2905064, -20.7924805, 20.7410431
7: -24.1473160, 1.3197708, -24.0852489, 1.3007460, -23.9131317, 23.9100723
8: -27.3612423, 1.0920277, -27.3462868, 1.0599833, -25.1698456, 25.2418976
9: -11.5606384, 11.8145885, -11.5434647, 11.8064804, -20.5313034, 20.4890900
10: -17.7065315, 12.3379192, -17.6649570, 12.3235798, -29.6470337, 29.5759811
11: -16.8238087, 10.4696789, -16.7354469, 10.4544458, -23.7497864, 23.6442146
12: -24.2200909, 11.2780676, -24.1910019, 11.2558174, -33.4047012, 33.3359680
13: -22.3989010, 12.2672930, -22.2990456, 12.2308149, -32.5276337, 32.4622803
14: -34.8983994, 6.4338374, -34.8772774, 6.3569593, -36.8202057, 36.7734528
15: -8.6936989, 16.8225288, -8.6707783, 16.7549744, -23.2370148, 23.3219528
16: -22.8086777, 3.0862913, -22.7564507, 3.0733511, -25.8820286, 25.8427429
17: -28.0547104, 8.1086140, -28.0014114, 8.0350361, -36.0897446, 36.1100235
18: -12.5327415, 18.6447563, -12.4913540, 18.5577278, -29.2064362, 29.2166367
19: -8.9243870, 8.0312595, -8.9007044, 8.0177879, -16.4655304, 16.4470673
20: -9.8956299, 8.7571182, -9.8751812, 8.7393236, -17.6400452, 17.6295395
21: -12.5390491, 9.1484709, -12.5018835, 9.1222591, -20.2135124, 20.1890564
22: -2.7561693, 18.5589046, -2.7258511, 18.4899826, -18.6495705, 18.6783257
23: -3.8895311, 15.2449856, -3.8609776, 15.2110186, -17.3790169, 17.3544769
24: -5.4689898, 17.3839188, -5.4500055, 17.3183289, -19.7026367, 19.7381363
25: 2.1164203, 24.3221664, 2.1392760, 24.2576942, -19.3047638, 19.3198929
26: -11.6318579, 21.6171646, -11.5887070, 21.5199623, -33.1518211, 33.2058716
27: -14.9565277, 10.0133801, -14.9192181, 9.9412155, -23.6858368, 23.7678528
28: -2.9550004, 18.0443859, -2.9250727, 17.9903717, -19.4248924, 19.4650688
29: -3.3505788, 15.5356493, -3.3107667, 15.4890156, -15.1284332, 15.1273079
30: -13.5568447, 13.8138199, -13.5365734, 13.7540674, -24.6224823, 24.6557350
31: -9.6567726, 11.3015890, -9.6319427, 11.2885275, -20.9453011, 20.9335327
32: -30.7730484, -3.8044972, -30.6876183, -3.8159447, -22.9530754, 22.8461418
33: -41.4830933, -3.2599325, -41.4610100, -3.2814589, -31.1209183, 31.1239395
34: -36.7509537, -3.8493953, -36.7337456, -3.8678389, -25.2561569, 25.2686043
35: -24.8940659, 5.5325518, -24.8711205, 5.5172939, -25.9487305, 25.9796982
36: -24.5848732, 6.4218287, -24.5363960, 6.4129262, -28.0827026, 28.0443497
37: -42.8426666, -6.5764408, -42.8308296, -6.5862389, -32.2327881, 32.2088470
38: -34.4109726, 3.4937868, -34.3353081, 3.4716382, -35.6872559, 35.5994949
39: -47.7397385, -7.1157446, -47.6633301, -7.1494098, -37.3605194, 37.2812119
40: -45.9484901, -18.9828682, -45.9255867, -19.0034504, -21.3910446, 21.3747787
41: -33.5359497, -4.6787291, -33.4692955, -4.6900764, -22.2907715, 22.2319679
42: -24.0844688, -0.4667130, -24.0208530, -0.4749558, -19.7173576, 19.6349716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=160, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6262220, upper bound: 10.6324997
time: 26.43 seconds

## Relational analysis of IS_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6262220, upper bound: 10.6508600
time: 29.12 seconds

## BFS IS instance: IS_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.5312271, 8.5945740, -22.5208874, 8.6110249, -31.1422520, 31.1154613
1: -11.8883829, 6.3474460, -11.8755312, 6.3563800, -18.2447624, 18.2229767
2: -13.1406536, 7.4214530, -13.1297779, 7.4292016, -19.4086838, 19.4257393
3: -18.6893215, 6.2471919, -18.6540260, 6.2575502, -24.2519302, 24.2254715
4: -20.2822971, 3.2793438, -20.2781487, 3.2754788, -21.1906509, 21.1886826
5: -18.1571274, 7.5483785, -18.1257248, 7.5608788, -25.1417236, 25.1559372
6: -36.7936172, -11.2758942, -36.7333527, -11.2804956, -20.8027115, 20.7518463
7: -24.1651154, 1.3219728, -24.1161995, 1.3340559, -23.9642944, 23.9357605
8: -27.3767281, 1.0937428, -27.3742371, 1.0892792, -25.2153091, 25.2582932
9: -11.5723734, 11.8157434, -11.5665398, 11.8300743, -20.5705490, 20.4996872
10: -17.7094097, 12.3407917, -17.6774044, 12.3377953, -29.6645966, 29.5914154
11: -16.8257408, 10.4828348, -16.7598419, 10.4791422, -23.7538757, 23.6834679
12: -24.2222061, 11.2898722, -24.2024632, 11.2817745, -33.4305954, 33.3588562
13: -22.4349785, 12.2709999, -22.3639698, 12.2991772, -32.6325378, 32.5093918
14: -34.9042931, 6.4378948, -34.8963852, 6.3700285, -36.8412933, 36.7983551
15: -8.6975183, 16.8245010, -8.6841927, 16.7623138, -23.2479248, 23.3364716
16: -22.8176956, 3.0879416, -22.7749176, 3.0955389, -25.9132347, 25.8628597
17: -28.0727997, 8.1107178, -28.0374165, 8.0592594, -36.1320572, 36.1481323
18: -12.5366344, 18.6735382, -12.5482912, 18.6087532, -29.2464752, 29.3022079
19: -8.9278736, 8.0374699, -8.9252958, 8.0293503, -16.4800301, 16.4825439
20: -9.8987474, 8.7713346, -9.9067135, 8.7648754, -17.6596756, 17.6750870
21: -12.5422544, 9.1632414, -12.5348740, 9.1493855, -20.2328911, 20.2365074
22: -2.7590432, 18.5689278, -2.7534995, 18.5077286, -18.6693001, 18.7160969
23: -3.8922343, 15.2643547, -3.9005404, 15.2464075, -17.3942108, 17.4132004
24: -5.4727130, 17.4034901, -5.4871597, 17.3525352, -19.7276306, 19.7947273
25: 2.1132126, 24.3373489, 2.1082482, 24.2849255, -19.3177338, 19.3665886
26: -11.6367941, 21.6500359, -11.6502934, 21.5769196, -33.2137146, 33.3003311
27: -14.9599419, 10.0407782, -14.9707680, 9.9898214, -23.7244263, 23.8473740
28: -2.9587069, 18.0614834, -2.9661727, 18.0220966, -19.4522781, 19.5232887
29: -3.3528366, 15.5457249, -3.3338914, 15.5075722, -15.1314430, 15.1613235
30: -13.5585461, 13.8294497, -13.5640697, 13.7852497, -24.6409760, 24.6986465
31: -9.6607180, 11.3114538, -9.6658344, 11.3074169, -20.9681358, 20.9772873
32: -30.7776947, -3.8020434, -30.7003613, -3.8048491, -22.9638977, 22.8724899
33: -41.4982491, -3.2576466, -41.4948273, -3.2567701, -31.1430283, 31.1424026
34: -36.7544975, -3.8440871, -36.7523193, -3.8580899, -25.2712173, 25.3069725
35: -24.9012299, 5.5337753, -24.8896027, 5.5239067, -25.9592819, 25.9965439
36: -24.5879440, 6.4227552, -24.5494080, 6.4165735, -28.0894394, 28.0581436
37: -42.8498840, -6.5743713, -42.8502808, -6.5767555, -32.2462769, 32.2259750
38: -34.4162750, 3.4997983, -34.3595772, 3.4832282, -35.7090607, 35.6186371
39: -47.7658997, -7.1143870, -47.7137680, -7.1155128, -37.4031067, 37.2923203
40: -45.9555817, -18.9818726, -45.9449959, -18.9916801, -21.3898544, 21.4191322
41: -33.5383720, -4.6758628, -33.4783249, -4.6806622, -22.3044281, 22.2485046
42: -24.0853691, -0.4594309, -24.0326366, -0.4572833, -19.7221107, 19.6546631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=160, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6361455, upper bound: 10.6324997
time: 58.54 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6361455, upper bound: 10.6508600
time: 31.87 seconds

## BFS IS instance: IS_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -22.5641499, 8.5850277, -22.4496765, 8.5810690, -31.1452179, 31.0347042
1: -11.9101524, 6.3404274, -11.8336906, 6.3346353, -18.2447872, 18.1741180
2: -13.1516962, 7.4111700, -13.0911770, 7.4118176, -19.4099579, 19.3856392
3: -18.7102699, 6.2358041, -18.5988274, 6.2286205, -24.2516251, 24.1612320
4: -20.2777481, 3.2533002, -20.2249184, 3.2461467, -21.1708069, 21.1304245
5: -18.1784172, 7.5386462, -18.0683956, 7.5321512, -25.1365662, 25.0865326
6: -36.8163910, -11.2648039, -36.7231750, -11.2823448, -20.8033295, 20.7310410
7: -24.1870594, 1.3079557, -24.0546532, 1.3077211, -23.9680099, 23.8622665
8: -27.3905849, 1.0804029, -27.3189220, 1.0600176, -25.2166061, 25.1921387
9: -11.5974007, 11.8074284, -11.5177488, 11.8007421, -20.5961838, 20.4350891
10: -17.7259369, 12.3641863, -17.6683102, 12.3187103, -29.6554718, 29.6057281
11: -16.8269768, 10.5021667, -16.7309837, 10.4376011, -23.7150269, 23.6883545
12: -24.2192993, 11.2921572, -24.1761131, 11.2385597, -33.3744965, 33.3440857
13: -22.4595413, 12.2545595, -22.2906170, 12.2836514, -32.6560211, 32.4282227
14: -34.9094810, 6.4340348, -34.8703232, 6.3577118, -36.7883987, 36.7917023
15: -8.7010002, 16.8183022, -8.6599770, 16.7366867, -23.2346497, 23.3031158
16: -22.8486938, 3.0911932, -22.7308464, 3.0824897, -25.9311829, 25.8220406
17: -28.0642185, 8.1004333, -27.9867744, 8.0364552, -36.1006737, 36.0872078
18: -12.5187321, 18.6960564, -12.5206833, 18.5353432, -29.1597366, 29.3078079
19: -8.9247446, 8.0493507, -8.8961697, 8.0054684, -16.4499283, 16.4645309
20: -9.8934383, 8.7702408, -9.8877497, 8.7357807, -17.6292534, 17.6593552
21: -12.5375767, 9.1601572, -12.5137787, 9.1235313, -20.1964569, 20.2153244
22: -2.7608261, 18.5881767, -2.7354889, 18.4775238, -18.6344986, 18.7236595
23: -3.8767862, 15.2716541, -3.8711019, 15.2000456, -17.3410835, 17.3972702
24: -5.4613457, 17.4261971, -5.4563131, 17.2961445, -19.6602325, 19.8055038
25: 2.1230364, 24.3438568, 2.1407127, 24.2413330, -19.2684288, 19.3464584
26: -11.6200924, 21.6727791, -11.6200247, 21.5014381, -33.1215286, 33.2928047
27: -14.9352570, 10.0417147, -14.9540520, 9.9357767, -23.6541519, 23.8353577
28: -2.9415922, 18.0674248, -2.9332232, 17.9795322, -19.3930893, 19.5060806
29: -3.3491664, 15.5578899, -3.3123808, 15.4783144, -15.1083107, 15.1686211
30: -13.5524349, 13.8440638, -13.5412874, 13.7421360, -24.5955811, 24.7100410
31: -9.6543713, 11.3244905, -9.6329365, 11.2729416, -20.9273129, 20.9574280
32: -30.7874126, -3.8014326, -30.6884613, -3.8128681, -22.9532127, 22.8559685
33: -41.5188141, -3.2459345, -41.4625244, -3.2595248, -31.1261215, 31.1080017
34: -36.7580528, -3.8127842, -36.7340622, -3.8946290, -25.2089691, 25.3095436
35: -24.8991585, 5.5362625, -24.8760281, 5.5072980, -25.9321289, 25.9864883
36: -24.5911694, 6.4277534, -24.5366096, 6.4034925, -28.0788345, 28.0535431
37: -42.8590279, -6.5677729, -42.8326759, -6.5823212, -32.1851730, 32.2200165
38: -34.4063263, 3.4996743, -34.3288536, 3.4468994, -35.6764603, 35.5988770
39: -47.7753296, -7.1083226, -47.6700020, -7.1137586, -37.3713837, 37.3002396
40: -45.9793015, -18.9646587, -45.9224854, -18.9954567, -21.3284225, 21.4167175
41: -33.5401382, -4.6768651, -33.4706650, -4.6895704, -22.2780800, 22.2375946
42: -24.1006203, -0.4556305, -24.0157738, -0.4690115, -19.7439232, 19.6387634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 637

## Relational analysis of IS_B2_A2_B1_A2_A1_B1

### Relational analysis result of IS_B2_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6324998, upper bound: 10.6212961
time: 27.01 seconds

## Relational analysis of IS_B2_A2_B1_A2_A1_B2

### Relational analysis result of IS_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6508601, upper bound: 10.6212961
time: 29.82 seconds

## BFS IS instance: IS_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -22.6044712, 8.6231337, -22.4713287, 8.5828819, -31.1873531, 31.0944633
1: -11.9272718, 6.3640652, -11.8436069, 6.3358617, -18.2631340, 18.2076721
2: -13.1733656, 7.4382010, -13.1039543, 7.4131703, -19.4197769, 19.4239426
3: -18.7387466, 6.2649717, -18.6146812, 6.2305083, -24.2748489, 24.2082291
4: -20.3163681, 3.2857704, -20.2458344, 3.2477074, -21.1801071, 21.1833191
5: -18.2105789, 7.5718994, -18.0863533, 7.5350165, -25.1707382, 25.1399536
6: -36.8233299, -11.2547827, -36.7251282, -11.2793045, -20.8140984, 20.7412987
7: -24.2180119, 1.3412514, -24.0724564, 1.3099077, -23.9936523, 23.9134293
8: -27.4185333, 1.1097198, -27.3343868, 1.0616779, -25.2329941, 25.2375565
9: -11.6204815, 11.8310385, -11.5294876, 11.8018799, -20.6068039, 20.4743958
10: -17.7384205, 12.3783875, -17.6711922, 12.3215389, -29.6709137, 29.6232986
11: -16.8513374, 10.5268955, -16.7329178, 10.4507771, -23.7542267, 23.6925011
12: -24.2307453, 11.3181973, -24.1782150, 11.2503891, -33.3972931, 33.3700714
13: -22.5244064, 12.3228664, -22.3267097, 12.2873688, -32.7032013, 32.5330734
14: -34.9285736, 6.4472041, -34.8762207, 6.3617454, -36.8132782, 36.8129044
15: -8.7144356, 16.8256721, -8.6638126, 16.7386799, -23.2491760, 23.3139992
16: -22.8671684, 3.1133988, -22.7398605, 3.0841372, -25.9513054, 25.8532600
17: -28.1002216, 8.1246605, -28.0048828, 8.0385342, -36.1387558, 36.1295433
18: -12.5756550, 18.7470551, -12.5245905, 18.5641232, -29.2452698, 29.3479004
19: -8.9493170, 8.0609093, -8.8996506, 8.0116768, -16.4853783, 16.4790192
20: -9.9249439, 8.7957649, -9.8908691, 8.7499971, -17.6747818, 17.6789894
21: -12.5705471, 9.1872826, -12.5169954, 9.1383018, -20.2438660, 20.2347260
22: -2.7884269, 18.6059113, -2.7383842, 18.4875336, -18.6722603, 18.7433891
23: -3.9163008, 15.3070316, -3.8738170, 15.2194233, -17.3998184, 17.4124756
24: -5.4984436, 17.4604225, -5.4600415, 17.3157482, -19.7168045, 19.8304825
25: 2.0920191, 24.3710842, 2.1374831, 24.2564774, -19.3150864, 19.3594513
26: -11.6816273, 21.7297668, -11.6250134, 21.5343285, -33.2159576, 33.3547821
27: -14.9867697, 10.0902863, -14.9574738, 9.9631615, -23.7336655, 23.8739548
28: -2.9826884, 18.0991478, -2.9369497, 17.9966183, -19.4513054, 19.5335007
29: -3.3722610, 15.5764780, -3.3146439, 15.4883823, -15.1422806, 15.1716652
30: -13.5799274, 13.8752604, -13.5429497, 13.7577705, -24.6384354, 24.7285423
31: -9.6882381, 11.3434143, -9.6368990, 11.2828436, -20.9710808, 20.9803123
32: -30.8001347, -3.7903256, -30.6930866, -3.8103986, -22.9795456, 22.8668060
33: -41.5527115, -3.2212348, -41.4776001, -3.2572956, -31.1444931, 31.1300964
34: -36.7766113, -3.8030872, -36.7375793, -3.8893733, -25.2472763, 25.3245888
35: -24.9177742, 5.5428405, -24.8831272, 5.5085659, -25.9489670, 25.9970245
36: -24.6041718, 6.4314189, -24.5396976, 6.4043908, -28.0926132, 28.0602493
37: -42.8785362, -6.5583572, -42.8399048, -6.5802364, -32.2024078, 32.2333832
38: -34.4306335, 3.5112467, -34.3342209, 3.4529042, -35.6956329, 35.6206741
39: -47.8258286, -7.0744333, -47.6961365, -7.1124263, -37.3825607, 37.3426971
40: -45.9987259, -18.9528751, -45.9295197, -18.9944496, -21.3727493, 21.4154701
41: -33.5491791, -4.6674109, -33.4730682, -4.6867185, -22.2946548, 22.2512436
42: -24.1124153, -0.4379580, -24.0166550, -0.4617205, -19.7636070, 19.6434898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 637

## Relational analysis of IS_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6324998, upper bound: 10.6312169
time: 29.85 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6508601, upper bound: 10.6312169
time: 28.60 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -22.5874043, 8.6500034, -22.5222340, 8.6458836, -31.2332878, 31.1722374
1: -11.9133606, 6.3787727, -11.8760252, 6.3760872, -18.2894478, 18.2547989
2: -13.1616974, 7.4458532, -13.1306639, 7.4468155, -19.4659348, 19.4876251
3: -18.7025070, 6.2713909, -18.6558800, 6.2749710, -24.3260880, 24.3030396
4: -20.3079262, 3.2944610, -20.2792854, 3.2913280, -21.2665405, 21.2900772
5: -18.1785774, 7.5811872, -18.1276264, 7.5829210, -25.2038269, 25.2092819
6: -36.7631531, -11.2847328, -36.7382812, -11.2778854, -20.7655640, 20.7374916
7: -24.1682873, 1.3468871, -24.1178684, 1.3509326, -24.0450821, 24.0363235
8: -27.4114876, 1.1110764, -27.3744583, 1.1074562, -25.3956604, 25.4080353
9: -11.6120090, 11.8564634, -11.5683174, 11.8551769, -20.7931061, 20.7044907
10: -17.7026901, 12.3672400, -17.6935310, 12.3414173, -29.6592255, 29.6351471
11: -16.7889137, 10.5126972, -16.7789307, 10.4761305, -23.7921371, 23.7891159
12: -24.2215672, 11.3025055, -24.2120018, 11.2832584, -33.3750000, 33.3327484
13: -22.4520893, 12.3236828, -22.3694229, 12.3326492, -32.7234192, 32.6383667
14: -34.9028282, 6.3784623, -34.8968582, 6.3753161, -36.7828827, 36.7010422
15: -8.6839342, 16.7733898, -8.6789827, 16.7705498, -23.2895126, 23.3171997
16: -22.8221512, 3.1221621, -22.7775402, 3.1168816, -25.9390335, 25.8997021
17: -28.0613232, 8.0697813, -28.0418205, 8.0640688, -36.1253929, 36.1116028
18: -12.5610790, 18.6799660, -12.5702190, 18.6117020, -29.3714676, 29.4111481
19: -8.9488420, 8.0564289, -8.9424725, 8.0305929, -16.5319901, 16.5461655
20: -9.9341192, 8.7912216, -9.9274101, 8.7666740, -17.7258301, 17.7297707
21: -12.5595598, 9.1753359, -12.5517635, 9.1516323, -20.2275314, 20.2245522
22: -2.7681241, 18.5455170, -2.7653217, 18.5100937, -18.7128143, 18.7388802
23: -3.9246454, 15.2903156, -3.9233055, 15.2485294, -17.5276375, 17.5261955
24: -5.5051336, 17.4110260, -5.5093803, 17.3552952, -19.8929443, 19.9444160
25: 2.0938754, 24.3185234, 2.0920687, 24.2867699, -19.4078178, 19.4054298
26: -11.6796761, 21.6553307, -11.6796227, 21.5808563, -33.2605324, 33.3349533
27: -14.9840879, 10.0395451, -14.9874239, 9.9930477, -23.7729721, 23.8596115
28: -2.9868526, 18.0590706, -2.9854813, 18.0248318, -19.5523720, 19.6001854
29: -3.3482370, 15.5388880, -3.3402157, 15.5099993, -15.1603718, 15.1681671
30: -13.5772552, 13.8304005, -13.5809155, 13.7879639, -24.6987686, 24.7406273
31: -9.6941118, 11.3414421, -9.6889105, 11.3097725, -21.0038834, 21.0303535
32: -30.7224884, -3.8243904, -30.7071533, -3.8089223, -22.8514633, 22.7898865
33: -41.5483246, -3.2238727, -41.5046234, -3.2341328, -31.0183029, 31.0029449
34: -36.7802849, -3.8179708, -36.7704964, -3.8563290, -25.3772430, 25.4057655
35: -24.9046898, 5.5326509, -24.8945808, 5.5253859, -25.9699707, 26.0130005
36: -24.5694180, 6.4105849, -24.5545769, 6.4141731, -28.0957260, 28.0827255
37: -42.8824081, -6.5649986, -42.8654137, -6.5707207, -32.0890121, 32.0863266
38: -34.3812408, 3.4736090, -34.3700905, 3.4766021, -35.7553558, 35.7126617
39: -47.7727699, -7.0990911, -47.7212486, -7.1009407, -37.0922165, 37.0594788
40: -45.9825134, -18.9753990, -45.9496536, -18.9782181, -21.1656647, 21.1472359
41: -33.4892769, -4.6903458, -33.4823418, -4.6840138, -22.2363129, 22.2349663
42: -24.0617561, -0.4574134, -24.0383492, -0.4510000, -19.7397728, 19.6688213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1670

## Relational analysis of IS_B2_A2_B2_A1_A2_B1

### Relational analysis result of IS_B2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6269856, upper bound: 10.6526874
time: 28.31 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2_B2

### Relational analysis result of IS_B2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6380734, upper bound: 10.6526874
time: 37.04 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -22.5663662, 8.6209126, -22.5023346, 8.6433592, -31.2097244, 31.1232471
1: -11.9112062, 6.3634849, -11.8666878, 6.3751740, -18.2863808, 18.2301731
2: -13.1526527, 7.4308429, -13.1180964, 7.4465251, -19.4691391, 19.4616852
3: -18.7117195, 6.2638855, -18.6404495, 6.2788706, -24.3424606, 24.2766190
4: -20.2797928, 3.2769628, -20.2598190, 3.2888584, -21.2686920, 21.2535248
5: -18.1799812, 7.5677810, -18.1100044, 7.5847225, -25.2057877, 25.1748352
6: -36.8214149, -11.2605858, -36.7359314, -11.2711582, -20.8313446, 20.7590485
7: -24.1889324, 1.3346097, -24.1003761, 1.3549676, -24.0803375, 24.0048294
8: -27.3922462, 1.1068048, -27.3610687, 1.1076021, -25.3885498, 25.3958435
9: -11.5998306, 11.8373737, -11.5577736, 11.8534069, -20.7919693, 20.6714478
10: -17.7338142, 12.3716125, -17.6906204, 12.3420401, -29.6908798, 29.6359406
11: -16.8457050, 10.5041590, -16.7762108, 10.4684429, -23.8395081, 23.7967072
12: -24.2347603, 11.2965441, -24.2085228, 11.2743578, -33.3830261, 33.3256226
13: -22.4620667, 12.2840652, -22.3334217, 12.3377914, -32.7611847, 32.5629044
14: -34.9166107, 6.4377360, -34.8974991, 6.3713918, -36.7886353, 36.7521973
15: -8.7048254, 16.8352394, -8.6855898, 16.7678452, -23.3058929, 23.3867493
16: -22.8527985, 3.1101050, -22.7696495, 3.1165285, -25.9693260, 25.8797550
17: -28.0719318, 8.1045303, -28.0253391, 8.0623112, -36.1342430, 36.1298676
18: -12.5483551, 18.6979637, -12.5778513, 18.5835915, -29.3292847, 29.4500656
19: -8.9463911, 8.0501566, -8.9406929, 8.0243292, -16.5177155, 16.5369644
20: -9.9120083, 8.7711544, -9.9243298, 8.7521124, -17.6897392, 17.7174683
21: -12.5530062, 9.1618185, -12.5501976, 9.1367054, -20.2056503, 20.2206421
22: -2.7780809, 18.5891838, -2.7714005, 18.4990654, -18.7074280, 18.7890854
23: -3.9043045, 15.2730007, -3.9231539, 15.2296658, -17.4878044, 17.5322952
24: -5.4917965, 17.4271202, -5.5114851, 17.3349342, -19.8584442, 19.9764481
25: 2.0945392, 24.3448334, 2.0877633, 24.2715187, -19.3887024, 19.4541550
26: -11.6538906, 21.6750050, -11.6833315, 21.5483379, -33.2022285, 33.3583374
27: -14.9538708, 10.0435934, -14.9895496, 9.9657097, -23.7140732, 23.8798447
28: -2.9703288, 18.0687504, -2.9873223, 18.0077438, -19.5180969, 19.6203842
29: -3.3609409, 15.5593052, -3.3430767, 15.4995356, -15.1600723, 15.2096252
30: -13.5753088, 13.8463440, -13.5841808, 13.7730646, -24.6810989, 24.7738266
31: -9.6823492, 11.3265209, -9.6876030, 11.2996998, -20.9820480, 21.0141239
32: -30.7915764, -3.7963624, -30.7017059, -3.8005800, -22.9184608, 22.8150253
33: -41.5287399, -3.2370806, -41.4897156, -3.2371144, -31.0168991, 30.9943314
34: -36.7756424, -3.8110900, -36.7673836, -3.8609328, -25.3487778, 25.4234428
35: -24.9042969, 5.5386147, -24.8882980, 5.5243988, -25.9717636, 26.0145721
36: -24.5979614, 6.4298787, -24.5512753, 6.4184494, -28.1291122, 28.0979462
37: -42.8670197, -6.5659122, -42.8507805, -6.5730953, -32.0866547, 32.0842438
38: -34.4234543, 3.5023403, -34.3644409, 3.4797745, -35.8120346, 35.7290268
39: -47.7818832, -7.1033568, -47.6958199, -7.0969944, -37.1493530, 37.0515900
40: -45.9841919, -18.9580688, -45.9440689, -18.9761620, -21.1411362, 21.1786423
41: -33.5441360, -4.6735539, -33.4794960, -4.6792912, -22.2928276, 22.2450867
42: -24.1058350, -0.4470849, -24.0361366, -0.4508028, -19.7851105, 19.6885834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1670

## Relational analysis of IS_B2_A2_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6415868, upper bound: 10.6427538
time: 27.49 seconds

## Relational analysis of IS_B2_A2_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6526844, upper bound: 10.6427538
time: 26.23 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -22.6067066, 8.6590385, -22.5239944, 8.6451969, -31.2519035, 31.1830330
1: -11.9283257, 6.3870907, -11.8766155, 6.3763776, -18.3047028, 18.2637062
2: -13.1743002, 7.4578838, -13.1308842, 7.4478941, -19.4789429, 19.5000153
3: -18.7402058, 6.2930517, -18.6562881, 6.2807236, -24.3657150, 24.3235931
4: -20.3184242, 3.3094885, -20.2807312, 3.2903893, -21.2780075, 21.3064651
5: -18.2121468, 7.6010542, -18.1279621, 7.5875807, -25.2399597, 25.2282715
6: -36.8283577, -11.2505627, -36.7378159, -11.2681170, -20.8421211, 20.7693253
7: -24.2198944, 1.3678913, -24.1181545, 1.3571231, -24.1059418, 24.0560150
8: -27.4202366, 1.1361341, -27.3765068, 1.1093025, -25.4049530, 25.4412613
9: -11.6228886, 11.8609781, -11.5695171, 11.8545704, -20.8025208, 20.7107162
10: -17.7462482, 12.3858099, -17.6935501, 12.3449497, -29.7063446, 29.6535187
11: -16.8701344, 10.5288582, -16.7781410, 10.4816170, -23.8787842, 23.8008041
12: -24.2462463, 11.3225822, -24.2106285, 11.2861500, -33.4058533, 33.3515930
13: -22.5269642, 12.3524132, -22.3694477, 12.3414936, -32.8083191, 32.6677475
14: -34.9357491, 6.4509168, -34.9034271, 6.3754067, -36.8135376, 36.7735825
15: -8.7182484, 16.8425903, -8.6894264, 16.7698250, -23.3204193, 23.3976479
16: -22.8712654, 3.1322894, -22.7786541, 3.1181955, -25.9894600, 25.9109440
17: -28.1079731, 8.1287622, -28.0434284, 8.0644598, -36.1724319, 36.1721916
18: -12.6052485, 18.7489834, -12.5817423, 18.6124172, -29.4148483, 29.4901428
19: -8.9709568, 8.0617218, -8.9441605, 8.0305510, -16.5531845, 16.5514603
20: -9.9435234, 8.7966709, -9.9274330, 8.7663469, -17.7352638, 17.7371101
21: -12.5859489, 9.1889257, -12.5534067, 9.1514645, -20.2531090, 20.2400322
22: -2.8057070, 18.6069221, -2.7742972, 18.5090828, -18.7451859, 18.8088245
23: -3.9438319, 15.3083868, -3.9258623, 15.2490387, -17.5465317, 17.5474968
24: -5.5289307, 17.4613457, -5.5151997, 17.3545074, -19.9150467, 20.0014114
25: 2.0635257, 24.3720684, 2.0845041, 24.2866879, -19.4353561, 19.4671440
26: -11.7154331, 21.7320175, -11.6882896, 21.5812016, -33.2966347, 33.4203072
27: -15.0053978, 10.0921783, -14.9929256, 9.9930964, -23.7936020, 23.9184265
28: -3.0114059, 18.1004543, -2.9910536, 18.0248451, -19.5763168, 19.6477699
29: -3.3840556, 15.5779047, -3.3453188, 15.5096073, -15.1940613, 15.2126427
30: -13.6028223, 13.8775425, -13.5858364, 13.7887020, -24.7240143, 24.7923393
31: -9.7162399, 11.3454151, -9.6915560, 11.3095846, -21.0258255, 21.0369720
32: -30.8043022, -3.7852244, -30.7063465, -3.7981281, -22.9447708, 22.8259010
33: -41.5626221, -3.2123604, -41.5047913, -3.2348919, -31.0352402, 31.0165482
34: -36.7942123, -3.8013659, -36.7709389, -3.8556514, -25.3870850, 25.4384727
35: -24.9229164, 5.5452394, -24.8954964, 5.5256352, -25.9885941, 26.0250778
36: -24.6109810, 6.4335856, -24.5543156, 6.4193478, -28.1428833, 28.1046524
37: -42.8865700, -6.5564175, -42.8580284, -6.5710526, -32.1038666, 32.0976105
38: -34.4477081, 3.5139103, -34.3698235, 3.4856977, -35.8312073, 35.7507629
39: -47.8323898, -7.0694275, -47.7220116, -7.0956287, -37.1605835, 37.0940781
40: -46.0036316, -18.9462490, -45.9511261, -18.9751358, -21.1854630, 21.1774788
41: -33.5531502, -4.6641059, -33.4819031, -4.6764059, -22.3093758, 22.2587509
42: -24.1176109, -0.4294083, -24.0370560, -0.4435112, -19.8047638, 19.6933250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1446

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1670

## Relational analysis of IS_B2_A2_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6415895, upper bound: 10.6526874
time: 30.10 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6526875, upper bound: 10.6526874
time: 25.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 57.99 seconds
IS_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6116048, upper bound: 10.6324997
IS_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6116048, upper bound: 10.6508600
IS_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6215255, upper bound: 10.6324997
IS_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6215255, upper bound: 10.6508600
IS_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6262220, upper bound: 10.6324997
IS_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6262220, upper bound: 10.6508600
IS_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6361455, upper bound: 10.6324997
IS_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6361455, upper bound: 10.6508600
IS_B2_A2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6324998, upper bound: 10.6212961
IS_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6508601, upper bound: 10.6212961
IS_B2_A2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6324998, upper bound: 10.6312169
IS_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6508601, upper bound: 10.6312169
IS_B2_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6269856, upper bound: 10.6526874
IS_B2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6380734, upper bound: 10.6526874
IS_B2_A2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6415868, upper bound: 10.6427538
IS_B2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6526844, upper bound: 10.6427538
IS_B2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6415895, upper bound: 10.6526874
IS_B2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 57.99
Output dim: 25, lower bound: -10.6526875, upper bound: 10.6526874

## BFS IS instance: IS_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -22.4940205, 8.6493664, -22.4766312, 8.5727444, -31.0667648, 31.1259975
1: -11.8633566, 6.3999095, -11.8561964, 6.3315020, -18.1948586, 18.2561054
2: -13.1159687, 7.4681597, -13.1064634, 7.4002352, -19.3569107, 19.4626236
3: -18.6387062, 6.3042822, -18.6225357, 6.2215896, -24.1662064, 24.2600937
4: -20.2534370, 3.3350344, -20.2360229, 3.2425745, -21.1213989, 21.2308731
5: -18.1071968, 7.6076646, -18.0904961, 7.5219536, -25.0510788, 25.1823578
6: -36.7387543, -11.2983599, -36.7165375, -11.3007174, -20.7324333, 20.7204971
7: -24.0965691, 1.3773923, -24.0823650, 1.2936130, -23.8490753, 23.9656754
8: -27.3542538, 1.1398125, -27.3418465, 1.0570812, -25.1578293, 25.2747421
9: -11.5557604, 11.8744602, -11.5403404, 11.8056335, -20.5192719, 20.5406799
10: -17.6781292, 12.3305550, -17.6639824, 12.3112345, -29.6097107, 29.5687866
11: -16.8016205, 10.4583902, -16.7350235, 10.4469376, -23.7190399, 23.6265335
12: -24.2460136, 11.2650547, -24.1907940, 11.2507915, -33.4366379, 33.3113708
13: -22.3286915, 12.2986889, -22.2968597, 12.2206354, -32.4407959, 32.4807816
14: -34.8776054, 6.3679285, -34.8689232, 6.3521280, -36.8370819, 36.6903381
15: -8.6684437, 16.7915859, -8.6590519, 16.7545853, -23.2133026, 23.2810135
16: -22.7703381, 3.1183655, -22.7533207, 3.0710914, -25.8414288, 25.8716869
17: -28.0499611, 8.0571651, -27.9974461, 8.0331440, -36.0831070, 36.0546112
18: -12.5649204, 18.5795860, -12.4784431, 18.5540485, -29.2364807, 29.1372604
19: -8.9641418, 8.0277653, -8.8977356, 8.0162125, -16.5072327, 16.4413261
20: -9.9183397, 8.7536831, -9.8742075, 8.7383595, -17.6669426, 17.6205368
21: -12.5486212, 9.1374388, -12.4990826, 9.1197777, -20.2382851, 20.1648178
22: -2.7670817, 18.5001450, -2.7159295, 18.4897442, -18.6710205, 18.6052952
23: -3.9600201, 15.2283115, -3.8573675, 15.2080307, -17.4480820, 17.3267021
24: -5.5159798, 17.3365364, -5.4432573, 17.3167324, -19.7501297, 19.6782036
25: 2.0810337, 24.2716103, 2.1478825, 24.2557411, -19.3448143, 19.2569542
26: -11.6834698, 21.5434780, -11.5784388, 21.5165482, -33.2000198, 33.1219177
27: -14.9855127, 9.9638958, -14.9125032, 9.9397392, -23.7181396, 23.7085953
28: -3.0019011, 18.0060139, -2.9181991, 17.9886646, -19.4705772, 19.4170151
29: -3.3577485, 15.4998341, -3.3043251, 15.4883003, -15.1353416, 15.0836868
30: -13.5685844, 13.7721786, -13.5305748, 13.7516785, -24.6351166, 24.6094398
31: -9.7041807, 11.3017635, -9.6279421, 11.2867393, -20.9909210, 20.9297066
32: -30.7132797, -3.8370800, -30.6851292, -3.8274760, -22.8897171, 22.8101006
33: -41.4859390, -3.2599673, -41.4554596, -3.2817349, -31.1424103, 31.1019592
34: -36.7957230, -3.8635097, -36.7318802, -3.8703585, -25.3051376, 25.2345886
35: -24.8977623, 5.5256381, -24.8691940, 5.5158629, -25.9552765, 25.9674072
36: -24.5893288, 6.4024224, -24.5351562, 6.4067359, -28.0823288, 28.0239487
37: -42.9023705, -6.5804758, -42.8360214, -6.5878038, -32.2871323, 32.1941223
38: -34.4030952, 3.4555626, -34.3333969, 3.4603019, -35.6686401, 35.5577698
39: -47.6978874, -7.1399002, -47.6612816, -7.1561527, -37.3444061, 37.2262268
40: -45.9434204, -19.0060577, -45.9215660, -19.0069847, -21.4109383, 21.3337669
41: -33.5015411, -4.7003965, -33.4680710, -4.6995134, -22.2510757, 22.2101402
42: -24.0476646, -0.4838109, -24.0203609, -0.4830387, -19.6634293, 19.6252975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=160, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1391

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1670

## Relational analysis of IS_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.5993516, upper bound: 10.6496970
time: 32.82 seconds

## Relational analysis of IS_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6104425, upper bound: 10.6496970
time: 24.89 seconds

## BFS IS instance: IS_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -22.5156784, 8.6512318, -22.5169258, 8.6108685, -31.1265469, 31.1681576
1: -11.8732729, 6.4011474, -11.8732910, 6.3551502, -18.2284241, 18.2744389
2: -13.1287556, 7.4694982, -13.1281090, 7.4272556, -19.3952408, 19.4724426
3: -18.6545715, 6.3061295, -18.6510315, 6.2507672, -24.2131348, 24.2832870
4: -20.2743645, 3.3365922, -20.2746429, 3.2750733, -21.1743240, 21.2401123
5: -18.1251678, 7.6105185, -18.1226616, 7.5552473, -25.1044693, 25.2165527
6: -36.7406883, -11.2953434, -36.7234688, -11.2907238, -20.7426529, 20.7312889
7: -24.1143684, 1.3795345, -24.1133251, 1.3269632, -23.9002609, 23.9913483
8: -27.3697166, 1.1415129, -27.3697777, 1.0864105, -25.2033081, 25.2911606
9: -11.5674973, 11.8756409, -11.5634298, 11.8292542, -20.5585556, 20.5512695
10: -17.6810341, 12.3334370, -17.6764221, 12.3254795, -29.6272888, 29.5842361
11: -16.8035507, 10.4715900, -16.7594032, 10.4716473, -23.7231522, 23.6657867
12: -24.2481079, 11.2768097, -24.2022438, 11.2767563, -33.4625397, 33.3341370
13: -22.3647461, 12.3024254, -22.3617878, 12.2889709, -32.5456696, 32.5279541
14: -34.8834763, 6.3720164, -34.8880653, 6.3652787, -36.8582001, 36.7151947
15: -8.6722794, 16.7935791, -8.6724663, 16.7619381, -23.2241745, 23.2955132
16: -22.7793674, 3.1200247, -22.7717628, 3.0932519, -25.8726196, 25.8917885
17: -28.0680714, 8.0592823, -28.0335007, 8.0573225, -36.1253929, 36.0927811
18: -12.5687943, 18.6083984, -12.5353975, 18.6050758, -29.2765656, 29.2228088
19: -8.9676113, 8.0339851, -8.9223185, 8.0277939, -16.5217323, 16.4768028
20: -9.9214420, 8.7678986, -9.9057350, 8.7638826, -17.6865768, 17.6660652
21: -12.5518141, 9.1522121, -12.5320787, 9.1469135, -20.2576981, 20.2122345
22: -2.7699499, 18.5101700, -2.7435617, 18.5074883, -18.6907310, 18.6430550
23: -3.9627509, 15.2477150, -3.8969212, 15.2434139, -17.4632950, 17.3854523
24: -5.5197220, 17.3560963, -5.4803810, 17.3509731, -19.7750854, 19.7348175
25: 2.0778046, 24.2867565, 2.1168551, 24.2829933, -19.3578072, 19.3036118
26: -11.6884756, 21.5763226, -11.6400299, 21.5735512, -33.2620277, 33.2163544
27: -14.9889278, 9.9913044, -14.9640455, 9.9883366, -23.7567062, 23.7881393
28: -3.0056129, 18.0231171, -2.9593086, 18.0203514, -19.4979706, 19.4752312
29: -3.3599911, 15.5099039, -3.3274345, 15.5068588, -15.1383495, 15.1176682
30: -13.5702515, 13.7878227, -13.5580883, 13.7828941, -24.6536102, 24.6523514
31: -9.7081490, 11.3116302, -9.6618309, 11.3056278, -21.0137768, 20.9734612
32: -30.7178822, -3.8345623, -30.6978569, -3.8163710, -22.9005203, 22.8364182
33: -41.5010567, -3.2577662, -41.4893570, -3.2569427, -31.1645432, 31.1203537
34: -36.7992516, -3.8582549, -36.7504463, -3.8606296, -25.3201904, 25.2729301
35: -24.9049149, 5.5268617, -24.8877125, 5.5225024, -25.9657822, 25.9842072
36: -24.5923309, 6.4033680, -24.5481586, 6.4104075, -28.0890732, 28.0377426
37: -42.9096069, -6.5783749, -42.8554382, -6.5783501, -32.3004913, 32.2112732
38: -34.4083481, 3.4615755, -34.3577042, 3.4718814, -35.6903992, 35.5768585
39: -47.7240372, -7.1385384, -47.7116776, -7.1222692, -37.3869019, 37.2373276
40: -45.9504776, -19.0050049, -45.9409981, -18.9952145, -21.4097137, 21.3781052
41: -33.5039635, -4.6975584, -33.4771233, -4.6900606, -22.2647324, 22.2266655
42: -24.0485954, -0.4765592, -24.0321503, -0.4653661, -19.6681480, 19.6449699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=160, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1522
type: A, layer: 1, pos: 1522
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1391

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1670

## Relational analysis of IS_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6092686, upper bound: 10.6496970
time: 24.74 seconds

## Relational analysis of IS_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6203613, upper bound: 10.6496970
time: 32.85 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 59.83 seconds
IS_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 59.83
Output dim: 25, lower bound: -10.5993516, upper bound: 10.6496970
IS_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 59.83
Output dim: 25, lower bound: -10.6104425, upper bound: 10.6496970
IS_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 59.83
Output dim: 25, lower bound: -10.6092686, upper bound: 10.6496970
IS_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 59.83
Output dim: 25, lower bound: -10.6203613, upper bound: 10.6496970
IS_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 59.83
Output dim: 25, lower bound: -10.6262220, upper bound: 10.6508600
IS_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 59.83
Output dim: 25, lower bound: -10.6361455, upper bound: 10.6508600
IS_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 59.83
Output dim: 25, lower bound: -10.6508601, upper bound: 10.6212961
IS_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 59.83
Output dim: 25, lower bound: -10.6508601, upper bound: 10.6312169
IS_B2_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 59.83
Output dim: 25, lower bound: -10.6269856, upper bound: 10.6526874
IS_B2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 59.83
Output dim: 25, lower bound: -10.6380734, upper bound: 10.6526874
IS_B2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 59.83
Output dim: 25, lower bound: -10.6526844, upper bound: 10.6427538
IS_B2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 59.83
Output dim: 25, lower bound: -10.6415895, upper bound: 10.6526874
IS_B2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 59.83
Output dim: 25, lower bound: -10.6526875, upper bound: 10.6526874

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 46.58 + 1778.55 = 1825.12 seconds
