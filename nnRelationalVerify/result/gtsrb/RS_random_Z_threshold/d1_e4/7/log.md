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
execution time: IAR + RelationalAnalysis = 2.45 + 44.04 = 46.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 25, lower bound: -10.6599395, upper bound: 10.6599395

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1694

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6588220, upper bound: 10.6594151
time: 25.69 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6594151, upper bound: 10.6588220
time: 27.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 53.55 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 53.55
Output dim: 25, lower bound: -10.6588220, upper bound: 10.6594151
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 53.55
Output dim: 25, lower bound: -10.6594151, upper bound: 10.6588220

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.5134506, 19.5135918
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3258667, 24.3243484
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3172989, 21.3181305
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2294540, 25.2273636
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8165588, 20.8156509
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0541916, 24.0519638
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4065475, 25.4029007
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.8004303, 20.8001862
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6759109, 29.6750793
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8205719, 23.8186722
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.4046402, 33.4043808
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7003098, 32.7019653
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8353806, 36.8295822
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3379593, 23.3374939
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4270935, 29.4270935
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5422974, 16.5431747
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7301407, 17.7288818
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2319183, 20.2287025
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7563896, 18.7577820
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5421143, 17.5418320
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9425583, 19.9431229
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4505997, 19.4503365
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8427124, 23.8400497
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5880737, 19.5898399
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1891785, 15.1891003
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7445602, 24.7438660
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8566017, 22.8566780
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0165863, 31.0183487
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3943253, 25.3965836
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9717407, 25.9752121
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1071243, 28.1090469
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1166992, 32.1167679
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7856445, 35.7856293
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1004486, 37.1020508
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2014465, 21.2010651
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2702751, 22.2705269
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7029686, 19.6989727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 669

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6482862, upper bound: 10.6488544
time: 24.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6482863, upper bound: 10.6488532
time: 31.21 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.5135956, 19.5134506
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3243561, 24.3258667
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3181305, 21.3172989
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2273636, 25.2294540
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8156586, 20.8165627
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0519638, 24.0541916
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4029007, 25.4065475
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.8001862, 20.8004303
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6750717, 29.6759033
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8186722, 23.8205719
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.4043961, 33.4046402
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7019577, 32.7003174
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8295822, 36.8353882
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3374939, 23.3379631
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4270935, 29.4270935
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5431747, 16.5422974
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7288818, 17.7301369
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2286987, 20.2319183
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7577820, 18.7563896
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5418320, 17.5421143
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9431229, 19.9425545
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4503403, 19.4505959
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8400497, 23.8427124
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5898438, 19.5880737
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1891022, 15.1891785
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7438660, 24.7445602
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8566780, 22.8566093
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0183487, 31.0165863
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3965836, 25.3943214
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9752121, 25.9717407
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1090469, 28.1071243
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1167755, 32.1166992
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7856293, 35.7856445
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1020355, 37.1004562
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2010727, 21.2014465
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2705269, 22.2702713
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6989746, 19.7029686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1464

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6553041, upper bound: 10.6585572
time: 27.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6591503, upper bound: 10.6547094
time: 28.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 58.23 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 58.23
Output dim: 25, lower bound: -10.6482862, upper bound: 10.6488544
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 58.23
Output dim: 25, lower bound: -10.6482863, upper bound: 10.6488532
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 58.23
Output dim: 25, lower bound: -10.6553041, upper bound: 10.6585572
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 58.23
Output dim: 25, lower bound: -10.6591503, upper bound: 10.6547094

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.5125961, 19.5123291
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3228455, 24.3241730
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3162079, 21.3152924
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2262497, 25.2281952
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8194160, 20.8191490
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0484543, 24.0502243
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4022446, 25.4057541
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.8000259, 20.8004608
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6745300, 29.6753922
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8197784, 23.8213043
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.4043808, 33.4046326
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7020569, 32.7004089
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8313065, 36.8369446
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3368073, 23.3380470
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4249496, 29.4250107
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5427513, 16.5418472
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7273750, 17.7282562
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2279205, 20.2310143
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7551765, 18.7542496
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5376930, 17.5384712
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9419098, 19.9414940
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4468536, 19.4475517
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8403854, 23.8430252
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5891418, 19.5874481
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1866302, 15.1870651
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7456207, 24.7461700
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8534317, 22.8527451
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0152969, 31.0138321
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3970871, 25.3948708
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9743958, 25.9709702
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1091919, 28.1072006
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1095123, 32.1100616
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7856445, 35.7856598
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1023254, 37.1007080
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2033653, 21.2033043
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2710800, 22.2706909
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6987114, 19.7027168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6545069, upper bound: 10.6538543
time: 29.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6505990, upper bound: 10.6577588
time: 22.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.5124741, 19.5124550
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3226547, 24.3243713
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3161240, 21.3153763
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2261047, 25.2283401
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8182411, 20.8203278
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0479889, 24.0506897
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4021149, 25.4058838
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.8002167, 20.8002701
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6745758, 29.6753616
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8194046, 23.8216743
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.4043808, 33.4046402
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7020569, 32.7004089
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8311539, 36.8370972
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3375778, 23.3372726
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4250031, 29.4249573
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5427284, 16.5418701
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7269936, 17.7286339
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2277985, 20.2311363
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7556419, 18.7537842
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5381889, 17.5379791
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9420624, 19.9413452
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4472961, 19.4471169
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8403702, 23.8430481
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5892181, 19.5873718
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1869888, 15.1867065
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7454834, 24.7463150
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8528214, 22.8533630
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0156021, 31.0135345
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3971329, 25.3948250
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9744415, 25.9709244
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1091232, 28.1072617
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1101379, 32.1094437
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7856445, 35.7856598
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1022949, 37.1007309
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2029228, 21.2037468
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2709427, 22.2708206
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6987228, 19.7027054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1589

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6574829, upper bound: 10.6540895
time: 33.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6585316, upper bound: 10.6530381
time: 27.46 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 62.82 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 62.82
Output dim: 25, lower bound: -10.6545069, upper bound: 10.6538543
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 62.82
Output dim: 25, lower bound: -10.6505990, upper bound: 10.6577588
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 62.82
Output dim: 25, lower bound: -10.6574829, upper bound: 10.6540895
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 62.82
Output dim: 25, lower bound: -10.6585316, upper bound: 10.6530381

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4910583, 19.4931564
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3169098, 24.3182297
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3011780, 21.3015060
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.1980362, 25.2023621
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8328705, 20.8335304
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0558548, 24.0571899
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4054871, 25.4088287
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7644653, 20.7598114
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6486893, 29.6466370
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7776260, 23.7738800
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3662033, 33.3607788
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7024918, 32.7000427
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.7642212, 36.7604218
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3250122, 23.3266754
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4419327, 29.4437485
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5424156, 16.5415077
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7209091, 17.7202530
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.1855316, 20.1819077
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7540207, 18.7535801
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5450439, 17.5458908
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9416428, 19.9430885
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4411774, 19.4414368
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8250046, 23.8293877
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5943642, 19.5917778
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1860867, 15.1863098
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7295227, 24.7253571
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8705292, 22.8712807
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -30.9718475, 30.9755630
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3614883, 25.3639717
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9753342, 25.9719391
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.0952911, 28.0960312
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1285553, 32.1330338
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7822723, 35.7833557
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1183929, 37.1213226
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2165413, 21.2216492
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.3001175, 22.3031197
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7097855, 19.7136154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1628

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 516

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6507012, upper bound: 10.6500267
time: 35.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6506879, upper bound: 10.6500401
time: 27.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4934235, 19.4907875
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3169022, 24.3182373
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3024216, 21.3002548
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2004089, 25.1999893
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8338013, 20.8325996
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0554199, 24.0576324
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4053192, 25.4090042
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7593842, 20.7649002
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6457748, 29.6495361
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7723541, 23.7791519
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3605270, 33.3664627
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7016830, 32.7008514
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.7547760, 36.7698593
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3254318, 23.3262520
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4436951, 29.4419861
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5424080, 16.5415115
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7193756, 17.7217865
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.1788101, 20.1886215
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7545090, 18.7530937
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5451202, 17.5458183
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9435043, 19.9412193
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4407425, 19.4418755
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8267441, 23.8276558
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5934639, 19.5926781
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1858730, 15.1865234
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7248077, 24.7300720
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8719635, 22.8698425
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -30.9770279, 30.9703827
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3661880, 25.3592682
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9753571, 25.9719086
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.0980225, 28.0932999
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1324768, 32.1291046
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7833405, 35.7822876
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1229401, 37.1167603
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2217140, 21.2164726
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.3035049, 22.2997360
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7096100, 19.7137871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 576

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1391

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6487756, upper bound: 10.6562940
time: 29.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6491328, upper bound: 10.6559359
time: 33.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4995346, 19.4980659
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3358536, 24.3393784
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3051300, 21.3042450
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2265701, 25.2288361
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7929344, 20.7975082
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0422440, 24.0450211
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3975067, 25.4007187
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7886124, 20.7874069
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6507721, 29.6546478
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8095169, 23.8121872
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3799896, 33.3827896
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6982880, 32.6962204
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8282166, 36.8339615
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3356552, 23.3352242
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4291763, 29.4281235
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5513191, 16.5498314
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7308807, 17.7324448
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2321167, 20.2354393
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7558556, 18.7539902
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5419197, 17.5407257
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9450378, 19.9439392
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4533157, 19.4525414
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8576355, 23.8583984
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5907440, 19.5883560
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1831207, 15.1829681
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7208328, 24.7247276
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8307571, 22.8332748
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0156326, 31.0153656
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3537598, 25.3595314
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9863129, 25.9863815
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1091232, 28.1072693
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1079865, 32.1075134
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7873459, 35.7875443
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1012344, 37.0997086
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1854324, 21.1870689
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2571030, 22.2581596
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6819305, 19.6863956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 531

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 766

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6561715, upper bound: 10.6537302
time: 33.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6571234, upper bound: 10.6527771
time: 36.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4980850, 19.4995117
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3376617, 24.3375626
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3049927, 21.3043823
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2265930, 25.2288055
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7954216, 20.7950287
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0423279, 24.0449448
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3969421, 25.4012833
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7873535, 20.7886658
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6538544, 29.6515732
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8099136, 23.8117905
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3825226, 33.3802490
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6978607, 32.6966400
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8280029, 36.8341675
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3355255, 23.3353539
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4281769, 29.4291306
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5506859, 16.5504608
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7308197, 17.7325134
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2321014, 20.2354622
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7558517, 18.7539959
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5409355, 17.5417137
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9446487, 19.9443245
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4527206, 19.4531364
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8557205, 23.8603134
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5902023, 19.5889015
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1832504, 15.1828365
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7238998, 24.7216682
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8327255, 22.8312988
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0174332, 31.0135651
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3618317, 25.3514519
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9899063, 25.9827881
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1091232, 28.1072693
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1082001, 32.1072998
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7875290, 35.7873764
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1012802, 37.0996552
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1862564, 21.1862526
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2582855, 22.2569809
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6824112, 19.6859150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 889

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6528847, upper bound: 10.6521652
time: 28.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6576614, upper bound: 10.6473922
time: 26.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 56.80 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 56.80
Output dim: 25, lower bound: -10.6507012, upper bound: 10.6500267
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 56.80
Output dim: 25, lower bound: -10.6506879, upper bound: 10.6500401
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 56.80
Output dim: 25, lower bound: -10.6487756, upper bound: 10.6562940
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 56.80
Output dim: 25, lower bound: -10.6491328, upper bound: 10.6559359
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 56.80
Output dim: 25, lower bound: -10.6561715, upper bound: 10.6537302
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 56.80
Output dim: 25, lower bound: -10.6571234, upper bound: 10.6527771
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 56.80
Output dim: 25, lower bound: -10.6528847, upper bound: 10.6521652
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 56.80
Output dim: 25, lower bound: -10.6576614, upper bound: 10.6473922

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4911041, 19.4931564
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3169098, 24.3182602
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3013382, 21.3014908
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.1980896, 25.2023468
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8328552, 20.8339157
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0558929, 24.0571899
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4055557, 25.4088211
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7644424, 20.7598190
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6486740, 29.6466141
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7776184, 23.7738686
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3662033, 33.3609009
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7024918, 32.7000351
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.7644958, 36.7603989
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3251190, 23.3266640
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4420776, 29.4437408
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5425606, 16.5414925
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7209244, 17.7202530
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.1857834, 20.1818924
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7541695, 18.7535744
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5450134, 17.5458870
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9417496, 19.9430847
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4412994, 19.4414291
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8251038, 23.8293648
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5943794, 19.5917854
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1861496, 15.1863079
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7295151, 24.7252884
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8705139, 22.8715668
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -30.9718399, 30.9757614
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3614578, 25.3643608
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9753189, 25.9720306
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.0952759, 28.0961380
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1285400, 32.1331482
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7822723, 35.7834167
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1183624, 37.1213455
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2165108, 21.2220497
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.3001099, 22.3034134
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7097588, 19.7139931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1361

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6281527, upper bound: 10.6286720
time: 25.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6296395, upper bound: 10.6268561
time: 30.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4910583, 19.4931564
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3169098, 24.3182373
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3011551, 21.3015060
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.1980286, 25.2023621
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8328705, 20.8335190
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0558548, 24.0571899
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4054871, 25.4088287
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7644730, 20.7598114
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6486893, 29.6466217
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7776260, 23.7738724
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3662033, 33.3607712
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7024918, 32.7000427
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.7641907, 36.7604218
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3250046, 23.3266754
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4419174, 29.4437485
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5423965, 16.5415077
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7209091, 17.7202530
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.1855087, 20.1819077
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7540169, 18.7535801
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5450439, 17.5458908
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9416275, 19.9430885
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4411774, 19.4414368
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8249969, 23.8293877
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5943642, 19.5917816
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1860847, 15.1863098
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7295227, 24.7253418
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8705292, 22.8712654
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -30.9718475, 30.9755554
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3614883, 25.3639450
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9753342, 25.9719162
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.0952911, 28.0960236
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1285553, 32.1330261
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7822723, 35.7833557
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1183929, 37.1213150
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2165413, 21.2216225
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.3001175, 22.3031044
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7097855, 19.7135925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1446

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6505890, upper bound: 10.6438147
time: 28.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6444635, upper bound: 10.6499414
time: 32.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4934235, 19.4907570
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3169098, 24.3182220
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3023834, 21.3000870
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2004013, 25.1999435
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8338165, 20.8324928
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0554199, 24.0576172
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4053116, 25.4089737
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7593613, 20.7648468
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6456985, 29.6494751
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7721252, 23.7792053
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3604965, 33.3664169
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7016754, 32.7007980
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.7546844, 36.7696686
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3254089, 23.3262177
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4436340, 29.4420090
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5424042, 16.5415154
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7193680, 17.7217865
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.1786804, 20.1883659
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7543793, 18.7530670
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5449295, 17.5458221
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9434891, 19.9412231
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4407043, 19.4418831
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8267288, 23.8276405
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5934410, 19.5926819
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1858654, 15.1865292
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7247925, 24.7300682
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8719749, 22.8697510
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -30.9771194, 30.9701691
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3660431, 25.3592148
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9752960, 25.9718704
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.0979919, 28.0932693
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1324921, 32.1291046
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7832947, 35.7823029
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1230850, 37.1166000
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2218018, 21.2163734
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.3035088, 22.2997246
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7095795, 19.7136803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6442973, upper bound: 10.6504290
time: 25.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6423139, upper bound: 10.6518306
time: 28.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4933929, 19.4907875
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3168945, 24.3182373
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3022537, 21.3002548
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2003708, 25.1999893
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8336945, 20.8325996
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0554047, 24.0576324
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4052887, 25.4090042
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7593842, 20.7648773
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6457748, 29.6494675
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7723541, 23.7789268
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3605270, 33.3664398
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7016296, 32.7008514
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.7547760, 36.7697678
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3253937, 23.3262520
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4436951, 29.4419327
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5424080, 16.5415077
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7193756, 17.7217789
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.1788101, 20.1884918
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7544785, 18.7530937
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5451202, 17.5456276
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9435043, 19.9411926
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4407425, 19.4418449
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8267441, 23.8276405
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5934639, 19.5926476
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1858730, 15.1865139
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7248077, 24.7300606
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8718758, 22.8698425
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -30.9768066, 30.9703827
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3661346, 25.3592682
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9753189, 25.9719086
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.0979996, 28.0932999
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1324768, 32.1291046
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7833405, 35.7822495
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1227798, 37.1167603
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2216187, 21.2164726
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.3034935, 22.2997360
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7096100, 19.7137585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1434

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6469409, upper bound: 10.6537374
time: 32.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6469409, upper bound: 10.6537374
time: 27.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4987335, 19.4966316
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3346481, 24.3378983
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3034592, 21.3012543
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2255707, 25.2270432
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7912712, 20.7965126
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0411911, 24.0431213
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3951569, 25.3964920
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7876740, 20.7861099
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6500854, 29.6541595
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8094559, 23.8123627
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3785706, 33.3820038
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6976624, 32.6959076
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8254700, 36.8290405
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3343964, 23.3329620
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4289093, 29.4277344
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5514412, 16.5497322
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7304306, 17.7317924
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2323990, 20.2350044
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7560234, 18.7539730
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5417290, 17.5406303
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9444885, 19.9431496
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4533157, 19.4525490
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8581009, 23.8583527
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5888519, 19.5873680
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1830826, 15.1829681
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7206802, 24.7250404
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8300095, 22.8328552
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0153427, 31.0153275
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3506241, 25.3577843
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9847412, 25.9855118
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1074982, 28.1063690
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1079254, 32.1074982
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7845688, 35.7859955
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1012573, 37.0995407
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1850357, 21.1870155
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2550354, 22.2570076
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6807480, 19.6857376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 857

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 531

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6557267, upper bound: 10.6490772
time: 31.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6515208, upper bound: 10.6532852
time: 23.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4981003, 19.4972687
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3343735, 24.3381729
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3021393, 21.3025818
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2247772, 25.2278366
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7919426, 20.7958412
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0403442, 24.0439682
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3932800, 25.3983688
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7873154, 20.7864685
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6502991, 29.6539459
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8096924, 23.8121300
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3791962, 33.3813705
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6979675, 32.6955948
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8232880, 36.8312225
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3333969, 23.3339615
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4287872, 29.4278488
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5512199, 16.5499535
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7302246, 17.7319946
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2316818, 20.2357101
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7558365, 18.7541599
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5418282, 17.5405312
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9442520, 19.9433861
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4533157, 19.4525490
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8575897, 23.8588638
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5897522, 19.5864716
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1831207, 15.1829300
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7211380, 24.7245674
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8303299, 22.8325310
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0155945, 31.0150757
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3520126, 25.3563881
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9854431, 25.9848175
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1082230, 28.1056519
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1079865, 32.1074524
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7858047, 35.7847519
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1010590, 37.0997467
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1853714, 21.1866798
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2559509, 22.2560883
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6812744, 19.6852112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 702

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1563

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6503444, upper bound: 10.6525855
time: 28.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6569315, upper bound: 10.6459989
time: 25.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4965744, 19.4979820
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3368607, 24.3366241
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3045273, 21.3048553
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2265625, 25.2289352
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7945251, 20.7947388
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0418701, 24.0445633
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3969345, 25.4013138
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7855301, 20.7865219
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6516266, 29.6490555
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8092804, 23.8113174
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3825760, 33.3802109
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6975861, 32.6962738
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8159485, 36.8247452
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3354263, 23.3344727
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4259949, 29.4272614
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5468674, 16.5475464
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7297974, 17.7322578
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2253571, 20.2304916
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7488556, 18.7480087
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5414696, 17.5416412
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9401398, 19.9406548
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4449081, 19.4460793
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8555756, 23.8602066
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5890350, 19.5879898
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1771622, 15.1782532
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7180405, 24.7173576
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8309326, 22.8272591
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0086212, 31.0018387
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3585205, 25.3460274
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9885559, 25.9810562
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1060333, 28.1028595
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1109238, 32.1068649
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7836838, 35.7831802
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.0967255, 37.0930099
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1838188, 21.1796112
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2606964, 22.2562752
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6853600, 19.6854706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 572

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 576

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6527984, upper bound: 10.6496390
time: 29.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6503583, upper bound: 10.6520788
time: 33.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4965515, 19.4980011
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3367310, 24.3367615
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3054657, 21.3039169
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2267227, 25.2287750
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7951279, 20.7941322
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0419464, 24.0444870
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3969727, 25.4012756
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7852097, 20.7868423
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6513367, 29.6493454
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8094406, 23.8111534
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3824844, 33.3802948
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6975098, 32.6963654
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8185883, 36.8221130
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3346481, 23.3352509
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4263077, 29.4269409
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5477715, 16.5466423
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7305603, 17.7314949
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2271271, 20.2287140
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7498627, 18.7470036
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5408669, 17.5422478
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9409866, 19.9398155
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4456635, 19.4453201
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8556061, 23.8601761
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5892944, 19.5877304
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1786652, 15.1767502
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7195816, 24.7158165
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8286896, 22.8294983
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0057068, 31.0047607
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3564148, 25.3481407
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9881668, 25.9814377
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1047134, 28.1041794
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1077652, 32.1100235
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7833176, 35.7835464
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.0946350, 37.0950928
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1796074, 21.1838188
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2575760, 22.2593918
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6819649, 19.6888638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 940

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 531

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6572139, upper bound: 10.6427451
time: 24.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6530096, upper bound: 10.6469430
time: 27.06 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 53.34 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6281527, upper bound: 10.6286720
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6296395, upper bound: 10.6268561
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6505890, upper bound: 10.6438147
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6444635, upper bound: 10.6499414
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6442973, upper bound: 10.6504290
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6423139, upper bound: 10.6518306
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6469409, upper bound: 10.6537374
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6469409, upper bound: 10.6537374
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6557267, upper bound: 10.6490772
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6515208, upper bound: 10.6532852
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6503444, upper bound: 10.6525855
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6569315, upper bound: 10.6459989
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6527984, upper bound: 10.6496390
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6503583, upper bound: 10.6520788
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6572139, upper bound: 10.6427451
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 53.34
Output dim: 25, lower bound: -10.6530096, upper bound: 10.6469430

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4935608, 19.4961166
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3177872, 24.3193665
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3051987, 21.3058853
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.1999130, 25.2046127
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8226089, 20.8245430
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0570755, 24.0585098
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4035950, 25.4066620
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7659149, 20.7611923
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6462021, 29.6437302
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7778244, 23.7731781
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3645935, 33.3592377
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6993866, 32.6973953
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.7550812, 36.7498856
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3206863, 23.3213959
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4420776, 29.4439316
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5430183, 16.5420341
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7200241, 17.7193489
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.1848183, 20.1811295
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7533073, 18.7522736
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5407066, 17.5409050
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9382629, 19.9388771
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4326668, 19.4316521
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8227310, 23.8265457
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5924301, 19.5894814
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1888313, 15.1880951
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7260132, 24.7207909
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8640862, 22.8658562
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -30.9626999, 30.9675674
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3571701, 25.3601608
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9665833, 25.9642639
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.0884933, 28.0900955
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1232758, 32.1282120
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7764053, 35.7782669
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1090012, 37.1134338
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2090492, 21.2150536
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2899475, 22.2942123
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7102127, 19.7141075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1644

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1548

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6441605, upper bound: 10.6376229
time: 32.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6443554, upper bound: 10.6374272
time: 19.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4940186, 19.4956589
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3180542, 24.3190994
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3055496, 21.3055344
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2002716, 25.2042542
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8238983, 20.8232498
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0571747, 24.0584183
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4033127, 25.4069443
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7658463, 20.7612686
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6457748, 29.6441498
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7769241, 23.7740784
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3646851, 33.3591461
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6998444, 32.6969299
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.7536621, 36.7513199
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3197250, 23.3223572
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4421005, 29.4439011
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5429306, 16.5421219
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7200089, 17.7193642
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.1847420, 20.1812134
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7527161, 18.7528648
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5400429, 17.5415688
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9374161, 19.9397278
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4313774, 19.4329414
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8221588, 23.8271103
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5920715, 19.5898438
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1878662, 15.1890583
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7249680, 24.7218361
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8651085, 22.8648338
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -30.9638748, 30.9664001
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3577042, 25.3596306
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9676743, 25.9631729
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.0893555, 28.0892258
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1237335, 32.1277542
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7771835, 35.7774963
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1104965, 37.1119461
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2099648, 21.2141304
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2912292, 22.2929306
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7102966, 19.7140217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 702

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 619

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6442917, upper bound: 10.6392429
time: 31.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6337528, upper bound: 10.6497690
time: 41.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4932022, 19.4908142
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3168945, 24.3182297
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3015518, 21.3003235
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2002487, 25.1999283
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8341827, 20.8311768
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0546875, 24.0577164
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4034653, 25.4092865
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7582855, 20.7650909
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6455765, 29.6494446
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7721100, 23.7791290
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3605957, 33.3659973
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7016449, 32.7008286
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.7515564, 36.7702560
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3241501, 23.3264961
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4436646, 29.4418259
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5423698, 16.5416641
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7193604, 17.7218819
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.1786041, 20.1893578
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7543449, 18.7535858
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5450211, 17.5457687
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9434738, 19.9412155
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4406433, 19.4427605
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8266602, 23.8283768
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5935287, 19.5922928
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1858425, 15.1866665
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7247925, 24.7300835
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8721848, 22.8688202
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -30.9770279, 30.9696121
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3665543, 25.3569260
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9754791, 25.9703445
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.0981445, 28.0920486
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1325531, 32.1286240
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7837448, 35.7804718
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1230545, 37.1168365
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2218018, 21.2162666
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.3040504, 22.2981758
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7101135, 19.7136173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 964

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 621

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6354371, upper bound: 10.6330609
time: 28.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6269428, upper bound: 10.6415605
time: 30.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4934235, 19.4905396
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3169098, 24.3182068
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3023834, 21.2992630
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2004013, 25.1997910
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8325043, 20.8324928
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0554199, 24.0568924
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4053116, 25.4071350
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7593613, 20.7637711
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6456985, 29.6493530
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7720490, 23.7792053
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3600922, 33.3664169
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7016754, 32.7007675
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.7546844, 36.7665405
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3254089, 23.3249588
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4434586, 29.4420090
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5424042, 16.5414810
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7193680, 17.7217789
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.1786804, 20.1882935
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7543793, 18.7530327
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5448685, 17.5458221
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9434891, 19.9412155
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4407043, 19.4418182
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8267288, 23.8275681
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5930481, 19.5926819
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1858654, 15.1865063
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7247925, 24.7300758
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8710403, 22.8697510
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -30.9765625, 30.9701691
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3637543, 25.3592148
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9737625, 25.9718704
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.0967636, 28.0932693
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1320038, 32.1291046
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7814713, 35.7823029
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1230850, 37.1165619
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2216949, 21.2163734
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.3019600, 22.2997246
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7095184, 19.7136803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1686

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1629

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6277912, upper bound: 10.6505632
time: 26.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6416290, upper bound: 10.6366988
time: 26.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4928741, 19.4900513
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3164520, 24.3178253
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3017502, 21.2993622
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.1994934, 25.1989822
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8288994, 20.8265381
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0528641, 24.0541992
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4042587, 25.4076233
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7577896, 20.7632980
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6443710, 29.6488647
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7677765, 23.7733536
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3604965, 33.3664093
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7006378, 32.7003555
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.7534409, 36.7682724
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3196793, 23.3215828
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4429474, 29.4410095
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5419235, 16.5409126
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7159271, 17.7174416
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.1767960, 20.1860657
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7515869, 18.7507973
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5437317, 17.5447311
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9433746, 19.9410591
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4397278, 19.4410477
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8252563, 23.8251610
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5934029, 19.5925941
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1853657, 15.1861610
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7228012, 24.7275581
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8707161, 22.8685684
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -30.9698792, 30.9650574
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3592377, 25.3536835
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9698639, 25.9677353
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.0977783, 28.0931244
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1277924, 32.1258240
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7830048, 35.7819061
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1196365, 37.1143799
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2214813, 21.2163391
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.3029976, 22.2993431
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7090645, 19.7134266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1459

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1522

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6466773, upper bound: 10.6527156
time: 33.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6459206, upper bound: 10.6534734
time: 35.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4926529, 19.4902725
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3164902, 24.3177795
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3013687, 21.2997360
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.1993713, 25.1991043
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.8276329, 20.8278008
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0519867, 24.0550766
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4039154, 25.4079666
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7578049, 20.7632828
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6451492, 29.6480865
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7667770, 23.7743530
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3604965, 33.3664093
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7011261, 32.6998520
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.7532730, 36.7684402
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3207245, 23.3205376
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4427795, 29.4411850
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5418167, 16.5410194
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7150345, 17.7183342
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.1763840, 20.1864853
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7521820, 18.7502022
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5442200, 17.5442390
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9433746, 19.9410553
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4399414, 19.4408264
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8242645, 23.8261604
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5934181, 19.5925903
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1855221, 15.1860046
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7222977, 24.7280540
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8705940, 22.8686790
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -30.9714737, 30.9634552
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3605499, 25.3523788
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9711456, 25.9664536
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.0978317, 28.0930710
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1292114, 32.1244125
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7830048, 35.7819061
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1203995, 37.1136169
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2214813, 21.2163315
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.3031044, 22.2992363
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7092743, 19.7132149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1454

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 964

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6341495, upper bound: 10.6515449
time: 27.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6447822, upper bound: 10.6409224
time: 27.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4990005, 19.4969635
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3327179, 24.3361588
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3003845, 21.2978516
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2231903, 25.2245789
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7915344, 20.7968216
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0337982, 24.0349350
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3963623, 25.3967743
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7857208, 20.7846298
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6514435, 29.6555862
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7941513, 23.7957458
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3732986, 33.3771591
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6861572, 32.6856079
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8159485, 36.8181458
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3331146, 23.3315048
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4230423, 29.4211349
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5504150, 16.5485840
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7244797, 17.7254219
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2257462, 20.2278481
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7577591, 18.7552185
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5341949, 17.5330925
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9388275, 19.9366646
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4512634, 19.4501534
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8460388, 23.8444290
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5888939, 19.5874062
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1811371, 15.1807747
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7148514, 24.7188148
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8243103, 22.8278503
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -30.9947662, 30.9970779
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3326340, 25.3415489
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9660110, 25.9690552
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1013412, 28.1009598
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1007233, 32.1007996
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7852631, 35.7869110
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.0859680, 37.0855713
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1806335, 21.1829720
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2508354, 22.2533073
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6819725, 19.6872883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1564

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 947

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6353361, upper bound: 10.6409939
time: 26.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6476226, upper bound: 10.6287127
time: 29.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4990616, 19.4969025
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3329163, 24.3359604
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3000565, 21.2981796
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2231140, 25.2246552
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7915802, 20.7967758
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0330048, 24.0357285
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3954315, 25.3977051
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7861938, 20.7841568
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6515045, 29.6555252
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7928391, 23.7970581
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3737259, 33.3767395
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6873627, 32.6843948
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8145752, 36.8195114
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3329391, 23.3316803
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4223099, 29.4218674
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5502968, 16.5487022
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7240601, 17.7258415
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2252426, 20.2283516
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7572708, 18.7557087
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5341873, 17.5330963
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9380035, 19.9374924
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4509277, 19.4505005
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8441772, 23.8462906
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5888939, 19.5874062
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1808891, 15.1810226
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7144547, 24.7192192
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8250046, 22.8271561
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -30.9970856, 30.9947586
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3343887, 25.3397942
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9682846, 25.9667816
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1020966, 28.1002045
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1012268, 32.1002884
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7854767, 35.7866898
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.0872955, 37.0842438
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1809921, 21.1826134
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2513390, 22.2528076
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6822968, 19.6869640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 857

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1282

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6481592, upper bound: 10.6500018
time: 29.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6481592, upper bound: 10.6500018
time: 29.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4989853, 19.4976730
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3329315, 24.3366699
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.2999878, 21.3008957
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2232285, 25.2263107
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7947006, 20.7984657
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0368042, 24.0407486
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3922043, 25.3976364
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7853088, 20.7843781
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6523819, 29.6562576
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8060913, 23.8089180
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3780518, 33.3802795
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6958771, 32.6931610
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8004684, 36.8111877
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3370056, 23.3378448
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4274979, 29.4266815
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5482597, 16.5472145
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7232933, 17.7258453
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2193451, 20.2249222
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7442169, 18.7437668
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5375900, 17.5355644
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9408875, 19.9405632
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4447021, 19.4450226
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8535385, 23.8552551
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5895309, 19.5862465
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1730194, 15.1741676
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7064590, 24.7118073
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8241768, 22.8253365
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0040054, 31.0018616
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3433762, 25.3465271
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9803314, 25.9791031
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.0996475, 28.0958328
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.0913544, 32.0885696
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7773361, 35.7754440
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.0896378, 37.0866699
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1698799, 21.1687927
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2383995, 22.2360268
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6673393, 19.6696281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 994

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1613

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6478578, upper bound: 10.6522221
time: 28.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6499724, upper bound: 10.6501060
time: 27.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4985046, 19.4981537
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3328705, 24.3367386
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3004608, 21.3004227
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2232437, 25.2262955
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7945709, 20.7986031
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0371246, 24.0404205
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3925552, 25.3972855
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7852249, 20.7844620
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6525955, 29.6560440
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8064728, 23.8085327
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3781128, 33.3802185
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6955414, 32.6935120
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8032608, 36.8083954
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3372803, 23.3375702
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4276276, 29.4265594
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5484810, 16.5469933
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7240868, 17.7250519
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2209015, 20.2233696
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7454453, 18.7425385
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5368729, 17.5362892
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9414215, 19.9400291
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4457932, 19.4439354
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8539886, 23.8548050
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5895386, 19.5862427
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1743546, 15.1728325
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7083817, 24.7098923
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8231468, 22.8263702
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0023804, 31.0034866
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3421555, 25.3477440
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9797287, 25.9797058
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.0984039, 28.0970764
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.0890961, 32.0908432
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7764969, 35.7762833
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.0879898, 37.0883179
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1674919, 21.1711845
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2358894, 22.2385368
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6656914, 19.6712761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1719

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1561

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6562683, upper bound: 10.6455873
time: 29.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6565211, upper bound: 10.6453342
time: 27.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4986801, 19.5005379
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3381577, 24.3381195
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3054886, 21.3059082
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2293396, 25.2321320
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7877083, 20.7886810
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0427551, 24.0454483
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3951416, 25.3992615
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7827682, 20.7834396
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6484680, 29.6454468
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8078766, 23.8097229
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3843689, 33.3818436
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6975861, 32.6963654
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8055573, 36.8128738
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3308716, 23.3291512
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4259949, 29.4275436
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5470505, 16.5477180
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7298355, 17.7322807
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2248650, 20.2297134
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7505379, 18.7491760
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5409088, 17.5410156
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9387054, 19.9390182
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4414444, 19.4419327
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8560638, 23.8605728
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5883713, 19.5871925
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1798382, 15.1803284
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7133179, 24.7119179
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8264160, 22.8232803
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0076065, 31.0009155
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3570786, 25.3447418
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9862976, 25.9790497
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1004639, 28.0979691
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1108093, 32.1067810
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7761307, 35.7765045
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.0936737, 37.0903320
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1814156, 21.1774445
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2536545, 22.2501144
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6840858, 19.6843319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1604

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 767

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6520296, upper bound: 10.6493383
time: 30.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6524980, upper bound: 10.6488699
time: 27.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4991302, 19.5000954
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3383560, 24.3379211
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3055801, 21.3058167
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2297592, 25.2317123
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7884560, 20.7879219
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0427628, 24.0454407
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3948822, 25.3995209
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7824478, 20.7837601
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6480408, 29.6458969
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8076859, 23.8099136
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3842163, 33.3820038
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6976776, 32.6962662
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8040924, 36.8143463
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3301010, 23.3299217
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4262772, 29.4272614
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5470352, 16.5477295
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7298203, 17.7322998
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2245827, 20.2300034
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7500267, 18.7496891
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5408478, 17.5410767
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9385071, 19.9392128
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4407578, 19.4426193
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8559418, 23.8606873
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5882339, 19.5873260
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1792355, 15.1809349
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7126160, 24.7126274
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8269501, 22.8227386
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0076981, 31.0008240
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3572388, 25.3445816
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9865417, 25.9787979
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1011505, 28.0972824
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1108398, 32.1067581
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7770309, 35.7756042
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.0940399, 37.0899506
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1816521, 21.1772079
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2545242, 22.2492332
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6842232, 19.6841927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1281

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 553

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6463891, upper bound: 10.6514281
time: 30.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6497099, upper bound: 10.6481071
time: 32.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4968185, 19.4983330
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3347855, 24.3350220
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3024063, 21.3005295
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2243423, 25.2263107
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7953873, 20.7944412
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0345306, 24.0362930
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3981781, 25.4015503
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7832489, 20.7853622
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6527100, 29.6507797
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7941284, 23.7945213
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3772278, 33.3754578
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6859894, 32.6860580
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8090668, 36.8112411
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3333664, 23.3337898
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4204407, 29.4203491
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5467491, 16.5455017
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7246017, 17.7251167
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2204819, 20.2215614
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7516022, 18.7482529
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5333405, 17.5347137
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9353256, 19.9333305
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4436188, 19.4429283
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8435440, 23.8462524
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5893211, 19.5877609
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1767197, 15.1745548
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7137680, 24.7095985
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8229942, 22.8244896
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -30.9851303, 30.9865036
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3384323, 25.3319092
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9694366, 25.9649811
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.0985489, 28.0987701
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1005630, 32.1033249
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7840271, 35.7844696
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.0793533, 37.0811310
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1752052, 21.1797714
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2533798, 22.2556953
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.6832047, 19.6904240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 589

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 939

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6498362, upper bound: 10.6372009
time: 28.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6430451, upper bound: 10.6353330
time: 30.84 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 61.39 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6441605, upper bound: 10.6376229
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6443554, upper bound: 10.6374272
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6442917, upper bound: 10.6392429
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6337528, upper bound: 10.6497690
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6354371, upper bound: 10.6330609
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6269428, upper bound: 10.6415605
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6277912, upper bound: 10.6505632
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6416290, upper bound: 10.6366988
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6466773, upper bound: 10.6527156
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6459206, upper bound: 10.6534734
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6341495, upper bound: 10.6515449
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6447822, upper bound: 10.6409224
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6353361, upper bound: 10.6409939
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6476226, upper bound: 10.6287127
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6481592, upper bound: 10.6500018
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6481592, upper bound: 10.6500018
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6478578, upper bound: 10.6522221
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6499724, upper bound: 10.6501060
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6562683, upper bound: 10.6455873
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6565211, upper bound: 10.6453342
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6520296, upper bound: 10.6493383
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6524980, upper bound: 10.6488699
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6463891, upper bound: 10.6514281
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6497099, upper bound: 10.6481071
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6498362, upper bound: 10.6372009
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 61.39
Output dim: 25, lower bound: -10.6430451, upper bound: 10.6353330
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 61.39
Output dim: 25, lower bound: -10.6530096, upper bound: 10.6469430

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 46.49 + 1809.73 = 1856.22 seconds
