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
execution time: IAR + RelationalAnalysis = 2.61 + 43.96 = 46.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 25, lower bound: -10.6599395, upper bound: 10.6599395

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1487

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 604

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6590250, upper bound: 10.6421912
time: 31.01 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6421912, upper bound: 10.6590250
time: 28.96 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 60.10 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 60.10
Output dim: 25, lower bound: -10.6590250, upper bound: 10.6421912
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 60.10
Output dim: 25, lower bound: -10.6421912, upper bound: 10.6590250

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.5110092, 19.5110855
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3366318, 24.3367004
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3084106, 21.3092575
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2407303, 25.2403030
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7975388, 20.7975693
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0694199, 24.0692902
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4320755, 25.4326401
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7967911, 20.7996368
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6821976, 29.6821823
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8088837, 23.8052101
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3981323, 33.3987808
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7102203, 32.7130432
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8136749, 36.8138199
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3410416, 23.3412094
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4240494, 29.4226837
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5472107, 16.5471458
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7384148, 17.7376976
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2469749, 20.2469254
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7592964, 18.7589474
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5379181, 17.5360413
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9388428, 19.9369926
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4483109, 19.4482727
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8566513, 23.8549309
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5932579, 19.5930367
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1875744, 15.1871204
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7445679, 24.7431602
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8543243, 22.8548050
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0243607, 31.0280609
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.4047852, 25.4046783
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9976807, 25.9993744
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1225967, 28.1230087
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1173553, 32.1184692
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7803268, 35.7791824
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1163559, 37.1195297
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1948013, 21.1956520
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2719421, 22.2720757
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7283516, 19.7288418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1487

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1661

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6367801, upper bound: 10.6414560
time: 32.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6583042, upper bound: 10.6199208
time: 26.93 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.5110855, 19.5110054
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3367004, 24.3366318
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.3092499, 21.3084106
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2403030, 25.2407303
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7975693, 20.7975388
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0692902, 24.0694199
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.4326401, 25.4320755
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7996368, 20.7967911
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6821823, 29.6822052
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8052139, 23.8088799
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.3987732, 33.3981400
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7130585, 32.7102203
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8138275, 36.8136749
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3412094, 23.3410416
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4226837, 29.4240494
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5471458, 16.5472107
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7376976, 17.7384186
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2469215, 20.2469788
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7589455, 18.7592983
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5360413, 17.5379181
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9369888, 19.9388390
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4482727, 19.4483109
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8549271, 23.8566551
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5930367, 19.5932617
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1871204, 15.1875725
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7431641, 24.7445641
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8548050, 22.8543243
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0280609, 31.0243607
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.4046783, 25.4047852
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9993744, 25.9976807
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1230087, 28.1226044
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1184692, 32.1173477
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7791824, 35.7803192
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1195297, 37.1163483
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.1956558, 21.1948051
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2720718, 22.2719421
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7288437, 19.7283516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1487

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1661

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6199208, upper bound: 10.6583042
time: 30.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6414560, upper bound: 10.6367801
time: 28.94 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 61.22 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 61.22
Output dim: 25, lower bound: -10.6367801, upper bound: 10.6414560
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 61.22
Output dim: 25, lower bound: -10.6583042, upper bound: 10.6199208
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 61.22
Output dim: 25, lower bound: -10.6199208, upper bound: 10.6583042
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 61.22
Output dim: 25, lower bound: -10.6414560, upper bound: 10.6367801

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.5009613, 19.5022812
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3251572, 24.3266449
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.2854691, 21.2891083
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2321320, 25.2322845
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7982292, 20.7980881
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0466461, 24.0494766
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3883362, 25.3942413
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7631378, 20.7698975
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6809311, 29.6813354
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.8001404, 23.7962723
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.4002762, 33.4008102
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7019653, 32.7059097
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8489761, 36.8558807
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3270645, 23.3288994
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4141922, 29.4117432
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5466423, 16.5466118
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7365494, 17.7357826
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2621994, 20.2646904
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7511673, 18.7496796
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5263901, 17.5234680
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9222107, 19.9180717
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4391251, 19.4383430
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8631363, 23.8618011
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5762482, 19.5736465
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1728172, 15.1704521
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7332687, 24.7303085
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8562889, 22.8566055
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0338593, 31.0386276
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3768158, 25.3728371
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9862289, 25.9861526
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1099548, 28.1085739
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1335297, 32.1358337
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7570267, 35.7526016
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1475677, 37.1553955
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2140579, 21.2178841
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2669983, 22.2664795
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7233505, 19.7237091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1487

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1628

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6426312, upper bound: 10.6178091
time: 27.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6572205, upper bound: 10.6136290
time: 23.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.5022812, 19.5009613
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3266449, 24.3251572
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.2891083, 21.2854691
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2322845, 25.2321320
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7980919, 20.7982292
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0494766, 24.0466461
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3942413, 25.3883362
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7698975, 20.7631378
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6813278, 29.6809311
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7962723, 23.8001404
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.4008102, 33.4002762
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.7059021, 32.7019653
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8558731, 36.8489761
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3289032, 23.3270645
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4117432, 29.4141922
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5466118, 16.5466423
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7357864, 17.7365532
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2646866, 20.2622070
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7496796, 18.7511673
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5234604, 17.5263901
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9180756, 19.9222107
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4383392, 19.4391251
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8617935, 23.8631363
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5736465, 19.5762520
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1704521, 15.1728172
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7303085, 24.7332687
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8566093, 22.8562927
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0386276, 31.0338593
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3728333, 25.3768120
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9861526, 25.9862289
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1085739, 28.1099548
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1358490, 32.1335220
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7526016, 35.7570267
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1553955, 37.1475677
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2178879, 21.2140579
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2664795, 22.2669983
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7237091, 19.7233524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1487

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1628

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6136290, upper bound: 10.6572205
time: 32.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6178091, upper bound: 10.6426312
time: 29.93 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 64.01 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 64.01
Output dim: 25, lower bound: -10.6426312, upper bound: 10.6178091
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 64.01
Output dim: 25, lower bound: -10.6572205, upper bound: 10.6136290
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 64.01
Output dim: 25, lower bound: -10.6136290, upper bound: 10.6572205
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 64.01
Output dim: 25, lower bound: -10.6178091, upper bound: 10.6426312

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4963150, 19.4981346
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3220673, 24.3236923
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.2793427, 21.2837219
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2328949, 25.2329865
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7984161, 20.7983589
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0437851, 24.0467834
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3802109, 25.3870621
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7505417, 20.7587814
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6803894, 29.6807404
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7910843, 23.7861099
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.4007721, 33.4013138
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6931686, 32.6981049
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8488007, 36.8557205
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3250580, 23.3269882
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4076080, 29.4045486
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5448303, 16.5446625
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7312088, 17.7298698
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2582016, 20.2603683
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7454567, 18.7435913
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5221939, 17.5187683
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9139328, 19.9086952
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4313889, 19.4295540
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8593826, 23.8577728
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5719681, 19.5688667
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1638680, 15.1602707
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7232666, 24.7189255
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8593826, 22.8602638
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0368195, 31.0436783
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3778229, 25.3737488
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9876785, 25.9880142
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1108017, 28.1096039
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1412506, 32.1452866
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7516937, 35.7464981
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1562805, 37.1670685
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2190590, 21.2256584
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2714233, 22.2714424
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7215462, 19.7223949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1487

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1660

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6516671, upper bound: 10.6128630
time: 28.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6564680, upper bound: 10.6083351
time: 27.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4981308, 19.4963188
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3237000, 24.3220749
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.2837219, 21.2793427
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2329865, 25.2328949
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7983551, 20.7984085
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0467834, 24.0437851
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3870621, 25.3802109
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7587814, 20.7505417
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6807404, 29.6803894
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7861099, 23.7910805
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.4013214, 33.4007645
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6981125, 32.6931686
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8557281, 36.8488083
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3269882, 23.3250580
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.4045486, 29.4076004
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5446625, 16.5448303
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7298660, 17.7312088
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2603683, 20.2582092
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7435913, 18.7454567
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5187683, 17.5221939
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9086914, 19.9139328
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4295578, 19.4313965
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8577728, 23.8593826
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5688705, 19.5719643
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1602707, 15.1638660
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7189255, 24.7232666
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8602676, 22.8593788
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0436783, 31.0368271
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3737488, 25.3778267
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9880142, 25.9876785
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1096039, 28.1108017
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1452789, 32.1412506
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7465057, 35.7516937
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1670685, 37.1562805
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2256660, 21.2190590
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2714462, 22.2714233
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7223930, 19.7215443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1487

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1660

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6083351, upper bound: 10.6564680
time: 21.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6128630, upper bound: 10.6516671
time: 26.95 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 50.79 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.79
Output dim: 25, lower bound: -10.6516671, upper bound: 10.6128630
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.79
Output dim: 25, lower bound: -10.6564680, upper bound: 10.6083351
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.79
Output dim: 25, lower bound: -10.6083351, upper bound: 10.6564680
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.79
Output dim: 25, lower bound: -10.6128630, upper bound: 10.6516671

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4953918, 19.4970970
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3185730, 24.3200378
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.2752380, 21.2791138
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2376099, 25.2383423
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7824860, 20.7855988
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0448151, 24.0477371
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3719940, 25.3776245
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7292252, 20.7350082
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6737747, 29.6735535
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7891998, 23.7843361
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.4014130, 33.4020233
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6798553, 32.6828766
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8497009, 36.8539963
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3166428, 23.3175735
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.3908768, 29.3884506
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5440559, 16.5438538
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7260895, 17.7247543
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2581596, 20.2603111
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7456894, 18.7438335
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5192032, 17.5158958
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9057312, 19.9008217
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4341888, 19.4319801
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8471298, 23.8458977
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5719299, 19.5688324
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1600208, 15.1568222
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7210541, 24.7173500
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8540268, 22.8553581
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0411911, 31.0481491
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3687057, 25.3655014
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9897614, 25.9902420
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1055679, 28.1049118
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1418304, 32.1459198
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7273560, 35.7250977
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1665268, 37.1764450
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2199669, 21.2266312
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2569580, 22.2585068
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7198372, 19.7208900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1487

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6386899, upper bound: 10.6099008
time: 25.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6490441, upper bound: 10.6038556
time: 28.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4952774, 19.4972115
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3184128, 24.3201981
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.2747421, 21.2796021
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2382965, 25.2376938
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7856445, 20.7824402
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0447311, 24.0478058
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3707809, 25.3788452
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7267609, 20.7374725
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6731949, 29.6741638
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7893753, 23.7842255
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.4014740, 33.4019699
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6779327, 32.6847916
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8470764, 36.8565369
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3156433, 23.3185806
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.3916702, 29.3878250
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5440216, 16.5438881
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7261276, 17.7247505
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2581520, 20.2603188
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7457008, 18.7438259
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5193710, 17.5157738
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9061127, 19.9004974
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4338226, 19.4323502
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8474960, 23.8455315
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5719299, 19.5688324
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1604176, 15.1564255
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7216797, 24.7167244
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8544693, 22.8549156
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0412979, 31.0481339
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3695755, 25.3646240
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9899063, 25.9900970
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1061020, 28.1043701
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1418915, 32.1458969
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7302856, 35.7221756
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1656723, 37.1773453
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2200279, 21.2266273
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2584686, 22.2569809
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7200470, 19.7206879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1487

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6405542, upper bound: 10.6051632
time: 25.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6540356, upper bound: 10.6015285
time: 27.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4972076, 19.4952812
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3201981, 24.3184128
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.2796021, 21.2747421
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2377014, 25.2382965
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7824402, 20.7856483
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0478058, 24.0447235
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3788452, 25.3707809
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7374725, 20.7267609
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6741714, 29.6731949
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7842255, 23.7893753
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.4019623, 33.4014740
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6847992, 32.6779327
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8565369, 36.8470764
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3185806, 23.3156433
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.3878250, 29.3916702
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5438881, 16.5440216
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7247543, 17.7261314
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2603188, 20.2581558
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7438278, 18.7457008
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5157700, 17.5193672
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9004974, 19.9061089
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4323502, 19.4338226
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8455276, 23.8474998
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5688324, 19.5719299
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1564274, 15.1604176
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7167206, 24.7216835
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8549118, 22.8544655
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0481339, 31.0412979
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3646317, 25.3695793
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9900970, 25.9899063
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1043701, 28.1061020
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1458893, 32.1418762
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7221680, 35.7302933
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1773605, 37.1656570
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2266197, 21.2200279
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2569733, 22.2584610
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7206879, 19.7200451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1487

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6015285, upper bound: 10.6540356
time: 30.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6051632, upper bound: 10.6405542
time: 29.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4970932, 19.4953957
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3200378, 24.3185730
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.2791214, 21.2752380
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2383423, 25.2376099
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7855988, 20.7824898
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0477371, 24.0448151
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3776245, 25.3719940
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7350082, 20.7292252
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6735611, 29.6737823
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7843399, 23.7891960
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.4020233, 33.4014206
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6828766, 32.6798553
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8540039, 36.8496857
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3175735, 23.3166466
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.3884506, 29.3908768
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5438538, 16.5440559
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7247543, 17.7260933
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2603111, 20.2581596
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7438354, 18.7456913
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5158920, 17.5191994
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9008179, 19.9057350
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4319763, 19.4341888
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8458939, 23.8471336
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5688324, 19.5719299
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1568241, 15.1600189
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7173462, 24.7210579
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8553543, 22.8540306
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0481491, 31.0411911
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3655014, 25.3687019
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9902420, 25.9897537
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1049118, 28.1055679
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1459198, 32.1418304
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7250977, 35.7273636
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1764450, 37.1665268
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2266350, 21.2199631
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2585144, 22.2569580
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7208900, 19.7198372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1487

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6038556, upper bound: 10.6490441
time: 27.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6099008, upper bound: 10.6386899
time: 35.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 64.77 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 64.77
Output dim: 25, lower bound: -10.6386899, upper bound: 10.6099008
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 64.77
Output dim: 25, lower bound: -10.6490441, upper bound: 10.6038556
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 64.77
Output dim: 25, lower bound: -10.6405542, upper bound: 10.6051632
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 64.77
Output dim: 25, lower bound: -10.6540356, upper bound: 10.6015285
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 64.77
Output dim: 25, lower bound: -10.6015285, upper bound: 10.6540356
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 64.77
Output dim: 25, lower bound: -10.6051632, upper bound: 10.6405542
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 64.77
Output dim: 25, lower bound: -10.6038556, upper bound: 10.6490441
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 64.77
Output dim: 25, lower bound: -10.6099008, upper bound: 10.6386899

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4949112, 19.4969864
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3181534, 24.3199768
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.2746201, 21.2795258
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2382889, 25.2376862
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7854767, 20.7822304
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0442581, 24.0475006
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3699112, 25.3783035
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7264252, 20.7372360
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6731720, 29.6741409
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7893295, 23.7841568
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.4014130, 33.4018936
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6778870, 32.6847763
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8470535, 36.8565445
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3154678, 23.3184395
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.3915024, 29.3876190
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5438538, 16.5437012
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7256775, 17.7241859
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2579536, 20.2600746
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7449341, 18.7429123
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5193291, 17.5157089
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9058685, 19.9001846
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4331131, 19.4313736
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8474731, 23.8455467
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5715332, 19.5683174
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1600208, 15.1557732
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7213440, 24.7162132
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8543663, 22.8549042
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0411682, 31.0482788
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3695374, 25.3644943
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9899063, 25.9901047
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1060638, 28.1043167
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1416245, 32.1461182
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7301788, 35.7219925
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1655273, 37.1776962
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2195358, 21.2267036
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2583389, 22.2569427
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7192154, 19.7200165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1487

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 636

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6452482, upper bound: 10.5919538
time: 28.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6398456, upper bound: 10.5944391
time: 25.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.5315895, 8.6601019, -22.5315895, 8.6601019, -31.1916924, 31.1916924
1: -11.8821363, 6.3869190, -11.8821363, 6.3869190, -18.2690544, 18.2690544
2: -13.1332664, 7.4625759, -13.1332664, 7.4625759, -19.4969864, 19.4949112
3: -18.6608334, 6.2980185, -18.6608334, 6.2980185, -24.3199768, 24.3181534
4: -20.2854996, 3.3015978, -20.2854996, 3.3015978, -21.2795258, 21.2746124
5: -18.1325493, 7.6051707, -18.1325493, 7.6051707, -25.2376785, 25.2382889
6: -36.7454491, -11.2355909, -36.7454491, -11.2355909, -20.7822266, 20.7854843
7: -24.1228085, 1.3723805, -24.1228085, 1.3723805, -24.0474930, 24.0442581
8: -27.3807297, 1.1222410, -27.3807297, 1.1222410, -25.3783035, 25.3699112
9: -11.5790672, 11.8656340, -11.5790672, 11.8656340, -20.7372360, 20.7264252
10: -17.7104530, 12.3543253, -17.7104530, 12.3543253, -29.6741486, 29.6731644
11: -16.7916298, 10.4938650, -16.7916298, 10.4938650, -23.7841568, 23.7893257
12: -24.2289658, 11.2936039, -24.2289658, 11.2936039, -33.4019012, 33.4014130
13: -22.3817253, 12.3568106, -22.3817253, 12.3568106, -32.6847687, 32.6778870
14: -34.9231949, 6.3831935, -34.9231949, 6.3831935, -36.8565445, 36.8470612
15: -8.7229404, 16.7796345, -8.7229404, 16.7796345, -23.3184433, 23.3154678
16: -22.7888603, 3.1424036, -22.7888603, 3.1424036, -25.9312630, 25.9312630
17: -28.0651875, 8.0703468, -28.0651875, 8.0703468, -36.1355362, 36.1355362
18: -12.6017017, 18.6201897, -12.6017017, 18.6201897, -29.3876190, 29.3915024
19: -8.9562683, 8.0431147, -8.9562683, 8.0431147, -16.5437012, 16.5438538
20: -9.9384737, 8.7778578, -9.9384737, 8.7778578, -17.7241898, 17.7256775
21: -12.5656128, 9.1593323, -12.5656128, 9.1593323, -20.2600746, 20.2579536
22: -2.8050990, 18.5131264, -2.8050990, 18.5131264, -18.7429123, 18.7449341
23: -3.9387264, 15.2583456, -3.9387264, 15.2583456, -17.5157127, 17.5193329
24: -5.5288115, 17.3602905, -5.5288115, 17.3602905, -19.9001846, 19.9058685
25: 2.0597486, 24.2907677, 2.0597486, 24.2907677, -19.4313736, 19.4331093
26: -11.7230453, 21.5887852, -11.7230453, 21.5887852, -33.3118286, 33.3118286
27: -15.0052557, 10.0010033, -15.0052557, 10.0010033, -23.8455505, 23.8474770
28: -3.0131588, 18.0299377, -3.0131588, 18.0299377, -19.5683136, 19.5715332
29: -3.3690634, 15.5123568, -3.3690634, 15.5123568, -15.1557713, 15.1600208
30: -13.5995684, 13.7980433, -13.5995684, 13.7980433, -24.7162170, 24.7213402
31: -9.7060299, 11.3188410, -9.7060299, 11.3188410, -21.0248718, 21.0248718
32: -30.7154942, -3.7853260, -30.7154942, -3.7853260, -22.8549004, 22.8543701
33: -41.5152588, -3.2222323, -41.5152588, -3.2222323, -31.0482712, 31.0411758
34: -36.7941132, -3.8487487, -36.7941132, -3.8487487, -25.3645020, 25.3695412
35: -24.9038906, 5.5323420, -24.9038906, 5.5323420, -25.9901047, 25.9899063
36: -24.5676956, 6.4279532, -24.5676956, 6.4279532, -28.1043167, 28.1060638
37: -42.8792114, -6.5617008, -42.8792114, -6.5617008, -32.1461105, 32.1416168
38: -34.3848343, 3.4991555, -34.3848343, 3.4991555, -35.7220001, 35.7301865
39: -47.7359390, -7.0786939, -47.7359390, -7.0786939, -37.1777039, 37.1655273
40: -45.9587326, -18.9628315, -45.9587326, -18.9628315, -21.2267075, 21.2195320
41: -33.4887924, -4.6557546, -33.4887924, -4.6557546, -22.2569427, 22.2583351
42: -24.0458565, -0.4308560, -24.0458565, -0.4308560, -19.7200165, 19.7192154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=163, inp2_unstable=163, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1487

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 636

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.5919538, upper bound: 10.6398456
time: 56.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.5919538, upper bound: 10.6452482
time: 28.25 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 87.28 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 87.28
Output dim: 25, lower bound: -10.6452482, upper bound: 10.5919538
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 87.28
Output dim: 25, lower bound: -10.6398456, upper bound: 10.5944391
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 87.28
Output dim: 25, lower bound: -10.5919538, upper bound: 10.6398456
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 87.28
Output dim: 25, lower bound: -10.5919538, upper bound: 10.6452482

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 46.56 + 790.89 = 837.46 seconds
