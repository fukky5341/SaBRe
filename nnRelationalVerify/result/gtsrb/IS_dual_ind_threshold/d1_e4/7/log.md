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
execution time: IAR + RelationalAnalysis = 2.45 + 43.95 = 46.41 seconds
status: Status.UNKNOWN
relational distance
Output dim: 25, lower bound: -10.6599395, upper bound: 10.6599395

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 874

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1645

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6408029, upper bound: 10.6593650
time: 27.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6593650, upper bound: 10.6593650
time: 26.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 53.95 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 53.95
Output dim: 25, lower bound: -10.6408029, upper bound: 10.6593650
IS_A2, status: Status.UNKNOWN, split count: 1, time: 53.95
Output dim: 25, lower bound: -10.6593650, upper bound: 10.6593650

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -22.4785347, 8.5933094, -22.5290718, 8.6215219, -31.1000557, 31.1223812
1: -11.8489523, 6.3436904, -11.8809671, 6.3621612, -18.2111130, 18.2246571
2: -13.1062031, 7.4251280, -13.1322756, 7.4412665, -19.4340820, 19.4442787
3: -18.6188927, 6.2440968, -18.6591568, 6.2677917, -24.2195663, 24.2366562
4: -20.2503338, 3.2570100, -20.2832451, 3.2768040, -21.1884995, 21.2018738
5: -18.0905457, 7.5489683, -18.1307602, 7.5738916, -25.1496582, 25.1648712
6: -36.7313995, -11.2499866, -36.7395020, -11.2417946, -20.7831230, 20.7848167
7: -24.0766563, 1.3220868, -24.1206627, 1.3437209, -23.9266968, 23.9490509
8: -27.3383808, 1.0709424, -27.3789253, 1.0936413, -25.2268753, 25.2432938
9: -11.5385618, 11.8091412, -11.5763321, 11.8334541, -20.5703659, 20.5862885
10: -17.6863480, 12.3296852, -17.7014046, 12.3461332, -29.6508865, 29.6471481
11: -16.7437706, 10.4625425, -16.7712860, 10.4915485, -23.6977463, 23.6956406
12: -24.1948662, 11.2570724, -24.2124214, 11.2888098, -33.4116211, 33.3931198
13: -22.3380451, 12.2994480, -22.3785782, 12.3252640, -32.5831528, 32.5988693
14: -34.8945389, 6.3688731, -34.9152908, 6.3791046, -36.8362503, 36.8234787
15: -8.6963139, 16.7460709, -8.7185392, 16.7613430, -23.2568359, 23.2642784
16: -22.7488480, 3.1058991, -22.7841454, 3.1219764, -25.8708248, 25.8900452
17: -28.0236244, 8.0436821, -28.0556717, 8.0656786, -36.0893021, 36.0993538
18: -12.5405607, 18.5713615, -12.5697393, 18.6178627, -29.2761765, 29.2579575
19: -8.9086113, 8.0240726, -8.9328194, 8.0421524, -16.4720230, 16.4784851
20: -9.8994694, 8.7612581, -9.9185047, 8.7767410, -17.6752472, 17.6794167
21: -12.5269699, 9.1458988, -12.5489197, 9.1574688, -20.2401886, 20.2414360
22: -2.7665558, 18.4914093, -2.7862849, 18.5120087, -18.6864853, 18.6851006
23: -3.8831911, 15.2284727, -3.9091520, 15.2567825, -17.4036217, 17.4007034
24: -5.4696226, 17.3213253, -5.4959874, 17.3592339, -19.7565765, 19.7448044
25: 2.1165781, 24.2602158, 2.0905027, 24.2895660, -19.3334160, 19.3306465
26: -11.6551762, 21.5411339, -11.6865749, 21.5860939, -33.2412720, 33.2277069
27: -14.9671755, 9.9707088, -14.9849987, 9.9989166, -23.8109970, 23.8005219
28: -2.9552116, 18.0012932, -2.9822087, 18.0283012, -19.4687653, 19.4688530
29: -3.3363476, 15.4908657, -3.3560967, 15.5107393, -15.1348495, 15.1348419
30: -13.5529976, 13.7665348, -13.5746899, 13.7953510, -24.6657333, 24.6593590
31: -9.6473989, 11.2917366, -9.6757317, 11.3165703, -20.9639702, 20.9674683
32: -30.7012081, -3.8002605, -30.7105579, -3.7918324, -22.8947372, 22.8921661
33: -41.4859962, -3.2488680, -41.5039673, -3.2335820, -31.1688538, 31.1648407
34: -36.7580566, -3.8828831, -36.7749748, -3.8507385, -25.2945175, 25.2795677
35: -24.8902397, 5.5148358, -24.8979874, 5.5297666, -25.9751358, 25.9667130
36: -24.5502834, 6.4122829, -24.5591793, 6.4255214, -28.0735626, 28.0683212
37: -42.8592377, -6.5712938, -42.8700027, -6.5637569, -32.2495880, 32.2349701
38: -34.3455467, 3.4651008, -34.3654861, 3.4958868, -35.6478271, 35.6400909
39: -47.7086067, -7.1001558, -47.7284317, -7.0855680, -37.3686066, 37.3534393
40: -45.9351654, -18.9866180, -45.9524155, -18.9721832, -21.4296532, 21.4143562
41: -33.4791260, -4.6666770, -33.4843674, -4.6593828, -22.2668648, 22.2633400
42: -24.0244293, -0.4504709, -24.0398598, -0.4402590, -19.6789970, 19.6830597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=162, inp2_unstable=163, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1629

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6384873, upper bound: 10.6435397
time: 27.13 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6384999, upper bound: 10.6580814
time: 31.82 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -22.5312290, 8.6555996, -22.5313759, 8.6574106, -31.1886406, 31.1869755
1: -11.8819637, 6.3843575, -11.8820381, 6.3851871, -18.2671509, 18.2663956
2: -13.1331758, 7.4598217, -13.1332283, 7.4609175, -19.5093460, 19.5031891
3: -18.6605530, 6.2943721, -18.6606369, 6.2958417, -24.3342438, 24.3276062
4: -20.2852364, 3.2997150, -20.2853317, 3.3004599, -21.3093567, 21.2995300
5: -18.1322994, 7.6015105, -18.1323700, 7.6030006, -25.2378998, 25.2342224
6: -36.7443428, -11.2391281, -36.7447777, -11.2378559, -20.7948189, 20.7933846
7: -24.1224785, 1.3693638, -24.1225739, 1.3703508, -24.0671844, 24.0608368
8: -27.3805485, 1.1185598, -27.3806095, 1.1200485, -25.4309769, 25.4163437
9: -11.5786209, 11.8618736, -11.5787830, 11.8633738, -20.8028488, 20.7835312
10: -17.7083645, 12.3530941, -17.7091713, 12.3535662, -29.6793518, 29.6810303
11: -16.7890759, 10.4934340, -16.7900906, 10.4936123, -23.8027725, 23.8137054
12: -24.2272453, 11.2929735, -24.2278328, 11.2932405, -33.3956909, 33.3977890
13: -22.3808479, 12.3537674, -22.3811798, 12.3547459, -32.7155151, 32.7035446
14: -34.9219780, 6.3825779, -34.9224510, 6.3828039, -36.8045654, 36.8157959
15: -8.7219553, 16.7773571, -8.7223568, 16.7782516, -23.3402023, 23.3349228
16: -22.7879219, 3.1399243, -22.7882843, 3.1407793, -25.9287014, 25.9282093
17: -28.0625687, 8.0696478, -28.0636520, 8.0699062, -36.1324768, 36.1333008
18: -12.5978584, 18.6196327, -12.5993853, 18.6198635, -29.4176559, 29.4243698
19: -8.9532995, 8.0429344, -8.9544687, 8.0430126, -16.5442772, 16.5454865
20: -9.9360895, 8.7776117, -9.9370422, 8.7777195, -17.7322693, 17.7379150
21: -12.5634928, 9.1590710, -12.5643444, 9.1591768, -20.2437286, 20.2506027
22: -2.8024693, 18.5129776, -2.8035064, 18.5130348, -18.7520943, 18.7578735
23: -3.9352703, 15.2580719, -3.9366493, 15.2581682, -17.5332108, 17.5391617
24: -5.5248590, 17.3600960, -5.5264235, 17.3601952, -19.9269409, 19.9395370
25: 2.0636187, 24.2904892, 2.0620575, 24.2906151, -19.4376793, 19.4456139
26: -11.7185307, 21.5881691, -11.7203426, 21.5883865, -33.3069153, 33.3085098
27: -15.0027924, 10.0006523, -15.0036659, 10.0007915, -23.8555450, 23.8603897
28: -3.0094385, 18.0295658, -3.0109248, 18.0297031, -19.5817337, 19.5905304
29: -3.3671350, 15.5121241, -3.3679004, 15.5122156, -15.1763744, 15.1868877
30: -13.5962849, 13.7975788, -13.5975857, 13.7977524, -24.7305374, 24.7442360
31: -9.7021999, 11.3184719, -9.7037249, 11.3186073, -21.0208073, 21.0221977
32: -30.7145290, -3.7880630, -30.7148952, -3.7869997, -22.8513947, 22.8501244
33: -41.5134888, -3.2273312, -41.5142136, -3.2254481, -31.0266266, 31.0212250
34: -36.7915344, -3.8491669, -36.7925568, -3.8489990, -25.3931580, 25.4032059
35: -24.9026775, 5.5320072, -24.9031715, 5.5321374, -25.9975204, 25.9996338
36: -24.5649109, 6.4275298, -24.5659714, 6.4276934, -28.1188126, 28.1207504
37: -42.8777046, -6.5619535, -42.8782959, -6.5618696, -32.1104736, 32.1229095
38: -34.3811722, 3.4983697, -34.3825989, 3.4986806, -35.7710495, 35.7791061
39: -47.7346153, -7.0839896, -47.7351074, -7.0818791, -37.1110001, 37.1144028
40: -45.9573402, -18.9672947, -45.9578629, -18.9655914, -21.1896057, 21.1916962
41: -33.4880447, -4.6563058, -33.4883423, -4.6560988, -22.2711182, 22.2721024
42: -24.0449123, -0.4322357, -24.0452785, -0.4316959, -19.7277946, 19.7243309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=162, inp2_unstable=163, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1629

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6581053, upper bound: 10.6435421
time: 31.18 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6581053, upper bound: 10.6581053
time: 25.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 59.04 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 59.04
Output dim: 25, lower bound: -10.6384873, upper bound: 10.6435397
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 59.04
Output dim: 25, lower bound: -10.6384999, upper bound: 10.6580814
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 59.04
Output dim: 25, lower bound: -10.6581053, upper bound: 10.6435421
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 59.04
Output dim: 25, lower bound: -10.6581053, upper bound: 10.6581053

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -22.4777870, 8.5881100, -22.5997982, 8.6187992, -31.0965862, 31.1879082
1: -11.8484583, 6.3404994, -11.9192753, 6.3620739, -18.2105331, 18.2597752
2: -13.1059914, 7.4220853, -13.1646919, 7.4407158, -19.4288101, 19.4730606
3: -18.6184120, 6.2404318, -18.7076283, 6.2680068, -24.2149124, 24.2809982
4: -20.2498856, 3.2543836, -20.3159790, 3.2775059, -21.1800461, 21.2309723
5: -18.0897865, 7.5451202, -18.1830215, 7.5746632, -25.1468201, 25.2122498
6: -36.7294426, -11.2554703, -36.7636299, -11.2435646, -20.7604485, 20.7828712
7: -24.0760422, 1.3187652, -24.1725693, 1.3447344, -23.9216690, 23.9962006
8: -27.3380127, 1.0679293, -27.4204388, 1.0975285, -25.2237091, 25.2780914
9: -11.5379391, 11.8053455, -11.6229401, 11.8303967, -20.5519867, 20.6317902
10: -17.6831894, 12.3279047, -17.7080498, 12.3763609, -29.6773148, 29.6482162
11: -16.7404785, 10.4620275, -16.7795944, 10.5344057, -23.7341309, 23.6861954
12: -24.1913376, 11.2556982, -24.2169476, 11.3137007, -33.4377899, 33.3867798
13: -22.3365269, 12.2927475, -22.4639816, 12.3227739, -32.5694733, 32.6776810
14: -34.8923187, 6.3679914, -34.9314156, 6.3854508, -36.8508682, 36.8275528
15: -8.6950903, 16.7430153, -8.7306404, 16.7616272, -23.2543869, 23.2745895
16: -22.7472553, 3.1026199, -22.8301392, 3.1222572, -25.8695126, 25.9327583
17: -28.0218315, 8.0423212, -28.0808411, 8.0729494, -36.0947800, 36.1231613
18: -12.5352888, 18.5707626, -12.5724192, 18.6903133, -29.3422699, 29.2502975
19: -8.9054489, 8.0238361, -8.9386930, 8.0650682, -16.4898033, 16.4804459
20: -9.8955069, 8.7606888, -9.9218550, 8.8006897, -17.6942139, 17.6776581
21: -12.5235071, 9.1454411, -12.5559845, 9.1810102, -20.2593842, 20.2453995
22: -2.7634282, 18.4909515, -2.7927732, 18.5483894, -18.7193756, 18.6844864
23: -3.8792992, 15.2280970, -3.9111395, 15.2984333, -17.4351692, 17.3871117
24: -5.4649906, 17.3209343, -5.4966121, 17.4159451, -19.8076782, 19.7316589
25: 2.1208143, 24.2597141, 2.0881491, 24.3225784, -19.3586731, 19.3201675
26: -11.6491642, 21.5404587, -11.6905174, 21.6642151, -33.3133774, 33.2309761
27: -14.9630337, 9.9700775, -14.9857159, 10.0475397, -23.8558731, 23.7955170
28: -2.9510322, 18.0009232, -2.9854455, 18.0644569, -19.4982185, 19.4634438
29: -3.3343554, 15.4906101, -3.3651199, 15.5404501, -15.1632309, 15.1355534
30: -13.5481453, 13.7659140, -13.5731535, 13.8387022, -24.7062225, 24.6469650
31: -9.6433144, 11.2913036, -9.6808987, 11.3478489, -20.9911633, 20.9722023
32: -30.6995296, -3.8044920, -30.7256737, -3.7934456, -22.8863373, 22.8897018
33: -41.4841080, -3.2508192, -41.5503082, -3.2269797, -31.1297150, 31.1665497
34: -36.7548141, -3.8835788, -36.7776680, -3.8107615, -25.3147507, 25.2613640
35: -24.8888760, 5.5129104, -24.9103947, 5.5353637, -25.9772186, 25.9665375
36: -24.5489788, 6.4099445, -24.5721970, 6.4276552, -28.0752563, 28.0795059
37: -42.8569069, -6.5759439, -42.8918991, -6.5631399, -32.2454071, 32.2192764
38: -34.3426208, 3.4639630, -34.3754692, 3.5067616, -35.6436768, 35.6535339
39: -47.7065926, -7.1051626, -47.7812958, -7.0825205, -37.3477783, 37.3738480
40: -45.9333801, -18.9890289, -45.9879532, -18.9676914, -21.4227829, 21.3754311
41: -33.4778900, -4.6705351, -33.4914017, -4.6588488, -22.2676010, 22.2537041
42: -24.0229340, -0.4528632, -24.0624828, -0.4365523, -19.6754570, 19.7021332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=162, inp2_unstable=162, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 874

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6354661, upper bound: 10.6367487
time: 25.54 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6354661, upper bound: 10.6551057
time: 29.29 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -22.5279922, 8.6163483, -22.5258427, 8.5901222, -31.1181145, 31.1421909
1: -11.8807755, 6.3610816, -11.8800201, 6.3453779, -18.2261543, 18.2411022
2: -13.1321983, 7.4381785, -13.1315928, 7.4239087, -19.4431076, 19.4495163
3: -18.6585846, 6.2676759, -18.6573982, 6.2501278, -24.2408295, 24.2495270
4: -20.2828484, 3.2822547, -20.2812424, 3.2709596, -21.2163315, 21.2106171
5: -18.1301346, 7.5711327, -18.1286945, 7.5509577, -25.1650238, 25.1809692
6: -36.7377853, -11.2555494, -36.7337799, -11.2658472, -20.7695312, 20.7729416
7: -24.1203766, 1.3430514, -24.1190567, 1.3252840, -23.9520798, 23.9587631
8: -27.3788338, 1.0957017, -27.3776894, 1.0813403, -25.2586670, 25.2461929
9: -11.5757961, 11.8336792, -11.5740194, 11.8150148, -20.5935974, 20.5792160
10: -17.6902122, 12.3443089, -17.6781979, 12.3386393, -29.6461868, 29.6411285
11: -16.7677689, 10.4910908, -16.7536125, 10.4896727, -23.7021713, 23.7087631
12: -24.2157803, 11.2878704, -24.2082863, 11.2845554, -33.4237595, 33.4228897
13: -22.3757534, 12.3047428, -22.3725491, 12.2706671, -32.5696259, 32.5831909
14: -34.9128036, 6.3777924, -34.9067764, 6.3746119, -36.8351212, 36.8568344
15: -8.7164021, 16.7668190, -8.7128582, 16.7603092, -23.2777634, 23.2723808
16: -22.7830181, 3.1141419, -22.7799568, 3.0967102, -25.8797283, 25.8940983
17: -28.0557537, 8.0631647, -28.0520344, 8.0589094, -36.1146622, 36.1152000
18: -12.5592308, 18.6169033, -12.5331640, 18.6152916, -29.2807846, 29.2673264
19: -8.9313774, 8.0418921, -8.9169636, 8.0412474, -16.4907303, 16.4776955
20: -9.9115267, 8.7763119, -9.8954792, 8.7755013, -17.6809921, 17.6718483
21: -12.5417061, 9.1572866, -12.5273657, 9.1561613, -20.2508774, 20.2446594
22: -2.7788162, 18.5116844, -2.7630892, 18.5108662, -18.6912727, 18.6823063
23: -3.9062366, 15.2560654, -3.8868675, 15.2547626, -17.4123840, 17.4050255
24: -5.4923401, 17.3587589, -5.4706540, 17.3579483, -19.7582550, 19.7561226
25: 2.0914011, 24.2889709, 2.1095533, 24.2880173, -19.3483582, 19.3402176
26: -11.6747379, 21.5848370, -11.6453438, 21.5827770, -33.2575150, 33.2301788
27: -14.9765244, 9.9981852, -14.9587069, 9.9965525, -23.8139572, 23.8016014
28: -2.9804373, 18.0272942, -2.9612765, 18.0259209, -19.4790573, 19.4733391
29: -3.3539486, 15.5103226, -3.3453398, 15.5091906, -15.1350861, 15.1425343
30: -13.5694265, 13.7942009, -13.5516119, 13.7920837, -24.6613388, 24.6644669
31: -9.6725140, 11.3164978, -9.6531162, 11.3152790, -20.9877930, 20.9696140
32: -30.7072906, -3.7988038, -30.7027645, -3.8052959, -22.8823204, 22.8838081
33: -41.5020523, -3.2500582, -41.4949951, -3.2636490, -31.1489563, 31.1581039
34: -36.7696953, -3.8513303, -36.7553253, -3.8526559, -25.2908630, 25.2915268
35: -24.8965168, 5.5284553, -24.8927212, 5.5260882, -25.9769058, 25.9766998
36: -24.5588684, 6.4222674, -24.5557861, 6.4188113, -28.0733643, 28.0761032
37: -42.8682251, -6.5722923, -42.8623199, -6.5793161, -32.2413483, 32.2597580
38: -34.3683281, 3.4955616, -34.3608170, 3.4940233, -35.6516266, 35.6570053
39: -47.7256050, -7.1080685, -47.7200089, -7.1227245, -37.3304291, 37.3611298
40: -45.9492493, -18.9860497, -45.9444122, -18.9969101, -21.4008827, 21.4180832
41: -33.4833488, -4.6635857, -33.4804459, -4.6682277, -22.2635345, 22.2647972
42: -24.0389957, -0.4479396, -24.0353394, -0.4585681, -19.6737633, 19.6734676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=162, inp2_unstable=162, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 874

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6551058, upper bound: 10.6220510
time: 25.53 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6551058, upper bound: 10.6404202
time: 27.99 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -22.5304394, 8.6504078, -22.6020317, 8.6547089, -31.1851482, 31.2524395
1: -11.8814526, 6.3810296, -11.9203291, 6.3851061, -18.2665596, 18.3013592
2: -13.1329269, 7.4567752, -13.1656017, 7.4603915, -19.5048904, 19.5322495
3: -18.6600304, 6.2906981, -18.7090969, 6.2961082, -24.3302841, 24.3719025
4: -20.2847652, 3.2970972, -20.3180161, 3.3012023, -21.3032074, 21.3288803
5: -18.1314087, 7.5976505, -18.1845989, 7.6038117, -25.2351685, 25.2814713
6: -36.7421608, -11.2443008, -36.7686653, -11.2393398, -20.7884560, 20.8108788
7: -24.1217308, 1.3660071, -24.1744118, 1.3713639, -24.0642776, 24.1085129
8: -27.3801384, 1.1155548, -27.4221382, 1.1239448, -25.4274902, 25.4500351
9: -11.5779572, 11.8580503, -11.6253586, 11.8603392, -20.7883377, 20.8275299
10: -17.7055244, 12.3512974, -17.7159309, 12.3837919, -29.7074814, 29.6836548
11: -16.7856922, 10.4928703, -16.7983932, 10.5363808, -23.8424301, 23.8107796
12: -24.2237511, 11.2914543, -24.2324867, 11.3180370, -33.4193268, 33.3953323
13: -22.3793049, 12.3468494, -22.4665432, 12.3522930, -32.7040939, 32.7828064
14: -34.9195099, 6.3816495, -34.9385834, 6.3891640, -36.8113403, 36.8277054
15: -8.7206974, 16.7742119, -8.7344551, 16.7785702, -23.3380508, 23.3458061
16: -22.7860565, 3.1366882, -22.8342323, 3.1411614, -25.9272175, 25.9709206
17: -28.0604324, 8.0681953, -28.0885048, 8.0770798, -36.1375122, 36.1567001
18: -12.5924301, 18.6190205, -12.6020432, 18.6922436, -29.4845352, 29.4198914
19: -8.9499846, 8.0426798, -8.9603548, 8.0658875, -16.5622330, 16.5482597
20: -9.9320850, 8.7770243, -9.9404411, 8.8016148, -17.7523346, 17.7381592
21: -12.5599222, 9.1585789, -12.5714388, 9.1826801, -20.2646713, 20.2546196
22: -2.7993422, 18.5125275, -2.8100395, 18.5493965, -18.7848091, 18.7574120
23: -3.9313526, 15.2577076, -3.9386797, 15.2998104, -17.5701828, 17.5338364
24: -5.5201883, 17.3596859, -5.5270567, 17.4168358, -19.9786148, 19.9299049
25: 2.0678787, 24.2898979, 2.0596576, 24.3235435, -19.4663620, 19.4404411
26: -11.7124462, 21.5873260, -11.7242918, 21.6664677, -33.3789139, 33.3116188
27: -14.9984989, 10.0000353, -15.0043631, 10.0494232, -23.9004288, 23.8554306
28: -3.0050945, 18.0291519, -3.0141773, 18.0657654, -19.6124954, 19.5884590
29: -3.3650374, 15.5118437, -3.3769288, 15.5418720, -15.2042084, 15.1874046
30: -13.5910206, 13.7968464, -13.5960789, 13.8409901, -24.7700729, 24.7324371
31: -9.6979771, 11.3180227, -9.7088957, 11.3498573, -21.0478344, 21.0269184
32: -30.7127819, -3.7922387, -30.7298508, -3.7883043, -22.8454933, 22.8549004
33: -41.5112991, -3.2284374, -41.5601273, -3.2181225, -31.0162506, 31.0574493
34: -36.7880821, -3.8498592, -36.7952957, -3.8089805, -25.4286194, 25.4011803
35: -24.9012051, 5.5300326, -24.9155159, 5.5377455, -26.0053253, 26.0062027
36: -24.5636063, 6.4249234, -24.5790272, 6.4298105, -28.1196518, 28.1297836
37: -42.8750839, -6.5667477, -42.8999100, -6.5612526, -32.1096344, 32.1207962
38: -34.3781891, 3.4967890, -34.3925552, 3.5093999, -35.7737579, 35.7891464
39: -47.7324333, -7.0883770, -47.7878952, -7.0775323, -37.0991364, 37.1518173
40: -45.9549713, -18.9697304, -45.9928589, -18.9610596, -21.1848717, 21.1881294
41: -33.4867287, -4.6602015, -33.4954147, -4.6555271, -22.2751160, 22.2684441
42: -24.0433083, -0.4346790, -24.0676880, -0.4279947, -19.7252693, 19.7432899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=162, inp2_unstable=162, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 874

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6551058, upper bound: 10.6367487
time: 21.08 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6551058, upper bound: 10.6551057
time: 28.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 51.45 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 51.45
Output dim: 25, lower bound: -10.6354661, upper bound: 10.6367487
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 51.45
Output dim: 25, lower bound: -10.6354661, upper bound: 10.6551057
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 51.45
Output dim: 25, lower bound: -10.6551058, upper bound: 10.6220510
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 51.45
Output dim: 25, lower bound: -10.6551058, upper bound: 10.6404202
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 51.45
Output dim: 25, lower bound: -10.6551058, upper bound: 10.6367487
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 51.45
Output dim: 25, lower bound: -10.6551058, upper bound: 10.6551057

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -22.4816875, 8.6538486, -22.5976791, 8.6180153, -31.0997028, 31.2515278
1: -11.8483095, 6.4027658, -11.9176292, 6.3612294, -18.2095394, 18.3203945
2: -13.1067610, 7.4821968, -13.1632462, 7.4398613, -19.4284821, 19.5324936
3: -18.6214619, 6.3210983, -18.7051048, 6.2670145, -24.2157516, 24.3594666
4: -20.2524605, 3.3274999, -20.3139515, 3.2762055, -21.1758728, 21.3004608
5: -18.0915852, 7.6271820, -18.1803513, 7.5736694, -25.1460037, 25.2919083
6: -36.7417488, -11.2409096, -36.7536278, -11.2440157, -20.7759666, 20.7966385
7: -24.0771637, 1.3973770, -24.1700554, 1.3438666, -23.9188614, 24.0723801
8: -27.3398037, 1.1408014, -27.4180317, 1.0965309, -25.2212448, 25.3464203
9: -11.5440426, 11.8698301, -11.6209984, 11.8289776, -20.5501404, 20.6915207
10: -17.6980286, 12.3389797, -17.7071133, 12.3676529, -29.6868134, 29.6596756
11: -16.7995682, 10.4669685, -16.7784538, 10.5323982, -23.7911301, 23.6803207
12: -24.2419281, 11.2630310, -24.2154636, 11.3116779, -33.5014648, 33.3815079
13: -22.3413010, 12.3534937, -22.4618359, 12.3215370, -32.5686417, 32.7275848
14: -34.9046326, 6.3745408, -34.9296951, 6.3807626, -36.8988342, 36.8177719
15: -8.7041874, 16.7814922, -8.7294130, 16.7606468, -23.2621841, 23.3153191
16: -22.7584343, 3.1448958, -22.8283272, 3.1213021, -25.8797359, 25.9732227
17: -28.0643864, 8.0498714, -28.0786438, 8.0714083, -36.1357956, 36.1285172
18: -12.6117678, 18.5746861, -12.5711145, 18.6873512, -29.4170761, 29.2502060
19: -8.9676437, 8.0256758, -8.9375429, 8.0634403, -16.5530663, 16.4800301
20: -9.9277992, 8.7627449, -9.9209347, 8.7993965, -17.7311592, 17.6758308
21: -12.5597897, 9.1480541, -12.5549221, 9.1783466, -20.3106689, 20.2364082
22: -2.8119850, 18.4936752, -2.7918129, 18.5471535, -18.7738075, 18.6820030
23: -3.9691334, 15.2295380, -3.9101162, 15.2959852, -17.5235596, 17.3813667
24: -5.5358529, 17.3238850, -5.4957085, 17.4135857, -19.8773422, 19.7288475
25: 2.0550575, 24.2627430, 2.0891876, 24.3205662, -19.4265518, 19.3193626
26: -11.7365627, 21.5438175, -11.6889706, 21.6612911, -33.3978539, 33.2327881
27: -15.0129633, 9.9733162, -14.9845991, 10.0461426, -23.9096756, 23.7951431
28: -3.0228281, 18.0039692, -2.9842281, 18.0627842, -19.5683212, 19.4633179
29: -3.3775139, 15.4938154, -3.3638530, 15.5393543, -15.2040520, 15.1367931
30: -13.5859547, 13.7714710, -13.5723372, 13.8370895, -24.7450638, 24.6526451
31: -9.7131519, 11.2954741, -9.6796665, 11.3458529, -21.0590057, 20.9751396
32: -30.7216225, -3.7977176, -30.7223091, -3.7941008, -22.9161224, 22.8902206
33: -41.5017471, -3.2396398, -41.5449905, -3.2279520, -31.1667175, 31.1589661
34: -36.8138351, -3.8810210, -36.7763367, -3.8125644, -25.3759308, 25.2597809
35: -24.9108200, 5.5186057, -24.9094715, 5.5342298, -26.0019989, 25.9672928
36: -24.5949478, 6.4136238, -24.5706902, 6.4267282, -28.1223831, 28.0811310
37: -42.9208908, -6.5715065, -42.8898468, -6.5650997, -32.3128815, 32.2181168
38: -34.4009094, 3.4664378, -34.3732452, 3.5048676, -35.7017365, 35.6506195
39: -47.7244797, -7.0993910, -47.7800903, -7.0838938, -37.3980255, 37.3562164
40: -45.9505730, -18.9836044, -45.9856453, -18.9681339, -21.4609642, 21.3676224
41: -33.5074120, -4.6658707, -33.4898071, -4.6605926, -22.3013306, 22.2555046
42: -24.0420780, -0.4417896, -24.0607014, -0.4371169, -19.6867409, 19.7172470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=162, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6266255, upper bound: 10.6445778
time: 30.95 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6326904, upper bound: 10.6523304
time: 27.66 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -22.4952126, 8.6125469, -22.5079498, 8.5880547, -31.0832672, 31.1204967
1: -11.8523283, 6.3568401, -11.8644886, 6.3430591, -18.1953869, 18.2213287
2: -13.1072330, 7.4338665, -13.1179571, 7.4215722, -19.4155960, 19.4315033
3: -18.6148758, 6.2612391, -18.6335239, 6.2466316, -24.1925125, 24.2185822
4: -20.2466583, 3.2761395, -20.2613525, 3.2676151, -21.1764755, 21.1841583
5: -18.0860863, 7.5651083, -18.1046562, 7.5476804, -25.1167984, 25.1505051
6: -36.7227325, -11.2602224, -36.7255249, -11.2683802, -20.7511902, 20.7586517
7: -24.0775223, 1.3379154, -24.0956535, 1.3225005, -23.9065094, 23.9301987
8: -27.3373260, 1.0904589, -27.3549957, 1.0785046, -25.2149734, 25.2178650
9: -11.5429325, 11.8282194, -11.5561466, 11.8120403, -20.5594330, 20.5571671
10: -17.6792679, 12.3319340, -17.6722374, 12.3316498, -29.6266861, 29.6200104
11: -16.7567406, 10.4580488, -16.7475681, 10.4716291, -23.6725540, 23.6688614
12: -24.2072945, 11.2560425, -24.2036858, 11.2671928, -33.3940735, 33.3801422
13: -22.3472385, 12.2974243, -22.3566284, 12.2666683, -32.5391846, 32.5607605
14: -34.8923950, 6.3633852, -34.8956985, 6.3662429, -36.7975159, 36.8133850
15: -8.6997280, 16.7619095, -8.7037497, 16.7576866, -23.2578125, 23.2577133
16: -22.7573910, 3.1097293, -22.7659931, 3.0942822, -25.8516731, 25.8757229
17: -28.0333366, 8.0364418, -28.0397415, 8.0443497, -36.0776863, 36.0761833
18: -12.5494604, 18.5666504, -12.5278358, 18.5878563, -29.2442474, 29.2117462
19: -8.9221811, 8.0214367, -8.9119205, 8.0300751, -16.4701805, 16.4512177
20: -9.9043350, 8.7700367, -9.8915396, 8.7720480, -17.6689491, 17.6595230
21: -12.5310497, 9.1482944, -12.5215569, 9.1510735, -20.2305069, 20.2209663
22: -2.7709222, 18.4945488, -2.7587514, 18.5014992, -18.6736565, 18.6589355
23: -3.8991966, 15.2169304, -3.8830252, 15.2334042, -17.3834991, 17.3612633
24: -5.4871264, 17.3211918, -5.4678202, 17.3374176, -19.7327118, 19.7157669
25: 2.0985174, 24.2610531, 2.1134462, 24.2727642, -19.3264999, 19.3081818
26: -11.6652966, 21.5395737, -11.6401787, 21.5580387, -33.2233353, 33.1797523
27: -14.9699202, 9.9752569, -14.9550743, 9.9839735, -23.7949066, 23.7743073
28: -2.9723606, 18.0031185, -2.9568424, 18.0126991, -19.4571838, 19.4440536
29: -3.3430824, 15.4936619, -3.3393908, 15.5001192, -15.1156673, 15.1211472
30: -13.5636282, 13.7737408, -13.5484371, 13.7808876, -24.6435623, 24.6408234
31: -9.6631432, 11.2923985, -9.6479998, 11.3021097, -20.9652519, 20.9403992
32: -30.6988468, -3.8045173, -30.6981125, -3.8084159, -22.8706970, 22.8727951
33: -41.4933929, -3.2590775, -41.4902306, -3.2685823, -31.1354599, 31.1423264
34: -36.7627869, -3.8847504, -36.7515602, -3.8708897, -25.2665634, 25.2556458
35: -24.8907776, 5.5107045, -24.8895798, 5.5164146, -25.9596863, 25.9524460
36: -24.5524521, 6.4060721, -24.5522614, 6.4099836, -28.0580826, 28.0560913
37: -42.8573227, -6.6040277, -42.8564224, -6.5966434, -32.2124100, 32.2187119
38: -34.3574295, 3.4612899, -34.3548927, 3.4752970, -35.6212540, 35.6163025
39: -47.7161713, -7.1179962, -47.7148819, -7.1281986, -37.3092117, 37.3332596
40: -45.9387741, -18.9913445, -45.9386520, -18.9998455, -21.3876801, 21.4040565
41: -33.4769363, -4.6751966, -33.4769325, -4.6745858, -22.2504501, 22.2487488
42: -24.0313282, -0.4528546, -24.0311031, -0.4612730, -19.6581154, 19.6602001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=162, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6462402, upper bound: 10.6115574
time: 28.63 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6523308, upper bound: 10.6192733
time: 34.98 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.5316982, 8.6820755, -22.5237160, 8.5893211, -31.1210194, 31.2057915
1: -11.8806038, 6.4226890, -11.8783522, 6.3444586, -18.2250633, 18.3010406
2: -13.1328745, 7.4982443, -13.1301365, 7.4230218, -19.4426498, 19.5085983
3: -18.6615429, 6.3482542, -18.6548004, 6.2490549, -24.2416077, 24.3278809
4: -20.2853813, 3.3540850, -20.2791977, 3.2696631, -21.2108307, 21.2793350
5: -18.1316833, 7.6531634, -18.1259766, 7.5499563, -25.1639938, 25.2605515
6: -36.7499199, -11.2408237, -36.7235336, -11.2663307, -20.7856064, 20.7840805
7: -24.1212044, 1.4212785, -24.1164856, 1.3243988, -23.9489212, 24.0339203
8: -27.3805733, 1.1685357, -27.3752308, 1.0802984, -25.2552338, 25.3135223
9: -11.5817986, 11.8980980, -11.5720749, 11.8135691, -20.5909805, 20.6372375
10: -17.7055111, 12.3554249, -17.6772308, 12.3298588, -29.6559219, 29.6522598
11: -16.8267708, 10.4960213, -16.7523842, 10.4876375, -23.7586975, 23.7027512
12: -24.2660332, 11.2948627, -24.2067223, 11.2824583, -33.4864960, 33.4171219
13: -22.3803368, 12.3644066, -22.3703785, 12.2693329, -32.5674591, 32.6311493
14: -34.9248772, 6.3842978, -34.9050064, 6.3699245, -36.8825226, 36.8454285
15: -8.7254028, 16.8050804, -8.7115726, 16.7592545, -23.2844925, 23.3121758
16: -22.7937660, 3.1560588, -22.7779617, 3.0957043, -25.8894711, 25.9340210
17: -28.0976219, 8.0706291, -28.0497208, 8.0573149, -36.1549377, 36.1203499
18: -12.6356220, 18.6207657, -12.5317745, 18.6123028, -29.3542938, 29.2669220
19: -8.9932480, 8.0436993, -8.9156818, 8.0396042, -16.5536156, 16.4772301
20: -9.9436321, 8.7783241, -9.8945274, 8.7742004, -17.7172623, 17.6702156
21: -12.5777302, 9.1598377, -12.5262318, 9.1535063, -20.3013382, 20.2358475
22: -2.8273559, 18.5143509, -2.7620897, 18.5096149, -18.7452202, 18.6792107
23: -3.9958954, 15.2574978, -3.8858337, 15.2523031, -17.5003624, 17.3985291
24: -5.5631285, 17.3616848, -5.4697208, 17.3555832, -19.8278427, 19.7531586
25: 2.0256891, 24.2918797, 2.1106253, 24.2859917, -19.4160080, 19.3389893
26: -11.7621651, 21.5877762, -11.6437206, 21.5796967, -33.3418617, 33.2314987
27: -15.0266438, 10.0013151, -14.9574862, 9.9951515, -23.8667145, 23.8011627
28: -3.0519066, 18.0302925, -2.9599710, 18.0242462, -19.5486832, 19.4728050
29: -3.3969941, 15.5134859, -3.3440018, 15.5080910, -15.1759014, 15.1430798
30: -13.6067133, 13.7996941, -13.5505190, 13.7904434, -24.6995544, 24.6698570
31: -9.7420387, 11.3206081, -9.6517410, 11.3132687, -21.0553074, 20.9723492
32: -30.7292728, -3.7921634, -30.6994152, -3.8060241, -22.9121666, 22.8837090
33: -41.5191650, -3.2387915, -41.4896698, -3.2646432, -31.1868286, 31.1495972
34: -36.8284378, -3.8488607, -36.7538605, -3.8545070, -25.3501740, 25.2900925
35: -24.9183407, 5.5340958, -24.8917427, 5.5249524, -26.0020218, 25.9765015
36: -24.6047554, 6.4258537, -24.5542717, 6.4178543, -28.1201096, 28.0775833
37: -42.9319992, -6.5678043, -42.8601303, -6.5813074, -32.3105011, 32.2563858
38: -34.4268341, 3.4975796, -34.3586349, 3.4918032, -35.7092438, 35.6533203
39: -47.7432976, -7.1024847, -47.7187424, -7.1240602, -37.3822708, 37.3409576
40: -45.9652786, -18.9799385, -45.9418983, -18.9973984, -21.4402466, 21.4071655
41: -33.5127983, -4.6590543, -33.4787827, -4.6700244, -22.2968521, 22.2667389
42: -24.0580444, -0.4370320, -24.0335445, -0.4591732, -19.6848373, 19.6883202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=162, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6462402, upper bound: 10.6299148
time: 24.79 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6523308, upper bound: 10.6376438
time: 25.88 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -22.4976349, 8.6466160, -22.5841007, 8.6526318, -31.1502666, 31.2307167
1: -11.8529968, 6.3767824, -11.9047804, 6.3827767, -18.2357731, 18.2815628
2: -13.1079836, 7.4524918, -13.1519737, 7.4580574, -19.4774170, 19.5142365
3: -18.6163177, 6.2842698, -18.6852493, 6.2926135, -24.2819748, 24.3409958
4: -20.2485981, 3.2909493, -20.2981319, 3.2978528, -21.2629547, 21.3018417
5: -18.0874081, 7.5916557, -18.1605797, 7.6005383, -25.1869125, 25.2510071
6: -36.7270660, -11.2489681, -36.7603111, -11.2418747, -20.7684937, 20.7953377
7: -24.0788651, 1.3608584, -24.1510372, 1.3685665, -24.0185699, 24.0799255
8: -27.3385983, 1.1103492, -27.3994446, 1.1210966, -25.3825226, 25.4213181
9: -11.5451088, 11.8526344, -11.6074543, 11.8573856, -20.7538834, 20.8056488
10: -17.6945629, 12.3389072, -17.7099762, 12.3767977, -29.6880188, 29.6626587
11: -16.7746887, 10.4598255, -16.7923737, 10.5183115, -23.8127594, 23.7706223
12: -24.2152634, 11.2595825, -24.2278290, 11.3007069, -33.3898010, 33.3531570
13: -22.3507919, 12.3395405, -22.4505692, 12.3482990, -32.6723633, 32.7596283
14: -34.8990974, 6.3672218, -34.9274673, 6.3807468, -36.7726974, 36.7857971
15: -8.7040405, 16.7693329, -8.7253609, 16.7759056, -23.3177719, 23.3309250
16: -22.7604446, 3.1322467, -22.8202343, 3.1387351, -25.8991795, 25.9524803
17: -28.0379677, 8.0415001, -28.0761528, 8.0624847, -36.1004524, 36.1176529
18: -12.5826912, 18.5687428, -12.5967464, 18.6648254, -29.4479523, 29.3643188
19: -8.9407787, 8.0222301, -8.9553289, 8.0547113, -16.5414925, 16.5214996
20: -9.9249010, 8.7707453, -9.9365301, 8.7981491, -17.7400284, 17.7254448
21: -12.5492783, 9.1495638, -12.5656090, 9.1775551, -20.2445221, 20.2313423
22: -2.7914219, 18.4953918, -2.8056979, 18.5400238, -18.7671242, 18.7337914
23: -3.9243422, 15.2185926, -3.9348078, 15.2784214, -17.5412598, 17.4899940
24: -5.5149517, 17.3221245, -5.5242062, 17.3963013, -19.9531021, 19.8895226
25: 2.0749903, 24.2620258, 2.0635328, 24.3082867, -19.4445648, 19.4084358
26: -11.7029552, 21.5420952, -11.7191334, 21.6417313, -33.3446884, 33.2612305
27: -14.9918919, 9.9771013, -15.0007467, 10.0368042, -23.8813324, 23.8281708
28: -2.9970222, 18.0049877, -3.0097537, 18.0525551, -19.5906372, 19.5591469
29: -3.3542118, 15.4951744, -3.3709731, 15.5327921, -15.1846886, 15.1659508
30: -13.5852394, 13.7764025, -13.5928593, 13.8297920, -24.7521744, 24.7082977
31: -9.6885853, 11.2939482, -9.7037926, 11.3366890, -21.0252743, 20.9977417
32: -30.7043362, -3.7979336, -30.7250843, -3.7914233, -22.8337784, 22.8438721
33: -41.5025787, -3.2374358, -41.5553551, -3.2230649, -31.0028000, 31.0424271
34: -36.7811623, -3.8832684, -36.7914772, -3.8271680, -25.4043121, 25.3646049
35: -24.8954391, 5.5122652, -24.9123688, 5.5280328, -25.9880524, 25.9817886
36: -24.5571785, 6.4087453, -24.5754757, 6.4209900, -28.1043930, 28.1097641
37: -42.8640938, -6.5984631, -42.8937950, -6.5785675, -32.0814362, 32.0811081
38: -34.3672371, 3.4625058, -34.3865967, 3.4907255, -35.7433319, 35.7482986
39: -47.7229843, -7.0982609, -47.7826996, -7.0829391, -37.0790329, 37.1262207
40: -45.9445763, -18.9750671, -45.9869766, -18.9639854, -21.1720505, 21.1762161
41: -33.4803009, -4.6718221, -33.4918633, -4.6618643, -22.2620239, 22.2523766
42: -24.0356636, -0.4396303, -24.0634289, -0.4306810, -19.7086678, 19.7294579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=162, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6462402, upper bound: 10.6262215
time: 32.74 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6523308, upper bound: 10.6339734
time: 21.87 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.5343170, 8.7161942, -22.5998611, 8.6539087, -31.1882248, 31.3160553
1: -11.8813248, 6.4427762, -11.9186659, 6.3841619, -18.2654877, 18.3614426
2: -13.1336184, 7.5169492, -13.1641903, 7.4595122, -19.5046616, 19.5916138
3: -18.6631279, 6.3713765, -18.7065392, 6.2950344, -24.3312836, 24.4503784
4: -20.2873840, 3.3689938, -20.3159580, 3.2998877, -21.2989655, 21.3987961
5: -18.1330013, 7.6797533, -18.1818771, 7.6027813, -25.2341995, 25.3610916
6: -36.7553101, -11.2294273, -36.7583847, -11.2398376, -20.8021469, 20.8271942
7: -24.1225700, 1.4443746, -24.1718636, 1.3704586, -24.0613480, 24.1843338
8: -27.3819389, 1.1885295, -27.4196835, 1.1228828, -25.4255066, 25.5195389
9: -11.5841427, 11.9225292, -11.6234150, 11.8588648, -20.7866974, 20.8890457
10: -17.7219353, 12.3624516, -17.7149658, 12.3750229, -29.7184143, 29.6947937
11: -16.8448181, 10.4981985, -16.7971725, 10.5343409, -23.8998947, 23.8043861
12: -24.2742348, 11.2985468, -24.2309265, 11.3159447, -33.4810486, 33.3907928
13: -22.3841553, 12.4065304, -22.4643269, 12.3509350, -32.7037659, 32.8353653
14: -34.9320335, 6.3881922, -34.9368172, 6.3844557, -36.8504257, 36.8227921
15: -8.7301178, 16.8125038, -8.7331619, 16.7775059, -23.3459244, 23.3869324
16: -22.7969017, 3.1788146, -22.8322296, 3.1401811, -25.9370823, 26.0110435
17: -28.1027412, 8.0764294, -28.0862236, 8.0754509, -36.1781921, 36.1626511
18: -12.6689129, 18.6230869, -12.6006470, 18.6892872, -29.5589066, 29.4197845
19: -9.0119677, 8.0446110, -8.9590712, 8.0642490, -16.6256943, 16.5473022
20: -9.9642820, 8.7793045, -9.9394855, 8.8003063, -17.7894592, 17.7361374
21: -12.5960674, 9.1612816, -12.5702858, 9.1799984, -20.3147888, 20.2468262
22: -2.8480377, 18.5152397, -2.8090453, 18.5481606, -18.8396950, 18.7537422
23: -4.0210795, 15.2591286, -3.9376516, 15.2973461, -17.6582146, 17.5269356
24: -5.5910492, 17.3626862, -5.5261335, 17.4144554, -20.0482712, 19.9276581
25: 2.0020523, 24.2929745, 2.0607324, 24.3215027, -19.5341644, 19.4395218
26: -11.8000774, 21.5904675, -11.7226772, 21.6633949, -33.4634705, 33.3131447
27: -15.0486994, 10.0032358, -15.0031185, 10.0480146, -23.9533844, 23.8551407
28: -3.0766497, 18.0325069, -3.0128713, 18.0640945, -19.6822929, 19.5883255
29: -3.4082580, 15.5152245, -3.3755827, 15.5407715, -15.2485809, 15.1893234
30: -13.6284313, 13.8028698, -13.5949593, 13.8393583, -24.8092651, 24.7383118
31: -9.7675667, 11.3223286, -9.7075272, 11.3478689, -21.1154366, 21.0298557
32: -30.7356987, -3.7853675, -30.7265091, -3.7890458, -22.8741760, 22.8561020
33: -41.5297470, -3.2170587, -41.5548172, -3.2190843, -31.0510025, 31.0555649
34: -36.8469620, -3.8473082, -36.7937737, -3.8108187, -25.4888000, 25.4003754
35: -24.9231110, 5.5357881, -24.9145489, 5.5365758, -26.0306473, 26.0059128
36: -24.6096935, 6.4287338, -24.5774746, 6.4287863, -28.1669006, 28.1316528
37: -42.9394302, -6.5623951, -42.8977280, -6.5631924, -32.1782608, 32.1206207
38: -34.4367409, 3.4990644, -34.3903160, 3.5072308, -35.8323669, 35.7863235
39: -47.7510757, -7.0826459, -47.7865982, -7.0788870, -37.1485748, 37.1380844
40: -45.9732323, -18.9635048, -45.9903259, -18.9615707, -21.2212639, 21.1842918
41: -33.5165253, -4.6556344, -33.4937248, -4.6573381, -22.3087234, 22.2704544
42: -24.0636253, -0.4236691, -24.0659103, -0.4286194, -19.7351074, 19.7603874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=162, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6462402, upper bound: 10.6445778
time: 34.76 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6523308, upper bound: 10.6523304
time: 31.55 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 68.50 seconds
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 68.50
Output dim: 25, lower bound: -10.6266255, upper bound: 10.6445778
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 68.50
Output dim: 25, lower bound: -10.6326904, upper bound: 10.6523304
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 68.50
Output dim: 25, lower bound: -10.6462402, upper bound: 10.6115574
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 68.50
Output dim: 25, lower bound: -10.6523308, upper bound: 10.6192733
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 68.50
Output dim: 25, lower bound: -10.6462402, upper bound: 10.6299148
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 68.50
Output dim: 25, lower bound: -10.6523308, upper bound: 10.6376438
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 68.50
Output dim: 25, lower bound: -10.6462402, upper bound: 10.6262215
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 68.50
Output dim: 25, lower bound: -10.6523308, upper bound: 10.6339734
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 68.50
Output dim: 25, lower bound: -10.6462402, upper bound: 10.6445778
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 68.50
Output dim: 25, lower bound: -10.6523308, upper bound: 10.6523304

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.4807816, 8.6519146, -22.5961494, 8.6147308, -31.0955124, 31.2480640
1: -11.8464766, 6.4010220, -11.9147110, 6.3583665, -18.2048435, 18.3157330
2: -13.1064358, 7.4794092, -13.1626787, 7.4352469, -19.3835907, 19.5288315
3: -18.6206837, 6.3167572, -18.7038345, 6.2606325, -24.1937485, 24.3552933
4: -20.2520123, 3.3252625, -20.3131714, 3.2725685, -21.1563187, 21.2975998
5: -18.0909309, 7.6235332, -18.1793175, 7.5678420, -25.0836029, 25.2874222
6: -36.7405586, -11.2515011, -36.7517014, -11.2616596, -20.7515755, 20.7836456
7: -24.0764008, 1.3940434, -24.1688423, 1.3383396, -23.8781052, 24.0676651
8: -27.3392715, 1.1389036, -27.4171791, 1.0933509, -25.1967850, 25.3435287
9: -11.5413532, 11.8688946, -11.6166039, 11.8274479, -20.5459366, 20.6181641
10: -17.6932392, 12.3369389, -17.6991386, 12.3643551, -29.6783905, 29.6069489
11: -16.7975903, 10.4620066, -16.7751999, 10.5241566, -23.7883759, 23.6720123
12: -24.2363243, 11.2621813, -24.2061214, 11.3103247, -33.4941406, 33.3148041
13: -22.3390083, 12.3522053, -22.4580803, 12.3193865, -32.5566101, 32.6975555
14: -34.8991852, 6.3737063, -34.9206772, 6.3793716, -36.8915405, 36.7321472
15: -8.6886711, 16.7803917, -8.7035084, 16.7588844, -23.2439423, 23.3150139
16: -22.7561111, 3.1356082, -22.8244286, 3.1058278, -25.8619385, 25.9600372
17: -28.0594139, 8.0491238, -28.0704384, 8.0701809, -36.1295929, 36.1195602
18: -12.6085205, 18.5735798, -12.5657330, 18.6855335, -29.4123001, 29.2196426
19: -8.9658184, 8.0190687, -8.9345379, 8.0527868, -16.5470886, 16.4734497
20: -9.9263201, 8.7582893, -9.9185009, 8.7920113, -17.7269707, 17.6706047
21: -12.5575199, 9.1452713, -12.5511971, 9.1737032, -20.2962265, 20.2156105
22: -2.7990346, 18.4932671, -2.7707829, 18.5464821, -18.7607117, 18.6536369
23: -3.9677715, 15.2256527, -3.9079027, 15.2896595, -17.5341415, 17.3723335
24: -5.5351024, 17.3226948, -5.4944668, 17.4115829, -19.8641434, 19.7218819
25: 2.0636749, 24.2622261, 2.1018748, 24.3197327, -19.4215698, 19.2912445
26: -11.7232685, 21.5430183, -11.6701193, 21.6599922, -33.3832626, 33.2131386
27: -15.0112820, 9.9715881, -14.9818459, 10.0433254, -23.8509903, 23.7855377
28: -3.0167398, 18.0030975, -2.9742999, 18.0613823, -19.5571709, 19.4753571
29: -3.3678136, 15.4935532, -3.3477783, 15.5389328, -15.1976337, 15.1297894
30: -13.5846195, 13.7682743, -13.5701294, 13.8317957, -24.7335129, 24.6535530
31: -9.7111778, 11.2919674, -9.6764488, 11.3400192, -21.0511971, 20.9684162
32: -30.7197838, -3.7982121, -30.7193451, -3.7949166, -22.9284172, 22.8768997
33: -41.5008621, -3.2421603, -41.5435562, -3.2320828, -31.1542053, 31.1616821
34: -36.8048782, -3.8827639, -36.7615280, -3.8154173, -25.3647461, 25.2540092
35: -24.9097748, 5.5166554, -24.9077320, 5.5310464, -25.9820633, 25.9894562
36: -24.5910244, 6.4118299, -24.5642509, 6.4237437, -28.1041412, 28.0668335
37: -42.9197769, -6.5733624, -42.8880234, -6.5681591, -32.3078003, 32.2102661
38: -34.3983994, 3.4633460, -34.3691940, 3.4997311, -35.6968613, 35.6211319
39: -47.7212181, -7.1018271, -47.7746353, -7.0879326, -37.3954468, 37.3484116
40: -45.9500580, -18.9852562, -45.9847794, -18.9709282, -21.4467163, 21.3636131
41: -33.5060425, -4.6740675, -33.4876022, -4.6723394, -22.2854424, 22.2445221
42: -24.0401573, -0.4443760, -24.0575676, -0.4413738, -19.7001915, 19.7030964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 874

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1628

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6182917, upper bound: 10.6508782
time: 28.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6315353, upper bound: 10.6511737
time: 22.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -22.4942818, 8.6105566, -22.5064335, 8.5847664, -31.0790482, 31.1169891
1: -11.8504887, 6.3550768, -11.8615475, 6.3401685, -18.1906567, 18.2166252
2: -13.1068897, 7.4310932, -13.1173735, 7.4169354, -19.3706894, 19.4278641
3: -18.6140976, 6.2569065, -18.6322441, 6.2402396, -24.1705093, 24.2143936
4: -20.2462044, 3.2738919, -20.2605972, 3.2639282, -21.1569290, 21.1812820
5: -18.0854588, 7.5614958, -18.1035976, 7.5418262, -25.0544434, 25.1459656
6: -36.7215462, -11.2708073, -36.7235756, -11.2860546, -20.7267265, 20.7456551
7: -24.0767822, 1.3345630, -24.0944405, 1.3169580, -23.8657532, 23.9254913
8: -27.3368111, 1.0885630, -27.3541718, 1.0752783, -25.1905136, 25.2149200
9: -11.5402727, 11.8272858, -11.5516958, 11.8105440, -20.5552750, 20.4838104
10: -17.6744499, 12.3298740, -17.6642647, 12.3283072, -29.6181946, 29.5673294
11: -16.7547798, 10.4530897, -16.7443199, 10.4633732, -23.6697769, 23.6605530
12: -24.2017136, 11.2551832, -24.1943626, 11.2658529, -33.3867264, 33.3133698
13: -22.3449078, 12.2961149, -22.3528709, 12.2645435, -32.5271759, 32.5306168
14: -34.8869858, 6.3625617, -34.8866196, 6.3648891, -36.7902069, 36.7277908
15: -8.6842079, 16.7608585, -8.6778355, 16.7559185, -23.2395935, 23.2574463
16: -22.7550659, 3.1004393, -22.7621040, 3.0788567, -25.8339233, 25.8625431
17: -28.0283222, 8.0356874, -28.0314827, 8.0431929, -36.0715141, 36.0671692
18: -12.5462027, 18.5654869, -12.5224171, 18.5860367, -29.2394333, 29.1811829
19: -8.9203377, 8.0148373, -8.9089251, 8.0194197, -16.4641953, 16.4446373
20: -9.9028492, 8.7655697, -9.8890839, 8.7646542, -17.6647377, 17.6543083
21: -12.5287666, 9.1454687, -12.5178204, 9.1464329, -20.2160568, 20.2001724
22: -2.7579355, 18.4941292, -2.7376966, 18.5008354, -18.6605682, 18.6305428
23: -3.8978338, 15.2130442, -3.8807855, 15.2270756, -17.3940964, 17.3522339
24: -5.4863567, 17.3199692, -5.4665575, 17.3354263, -19.7195282, 19.7087822
25: 2.1071587, 24.2605553, 2.1261225, 24.2719326, -19.3215256, 19.2800293
26: -11.6519861, 21.5387878, -11.6213646, 21.5567799, -33.2087669, 33.1601524
27: -14.9682407, 9.9734888, -14.9523258, 9.9811172, -23.7361526, 23.7647095
28: -2.9662666, 18.0022392, -2.9469175, 18.0112915, -19.4459915, 19.4561081
29: -3.3334050, 15.4933968, -3.3232713, 15.4996967, -15.1092529, 15.1141396
30: -13.5622654, 13.7705555, -13.5462341, 13.7755833, -24.6320038, 24.6417236
31: -9.6611919, 11.2888927, -9.6447706, 11.2962761, -20.9574680, 20.9336624
32: -30.6970158, -3.8050232, -30.6951504, -3.8092108, -22.8830109, 22.8595238
33: -41.4924774, -3.2616258, -41.4887657, -3.2727766, -31.1230011, 31.1450348
34: -36.7538452, -3.8865085, -36.7367401, -3.8737874, -25.2553558, 25.2498703
35: -24.8896961, 5.5087652, -24.8878021, 5.5132232, -25.9397430, 25.9746323
36: -24.5485535, 6.4042706, -24.5458908, 6.4070063, -28.0398407, 28.0417938
37: -42.8561974, -6.6058912, -42.8545914, -6.5997105, -32.2072601, 32.2108231
38: -34.3549500, 3.4582243, -34.3508301, 3.4701858, -35.6163025, 35.5868683
39: -47.7128677, -7.1204824, -47.7094269, -7.1322823, -37.3066406, 37.3254547
40: -45.9382553, -18.9930382, -45.9377670, -19.0026016, -21.3734360, 21.4000282
41: -33.4756012, -4.6833954, -33.4747162, -4.6862912, -22.2344475, 22.2377853
42: -24.0294075, -0.4554536, -24.0279617, -0.4655406, -19.6715736, 19.6460552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 874

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 604

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6513635, upper bound: 10.6028635
time: 27.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6513654, upper bound: 10.6183830
time: 33.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -22.5307446, 8.6800413, -22.5222034, 8.5860224, -31.1167679, 31.2022438
1: -11.8787746, 6.4209528, -11.8754349, 6.3415742, -18.2203484, 18.2963867
2: -13.1325445, 7.4954515, -13.1295729, 7.4184084, -19.3977890, 19.5049591
3: -18.6607819, 6.3439121, -18.6535206, 6.2426882, -24.2195892, 24.3236694
4: -20.2848930, 3.3518624, -20.2784252, 3.2659998, -21.1913071, 21.2764206
5: -18.1310425, 7.6495175, -18.1248760, 7.5441103, -25.1016235, 25.2560501
6: -36.7487183, -11.2514076, -36.7215805, -11.2839661, -20.7611618, 20.7711143
7: -24.1204720, 1.4179554, -24.1152782, 1.3188586, -23.9081650, 24.0292740
8: -27.3800411, 1.1665487, -27.3743935, 1.0770569, -25.2308044, 25.3105698
9: -11.5791149, 11.8971643, -11.5676498, 11.8120613, -20.5867615, 20.5638962
10: -17.7006969, 12.3534088, -17.6692715, 12.3265047, -29.6474609, 29.5994720
11: -16.8248291, 10.4910583, -16.7491436, 10.4793873, -23.7558899, 23.6944237
12: -24.2604465, 11.2940540, -24.1974564, 11.2810888, -33.4791565, 33.3504105
13: -22.3780193, 12.3630772, -22.3666401, 12.2671881, -32.5553970, 32.6010361
14: -34.9194450, 6.3834500, -34.8959351, 6.3685451, -36.8752136, 36.7598114
15: -8.7098656, 16.8040066, -8.6856699, 16.7574768, -23.2662964, 23.3119202
16: -22.7914410, 3.1467738, -22.7740784, 3.0803151, -25.8717556, 25.9208527
17: -28.0926113, 8.0698776, -28.0414772, 8.0561628, -36.1487732, 36.1113548
18: -12.6323500, 18.6196365, -12.5263319, 18.6104908, -29.3494873, 29.2363358
19: -8.9914417, 8.0370903, -8.9126797, 8.0289421, -16.5476303, 16.4706535
20: -9.9421406, 8.7738686, -9.8920879, 8.7667999, -17.7130661, 17.6649933
21: -12.5754347, 9.1570454, -12.5224724, 9.1488724, -20.2868881, 20.2150307
22: -2.8143988, 18.5139351, -2.7410393, 18.5089512, -18.7321167, 18.6507874
23: -3.9945393, 15.2535810, -3.8835902, 15.2459621, -17.5109406, 17.3894958
24: -5.5623879, 17.3604622, -5.4684834, 17.3535957, -19.8146057, 19.7461967
25: 2.0343008, 24.2913723, 2.1233373, 24.2851467, -19.4110374, 19.3108444
26: -11.7488337, 21.5869675, -11.6248798, 21.5784302, -33.3272629, 33.2118454
27: -15.0249825, 9.9995737, -14.9547215, 9.9923267, -23.8080292, 23.7915649
28: -3.0458298, 18.0294247, -2.9500432, 18.0228195, -19.5375061, 19.4848251
29: -3.3872862, 15.5132275, -3.3279138, 15.5076571, -15.1694832, 15.1360931
30: -13.6054020, 13.7964993, -13.5483007, 13.7851696, -24.6879883, 24.6707878
31: -9.7400646, 11.3171043, -9.6485004, 11.3074484, -21.0475121, 20.9656048
32: -30.7274513, -3.7926388, -30.6964493, -3.8068366, -22.9244690, 22.8703613
33: -41.5182724, -3.2413006, -41.4881935, -3.2688046, -31.1743469, 31.1522598
34: -36.8195190, -3.8505754, -36.7390823, -3.8574405, -25.3389740, 25.2843323
35: -24.9172401, 5.5322046, -24.8899651, 5.5217743, -25.9820862, 25.9986877
36: -24.6008606, 6.4240656, -24.5478516, 6.4148417, -28.1018906, 28.0632935
37: -42.9308815, -6.5696297, -42.8582993, -6.5843744, -32.3053055, 32.2484741
38: -34.4243622, 3.4945040, -34.3545151, 3.4867048, -35.7043152, 35.6239624
39: -47.7399979, -7.1049194, -47.7132759, -7.1281714, -37.3796387, 37.3331680
40: -45.9647141, -18.9816132, -45.9410324, -19.0001659, -21.4260292, 21.4031372
41: -33.5114708, -4.6672792, -33.4765549, -4.6817369, -22.2808952, 22.2557793
42: -24.0561523, -0.4396236, -24.0304031, -0.4634359, -19.6982880, 19.6741810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 874

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1628

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6379063, upper bound: 10.6363131
time: 25.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6511740, upper bound: 10.6366189
time: 28.96 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -22.4967041, 8.6446333, -22.5825863, 8.6493406, -31.1460457, 31.2272186
1: -11.8511629, 6.3750486, -11.9018698, 6.3799138, -18.2310772, 18.2769184
2: -13.1076365, 7.4497252, -13.1514187, 7.4534206, -19.4325104, 19.5106125
3: -18.6154900, 6.2799368, -18.6839638, 6.2862182, -24.2599487, 24.3367920
4: -20.2481537, 3.2887423, -20.2973728, 3.2941978, -21.2434235, 21.2989502
5: -18.0867290, 7.5880222, -18.1595364, 7.5946760, -25.1245422, 25.2464676
6: -36.7258797, -11.2595482, -36.7583885, -11.2595463, -20.7440910, 20.7823486
7: -24.0781288, 1.3575165, -24.1498260, 1.3630140, -23.9778061, 24.0752640
8: -27.3381157, 1.1084223, -27.3986149, 1.1178885, -25.3580475, 25.4183884
9: -11.5424252, 11.8517141, -11.6030273, 11.8558512, -20.7496796, 20.7322998
10: -17.6897850, 12.3368549, -17.7019730, 12.3734932, -29.6795731, 29.6099167
11: -16.7726974, 10.4548759, -16.7890930, 10.5100718, -23.8099747, 23.7623062
12: -24.2096539, 11.2587500, -24.2185001, 11.2992744, -33.3824921, 33.2864304
13: -22.3484783, 12.3382053, -22.4468002, 12.3461437, -32.6603851, 32.7295456
14: -34.8936844, 6.3664441, -34.9184113, 6.3793936, -36.7654495, 36.7002106
15: -8.6885214, 16.7682495, -8.6994228, 16.7741318, -23.2995758, 23.3306313
16: -22.7580891, 3.1229985, -22.8163490, 3.1233091, -25.8813972, 25.9393482
17: -28.0330276, 8.0407715, -28.0678940, 8.0613289, -36.0943565, 36.1086655
18: -12.5794191, 18.5676155, -12.5913429, 18.6630020, -29.4431458, 29.3337555
19: -8.9389572, 8.0156317, -8.9523249, 8.0440550, -16.5355225, 16.5148964
20: -9.9234085, 8.7662888, -9.9340544, 8.7907639, -17.7358322, 17.7202110
21: -12.5469904, 9.1467609, -12.5618668, 9.1729164, -20.2300720, 20.2105598
22: -2.7784591, 18.4949646, -2.7846670, 18.5393543, -18.7540398, 18.7053986
23: -3.9229689, 15.2146778, -3.9325781, 15.2720909, -17.5518341, 17.4809570
24: -5.5141902, 17.3209114, -5.5229836, 17.3943214, -19.9398804, 19.8825531
25: 2.0836091, 24.2615128, 2.0762262, 24.3074474, -19.4395561, 19.3803062
26: -11.6896486, 21.5413017, -11.7002983, 21.6404629, -33.3301125, 33.2416000
27: -14.9902344, 9.9753723, -14.9979687, 10.0339909, -23.8226013, 23.8185730
28: -2.9909601, 18.0041161, -2.9998264, 18.0511532, -19.5794754, 19.5711899
29: -3.3445139, 15.4948969, -3.3548660, 15.5323696, -15.1782722, 15.1589241
30: -13.5838900, 13.7731981, -13.5906525, 13.8245077, -24.7406006, 24.7092247
31: -9.6866264, 11.2904186, -9.7005444, 11.3308411, -21.0174675, 20.9909630
32: -30.7025013, -3.7984486, -30.7220917, -3.7922330, -22.8461151, 22.8305740
33: -41.5016937, -3.2399683, -41.5539131, -3.2271729, -30.9902191, 31.0450974
34: -36.7722702, -3.8850055, -36.7766914, -3.8300390, -25.3931198, 25.3587990
35: -24.8943310, 5.5103683, -24.9106178, 5.5248623, -25.9681244, 26.0039825
36: -24.5533028, 6.4069262, -24.5690727, 6.4180346, -28.0861053, 28.0954132
37: -42.8629875, -6.6003132, -42.8919525, -6.5816345, -32.0763092, 32.0731888
38: -34.3647766, 3.4593816, -34.3825073, 3.4855652, -35.7384338, 35.7189026
39: -47.7196655, -7.1007848, -47.7772522, -7.0870433, -37.0764160, 37.1184387
40: -45.9440308, -18.9767303, -45.9860725, -18.9667511, -21.1577911, 21.1721764
41: -33.4789581, -4.6800475, -33.4896355, -4.6736150, -22.2460556, 22.2414093
42: -24.0337486, -0.4421918, -24.0602913, -0.4349606, -19.7221336, 19.7153168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 874

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 604

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6513635, upper bound: 10.6161241
time: 30.12 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6513654, upper bound: 10.6330155
time: 29.47 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.5334015, 8.7142048, -22.5983562, 8.6505699, -31.1839714, 31.3125610
1: -11.8794918, 6.4410386, -11.9157486, 6.3813071, -18.2607994, 18.3567867
2: -13.1332722, 7.5141530, -13.1635990, 7.4549093, -19.4597855, 19.5879898
3: -18.6623306, 6.3670053, -18.7052383, 6.2886701, -24.3092804, 24.4461594
4: -20.2869186, 3.3668032, -20.3151817, 3.2962675, -21.2794113, 21.3958893
5: -18.1323700, 7.6761284, -18.1808090, 7.5969305, -25.1717834, 25.3565979
6: -36.7541199, -11.2399807, -36.7564278, -11.2574663, -20.7777557, 20.8142090
7: -24.1218395, 1.4410460, -24.1706467, 1.3649504, -24.0206375, 24.1796417
8: -27.3814201, 1.1865997, -27.4188900, 1.1196718, -25.4010620, 25.5165710
9: -11.5814724, 11.9216070, -11.6189976, 11.8573799, -20.7825089, 20.8157120
10: -17.7171249, 12.3603992, -17.7069550, 12.3716679, -29.7099838, 29.6421280
11: -16.8428879, 10.4932356, -16.7939091, 10.5261059, -23.8971176, 23.7960701
12: -24.2686081, 11.2977142, -24.2216511, 11.3145676, -33.4737396, 33.3240662
13: -22.3818302, 12.4052601, -22.4606304, 12.3488064, -32.6917572, 32.8052979
14: -34.9265633, 6.3873591, -34.9277458, 6.3830938, -36.8431244, 36.7371292
15: -8.7145824, 16.8114281, -8.7072725, 16.7757301, -23.3276978, 23.3866348
16: -22.7945633, 3.1695118, -22.8283272, 3.1247580, -25.9193211, 25.9978390
17: -28.0977993, 8.0757008, -28.0779629, 8.0742874, -36.1720886, 36.1536636
18: -12.6656647, 18.6219501, -12.5952835, 18.6874619, -29.5541077, 29.3892517
19: -9.0101395, 8.0379972, -8.9560833, 8.0535975, -16.6197243, 16.5407219
20: -9.9627943, 8.7748365, -9.9370270, 8.7928848, -17.7852554, 17.7308998
21: -12.5937729, 9.1585102, -12.5665340, 9.1753502, -20.3003082, 20.2260208
22: -2.8350759, 18.5148277, -2.7880259, 18.5474834, -18.8266068, 18.7253513
23: -4.0197277, 15.2552376, -3.9353943, 15.2910080, -17.6688004, 17.5178871
24: -5.5902910, 17.3614731, -5.5248904, 17.4125023, -20.0350876, 19.9207115
25: 2.0106573, 24.2924423, 2.0734363, 24.3206635, -19.5292130, 19.4113846
26: -11.7867756, 21.5897045, -11.7038307, 21.6621094, -33.4488831, 33.2935333
27: -15.0470219, 10.0014877, -15.0003910, 10.0451698, -23.8946838, 23.8455505
28: -3.0705752, 18.0316429, -3.0029325, 18.0626945, -19.6711006, 19.6003418
29: -3.3985620, 15.5149622, -3.3594770, 15.5403290, -15.2421570, 15.1823196
30: -13.6270885, 13.7996950, -13.5927534, 13.8340521, -24.7977371, 24.7392426
31: -9.7656002, 11.3188086, -9.7042818, 11.3420238, -21.1076241, 21.0230904
32: -30.7338829, -3.7859063, -30.7234974, -3.7898350, -22.8864746, 22.8428154
33: -41.5289001, -3.2195578, -41.5533524, -3.2232170, -31.0384827, 31.0582733
34: -36.8380661, -3.8490319, -36.7789879, -3.8137236, -25.4776230, 25.3946419
35: -24.9220371, 5.5338993, -24.9128208, 5.5333824, -26.0106812, 26.0280838
36: -24.6057816, 6.4269414, -24.5711117, 6.4258270, -28.1486053, 28.1173630
37: -42.9383087, -6.5642281, -42.8958740, -6.5662889, -32.1731262, 32.1126862
38: -34.4342804, 3.4959769, -34.3862801, 3.5021276, -35.8275146, 35.7568817
39: -47.7477989, -7.0850601, -47.7811127, -7.0829449, -37.1459351, 37.1302795
40: -45.9726944, -18.9651833, -45.9894791, -18.9643517, -21.2070160, 21.1802483
41: -33.5151520, -4.6638594, -33.4915466, -4.6690559, -22.2928085, 22.2594986
42: -24.0617199, -0.4262495, -24.0627575, -0.4328833, -19.7485542, 19.7462482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=161, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 874

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1628

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6379063, upper bound: 10.6508782
time: 29.05 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6511740, upper bound: 10.6511737
time: 55.91 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 87.17 seconds
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 87.17
Output dim: 25, lower bound: -10.6182917, upper bound: 10.6508782
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 87.17
Output dim: 25, lower bound: -10.6315353, upper bound: 10.6511737
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 87.17
Output dim: 25, lower bound: -10.6513635, upper bound: 10.6028635
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 87.17
Output dim: 25, lower bound: -10.6513654, upper bound: 10.6183830
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 87.17
Output dim: 25, lower bound: -10.6379063, upper bound: 10.6363131
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 87.17
Output dim: 25, lower bound: -10.6511740, upper bound: 10.6366189
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 87.17
Output dim: 25, lower bound: -10.6513635, upper bound: 10.6161241
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 87.17
Output dim: 25, lower bound: -10.6513654, upper bound: 10.6330155
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 87.17
Output dim: 25, lower bound: -10.6379063, upper bound: 10.6508782
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 87.17
Output dim: 25, lower bound: -10.6511740, upper bound: 10.6511737

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -22.4297523, 8.5718813, -22.5928898, 8.5694847, -30.9992371, 31.1647720
1: -11.8156986, 6.3480287, -11.9130630, 6.3286018, -18.1443005, 18.2610912
2: -13.0800686, 7.4284058, -13.1617126, 7.4064116, -19.3279724, 19.4763870
3: -18.5845451, 6.2577658, -18.7021503, 6.2282085, -24.1227264, 24.2923660
4: -20.2217636, 3.2869363, -20.3098927, 3.2516470, -21.1030350, 21.2530975
5: -18.0534458, 7.5609837, -18.1771965, 7.5334001, -25.0084534, 25.2209396
6: -36.7096405, -11.2638836, -36.7392960, -11.2657547, -20.7143784, 20.7533684
7: -24.0375900, 1.3402956, -24.1669350, 1.3082397, -23.8077393, 24.0107193
8: -27.2981491, 1.0837300, -27.4148140, 1.0634680, -25.1259842, 25.2853851
9: -11.5022516, 11.8069878, -11.6132851, 11.7930756, -20.4720840, 20.5530930
10: -17.6564407, 12.3021049, -17.6824760, 12.3541374, -29.6312256, 29.5549164
11: -16.7555428, 10.4362583, -16.7571602, 10.5214539, -23.7418442, 23.6254578
12: -24.1916142, 11.2230072, -24.1831455, 11.3045206, -33.4419708, 33.2479782
13: -22.2913094, 12.2706032, -22.4539261, 12.2761879, -32.4642792, 32.6110764
14: -34.8640823, 6.3551831, -34.9114838, 6.3714561, -36.8506927, 36.6984863
15: -8.6606884, 16.7461491, -8.6971722, 16.7407570, -23.1959610, 23.2722931
16: -22.7106209, 3.0815516, -22.8185387, 3.0761271, -25.7867470, 25.9000893
17: -28.0116730, 8.0211411, -28.0622177, 8.0578384, -36.0695114, 36.0833588
18: -12.5350666, 18.5242729, -12.5277786, 18.6830826, -29.3356400, 29.1312866
19: -8.9090023, 7.9997873, -8.9067755, 8.0518675, -16.4840126, 16.4209023
20: -9.8605394, 8.7349815, -9.8837862, 8.7905140, -17.6571960, 17.6103554
21: -12.4957027, 9.1232224, -12.5211802, 9.1716757, -20.2302475, 20.1601410
22: -2.7444677, 18.4661255, -2.7438688, 18.5448837, -18.7028427, 18.5974770
23: -3.9108157, 15.2000589, -3.8778343, 15.2878408, -17.4721222, 17.3118401
24: -5.4746461, 17.2901154, -5.4620380, 17.4103699, -19.8016129, 19.6555748
25: 2.1271763, 24.2376785, 2.1354551, 24.3183537, -19.3549347, 19.2307625
26: -11.6430492, 21.4967918, -11.6279621, 21.6572685, -33.3003159, 33.1247559
27: -14.9540310, 9.9346037, -14.9527760, 10.0406141, -23.7883530, 23.7153435
28: -2.9586854, 17.9796753, -2.9445801, 18.0595531, -19.4958191, 19.4205437
29: -3.3373818, 15.4748058, -3.3362870, 15.5368700, -15.1623173, 15.0993500
30: -13.5383053, 13.7406998, -13.5452318, 13.8281898, -24.6770859, 24.5960236
31: -9.6335068, 11.2622633, -9.6365299, 11.3380184, -20.9715252, 20.8987923
32: -30.6814327, -3.8153601, -30.7009792, -3.8006282, -22.8800201, 22.8249168
33: -41.4631081, -3.2574477, -41.5274429, -3.2368436, -31.1012268, 31.1049805
34: -36.7380219, -3.9291396, -36.7250557, -3.8183446, -25.2931900, 25.1685753
35: -24.8831005, 5.4910707, -24.8954048, 5.5270462, -25.9474335, 25.9440765
36: -24.5568466, 6.3884292, -24.5471458, 6.4211378, -28.0663223, 28.0249939
37: -42.8844452, -6.5874672, -42.8728790, -6.5708990, -32.2672119, 32.1559677
38: -34.3244476, 3.4137750, -34.3316727, 3.4973044, -35.6177597, 35.5338669
39: -47.6855049, -7.1192589, -47.7608109, -7.0909705, -37.3426971, 37.2761230
40: -45.9245148, -19.0007687, -45.9737816, -18.9759483, -21.4089966, 21.3139038
41: -33.4790459, -4.6932368, -33.4749603, -4.6764336, -22.2527466, 22.2034607
42: -24.0164566, -0.4611657, -24.0471325, -0.4485343, -19.6672325, 19.6725140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 604

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6066846, upper bound: 10.6499100
time: 29.24 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6157024, upper bound: 10.6500574
time: 29.01 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.4800415, 8.6461973, -22.5957432, 8.6113548, -31.0913963, 31.2419395
1: -11.8458891, 6.3973417, -11.9143620, 6.3561921, -18.2020817, 18.3117027
2: -13.1060848, 7.4758310, -13.1625080, 7.4331279, -19.3810883, 19.5188828
3: -18.6199322, 6.3127446, -18.7034607, 6.2582674, -24.1902695, 24.3445435
4: -20.2513695, 3.3225608, -20.3128071, 3.2709544, -21.1538849, 21.2820358
5: -18.0899143, 7.6194234, -18.1788101, 7.5653868, -25.0798721, 25.2823334
6: -36.7363091, -11.2524872, -36.7490616, -11.2622414, -20.7434273, 20.7808228
7: -24.0756569, 1.3904667, -24.1684456, 1.3362079, -23.8748932, 24.0558853
8: -27.3386669, 1.1351795, -27.4168587, 1.0911770, -25.1938171, 25.3206787
9: -11.5407066, 11.8646441, -11.6161814, 11.8249359, -20.5427094, 20.5898895
10: -17.6908302, 12.3352251, -17.6976471, 12.3633385, -29.6743011, 29.6034775
11: -16.7942600, 10.4613810, -16.7730827, 10.5238228, -23.7668228, 23.6695023
12: -24.2336121, 11.2606840, -24.2045364, 11.3094673, -33.4882278, 33.3137665
13: -22.3375320, 12.3462982, -22.4571724, 12.3158760, -32.5515900, 32.6767349
14: -34.8973312, 6.3713188, -34.9195442, 6.3779435, -36.8856125, 36.7249298
15: -8.6874428, 16.7773666, -8.7027740, 16.7569885, -23.2410126, 23.3052979
16: -22.7545547, 3.1321051, -22.8235054, 3.1037762, -25.8583317, 25.9556103
17: -28.0576019, 8.0457869, -28.0693207, 8.0680752, -36.1256790, 36.1151085
18: -12.6033907, 18.5728760, -12.5627098, 18.6851578, -29.3936920, 29.2158585
19: -8.9621162, 8.0187321, -8.9323502, 8.0526257, -16.5388145, 16.4704094
20: -9.9216614, 8.7577562, -9.9157076, 8.7916756, -17.7146759, 17.6669579
21: -12.5533333, 9.1447830, -12.5487452, 9.1734467, -20.2905502, 20.2126083
22: -2.7956309, 18.4928169, -2.7687554, 18.5462265, -18.7490768, 18.6510429
23: -3.9639058, 15.2251320, -3.9056025, 15.2894115, -17.5178070, 17.3688431
24: -5.5304289, 17.3221207, -5.4916797, 17.4113121, -19.8416595, 19.7184677
25: 2.0683565, 24.2616806, 2.1046920, 24.3194466, -19.4053726, 19.2878456
26: -11.7175703, 21.5423393, -11.6667204, 21.6596661, -33.3772354, 33.2090607
27: -15.0069914, 9.9709120, -14.9792690, 10.0429230, -23.8416824, 23.7821121
28: -3.0125628, 18.0026417, -2.9718242, 18.0611382, -19.5428467, 19.4721031
29: -3.3660274, 15.4932022, -3.3466940, 15.5387249, -15.1830215, 15.1286392
30: -13.5800390, 13.7676277, -13.5673809, 13.8314400, -24.7127991, 24.6497765
31: -9.7061901, 11.2914982, -9.6734591, 11.3397894, -21.0459785, 20.9649582
32: -30.7159195, -3.8002839, -30.7165356, -3.7962036, -22.9183998, 22.8804474
33: -41.4980240, -3.2430949, -41.5420074, -3.2326717, -31.1342621, 31.1674652
34: -36.8005486, -3.8834362, -36.7589951, -3.8158712, -25.3558655, 25.2504349
35: -24.9057999, 5.5157695, -24.9054489, 5.5305614, -25.9763718, 25.9873199
36: -24.5875587, 6.4111414, -24.5622025, 6.4233360, -28.0997086, 28.0639191
37: -42.9170189, -6.5743523, -42.8863602, -6.5686836, -32.2944946, 32.2222366
38: -34.3912544, 3.4622645, -34.3649521, 3.4990921, -35.6748047, 35.6148529
39: -47.7184715, -7.1032100, -47.7731209, -7.0886664, -37.3760834, 37.3634186
40: -45.9482422, -18.9892616, -45.9837112, -18.9732590, -21.4289856, 21.3764725
41: -33.5040398, -4.6751118, -33.4864960, -4.6729555, -22.2797012, 22.2475204
42: -24.0384636, -0.4478898, -24.0565643, -0.4435372, -19.6940193, 19.6986427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 604

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6114301, upper bound: 10.6502110
time: 25.16 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6306283, upper bound: 10.6502612
time: 49.25 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -22.4586525, 8.6067066, -22.4880505, 8.5827570, -31.0414085, 31.0947571
1: -11.8333378, 6.3521008, -11.8527317, 6.3386459, -18.1719837, 18.2048321
2: -13.0847692, 7.4271579, -13.1060038, 7.4149160, -19.3462982, 19.4124451
3: -18.5728035, 6.2514052, -18.6110268, 6.2373924, -24.1256256, 24.1873016
4: -20.2194366, 3.2702751, -20.2468224, 3.2620466, -21.1278610, 21.1634293
5: -18.0494118, 7.5566545, -18.0850563, 7.5393353, -25.0128860, 25.1210709
6: -36.7160225, -11.2758274, -36.7207909, -11.2886028, -20.7174187, 20.7359238
7: -24.0526085, 1.3310649, -24.0820103, 1.3151262, -23.8370895, 23.9080353
8: -27.3083000, 1.0843973, -27.3395691, 1.0731306, -25.1596985, 25.1959763
9: -11.5099087, 11.8241444, -11.5360699, 11.8088760, -20.5231705, 20.4652481
10: -17.6684036, 12.3161917, -17.6611328, 12.3211279, -29.6045837, 29.5496597
11: -16.7492085, 10.4158735, -16.7414703, 10.4442320, -23.6450729, 23.6202545
12: -24.1960793, 11.2244854, -24.1914024, 11.2500353, -33.3624268, 33.2745285
13: -22.3077755, 12.2899017, -22.3337193, 12.2613316, -32.4862442, 32.5047150
14: -34.8731232, 6.3514566, -34.8793716, 6.3585119, -36.7665482, 36.7022018
15: -8.6627903, 16.7569580, -8.6668234, 16.7538910, -23.2160797, 23.2426300
16: -22.7425327, 3.0980899, -22.7556648, 3.0776291, -25.8201618, 25.8537540
17: -28.0182686, 8.0276928, -28.0262337, 8.0389366, -36.0572052, 36.0539246
18: -12.5383205, 18.5183449, -12.5182877, 18.5617561, -29.2081833, 29.1302795
19: -8.9137173, 7.9965687, -8.9055157, 8.0100212, -16.4480972, 16.4223099
20: -9.8969603, 8.7502270, -9.8860817, 8.7567272, -17.6509209, 17.6358948
21: -12.5218773, 9.1318245, -12.5142660, 9.1393623, -20.2021751, 20.1826591
22: -2.7523079, 18.4868469, -2.7348337, 18.4970512, -18.6508636, 18.6184101
23: -3.8930764, 15.1871758, -3.8783569, 15.2137718, -17.3749390, 17.3221893
24: -5.4810319, 17.2861423, -5.4638453, 17.3180428, -19.6970978, 19.6721687
25: 2.1135821, 24.2370911, 2.1294394, 24.2598820, -19.3034134, 19.2531052
26: -11.6447372, 21.4951115, -11.6176167, 21.5343056, -33.1790428, 33.1127281
27: -14.9617825, 9.9450855, -14.9490185, 9.9664927, -23.7155762, 23.7327728
28: -2.9594460, 17.9832916, -2.9434090, 18.0015297, -19.4290848, 19.4332886
29: -3.3276715, 15.4814186, -3.3203187, 15.4934988, -15.0978508, 15.1000023
30: -13.5568819, 13.7463255, -13.5434494, 13.7631245, -24.6140823, 24.6156921
31: -9.6542053, 11.2658882, -9.6412125, 11.2844296, -20.9386349, 20.9071007
32: -30.6906509, -3.8099785, -30.6918488, -3.8118153, -22.8735809, 22.8496819
33: -41.4678307, -3.2673955, -41.4750824, -3.2757506, -31.1051865, 31.1301804
34: -36.7484665, -3.8934441, -36.7339783, -3.8773446, -25.2465973, 25.2392540
35: -24.8799381, 5.5042677, -24.8824921, 5.5108919, -25.9281464, 25.9647141
36: -24.5439320, 6.4001102, -24.5434837, 6.4048853, -28.0319366, 28.0333481
37: -42.8482895, -6.6207008, -42.8504372, -6.6073761, -32.1878281, 32.1840515
38: -34.3461380, 3.4352822, -34.3462448, 3.4584270, -35.5943069, 35.5580826
39: -47.6977692, -7.1242895, -47.7014236, -7.1342244, -37.2938995, 37.3145370
40: -45.9318237, -18.9954796, -45.9344254, -19.0038452, -21.3637581, 21.3886681
41: -33.4710503, -4.6940188, -33.4724121, -4.6917648, -22.2250328, 22.2248611
42: -24.0249710, -0.4600620, -24.0257015, -0.4679158, -19.6644936, 19.6382923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1628

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6499103, upper bound: 10.5959133
time: 27.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6502114, upper bound: 10.6006320
time: 129.50 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -22.4967270, 8.6964693, -22.5039177, 8.5833969, -31.0801239, 31.2003860
1: -11.8505344, 6.4028420, -11.8600712, 6.3393450, -18.1898804, 18.2629128
2: -13.1076965, 7.5029583, -13.1159916, 7.4160614, -19.3698196, 19.5004272
3: -18.6170349, 6.3519011, -18.6296349, 6.2389374, -24.1714554, 24.3081055
4: -20.2502480, 3.3382363, -20.2588959, 3.2631905, -21.1568604, 21.2465286
5: -18.0874023, 7.6473956, -18.1011600, 7.5409288, -25.0494690, 25.2355728
6: -36.7459145, -11.2636490, -36.7222977, -11.2873077, -20.7559166, 20.7464180
7: -24.0786514, 1.3968103, -24.0926781, 1.3163166, -23.8614731, 23.9907608
8: -27.3410645, 1.1497059, -27.3524246, 1.0740004, -25.1924438, 25.2742157
9: -11.5467892, 11.8930798, -11.5496750, 11.8096161, -20.5541153, 20.5479431
10: -17.7178478, 12.3510885, -17.6635323, 12.3269119, -29.6616821, 29.5867462
11: -16.8333454, 10.4560604, -16.7425537, 10.4610167, -23.7472687, 23.6537933
12: -24.2707615, 11.2640343, -24.1935768, 11.2634668, -33.4672470, 33.3101120
13: -22.3496971, 12.3874006, -22.3496284, 12.2630806, -32.5255585, 32.6193466
14: -34.9004059, 6.3681126, -34.8848495, 6.3593969, -36.8191376, 36.7317047
15: -8.6920414, 16.8105049, -8.6759548, 16.7549343, -23.2441788, 23.3076324
16: -22.7662678, 3.1300976, -22.7604351, 3.0783548, -25.8446236, 25.8905334
17: -28.0367317, 8.0435867, -28.0296173, 8.0385017, -36.0752335, 36.0732040
18: -12.6396589, 18.5696678, -12.5206442, 18.5831146, -29.3317795, 29.1799393
19: -8.9799290, 8.0158777, -8.9073925, 8.0182686, -16.5273399, 16.4429474
20: -9.9556398, 8.7683754, -9.8878222, 8.7633352, -17.7204514, 17.6544838
21: -12.5823088, 9.1485777, -12.5162334, 9.1454105, -20.2725525, 20.1991272
22: -2.7939763, 18.4971733, -2.7366529, 18.5001984, -18.7060585, 18.6269398
23: -3.9708533, 15.2146816, -3.8797960, 15.2254391, -17.4658127, 17.3478241
24: -5.5627160, 17.3235054, -5.4648776, 17.3333015, -19.7952957, 19.7065353
25: 2.0409184, 24.2637882, 2.1277294, 24.2704353, -19.3886948, 19.2803497
26: -11.7595062, 21.5425816, -11.6197100, 21.5540695, -33.3135757, 33.1622925
27: -15.0492764, 9.9765072, -14.9507408, 9.9791594, -23.8191147, 23.7609978
28: -3.0377665, 18.0051231, -2.9454103, 18.0100555, -19.5161591, 19.4570694
29: -3.3722568, 15.4983616, -3.3219738, 15.4989481, -15.1496239, 15.1177177
30: -13.6181231, 13.7745781, -13.5440006, 13.7739582, -24.6883698, 24.6428490
31: -9.7307358, 11.2924757, -9.6431313, 11.2947636, -21.0254993, 20.9356079
32: -30.7256222, -3.7962718, -30.6939945, -3.8109293, -22.9211426, 22.8610229
33: -41.5076218, -3.1983533, -41.4837875, -3.2736402, -31.1345596, 31.1758347
34: -36.7949104, -3.8834057, -36.7360153, -3.8747611, -25.3021469, 25.2499619
35: -24.8914280, 5.5133572, -24.8773880, 5.5125589, -25.9459000, 25.9658203
36: -24.5837002, 6.4080272, -24.5449715, 6.4062586, -28.0781021, 28.0420456
37: -42.8925629, -6.5998878, -42.8531151, -6.6022472, -32.2597885, 32.2035904
38: -34.4204025, 3.4611359, -34.3494949, 3.4681921, -35.6780701, 35.5848083
39: -47.7222710, -7.1103868, -47.7015076, -7.1331849, -37.3431396, 37.3114166
40: -45.9539261, -18.9884567, -45.9366379, -19.0033817, -21.4136810, 21.3898239
41: -33.5159950, -4.6765728, -33.4739838, -4.6874661, -22.2782593, 22.2425804
42: -24.0487690, -0.4474690, -24.0271950, -0.4665654, -19.6933289, 19.6532326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 874

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1628

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6438343, upper bound: 10.5952912
time: 26.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6502615, upper bound: 10.6173453
time: 130.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.5300350, 8.6744251, -22.5217762, 8.5826731, -31.1127090, 31.1962013
1: -11.8781929, 6.4172907, -11.8750935, 6.3394060, -18.2175980, 18.2923851
2: -13.1322060, 7.4918761, -13.1293907, 7.4163055, -19.3952713, 19.4949646
3: -18.6600456, 6.3399048, -18.6531601, 6.2403002, -24.2161484, 24.3129425
4: -20.2842941, 3.3491592, -20.2780762, 3.2644095, -21.1888885, 21.2608871
5: -18.1300621, 7.6454339, -18.1243744, 7.5416732, -25.0978851, 25.2509537
6: -36.7445412, -11.2523746, -36.7191391, -11.2845268, -20.7530823, 20.7690277
7: -24.1197166, 1.4144046, -24.1148643, 1.3167498, -23.9049530, 24.0174561
8: -27.3794708, 1.1628733, -27.3741112, 1.0748873, -25.2277908, 25.2877655
9: -11.5784740, 11.8929605, -11.5672598, 11.8095484, -20.5835800, 20.5347061
10: -17.6983662, 12.3517065, -17.6678200, 12.3255367, -29.6434326, 29.5960999
11: -16.8214817, 10.4903946, -16.7470589, 10.4790468, -23.7341156, 23.6919632
12: -24.2577629, 11.2925797, -24.1958351, 11.2802010, -33.4732513, 33.3494263
13: -22.3765602, 12.3571949, -22.3657532, 12.2637405, -32.5504150, 32.5800705
14: -34.9175377, 6.3811197, -34.8948402, 6.3671088, -36.8692780, 36.7528992
15: -8.7086258, 16.8009644, -8.6849203, 16.7555962, -23.2633896, 23.3022156
16: -22.7898769, 3.1432667, -22.7731628, 3.0782218, -25.8680992, 25.9164295
17: -28.0907707, 8.0666180, -28.0404320, 8.0540867, -36.1448593, 36.1070480
18: -12.6271982, 18.6189423, -12.5233459, 18.6101036, -29.3308868, 29.2325668
19: -8.9877548, 8.0367374, -8.9105091, 8.0287971, -16.5393677, 16.4676399
20: -9.9375076, 8.7733240, -9.8893414, 8.7664671, -17.7009735, 17.6615295
21: -12.5712824, 9.1565819, -12.5200338, 9.1486216, -20.2812347, 20.2120361
22: -2.8109684, 18.5134468, -2.7390447, 18.5086823, -18.7204933, 18.6482182
23: -3.9906683, 15.2530746, -3.8813314, 15.2457314, -17.4946060, 17.3860207
24: -5.5577106, 17.3599339, -5.4657421, 17.3533249, -19.7921600, 19.7427979
25: 2.0389996, 24.2907963, 2.1260967, 24.2848549, -19.3947372, 19.3074493
26: -11.7431698, 21.5862999, -11.6215258, 21.5780449, -33.3212128, 33.2078247
27: -15.0206852, 9.9989319, -14.9521866, 9.9919558, -23.7987595, 23.7881699
28: -3.0416493, 18.0289993, -2.9475904, 18.0225754, -19.5232162, 19.4816322
29: -3.3854747, 15.5128918, -3.3268409, 15.5074682, -15.1543045, 15.1348305
30: -13.6008368, 13.7958746, -13.5456133, 13.7847862, -24.6668320, 24.6671524
31: -9.7351170, 11.3166275, -9.6455975, 11.3071756, -21.0422935, 20.9622250
32: -30.7235947, -3.7946830, -30.6941376, -3.8080564, -22.9150162, 22.8738213
33: -41.5154572, -3.2422943, -41.4865799, -3.2693796, -31.1547012, 31.1614914
34: -36.8152084, -3.8513465, -36.7365417, -3.8578701, -25.3301010, 25.2807388
35: -24.9133492, 5.5312991, -24.8876572, 5.5212989, -25.9763031, 25.9967880
36: -24.5973663, 6.4233990, -24.5458126, 6.4144654, -28.0974274, 28.0603867
37: -42.9281464, -6.5705948, -42.8566818, -6.5848408, -32.2915497, 32.2612839
38: -34.4171867, 3.4934025, -34.3503113, 3.4860263, -35.6826706, 35.6176910
39: -47.7373581, -7.1062813, -47.7117615, -7.1289392, -37.3603439, 37.3495636
40: -45.9629745, -18.9859352, -45.9399605, -19.0028248, -21.4085808, 21.4167442
41: -33.5094681, -4.6683283, -33.4753799, -4.6823664, -22.2751884, 22.2587624
42: -24.0544472, -0.4431462, -24.0294285, -0.4655843, -19.6921005, 19.6716747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 874

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 604

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6324587, upper bound: 10.6356156
time: 26.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6502613, upper bound: 10.6356940
time: 30.91 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -22.4610615, 8.6407566, -22.5642776, 8.6473370, -31.1083984, 31.2050343
1: -11.8340292, 6.3720465, -11.8930492, 6.3783741, -18.2124023, 18.2650948
2: -13.0855236, 7.4457850, -13.1400490, 7.4514132, -19.4074631, 19.4948730
3: -18.5742455, 6.2744799, -18.6627045, 6.2834082, -24.2141571, 24.3092499
4: -20.2213745, 3.2851145, -20.2835808, 3.2923357, -21.2143631, 21.2811127
5: -18.0507374, 7.5831814, -18.1410332, 7.5921869, -25.0827332, 25.2214279
6: -36.7203369, -11.2645359, -36.7554550, -11.2621069, -20.7356339, 20.7739296
7: -24.0539341, 1.3540428, -24.1373787, 1.3612397, -23.9477310, 24.0570450
8: -27.3096218, 1.1043005, -27.3839607, 1.1157889, -25.3245468, 25.3979492
9: -11.5121050, 11.8485603, -11.5874081, 11.8542233, -20.7177048, 20.7139435
10: -17.6836853, 12.3232327, -17.6988983, 12.3663311, -29.6659927, 29.5922623
11: -16.7671547, 10.4176636, -16.7862778, 10.4909391, -23.7853928, 23.7219543
12: -24.2040272, 11.2280560, -24.2155933, 11.2835331, -33.3587875, 33.2485962
13: -22.3113289, 12.3320427, -22.4277058, 12.3429527, -32.6194763, 32.7038193
14: -34.8798218, 6.3553119, -34.9112244, 6.3732567, -36.7452240, 36.6794510
15: -8.6671505, 16.7642975, -8.6884289, 16.7721462, -23.2761002, 23.3158569
16: -22.7455521, 3.1206732, -22.8099232, 3.1220834, -25.8676357, 25.9305954
17: -28.0229111, 8.0327492, -28.0626564, 8.0570068, -36.0799179, 36.0954056
18: -12.5715513, 18.5204659, -12.5873079, 18.6387348, -29.4119339, 29.2828903
19: -8.9323416, 7.9973750, -8.9489365, 8.0346632, -16.5194168, 16.4925919
20: -9.9175367, 8.7509680, -9.9310436, 8.7828903, -17.7220573, 17.7018509
21: -12.5400887, 9.1331501, -12.5583525, 9.1658726, -20.2162323, 20.1930847
22: -2.7728381, 18.4876862, -2.7817979, 18.5355911, -18.7440720, 18.6926689
23: -3.9182191, 15.1888218, -3.9301653, 15.2588072, -17.5326996, 17.4508972
24: -5.5088944, 17.2870598, -5.5202417, 17.3769207, -19.9174652, 19.8459358
25: 2.0900245, 24.2380428, 2.0795021, 24.2954102, -19.4215775, 19.3534164
26: -11.6824207, 21.4976578, -11.6965847, 21.6180077, -33.3004303, 33.1942444
27: -14.9837780, 9.9469433, -14.9946918, 10.0193748, -23.8020020, 23.7866096
28: -2.9841452, 17.9851494, -2.9963226, 18.0414047, -19.5621567, 19.5475197
29: -3.3388047, 15.4829292, -3.3519430, 15.5261822, -15.1670551, 15.1446381
30: -13.5785303, 13.7489843, -13.5878525, 13.8120537, -24.7224426, 24.6825409
31: -9.6796646, 11.2674370, -9.6970196, 11.3190117, -20.9986763, 20.9644566
32: -30.6960888, -3.8033743, -30.7186298, -3.7948003, -22.8373489, 22.8222504
33: -41.4770813, -3.2457089, -41.5402985, -3.2301564, -30.9669571, 31.0270081
34: -36.7668610, -3.8919802, -36.7739143, -3.8336210, -25.3835678, 25.3467522
35: -24.8846054, 5.5057778, -24.9052601, 5.5225248, -25.9566269, 25.9940262
36: -24.5487003, 6.4027724, -24.5666866, 6.4158702, -28.0778351, 28.0863419
37: -42.8550644, -6.6152172, -42.8877602, -6.5892887, -32.0585251, 32.0493927
38: -34.3559380, 3.4364486, -34.3779678, 3.4737763, -35.7164459, 35.6901398
39: -47.7045708, -7.1045794, -47.7692299, -7.0890026, -37.0633240, 37.1075363
40: -45.9375763, -18.9790516, -45.9826393, -18.9679604, -21.1506729, 21.1651230
41: -33.4744034, -4.6906610, -33.4873199, -4.6790538, -22.2366104, 22.2284355
42: -24.0292511, -0.4467936, -24.0579948, -0.4373386, -19.7146912, 19.7071304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1628

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6499103, upper bound: 10.6093008
time: 28.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6502114, upper bound: 10.6140888
time: 18.37 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -22.4996452, 8.7307653, -22.5800858, 8.6479483, -31.1475945, 31.3108521
1: -11.8514118, 6.4231343, -11.9004421, 6.3790674, -18.2304802, 18.3235760
2: -13.1084585, 7.5217853, -13.1500177, 7.4525704, -19.4303818, 19.5846481
3: -18.6187935, 6.3752265, -18.6813450, 6.2849846, -24.2591171, 24.4324799
4: -20.2524567, 3.3532867, -20.2956886, 3.2934306, -21.2448349, 21.3644257
5: -18.0887871, 7.6740541, -18.1570759, 7.5938125, -25.1190186, 25.3369598
6: -36.7530518, -11.2521381, -36.7570572, -11.2607546, -20.7732010, 20.7861214
7: -24.0800648, 1.4201391, -24.1480446, 1.3624139, -23.9704590, 24.1442261
8: -27.3426018, 1.1699324, -27.3968697, 1.1166754, -25.3546219, 25.4827881
9: -11.5494308, 11.9175949, -11.6010094, 11.8549500, -20.7509003, 20.7960587
10: -17.7354927, 12.3582916, -17.7012444, 12.3719873, -29.7254791, 29.6295547
11: -16.8515778, 10.4588795, -16.7873726, 10.5077248, -23.8876419, 23.7576637
12: -24.2789917, 11.2679605, -24.2177620, 11.2969761, -33.4627380, 33.2859726
13: -22.3539581, 12.4296827, -22.4436302, 12.3446932, -32.6604233, 32.8182907
14: -34.9081993, 6.3719811, -34.9166451, 6.3739028, -36.7828903, 36.7082443
15: -8.6973515, 16.8178253, -8.6975431, 16.7731323, -23.3054276, 23.3809433
16: -22.7695999, 3.1529839, -22.8147163, 3.1227710, -25.8923702, 25.9677010
17: -28.0422535, 8.0504265, -28.0660076, 8.0571136, -36.0993652, 36.1164322
18: -12.6732359, 18.5722790, -12.5896273, 18.6600533, -29.5358353, 29.3336029
19: -8.9987345, 8.0170107, -8.9508228, 8.0429039, -16.5988426, 16.5138474
20: -9.9763889, 8.7697506, -9.9327574, 8.7893724, -17.7915573, 17.7211151
21: -12.6007185, 9.1503353, -12.5602961, 9.1718960, -20.2881203, 20.2095909
22: -2.8148546, 18.4982319, -2.7836094, 18.5387268, -18.8012466, 18.7005920
23: -3.9961481, 15.2164078, -3.9316216, 15.2704601, -17.6237526, 17.4777794
24: -5.5908880, 17.3246651, -5.5213695, 17.3921928, -20.0159912, 19.8817825
25: 2.0170174, 24.2650566, 2.0777497, 24.3059425, -19.5070877, 19.3813705
26: -11.7979326, 21.5455284, -11.6987038, 21.6377182, -33.4356499, 33.2442322
27: -15.0714283, 9.9785204, -14.9964256, 10.0320454, -23.9057388, 23.8151016
28: -3.0626607, 18.0077972, -2.9983573, 18.0499268, -19.6513824, 19.5710220
29: -3.3837576, 15.5004063, -3.3535686, 15.5316153, -15.2211418, 15.1641464
30: -13.6401243, 13.7785454, -13.5884762, 13.8228521, -24.7978592, 24.7132645
31: -9.7563496, 11.2945137, -9.6989079, 11.3293457, -21.0856953, 20.9934216
32: -30.7332573, -3.7884803, -30.7209702, -3.7938018, -22.8829689, 22.8357391
33: -41.5201569, -3.1779118, -41.5487175, -3.2280684, -31.0078964, 31.0949173
34: -36.8136253, -3.8817229, -36.7759857, -3.8310070, -25.4432220, 25.3557968
35: -24.8963413, 5.5156260, -24.9010315, 5.5242176, -25.9747162, 25.9954681
36: -24.5888119, 6.4113436, -24.5681553, 6.4172211, -28.1260529, 28.0948105
37: -42.9008179, -6.5944271, -42.8905373, -6.5842333, -32.1278687, 32.0722351
38: -34.4305267, 3.4629817, -34.3812332, 3.4836011, -35.8014679, 35.7186890
39: -47.7311058, -7.0889802, -47.7692146, -7.0880108, -37.1108246, 37.1159363
40: -45.9650803, -18.9722748, -45.9849396, -18.9676838, -21.1963310, 21.1721649
41: -33.5201492, -4.6729980, -33.4889450, -4.6747293, -22.2906799, 22.2462959
42: -24.0561180, -0.4339397, -24.0595093, -0.4359715, -19.7444038, 19.7247925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1628

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6500578, upper bound: 10.6169399
time: 29.16 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6502615, upper bound: 10.6319193
time: 31.69 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -22.4821129, 8.6341248, -22.5950489, 8.6053276, -31.0874405, 31.2291737
1: -11.8486242, 6.3879132, -11.9141130, 6.3515120, -18.2001362, 18.3020267
2: -13.1068840, 7.4630609, -13.1626043, 7.4260464, -19.3748093, 19.5048523
3: -18.6259880, 6.3079605, -18.7035599, 6.2562475, -24.1924438, 24.3357315
4: -20.2565613, 3.3283796, -20.3119392, 3.2753048, -21.1632233, 21.2832947
5: -18.0948238, 7.6135187, -18.1786919, 7.5625176, -25.0792618, 25.2721939
6: -36.7219162, -11.2525139, -36.7438736, -11.2616024, -20.7478790, 20.7809334
7: -24.0830021, 1.3871062, -24.1687088, 1.3347788, -23.8811035, 24.0502396
8: -27.3402119, 1.1312885, -27.4164867, 1.0897951, -25.1916046, 25.3084793
9: -11.5421238, 11.8596382, -11.6156464, 11.8229580, -20.5454254, 20.5780945
10: -17.6791000, 12.3254766, -17.6900978, 12.3614731, -29.6619263, 29.5902023
11: -16.8007088, 10.4669552, -16.7758846, 10.5233698, -23.7758255, 23.6815948
12: -24.2237358, 11.2584038, -24.1986160, 11.3087521, -33.4783783, 33.3103180
13: -22.3338585, 12.3235893, -22.4564095, 12.3055458, -32.5390930, 32.6516647
14: -34.8909187, 6.3688221, -34.9184418, 6.3751292, -36.8702698, 36.7554855
15: -8.6861238, 16.7771149, -8.7008858, 16.7575798, -23.2410660, 23.3003197
16: -22.7489262, 3.1152406, -22.8224144, 3.0949788, -25.8439045, 25.9376545
17: -28.0494919, 8.0466938, -28.0697174, 8.0617714, -36.1112633, 36.1164093
18: -12.5921202, 18.5723991, -12.5572872, 18.6849403, -29.3837967, 29.2117004
19: -8.9532604, 8.0185452, -8.9282894, 8.0526733, -16.5284233, 16.4622993
20: -9.8969221, 8.7512026, -9.9022827, 8.7913742, -17.6902390, 17.6476936
21: -12.5319090, 9.1362591, -12.5364637, 9.1733055, -20.2687187, 20.2057571
22: -2.7803702, 18.4875698, -2.7610683, 18.5458488, -18.7339592, 18.6371651
23: -3.9626746, 15.2295933, -3.9052925, 15.2891903, -17.5207672, 17.3761559
24: -5.5297618, 17.3287830, -5.4924440, 17.4112587, -19.8393555, 19.7257233
25: 2.0743103, 24.2677727, 2.1070142, 24.3192711, -19.4046631, 19.2949524
26: -11.7062731, 21.5432224, -11.6616297, 21.6593552, -33.3656273, 33.2048531
27: -14.9897060, 9.9644279, -14.9712877, 10.0424690, -23.8221130, 23.7658463
28: -3.0124488, 18.0078163, -2.9731703, 18.0607967, -19.5406685, 19.4793968
29: -3.3679414, 15.4959240, -3.3479781, 15.5382490, -15.1796665, 15.1312981
30: -13.5806656, 13.7714787, -13.5678301, 13.8303623, -24.7061462, 24.6504822
31: -9.6878738, 11.2888861, -9.6643753, 11.3399944, -21.0278683, 20.9532623
32: -30.6944122, -3.8028603, -30.7050419, -3.7955585, -22.8961983, 22.8463478
33: -41.4893913, -3.2349906, -41.5371246, -3.2280736, -31.1553116, 31.1620865
34: -36.7710381, -3.8954787, -36.7425003, -3.8166246, -25.3298035, 25.2365494
35: -24.8952045, 5.5081525, -24.9004841, 5.5293856, -25.9682693, 25.9753952
36: -24.5713806, 6.4031816, -24.5539474, 6.4231381, -28.0793381, 28.0463409
37: -42.9022446, -6.5784206, -42.8806534, -6.5690098, -32.2895126, 32.2115860
38: -34.3601151, 3.4460268, -34.3487053, 3.4996195, -35.6435089, 35.5766983
39: -47.7109108, -7.1025939, -47.7672348, -7.0860410, -37.3613434, 37.3259811
40: -45.9440002, -18.9808197, -45.9782104, -18.9694214, -21.4265137, 21.3693924
41: -33.4877472, -4.6831264, -33.4788666, -4.6731691, -22.2678604, 22.2258720
42: -24.0365829, -0.4431338, -24.0521965, -0.4400532, -19.6946983, 19.6876373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 604

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6277269, upper bound: 10.6499100
time: 23.27 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6353176, upper bound: 10.6500574
time: 29.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.5326157, 8.7085285, -22.5979004, 8.6471977, -31.1798134, 31.3064289
1: -11.8788929, 6.4373693, -11.9153957, 6.3791175, -18.2580109, 18.3527641
2: -13.1329241, 7.5105572, -13.1634293, 7.4527740, -19.4572525, 19.5801544
3: -18.6615696, 6.3629999, -18.7048798, 6.2862921, -24.3058014, 24.4386063
4: -20.2862740, 3.3640847, -20.3148499, 3.2946303, -21.2769547, 21.3872910
5: -18.1313381, 7.6720181, -18.1803398, 7.5945139, -25.1686707, 25.3510895
6: -36.7496529, -11.2410097, -36.7536926, -11.2580624, -20.7717972, 20.8105507
7: -24.1210575, 1.4374845, -24.1702347, 1.3628142, -24.0173950, 24.1727066
8: -27.3808327, 1.1828604, -27.4185715, 1.1174307, -25.3980865, 25.5055618
9: -11.5807686, 11.9173470, -11.6185713, 11.8548470, -20.7792358, 20.8004303
10: -17.7145176, 12.3586769, -17.7053528, 12.3706703, -29.7056656, 29.6384048
11: -16.8395271, 10.4925003, -16.7918129, 10.5257483, -23.8848953, 23.7933655
12: -24.2658844, 11.2961788, -24.2200050, 11.3137045, -33.4691162, 33.3207932
13: -22.3803177, 12.3993587, -22.4597015, 12.3452435, -32.6866379, 32.7911148
14: -34.9246063, 6.3849969, -34.9265938, 6.3816552, -36.8393173, 36.7329865
15: -8.7133026, 16.8083611, -8.7064810, 16.7738113, -23.3246536, 23.3811226
16: -22.7929764, 3.1659696, -22.8274155, 3.1226516, -25.9156284, 25.9933853
17: -28.0958862, 8.0722923, -28.0768242, 8.0720654, -36.1679535, 36.1491165
18: -12.6605244, 18.6212196, -12.5922346, 18.6870766, -29.5422363, 29.3853989
19: -9.0064077, 8.0376234, -8.9538803, 8.0534134, -16.6135063, 16.5376663
20: -9.9581261, 8.7742605, -9.9342594, 8.7925539, -17.7746735, 17.7272034
21: -12.5895844, 9.1579943, -12.5640373, 9.1750736, -20.2916603, 20.2229004
22: -2.8316445, 18.5143394, -2.7859864, 18.5472088, -18.8172150, 18.7227402
23: -4.0158463, 15.2547035, -3.9330778, 15.2907362, -17.6599846, 17.5143890
24: -5.5856109, 17.3609009, -5.5220995, 17.4122124, -20.0221710, 19.9172745
25: 2.0153713, 24.2918510, 2.0762396, 24.3203716, -19.5167732, 19.4079590
26: -11.7810755, 21.5889435, -11.7004595, 21.6617222, -33.4427986, 33.2894020
27: -15.0427036, 10.0008087, -14.9977942, 10.0447893, -23.8860779, 23.8421211
28: -3.0663719, 18.0311279, -3.0004444, 18.0624123, -19.6622696, 19.5970612
29: -3.3967214, 15.5145788, -3.3583937, 15.5401335, -15.2313557, 15.1805954
30: -13.6224823, 13.7989540, -13.5900049, 13.8336239, -24.7826920, 24.7353058
31: -9.7606144, 11.3183079, -9.7012997, 11.3417301, -21.1023445, 21.0196075
32: -30.7298164, -3.7879701, -30.7205944, -3.7911143, -22.8784676, 22.8425674
33: -41.5258141, -3.2205462, -41.5516739, -3.2238574, -31.0258560, 31.0575943
34: -36.8336754, -3.8497958, -36.7764359, -3.8141685, -25.4717331, 25.3918190
35: -24.9181156, 5.5330253, -24.9105110, 5.5328994, -26.0047455, 26.0261383
36: -24.6022797, 6.4261804, -24.5690556, 6.4253664, -28.1436844, 28.1153107
37: -42.9354744, -6.5652599, -42.8941650, -6.5668135, -32.1630402, 32.1186676
38: -34.4270935, 3.4947858, -34.3820496, 3.5014148, -35.8137589, 35.7505646
39: -47.7448997, -7.0864840, -47.7794952, -7.0837445, -37.1325378, 37.1360779
40: -45.9705353, -18.9691925, -45.9882202, -18.9666748, -21.1961784, 21.1848869
41: -33.5130844, -4.6649241, -33.4904099, -4.6696873, -22.2872505, 22.2620430
42: -24.0598030, -0.4297826, -24.0616474, -0.4350507, -19.7436028, 19.7406960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=161, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=209, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 874

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 604

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6324587, upper bound: 10.6502110
time: 33.30 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -10.6502613, upper bound: 10.6502612
time: 28.42 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 63.90 seconds
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6066846, upper bound: 10.6499100
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6157024, upper bound: 10.6500574
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6114301, upper bound: 10.6502110
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6306283, upper bound: 10.6502612
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6499103, upper bound: 10.5959133
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6502114, upper bound: 10.6006320
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6438343, upper bound: 10.5952912
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6502615, upper bound: 10.6173453
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6324587, upper bound: 10.6356156
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6502613, upper bound: 10.6356940
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6499103, upper bound: 10.6093008
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6502114, upper bound: 10.6140888
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6500578, upper bound: 10.6169399
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6502615, upper bound: 10.6319193
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6277269, upper bound: 10.6499100
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6353176, upper bound: 10.6500574
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6324587, upper bound: 10.6502110
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 63.90
Output dim: 25, lower bound: -10.6502613, upper bound: 10.6502612

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -22.4116707, 8.5698872, -22.5572186, 8.5655956, -30.9772663, 31.1271057
1: -11.8068695, 6.3465014, -11.8958797, 6.3256025, -18.1324730, 18.2423820
2: -13.0687199, 7.4263616, -13.1395626, 7.4024820, -19.3126068, 19.4519882
3: -18.5633736, 6.2549677, -18.6608524, 6.2227459, -24.0956955, 24.2474823
4: -20.2080269, 3.2851062, -20.2830772, 3.2480271, -21.0852661, 21.2240295
5: -18.0349541, 7.5584617, -18.1411781, 7.5285687, -24.9835663, 25.1794052
6: -36.7067261, -11.2664566, -36.7335434, -11.2708473, -20.7046928, 20.7436905
7: -24.0251923, 1.3385122, -24.1427708, 1.3047233, -23.7903061, 23.9820862
8: -27.2835579, 1.0815783, -27.3862686, 1.0593996, -25.1070862, 25.2545471
9: -11.4873571, 11.8054371, -11.5828772, 11.7899475, -20.4541397, 20.5225143
10: -17.6533661, 12.2950239, -17.6764584, 12.3404255, -29.6135864, 29.5414276
11: -16.7526703, 10.4174232, -16.7516251, 10.4842300, -23.7019348, 23.6008644
12: -24.1887531, 11.2073269, -24.1775208, 11.2737379, -33.4029694, 33.2241592
13: -22.2721748, 12.2674437, -22.4166241, 12.2700071, -32.4383163, 32.5702591
14: -34.8570480, 6.3486719, -34.8977127, 6.3597369, -36.8261566, 36.6742477
15: -8.6501493, 16.7441578, -8.6757555, 16.7369118, -23.1821594, 23.2490463
16: -22.7041245, 3.0803549, -22.8059483, 3.0737906, -25.7779160, 25.8863029
17: -28.0065918, 8.0165329, -28.0522232, 8.0491314, -36.0557251, 36.0687561
18: -12.5310383, 18.5000916, -12.5199394, 18.6359100, -29.2847977, 29.1000977
19: -8.9055929, 7.9904222, -8.9001427, 8.0335989, -16.4616280, 16.4048386
20: -9.8575153, 8.7270737, -9.8779287, 8.7751837, -17.6387405, 17.5967827
21: -12.4921579, 9.1162605, -12.5143328, 9.1580143, -20.2127190, 20.1463928
22: -2.7415662, 18.4623795, -2.7382817, 18.5375481, -18.6906509, 18.5878296
23: -3.9083271, 15.1868029, -3.8730755, 15.2619801, -17.4420776, 17.2927475
24: -5.4718537, 17.2727566, -5.4566159, 17.3765106, -19.7649384, 19.6331177
25: 2.1305227, 24.2259750, 2.1419454, 24.2948647, -19.3280563, 19.2132072
26: -11.6393042, 21.4743843, -11.6207047, 21.6135979, -33.2529030, 33.0950890
27: -14.9507322, 9.9199753, -14.9463491, 10.0121288, -23.7564011, 23.6947098
28: -2.9551892, 17.9699402, -2.9377756, 18.0405464, -19.4730072, 19.4036713
29: -3.3344107, 15.4686775, -3.3306131, 15.5248547, -15.1484432, 15.0884171
30: -13.5355663, 13.7288780, -13.5396776, 13.8039818, -24.6520538, 24.5788612
31: -9.6299276, 11.2505608, -9.6296120, 11.3149862, -20.9449139, 20.8801727
32: -30.6780663, -3.8179712, -30.6942596, -3.8057323, -22.8702507, 22.8155861
33: -41.4497032, -3.2604027, -41.5030556, -3.2425919, -31.0883789, 31.0810928
34: -36.7352486, -3.9327011, -36.7197151, -3.8253231, -25.2826385, 25.1598434
35: -24.8778076, 5.4887519, -24.8853874, 5.5225272, -25.9376678, 25.9319229
36: -24.5544930, 6.3862805, -24.5425243, 6.4169683, -28.0579071, 28.0170364
37: -42.8803787, -6.5950327, -42.8646317, -6.5857754, -32.2407532, 32.1360245
38: -34.3199005, 3.4021363, -34.3227921, 3.4743686, -35.5890274, 35.5126572
39: -47.6774597, -7.1213007, -47.7457466, -7.0947485, -37.3317337, 37.2605896
40: -45.9214554, -19.0019760, -45.9671745, -18.9782257, -21.3976021, 21.3040810
41: -33.4767265, -4.6986113, -33.4704132, -4.6870785, -22.2397041, 22.1940765
42: -24.0141888, -0.4635370, -24.0427704, -0.4531674, -19.6594467, 19.6651249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=160, inp2_unstable=160, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 874

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 569

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.5912640, upper bound: 10.6491109
time: 30.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 25, lower bound: -10.6058852, upper bound: 10.6491109
time: 29.30 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 61.72 seconds
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 61.72
Output dim: 25, lower bound: -10.5912640, upper bound: 10.6491109
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 61.72
Output dim: 25, lower bound: -10.6058852, upper bound: 10.6491109
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 61.72
Output dim: 25, lower bound: -10.6157024, upper bound: 10.6500574
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 61.72
Output dim: 25, lower bound: -10.6114301, upper bound: 10.6502110
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 61.72
Output dim: 25, lower bound: -10.6306283, upper bound: 10.6502612
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 61.72
Output dim: 25, lower bound: -10.6499103, upper bound: 10.5959133
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 61.72
Output dim: 25, lower bound: -10.6502114, upper bound: 10.6006320
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 61.72
Output dim: 25, lower bound: -10.6502615, upper bound: 10.6173453
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 61.72
Output dim: 25, lower bound: -10.6502613, upper bound: 10.6356940
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 61.72
Output dim: 25, lower bound: -10.6499103, upper bound: 10.6093008
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 61.72
Output dim: 25, lower bound: -10.6502114, upper bound: 10.6140888
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 61.72
Output dim: 25, lower bound: -10.6500578, upper bound: 10.6169399
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 61.72
Output dim: 25, lower bound: -10.6502615, upper bound: 10.6319193
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 61.72
Output dim: 25, lower bound: -10.6277269, upper bound: 10.6499100
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 61.72
Output dim: 25, lower bound: -10.6353176, upper bound: 10.6500574
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 61.72
Output dim: 25, lower bound: -10.6324587, upper bound: 10.6502110
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 61.72
Output dim: 25, lower bound: -10.6502613, upper bound: 10.6502612

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 46.41 + 1775.07 = 1821.48 seconds
