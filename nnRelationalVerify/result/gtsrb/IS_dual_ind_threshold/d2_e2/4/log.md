## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 3600 seconds
Split limit: 100
Threshold: 24.3060988707


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627)
1: (-21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790)
2: (-15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286)
3: (-20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897)
4: (-24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519)
5: (-19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280)
6: (-33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090)
7: (-24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857)
8: (-27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963)
9: (-24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217)
10: (-32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655)
11: (-27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078)
12: (-32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3501511, 54.3501358)
13: (-31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802)
14: (-51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714)
15: (-26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130)
16: (-34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668)
17: (-50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355)
18: (-35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095)
19: (-20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903)
20: (-20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152)
21: (-26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219)
22: (-25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513)
23: (-19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472)
24: (-27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234)
25: (-21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741)
26: (-35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622)
27: (-26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534)
28: (-20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425)
29: (-24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824)
30: (-26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868)
31: (-28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047)
32: (-31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8737335, 46.8737335)
33: (-52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6555099, 68.6555023)
34: (-45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8373260, 49.8373260)
35: (-41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2313690, 54.2313766)
36: (-36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5736771, 54.5736771)
37: (-59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5318146, 67.5318069)
38: (-45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743)
39: (-55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1515961, 73.1516037)
40: (-44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7571106, 50.7571182)
41: (-35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3369751, 52.3369713)
42: (-23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.81 + 77.96 = 80.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 20, lower bound: -24.3304293, upper bound: 24.3304293

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 701

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2934227, upper bound: 24.3293512
time: 63.62 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2934227, upper bound: 24.3294822
time: 60.16 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 123.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 123.89
Output dim: 20, lower bound: -24.2934227, upper bound: 24.3293512
IS_A2, status: Status.UNKNOWN, split count: 1, time: 123.89
Output dim: 20, lower bound: -24.2934227, upper bound: 24.3294822

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -42.9980545, 19.3566475, -43.0101089, 19.3916149, -62.3896713, 62.3667564
1: -21.2886810, 18.1858768, -21.2933483, 18.2048092, -39.4934921, 39.4792252
2: -15.0332346, 20.1806679, -15.0391407, 20.2038364, -35.2370720, 35.2198105
3: -20.4405804, 22.4295311, -20.4452705, 22.4561901, -42.8967705, 42.8748016
4: -24.0957203, 20.0222893, -24.1042747, 20.0442600, -44.1399803, 44.1265640
5: -19.1840897, 21.6075783, -19.1899796, 21.6313553, -40.8154449, 40.7975578
6: -33.2146492, 15.8682251, -33.2301941, 15.8768749, -49.0915222, 49.0984192
7: -24.6238289, 19.6139297, -24.6297913, 19.6322060, -44.2560349, 44.2437210
8: -27.8630905, 26.8773613, -27.8707047, 26.9066505, -54.7697411, 54.7480659
9: -24.0501404, 23.0480461, -24.0597572, 23.0745430, -47.1246834, 47.1078033
10: -32.4192085, 25.0171795, -32.4282990, 25.0380859, -57.4572945, 57.4454803
11: -26.9833603, 16.3506088, -27.0131931, 16.3571815, -43.3405418, 43.3638000
12: -32.1872330, 22.4957886, -32.1944962, 22.5122185, -54.3314285, 54.3225021
13: -31.2771702, 30.7403297, -31.2842903, 30.7945290, -62.0717010, 62.0246201
14: -51.4270172, 16.1432285, -51.4442368, 16.1676521, -67.5946655, 67.5874634
15: -26.7171726, 17.8270683, -26.7263508, 17.8349266, -44.5521011, 44.5534210
16: -34.0057716, 18.1374760, -34.0170593, 18.1521931, -52.1579666, 52.1545334
17: -50.6526375, 17.8114700, -50.6638336, 17.8500824, -68.5027161, 68.4753036
18: -35.8139191, 18.0467224, -35.8617668, 18.0520477, -53.8659668, 53.9084892
19: -20.5988197, 14.1480522, -20.6230965, 14.1503820, -34.7492027, 34.7711487
20: -20.5822525, 17.8736401, -20.6147938, 17.8790016, -38.4612541, 38.4884338
21: -26.2789974, 15.5798597, -26.3089733, 15.5850210, -41.8640175, 41.8888321
22: -25.9669437, 15.7352953, -25.9882565, 15.7408447, -41.7077866, 41.7235527
23: -19.3168240, 19.3687897, -19.3408356, 19.3747444, -38.6915665, 38.7096252
24: -27.6026077, 18.0975971, -27.6323433, 18.1032200, -45.7058258, 45.7299423
25: -21.4673271, 21.4144058, -21.4976349, 21.4232178, -42.8905449, 42.9120407
26: -35.1649246, 25.5048714, -35.2037964, 25.5099831, -60.6749077, 60.7086678
27: -26.2221298, 17.0113602, -26.2699203, 17.0148983, -43.2370300, 43.2812805
28: -20.1211700, 20.4450417, -20.1542683, 20.4506187, -40.5717888, 40.5993118
29: -24.3183022, 15.3551989, -24.3397388, 15.3614359, -39.6797371, 39.6949387
30: -26.7121010, 19.4712696, -26.7290020, 19.4803524, -46.1924515, 46.2002716
31: -28.4323883, 19.9188385, -28.4762287, 19.9244614, -48.3568497, 48.3950653
32: -31.9791927, 14.9975691, -31.9888992, 15.0093899, -46.8537064, 46.8516769
33: -52.4415588, 17.1806717, -52.4538345, 17.2187176, -68.6200790, 68.5960159
34: -45.5426445, 5.7441483, -45.5530777, 5.7606459, -49.8110123, 49.8079300
35: -41.0947037, 14.1040659, -41.1034889, 14.1202526, -54.2096786, 54.2029953
36: -36.6977501, 17.8594055, -36.7111435, 17.8636436, -54.5519943, 54.5612411
37: -59.2306404, 8.2995472, -59.2487183, 8.3345594, -67.4879150, 67.4719696
38: -45.8074799, 17.6034508, -45.8368378, 17.6109543, -63.4184341, 63.4402885
39: -55.0614586, 18.1200790, -55.0765305, 18.1582184, -73.1122589, 73.0890579
40: -44.7659149, 6.0142040, -44.7833214, 6.0325623, -50.7231293, 50.7223015
41: -35.4453201, 16.8933392, -35.4575882, 16.9117889, -52.3101654, 52.3041153
42: -23.7524757, 14.9652224, -23.7600937, 14.9759102, -38.7283859, 38.7253151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2592121, upper bound: 24.3279231
time: 80.21 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2592121, upper bound: 24.3279231
time: 64.59 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -43.0630379, 19.4106846, -43.0140152, 19.4061089, -62.4691467, 62.4246979
1: -21.3153839, 18.2177410, -21.2947617, 18.2127457, -39.5281296, 39.5125046
2: -15.0623693, 20.2168884, -15.0410824, 20.2128677, -35.2752380, 35.2579727
3: -20.4746132, 22.4723587, -20.4462624, 22.4670982, -42.9417114, 42.9186211
4: -24.1438599, 20.0543308, -24.1074314, 20.0526562, -44.1965179, 44.1617622
5: -19.2104492, 21.6470985, -19.1914234, 21.6414337, -40.8518829, 40.8385239
6: -33.2428055, 15.8885746, -33.2348480, 15.8802576, -49.1230621, 49.1234207
7: -24.6512070, 19.6453876, -24.6315155, 19.6394768, -44.2906837, 44.2769012
8: -27.9217548, 26.9249916, -27.8731613, 26.9187813, -54.8405380, 54.7981529
9: -24.0957565, 23.0894451, -24.0633545, 23.0853882, -47.1811447, 47.1528015
10: -32.4511261, 25.0555954, -32.4316330, 25.0465660, -57.4976921, 57.4872284
11: -27.0276184, 16.3934441, -27.0253887, 16.3595486, -43.3871689, 43.4188309
12: -32.2062988, 22.5293770, -32.1970482, 22.5161304, -54.3544846, 54.3608856
13: -31.3569469, 30.8262177, -31.2860928, 30.8170853, -62.1740341, 62.1123123
14: -51.4745560, 16.1837139, -51.4500122, 16.1779861, -67.6525421, 67.6337280
15: -26.7475834, 17.8368702, -26.7298317, 17.8357048, -44.5832901, 44.5667038
16: -34.0301323, 18.1639843, -34.0208359, 18.1579685, -52.1881027, 52.1848221
17: -50.7140083, 17.8724022, -50.6675339, 17.8659916, -68.5800018, 68.5399323
18: -35.8922119, 18.1227360, -35.8815346, 18.0532475, -53.9454575, 54.0042725
19: -20.6393280, 14.1760340, -20.6329002, 14.1508331, -34.7901611, 34.8089333
20: -20.6336060, 17.9239655, -20.6281300, 17.8805504, -38.5141563, 38.5520935
21: -26.3268147, 15.6215849, -26.3209705, 15.5866184, -41.9134331, 41.9425545
22: -26.0059528, 15.7603989, -25.9967175, 15.7424202, -41.7483749, 41.7571182
23: -19.3547363, 19.4042435, -19.3506012, 19.3767242, -38.7314606, 38.7548447
24: -27.6496067, 18.1416950, -27.6443424, 18.1045551, -45.7541618, 45.7860374
25: -21.5145111, 21.4724426, -21.5100727, 21.4262199, -42.9407310, 42.9825134
26: -35.2319336, 25.5555611, -35.2197876, 25.5109768, -60.7429123, 60.7753487
27: -26.2985764, 17.0745926, -26.2894745, 17.0156937, -43.3142700, 43.3640671
28: -20.1738987, 20.4911766, -20.1677952, 20.4522743, -40.6261749, 40.6589737
29: -24.3553963, 15.3771677, -24.3479176, 15.3635778, -39.7189751, 39.7250862
30: -26.7371330, 19.5112724, -26.7355251, 19.4834213, -46.2205544, 46.2467957
31: -28.5036201, 19.9791641, -28.4944172, 19.9261055, -48.4297256, 48.4735794
32: -32.0078583, 15.0204887, -31.9922237, 15.0139732, -46.8884277, 46.8766022
33: -52.5081711, 17.2420654, -52.4585991, 17.2345524, -68.7029572, 68.6548157
34: -45.5764961, 5.7670898, -45.5569229, 5.7651205, -49.8518143, 49.8373337
35: -41.1290054, 14.1280861, -41.1066132, 14.1246262, -54.2470856, 54.2315598
36: -36.7279663, 17.8703651, -36.7150192, 17.8650799, -54.5837021, 54.5761871
37: -59.3005981, 8.3543768, -59.2555351, 8.3485098, -67.5717163, 67.5293732
38: -45.8663101, 17.6267815, -45.8473587, 17.6136971, -63.4800072, 63.4741402
39: -55.1409378, 18.1762276, -55.0821991, 18.1741619, -73.2082062, 73.1482925
40: -44.8233414, 6.0430117, -44.7899323, 6.0399103, -50.7893982, 50.7565384
41: -35.4799042, 16.9255314, -35.4620743, 16.9195004, -52.3526306, 52.3400040
42: -23.7666626, 14.9875546, -23.7626953, 14.9795971, -38.7462616, 38.7502518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2952408, upper bound: 24.3280538
time: 73.59 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2592121, upper bound: 24.3288701
time: 69.57 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 145.41 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 145.41
Output dim: 20, lower bound: -24.2592121, upper bound: 24.3279231
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 145.41
Output dim: 20, lower bound: -24.2592121, upper bound: 24.3279231
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 145.41
Output dim: 20, lower bound: -24.2952408, upper bound: 24.3280538
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 145.41
Output dim: 20, lower bound: -24.2592121, upper bound: 24.3288701

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -42.9687157, 19.3510513, -42.9056740, 19.3267059, -62.2954216, 62.2567253
1: -21.2695618, 18.1812267, -21.2336731, 18.1588860, -39.4284477, 39.4149017
2: -15.0211267, 20.1779137, -14.9981327, 20.1757164, -35.1968422, 35.1760483
3: -20.4157028, 22.4253922, -20.3649254, 22.3863602, -42.8020630, 42.7903175
4: -24.0705166, 20.0172520, -24.0214481, 19.9814339, -44.0519485, 44.0387001
5: -19.1593723, 21.6034927, -19.1063900, 21.5639000, -40.7232742, 40.7098846
6: -33.2080803, 15.8606234, -33.2003593, 15.8691978, -49.0772781, 49.0609818
7: -24.6060257, 19.6105556, -24.5711632, 19.5876579, -44.1936836, 44.1817169
8: -27.8451500, 26.8722916, -27.8119335, 26.8588581, -54.7040100, 54.6842270
9: -24.0181389, 23.0430565, -23.9557934, 22.9919586, -47.0100975, 46.9988480
10: -32.3799057, 25.0110683, -32.3003387, 24.9428902, -57.3227959, 57.3114090
11: -26.9713383, 16.3285732, -26.9534416, 16.2853889, -43.2567291, 43.2820129
12: -32.1820869, 22.4851303, -32.1715012, 22.4706039, -54.2828064, 54.2873688
13: -31.2537899, 30.7306480, -31.2082710, 30.7333221, -61.9871140, 61.9389191
14: -51.4083138, 16.1364670, -51.3753204, 16.1354351, -67.5437469, 67.5117874
15: -26.6890049, 17.8214264, -26.6316929, 17.7684937, -44.4574966, 44.4531174
16: -33.9777451, 18.1315346, -33.9230728, 18.0878792, -52.0656242, 52.0546074
17: -50.6316872, 17.7970600, -50.5912704, 17.7932167, -68.4249039, 68.3883286
18: -35.8039246, 18.0300980, -35.8092117, 17.9973831, -53.8013077, 53.8393097
19: -20.5912762, 14.1229801, -20.5465431, 14.0758486, -34.6671257, 34.6695251
20: -20.5745468, 17.8503323, -20.5486031, 17.8063049, -38.3808517, 38.3989334
21: -26.2682457, 15.5491810, -26.2361622, 15.4928455, -41.7610931, 41.7853432
22: -25.9590836, 15.7175465, -25.9290485, 15.6854801, -41.6445618, 41.6465950
23: -19.3084412, 19.3316727, -19.2329025, 19.2584267, -38.5668678, 38.5645752
24: -27.5957470, 18.0639973, -27.5379391, 17.9987068, -45.5944519, 45.6019363
25: -21.4604530, 21.3803177, -21.4134483, 21.3159218, -42.7763748, 42.7937660
26: -35.1549149, 25.4728413, -35.1162567, 25.4111137, -60.5660286, 60.5890961
27: -26.2158089, 16.9874992, -26.1951218, 16.9394245, -43.1552353, 43.1826210
28: -20.1148586, 20.4132614, -20.0685577, 20.3499203, -40.4647789, 40.4818192
29: -24.3096218, 15.3398027, -24.2812462, 15.3116302, -39.6212540, 39.6210480
30: -26.7053204, 19.4441452, -26.6562691, 19.3870659, -46.0923843, 46.1004143
31: -28.4233456, 19.8943520, -28.3922539, 19.8484993, -48.2718430, 48.2866058
32: -31.9720802, 14.9911718, -31.9564075, 15.0030050, -46.8397369, 46.8122864
33: -52.4341965, 17.1612892, -52.3964310, 17.1557159, -68.5436859, 68.5081482
34: -45.5367432, 5.7364483, -45.5262527, 5.7346516, -49.7729340, 49.7675400
35: -41.0892487, 14.0864325, -41.0586166, 14.0667849, -54.1446457, 54.1289215
36: -36.6928558, 17.8418465, -36.6685104, 17.8079643, -54.4913101, 54.5008926
37: -59.2211609, 8.2721329, -59.1568184, 8.2475510, -67.3888397, 67.3472824
38: -45.8002396, 17.5833321, -45.7791443, 17.5448036, -63.3450432, 63.3624763
39: -55.0520477, 18.1040192, -55.0150986, 18.1071301, -73.0496826, 73.0062180
40: -44.7541466, 6.0059786, -44.7322617, 6.0219069, -50.7005615, 50.6628189
41: -35.4393959, 16.8843117, -35.4276276, 16.8789673, -52.2708435, 52.2633324
42: -23.7459087, 14.9586220, -23.7285976, 14.9523811, -38.6982880, 38.6872177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=401, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2984972
time: 73.29 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3196501
time: 70.58 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -42.9961777, 19.3556843, -43.0047188, 19.3887558, -62.3849335, 62.3604050
1: -21.2875690, 18.1851692, -21.2901192, 18.2027683, -39.4903374, 39.4752884
2: -15.0324411, 20.1802063, -15.0368261, 20.2024841, -35.2349243, 35.2170334
3: -20.4389496, 22.4290085, -20.4405861, 22.4547043, -42.8936539, 42.8695946
4: -24.0943356, 20.0212631, -24.1002235, 20.0413551, -44.1356888, 44.1214867
5: -19.1825485, 21.6070786, -19.1854610, 21.6299782, -40.8125267, 40.7925415
6: -33.2064743, 15.8674784, -33.2057419, 15.8749084, -49.0813828, 49.0732193
7: -24.6226501, 19.6135216, -24.6263676, 19.6310654, -44.2537155, 44.2398911
8: -27.8618507, 26.8767376, -27.8671074, 26.9049168, -54.7667694, 54.7438431
9: -24.0480804, 23.0472794, -24.0539074, 23.0722733, -47.1203537, 47.1011887
10: -32.4167175, 25.0162277, -32.4212112, 25.0354099, -57.4521255, 57.4374390
11: -26.9821339, 16.3492737, -27.0096855, 16.3533249, -43.3354568, 43.3589592
12: -32.1834717, 22.4948063, -32.1832657, 22.5094185, -54.3245316, 54.3147125
13: -31.2755184, 30.7392502, -31.2794933, 30.7914715, -62.0669899, 62.0187454
14: -51.4253693, 16.1415977, -51.4395866, 16.1629467, -67.5883179, 67.5811844
15: -26.7155628, 17.8260860, -26.7217407, 17.8321667, -44.5477295, 44.5478287
16: -34.0036469, 18.1366501, -34.0110550, 18.1497955, -52.1534424, 52.1477051
17: -50.6511841, 17.8102570, -50.6595306, 17.8466148, -68.4978027, 68.4697876
18: -35.8130264, 18.0442924, -35.8592758, 18.0452118, -53.8582382, 53.9035683
19: -20.5980263, 14.1465282, -20.6208134, 14.1459665, -34.7439919, 34.7673416
20: -20.5813217, 17.8723888, -20.6121445, 17.8753471, -38.4566689, 38.4845352
21: -26.2778606, 15.5785484, -26.3057709, 15.5813856, -41.8592453, 41.8843193
22: -25.9660931, 15.7336178, -25.9858818, 15.7358904, -41.7019844, 41.7194977
23: -19.3158245, 19.3668060, -19.3380260, 19.3688660, -38.6846924, 38.7048340
24: -27.6016655, 18.0956745, -27.6297226, 18.0976276, -45.6992950, 45.7253952
25: -21.4664211, 21.4125271, -21.4951000, 21.4177456, -42.8841667, 42.9076271
26: -35.1638298, 25.5028572, -35.2007599, 25.5041065, -60.6679382, 60.7036171
27: -26.2211227, 17.0099220, -26.2671146, 17.0107212, -43.2318420, 43.2770386
28: -20.1204700, 20.4433689, -20.1522865, 20.4456902, -40.5661621, 40.5956573
29: -24.3173466, 15.3542366, -24.3370380, 15.3586922, -39.6760406, 39.6912766
30: -26.7108688, 19.4697456, -26.7255878, 19.4759407, -46.1868095, 46.1953354
31: -28.4314480, 19.9173107, -28.4735565, 19.9201508, -48.3516006, 48.3908691
32: -31.9715328, 14.9968395, -31.9661140, 15.0073795, -46.8439941, 46.8302536
33: -52.4404221, 17.1792908, -52.4507484, 17.2147007, -68.6118774, 68.5957336
34: -45.5371017, 5.7433338, -45.5367241, 5.7583494, -49.8018570, 49.8070412
35: -41.0938950, 14.1027670, -41.1014061, 14.1165676, -54.2020111, 54.2042389
36: -36.6971321, 17.8583412, -36.7093849, 17.8606339, -54.5482559, 54.5585632
37: -59.2294922, 8.2978106, -59.2455635, 8.3294582, -67.4802704, 67.4690704
38: -45.8066635, 17.6018562, -45.8344955, 17.6065025, -63.4131660, 63.4363518
39: -55.0602684, 18.1188488, -55.0733719, 18.1547661, -73.1063995, 73.0864563
40: -44.7574921, 6.0133533, -44.7596397, 6.0302238, -50.7126999, 50.6996422
41: -35.4443779, 16.8924637, -35.4549255, 16.9094009, -52.3064346, 52.3012161
42: -23.7471943, 14.9645805, -23.7444210, 14.9740753, -38.7212677, 38.7089996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2993177
time: 78.25 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3204129
time: 73.34 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -43.0337029, 19.4051113, -42.9095688, 19.3411942, -62.3748970, 62.3146820
1: -21.2962742, 18.2130890, -21.2351227, 18.1667976, -39.4630737, 39.4482117
2: -15.0502319, 20.2141342, -15.0000648, 20.1847305, -35.2349625, 35.2141991
3: -20.4497509, 22.4682121, -20.3659115, 22.3973064, -42.8470573, 42.8341217
4: -24.1186562, 20.0492821, -24.0246086, 19.9898224, -44.1084785, 44.0738907
5: -19.1857414, 21.6430206, -19.1078262, 21.5739784, -40.7597198, 40.7508469
6: -33.2362213, 15.8809519, -33.2050018, 15.8725939, -49.1088142, 49.0859528
7: -24.6334038, 19.6420212, -24.5728741, 19.5949020, -44.2283058, 44.2148972
8: -27.9038086, 26.9199238, -27.8143749, 26.8710136, -54.7748222, 54.7342987
9: -24.0637245, 23.0844555, -23.9593887, 23.0028172, -47.0665436, 47.0438461
10: -32.4118805, 25.0494995, -32.3036652, 24.9513969, -57.3632774, 57.3531647
11: -27.0155926, 16.3714180, -26.9656086, 16.2877541, -43.3033447, 43.3370285
12: -32.2011642, 22.5187454, -32.1740456, 22.4745789, -54.3058777, 54.3257675
13: -31.3335781, 30.8165379, -31.2100430, 30.7559090, -62.0894852, 62.0265808
14: -51.4558792, 16.1769485, -51.3810806, 16.1457672, -67.6016464, 67.5580292
15: -26.7194138, 17.8312340, -26.6351833, 17.7692566, -44.4886703, 44.4664154
16: -34.0021133, 18.1580372, -33.9268532, 18.0936394, -52.0957527, 52.0848923
17: -50.6930847, 17.8580074, -50.5949402, 17.8091316, -68.5022125, 68.4529495
18: -35.8822327, 18.1061211, -35.8289719, 17.9985847, -53.8808174, 53.9350929
19: -20.6317940, 14.1509523, -20.5563545, 14.0763054, -34.7080994, 34.7073059
20: -20.6259079, 17.9006691, -20.5619488, 17.8078403, -38.4337463, 38.4626160
21: -26.3160782, 15.5909033, -26.2481842, 15.4944468, -41.8105240, 41.8390884
22: -25.9980793, 15.7426262, -25.9375286, 15.6870356, -41.6851158, 41.6801529
23: -19.3463478, 19.3671341, -19.2426796, 19.2604179, -38.6067657, 38.6098137
24: -27.6427479, 18.1081085, -27.5499687, 18.0000305, -45.6427765, 45.6580772
25: -21.5076332, 21.4383545, -21.4258842, 21.3189354, -42.8265686, 42.8642387
26: -35.2219162, 25.5234966, -35.1322441, 25.4121304, -60.6340485, 60.6557388
27: -26.2922592, 17.0507259, -26.2146721, 16.9402084, -43.2324677, 43.2653961
28: -20.1676140, 20.4593925, -20.0821037, 20.3515701, -40.5191841, 40.5414963
29: -24.3467064, 15.3617659, -24.2894516, 15.3137684, -39.6604767, 39.6512184
30: -26.7303371, 19.4841747, -26.6627560, 19.3901138, -46.1204529, 46.1469307
31: -28.4945889, 19.9547157, -28.4104538, 19.8501625, -48.3447495, 48.3651695
32: -32.0007935, 15.0140772, -31.9597301, 15.0075750, -46.8744431, 46.8371925
33: -52.5007744, 17.2226238, -52.4011955, 17.1715813, -68.6265869, 68.5670242
34: -45.5705986, 5.7593813, -45.5301132, 5.7391386, -49.8137207, 49.7969360
35: -41.1235237, 14.1104136, -41.0617485, 14.0711775, -54.1820221, 54.1574860
36: -36.7230530, 17.8528175, -36.6723976, 17.8094101, -54.5229797, 54.5158463
37: -59.2911110, 8.3269157, -59.1636276, 8.2614803, -67.4725800, 67.4046936
38: -45.8590622, 17.6066475, -45.7896690, 17.5475731, -63.4066353, 63.3963165
39: -55.1315155, 18.1601982, -55.0207253, 18.1230583, -73.1455994, 73.0654144
40: -44.8115616, 6.0348110, -44.7388611, 6.0292673, -50.7668381, 50.6970749
41: -35.4739876, 16.9164963, -35.4321442, 16.8866615, -52.3132477, 52.2992325
42: -23.7601051, 14.9809570, -23.7312012, 14.9560699, -38.7161751, 38.7121582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=401, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2986522
time: 70.84 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3198122
time: 70.23 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -43.0611572, 19.4097252, -43.0086060, 19.4032478, -62.4644051, 62.4183311
1: -21.3142834, 18.2170296, -21.2915268, 18.2107048, -39.5249863, 39.5085564
2: -15.0615683, 20.2164116, -15.0387869, 20.2115097, -35.2730789, 35.2551994
3: -20.4729691, 22.4718304, -20.4415531, 22.4656601, -42.9386292, 42.9133835
4: -24.1424828, 20.0533276, -24.1033707, 20.0497513, -44.1922340, 44.1567001
5: -19.2089157, 21.6465950, -19.1869011, 21.6400414, -40.8489571, 40.8334961
6: -33.2346191, 15.8878422, -33.2103653, 15.8782930, -49.1129112, 49.0982056
7: -24.6500320, 19.6449966, -24.6280670, 19.6383419, -44.2883759, 44.2730637
8: -27.9205246, 26.9243698, -27.8695545, 26.9170818, -54.8376083, 54.7939224
9: -24.0937023, 23.0886536, -24.0575123, 23.0831509, -47.1768532, 47.1461639
10: -32.4486771, 25.0546417, -32.4245262, 25.0439129, -57.4925919, 57.4791679
11: -27.0263786, 16.3921051, -27.0218716, 16.3556938, -43.3820724, 43.4139786
12: -32.2025223, 22.5283279, -32.1857796, 22.5133076, -54.3475876, 54.3530655
13: -31.3552818, 30.8251495, -31.2813206, 30.8140869, -62.1693687, 62.1064682
14: -51.4729004, 16.1820946, -51.4453659, 16.1732483, -67.6461487, 67.6274567
15: -26.7459812, 17.8358784, -26.7252197, 17.8329334, -44.5789146, 44.5610962
16: -34.0279999, 18.1631432, -34.0148087, 18.1555786, -52.1835785, 52.1779518
17: -50.7125282, 17.8712006, -50.6631851, 17.8625603, -68.5750885, 68.5343857
18: -35.8913193, 18.1202679, -35.8790283, 18.0464287, -53.9377480, 53.9992981
19: -20.6385326, 14.1745052, -20.6306229, 14.1464176, -34.7849503, 34.8051300
20: -20.6326694, 17.9227180, -20.6254692, 17.8768845, -38.5095520, 38.5481873
21: -26.3256817, 15.6202536, -26.3177834, 15.5829897, -41.9086723, 41.9380379
22: -26.0050907, 15.7586975, -25.9943619, 15.7374744, -41.7425652, 41.7530594
23: -19.3537292, 19.4022369, -19.3477764, 19.3708534, -38.7245827, 38.7500153
24: -27.6486702, 18.1397839, -27.6417389, 18.0989799, -45.7476501, 45.7815247
25: -21.5136185, 21.4705639, -21.5075302, 21.4207420, -42.9343605, 42.9780960
26: -35.2308578, 25.5535469, -35.2167130, 25.5051270, -60.7359848, 60.7702599
27: -26.2975750, 17.0731564, -26.2866726, 17.0115204, -43.3090973, 43.3598289
28: -20.1732216, 20.4895020, -20.1658268, 20.4473457, -40.6205673, 40.6553268
29: -24.3544540, 15.3762007, -24.3452473, 15.3608274, -39.7152824, 39.7214470
30: -26.7358894, 19.5097580, -26.7320881, 19.4790115, -46.2149010, 46.2418442
31: -28.5026970, 19.9776669, -28.4917526, 19.9217834, -48.4244804, 48.4694214
32: -32.0002136, 15.0197659, -31.9694347, 15.0119724, -46.8787079, 46.8551712
33: -52.5070229, 17.2406578, -52.4555244, 17.2305832, -68.6947327, 68.6545334
34: -45.5709801, 5.7662401, -45.5405693, 5.7627754, -49.8426361, 49.8364105
35: -41.1281929, 14.1268291, -41.1045303, 14.1209402, -54.2393799, 54.2327881
36: -36.7273293, 17.8693161, -36.7132568, 17.8620644, -54.5799942, 54.5734825
37: -59.2994308, 8.3526134, -59.2523766, 8.3434086, -67.5640030, 67.5264435
38: -45.8654823, 17.6252060, -45.8450394, 17.6091938, -63.4746780, 63.4702454
39: -55.1397667, 18.1750145, -55.0790253, 18.1706963, -73.2023010, 73.1457062
40: -44.8149071, 6.0421648, -44.7662582, 6.0375757, -50.7789612, 50.7338867
41: -35.4789352, 16.9246368, -35.4594231, 16.9170647, -52.3488007, 52.3371010
42: -23.7613869, 14.9868975, -23.7470207, 14.9777584, -38.7391434, 38.7339172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3205721, upper bound: 24.2994783
time: 92.90 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3205725
time: 75.97 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 171.14 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 171.14
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2984972
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 171.14
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3196501
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 171.14
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2993177
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 171.14
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3204129
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 171.14
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2986522
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 171.14
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3198122
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 171.14
Output dim: 20, lower bound: -24.3205721, upper bound: 24.2994783
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 171.14
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3205725

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -42.9865990, 19.3739090, -42.9005013, 19.3256531, -62.3122520, 62.2744102
1: -21.2800941, 18.2209797, -21.2309151, 18.1574459, -39.4375381, 39.4518967
2: -15.0304623, 20.2472076, -14.9962158, 20.1746998, -35.2051620, 35.2434235
3: -20.4158669, 22.5249786, -20.3605003, 22.3848572, -42.8007240, 42.8854790
4: -24.0784492, 20.1029129, -24.0193443, 19.9805489, -44.0589981, 44.1222572
5: -19.1645393, 21.6698990, -19.1037483, 21.5629807, -40.7275200, 40.7736473
6: -33.2484589, 15.8695164, -33.1990471, 15.8652000, -49.1136589, 49.0685654
7: -24.6211319, 19.6494942, -24.5678940, 19.5854454, -44.2065773, 44.2173882
8: -27.8546906, 26.9377308, -27.8094368, 26.8570099, -54.7117004, 54.7471695
9: -24.0235710, 23.0993786, -23.9505157, 22.9909420, -47.0145111, 47.0498962
10: -32.4246445, 25.0440025, -32.2986450, 24.9402065, -57.3648529, 57.3426476
11: -27.1083183, 16.3248539, -26.9517574, 16.2790642, -43.3873825, 43.2766113
12: -32.2975807, 22.4911652, -32.1694145, 22.4653111, -54.3965378, 54.2865829
13: -31.2452984, 30.7859154, -31.1969738, 30.7310181, -61.9763184, 61.9828873
14: -51.5737915, 16.1362762, -51.3716888, 16.1299400, -67.7037354, 67.5079651
15: -26.6921768, 17.8677101, -26.6255932, 17.7673092, -44.4594879, 44.4933014
16: -34.0190926, 18.1732292, -33.9189224, 18.0853348, -52.1044273, 52.0921516
17: -50.8156662, 17.8056316, -50.5885353, 17.7883682, -68.6040344, 68.3941650
18: -35.9219017, 18.0095444, -35.8070297, 17.9818649, -53.9037666, 53.8165741
19: -20.6675167, 14.1208372, -20.5451603, 14.0729437, -34.7404594, 34.6659966
20: -20.6459503, 17.8517609, -20.5471134, 17.8034840, -38.4494324, 38.3988724
21: -26.3841190, 15.5505733, -26.2345676, 15.4901981, -41.8743172, 41.7851410
22: -26.0449677, 15.7252674, -25.9258709, 15.6831284, -41.7280960, 41.6511383
23: -19.3846779, 19.3329926, -19.2316170, 19.2549229, -38.6396027, 38.5646095
24: -27.6525097, 18.0556107, -27.5361385, 17.9920692, -45.6445770, 45.5917511
25: -21.5232334, 21.3844433, -21.4115753, 21.3132057, -42.8364410, 42.7960205
26: -35.2989693, 25.4753361, -35.1132965, 25.4056587, -60.7046280, 60.5886307
27: -26.2804375, 16.9729385, -26.1931267, 16.9311638, -43.2116013, 43.1660652
28: -20.1695213, 20.4149265, -20.0673103, 20.3478699, -40.5173912, 40.4822388
29: -24.4236870, 15.3441830, -24.2785072, 15.3091011, -39.7327881, 39.6226883
30: -26.7888088, 19.4441013, -26.6547031, 19.3822956, -46.1711044, 46.0988045
31: -28.4940987, 19.8831749, -28.3901978, 19.8411865, -48.3352852, 48.2733727
32: -32.0017471, 15.0059938, -31.9538422, 15.0017948, -46.8687897, 46.8244858
33: -52.4499893, 17.2544022, -52.3925705, 17.1541443, -68.5548248, 68.5988998
34: -45.5480156, 5.7977829, -45.5239754, 5.7333412, -49.7794952, 49.8215103
35: -41.0938225, 14.1540031, -41.0535240, 14.0660591, -54.1532288, 54.1971436
36: -36.7198029, 17.8637295, -36.6665573, 17.8073730, -54.5176926, 54.5208397
37: -59.2721062, 8.2812576, -59.1544685, 8.2430897, -67.4305878, 67.3529358
38: -45.8320694, 17.6039963, -45.7763519, 17.5397205, -63.3717880, 63.3803482
39: -55.0781326, 18.2129765, -55.0116310, 18.1062832, -73.0734711, 73.1124420
40: -44.7791443, 6.0497789, -44.7295952, 6.0198479, -50.7238770, 50.7046280
41: -35.4585457, 16.9166946, -35.4250641, 16.8774319, -52.2877197, 52.2931366
42: -23.7697430, 14.9718914, -23.7275658, 14.9500952, -38.7198372, 38.6994553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=401, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 701

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3002564
time: 134.32 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3196501
time: 68.69 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -43.0140610, 19.3785343, -42.9995384, 19.3876801, -62.4017410, 62.3780746
1: -21.2981148, 18.2249298, -21.2873554, 18.2013435, -39.4994583, 39.5122833
2: -15.0417662, 20.2495060, -15.0349369, 20.2014809, -35.2432480, 35.2844429
3: -20.4391060, 22.5285797, -20.4361401, 22.4532280, -42.8923340, 42.9647217
4: -24.1022758, 20.1069489, -24.0980911, 20.0404644, -44.1427383, 44.2050400
5: -19.1877174, 21.6735058, -19.1828194, 21.6290550, -40.8167725, 40.8563232
6: -33.2468376, 15.8764133, -33.2044106, 15.8708925, -49.1177292, 49.0808258
7: -24.6377792, 19.6524944, -24.6230812, 19.6288567, -44.2666359, 44.2755737
8: -27.8714123, 26.9421234, -27.8645954, 26.9030647, -54.7744751, 54.8067169
9: -24.0535278, 23.1035995, -24.0486317, 23.0712490, -47.1247787, 47.1522293
10: -32.4614677, 25.0491600, -32.4195023, 25.0327053, -57.4941711, 57.4686623
11: -27.1191006, 16.3455658, -27.0080280, 16.3470211, -43.4661217, 43.3535919
12: -32.2989273, 22.5008087, -32.1811752, 22.5041084, -54.4382706, 54.3139572
13: -31.2670097, 30.7945251, -31.2682228, 30.7891674, -62.0561752, 62.0627480
14: -51.5907936, 16.1413670, -51.4359512, 16.1574707, -67.7482605, 67.5773163
15: -26.7187481, 17.8723621, -26.7156334, 17.8309841, -44.5497322, 44.5879974
16: -34.0449600, 18.1783218, -34.0068932, 18.1472759, -52.1922379, 52.1852150
17: -50.8351212, 17.8188362, -50.6568108, 17.8417931, -68.6769104, 68.4756470
18: -35.9309692, 18.0237522, -35.8570938, 18.0297165, -53.9606857, 53.8808441
19: -20.6742554, 14.1444025, -20.6194344, 14.1430445, -34.8172989, 34.7638359
20: -20.6527138, 17.8738174, -20.6106358, 17.8725281, -38.5252419, 38.4844513
21: -26.3937492, 15.5799379, -26.3041916, 15.5787201, -41.9724693, 41.8841286
22: -26.0519676, 15.7413692, -25.9827194, 15.7335548, -41.7855225, 41.7240906
23: -19.3920479, 19.3681030, -19.3367729, 19.3653641, -38.7574120, 38.7048759
24: -27.6584587, 18.0872974, -27.6279087, 18.0909958, -45.7494545, 45.7152061
25: -21.5291939, 21.4166374, -21.4932442, 21.4150162, -42.9442101, 42.9098816
26: -35.3078613, 25.5053749, -35.1977844, 25.4986229, -60.8064842, 60.7031593
27: -26.2857914, 16.9953423, -26.2651138, 17.0024757, -43.2882690, 43.2604561
28: -20.1751099, 20.4450340, -20.1510353, 20.4436378, -40.6187477, 40.5960693
29: -24.4314156, 15.3586216, -24.3343124, 15.3561630, -39.7875786, 39.6929321
30: -26.7943554, 19.4696827, -26.7240086, 19.4711876, -46.2655411, 46.1936913
31: -28.5021782, 19.9061584, -28.4714985, 19.9128342, -48.4150124, 48.3776550
32: -32.0011978, 15.0116777, -31.9635468, 15.0061817, -46.8730545, 46.8424377
33: -52.4562302, 17.2723980, -52.4469299, 17.2131252, -68.6230621, 68.6865158
34: -45.5483742, 5.8046932, -45.5344086, 5.7569799, -49.8084030, 49.8609886
35: -41.0984650, 14.1703386, -41.0963058, 14.1158257, -54.2105713, 54.2724724
36: -36.7241058, 17.8802528, -36.7074165, 17.8600426, -54.5746765, 54.5785217
37: -59.2804184, 8.3069468, -59.2432289, 8.3249989, -67.5219955, 67.4746857
38: -45.8385086, 17.6225700, -45.8317070, 17.6013966, -63.4399033, 63.4542770
39: -55.0863800, 18.2278099, -55.0699081, 18.1539059, -73.1301880, 73.1928024
40: -44.7825241, 6.0571518, -44.7569580, 6.0281811, -50.7360382, 50.7414474
41: -35.4635239, 16.9248428, -35.4523735, 16.9078407, -52.3233032, 52.3309784
42: -23.7710381, 14.9778204, -23.7433739, 14.9717722, -38.7428093, 38.7211952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 701

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2844175, upper bound: 24.3010221
time: 72.26 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2844175, upper bound: 24.3204129
time: 68.37 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -43.0515709, 19.4279671, -42.9043999, 19.3401451, -62.3917160, 62.3323669
1: -21.3067932, 18.2528343, -21.2323341, 18.1653938, -39.4721870, 39.4851685
2: -15.0595598, 20.2834167, -14.9981642, 20.1837349, -35.2432938, 35.2815819
3: -20.4499130, 22.5677910, -20.3614902, 22.3958054, -42.8457184, 42.9292831
4: -24.1265621, 20.1350098, -24.0224781, 19.9889355, -44.1154976, 44.1574860
5: -19.1909237, 21.7094784, -19.1051807, 21.5730648, -40.7639885, 40.8146591
6: -33.2766113, 15.8898811, -33.2036972, 15.8685646, -49.1451759, 49.0935783
7: -24.6485176, 19.6809578, -24.5695915, 19.5927086, -44.2412262, 44.2505493
8: -27.9133472, 26.9853134, -27.8118820, 26.8691730, -54.7825203, 54.7971954
9: -24.0691681, 23.1407566, -23.9541206, 23.0018234, -47.0709915, 47.0948792
10: -32.4565849, 25.0824146, -32.3019867, 24.9486885, -57.4052734, 57.3843994
11: -27.1525669, 16.3676624, -26.9639435, 16.2814102, -43.4339752, 43.3316040
12: -32.3166275, 22.5247841, -32.1719589, 22.4692612, -54.4195480, 54.3250198
13: -31.3250656, 30.8717957, -31.1987400, 30.7535686, -62.0786362, 62.0705338
14: -51.6212730, 16.1767044, -51.3774185, 16.1402969, -67.7615662, 67.5541229
15: -26.7225914, 17.8775196, -26.6290855, 17.7680779, -44.4906693, 44.5066071
16: -34.0434570, 18.1997147, -33.9226875, 18.0911064, -52.1345634, 52.1224022
17: -50.8770103, 17.8665333, -50.5922089, 17.8043327, -68.6813431, 68.4587402
18: -36.0001831, 18.0855293, -35.8267899, 17.9830971, -53.9832802, 53.9123192
19: -20.7080231, 14.1488190, -20.5549698, 14.0734053, -34.7814293, 34.7037888
20: -20.6972961, 17.9020805, -20.5604649, 17.8050423, -38.5023384, 38.4625473
21: -26.4319305, 15.5922966, -26.2465916, 15.4917955, -41.9237251, 41.8388901
22: -26.0839615, 15.7503166, -25.9343414, 15.6847048, -41.7686653, 41.6846581
23: -19.4225903, 19.3684349, -19.2414150, 19.2569218, -38.6795120, 38.6098480
24: -27.6995506, 18.0996990, -27.5481224, 17.9934025, -45.6929550, 45.6478195
25: -21.5704174, 21.4424400, -21.4240055, 21.3162079, -42.8866272, 42.8664474
26: -35.3659668, 25.5260162, -35.1292801, 25.4066582, -60.7726250, 60.6552963
27: -26.3569412, 17.0361633, -26.2126751, 16.9319592, -43.2889023, 43.2488403
28: -20.2222462, 20.4610558, -20.0808105, 20.3495159, -40.5717621, 40.5418663
29: -24.4607677, 15.3661242, -24.2866840, 15.3112421, -39.7720108, 39.6528091
30: -26.8138161, 19.4840717, -26.6611996, 19.3853989, -46.1992149, 46.1452713
31: -28.5653458, 19.9435196, -28.4083748, 19.8428459, -48.4081917, 48.3518944
32: -32.0304031, 15.0288820, -31.9571304, 15.0063429, -46.9034729, 46.8494415
33: -52.5164833, 17.3158131, -52.3973770, 17.1700516, -68.6376190, 68.6577301
34: -45.5818710, 5.8207283, -45.5278244, 5.7377825, -49.8202515, 49.8509140
35: -41.1280518, 14.1780233, -41.0566368, 14.0704489, -54.1905823, 54.2257233
36: -36.7499924, 17.8747005, -36.6704369, 17.8088036, -54.5493927, 54.5357819
37: -59.3420792, 8.3361063, -59.1612816, 8.2570200, -67.5144348, 67.4104156
38: -45.8908997, 17.6272945, -45.7868958, 17.5424843, -63.4333839, 63.4141922
39: -55.1576157, 18.2691498, -55.0172501, 18.1221943, -73.1693573, 73.1716690
40: -44.8365135, 6.0786161, -44.7361794, 6.0272150, -50.7900696, 50.7389412
41: -35.4930763, 16.9488964, -35.4295654, 16.8851166, -52.3300629, 52.3290329
42: -23.7839394, 14.9942312, -23.7301598, 14.9537659, -38.7377052, 38.7243919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=401, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 701

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2836641
time: 77.32 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2845896
time: 61.09 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -43.0118408, 19.3961735, -42.9945602, 19.3992615, -62.4111023, 62.3907318
1: -21.2832394, 18.2076874, -21.2825985, 18.2078819, -39.4911194, 39.4902878
2: -15.0126762, 20.2078400, -15.0249128, 20.2089806, -35.2216568, 35.2327538
3: -20.4038391, 22.4569016, -20.4221573, 22.4611931, -42.8650322, 42.8790588
4: -24.0885448, 20.0429173, -24.0872307, 20.0466881, -44.1352310, 44.1301498
5: -19.1522541, 21.6341858, -19.1707706, 21.6362877, -40.7885437, 40.8049545
6: -33.2192459, 15.8707142, -33.2057648, 15.8728857, -49.0921326, 49.0764771
7: -24.5924454, 19.6329155, -24.6116333, 19.6346970, -44.2271423, 44.2445488
8: -27.8717957, 26.9106312, -27.8556614, 26.9130287, -54.7848244, 54.7662926
9: -24.0492840, 23.0752411, -24.0448990, 23.0786686, -47.1279526, 47.1201401
10: -32.4274788, 25.0300522, -32.4183083, 25.0368595, -57.4643402, 57.4483604
11: -27.0058346, 16.3371353, -27.0158348, 16.3395405, -43.3453751, 43.3529701
12: -32.1820984, 22.4085236, -32.1798401, 22.4798431, -54.3177795, 54.2485428
13: -31.3067703, 30.7987061, -31.2675762, 30.8061886, -62.1129608, 62.0662842
14: -51.4269905, 16.0504189, -51.4322548, 16.1364975, -67.5634918, 67.4826736
15: -26.7071247, 17.8204288, -26.7135925, 17.8283348, -44.5354614, 44.5340195
16: -33.9919205, 18.1491108, -34.0041656, 18.1511230, -52.1430435, 52.1532745
17: -50.6751556, 17.7667484, -50.6525955, 17.8331528, -68.5083084, 68.4193420
18: -35.8693390, 18.0538216, -35.8725853, 18.0269966, -53.8963356, 53.9264069
19: -20.6185951, 14.1552467, -20.6246910, 14.1409655, -34.7595596, 34.7799377
20: -20.6100445, 17.8607292, -20.6189346, 17.8595982, -38.4696426, 38.4796638
21: -26.3018589, 15.5633545, -26.3108730, 15.5671625, -41.8690224, 41.8742294
22: -25.9827061, 15.7154541, -25.9877377, 15.7249966, -41.7077026, 41.7031937
23: -19.3371391, 19.3661327, -19.3429298, 19.3604012, -38.6975403, 38.7090607
24: -27.6308899, 18.1002140, -27.6358051, 18.0878696, -45.7187576, 45.7360191
25: -21.4955540, 21.4204063, -21.5021820, 21.4065361, -42.9020920, 42.9225883
26: -35.2056847, 25.4241428, -35.2093353, 25.4689293, -60.6746140, 60.6334763
27: -26.2739792, 17.0501137, -26.2792110, 17.0050049, -43.2789841, 43.3293228
28: -20.1514511, 20.4565449, -20.1596012, 20.4380798, -40.5895309, 40.6161461
29: -24.3322544, 15.3281317, -24.3388615, 15.3470993, -39.6793518, 39.6669922
30: -26.7175865, 19.4445133, -26.7267342, 19.4600182, -46.1776047, 46.1712494
31: -28.4747200, 19.9434700, -28.4832382, 19.9120750, -48.3867950, 48.4267082
32: -31.9810925, 15.0021667, -31.9638500, 15.0068197, -46.8605652, 46.8381310
33: -52.4351349, 17.2174816, -52.4352722, 17.2238960, -68.5637970, 68.5611267
34: -45.5405998, 5.7485924, -45.5318146, 5.7576647, -49.7536011, 49.7545471
35: -41.0964699, 14.1156387, -41.0954285, 14.1176014, -54.1518936, 54.1648560
36: -36.7115250, 17.8594818, -36.7083817, 17.8592491, -54.5616684, 54.5590363
37: -59.2750778, 8.3273439, -59.2447052, 8.3362331, -67.5359497, 67.4963531
38: -45.8410187, 17.6023483, -45.8375626, 17.6026802, -63.4436989, 63.4399109
39: -55.0774384, 18.1601677, -55.0607224, 18.1664562, -73.1175385, 73.0962143
40: -44.7907639, 6.0320168, -44.7591476, 6.0346260, -50.7451324, 50.7098427
41: -35.4502945, 16.9099979, -35.4510422, 16.9128723, -52.3105469, 52.3085823
42: -23.7469673, 14.9649773, -23.7427959, 14.9709587, -38.7179260, 38.7077713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 873

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 701

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2633831
time: 75.87 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2633831
time: 79.83 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -43.0790176, 19.4325657, -43.0034256, 19.4021835, -62.4812012, 62.4359894
1: -21.3248100, 18.2567787, -21.2887611, 18.2092915, -39.5341034, 39.5455399
2: -15.0708828, 20.2857170, -15.0368605, 20.2105122, -35.2813950, 35.3225784
3: -20.4731483, 22.5714054, -20.4371319, 22.4641533, -42.9373016, 43.0085373
4: -24.1503983, 20.1389980, -24.1012497, 20.0488396, -44.1992378, 44.2402496
5: -19.2140846, 21.7130604, -19.1842670, 21.6391239, -40.8532104, 40.8973274
6: -33.2749939, 15.8967819, -33.2090645, 15.8742762, -49.1492691, 49.1058464
7: -24.6651592, 19.6839561, -24.6247673, 19.6361465, -44.3013077, 44.3087234
8: -27.9300232, 26.9897766, -27.8670521, 26.9152260, -54.8452492, 54.8568268
9: -24.0991077, 23.1450005, -24.0522366, 23.0821285, -47.1812363, 47.1972351
10: -32.4933739, 25.0875874, -32.4228363, 25.0411949, -57.5345688, 57.5104218
11: -27.1633682, 16.3883705, -27.0202141, 16.3493652, -43.5127335, 43.4085846
12: -32.3179779, 22.5344162, -32.1837082, 22.5080070, -54.4612961, 54.3523026
13: -31.3467808, 30.8803978, -31.2700138, 30.8117485, -62.1585312, 62.1504135
14: -51.6382904, 16.1818771, -51.4417038, 16.1677723, -67.8060608, 67.6235809
15: -26.7491493, 17.8821754, -26.7191353, 17.8317451, -44.5808945, 44.6013107
16: -34.0693359, 18.2048168, -34.0106583, 18.1530342, -52.2223701, 52.2154770
17: -50.8964195, 17.8797703, -50.6604614, 17.8577251, -68.7541428, 68.5402298
18: -36.0093002, 18.0997086, -35.8768463, 18.0309143, -54.0402145, 53.9765549
19: -20.7147598, 14.1723700, -20.6292362, 14.1435204, -34.8582802, 34.8016052
20: -20.7040367, 17.9241371, -20.6239834, 17.8740749, -38.5781097, 38.5481186
21: -26.4415436, 15.6216564, -26.3161945, 15.5803337, -42.0218773, 41.9378510
22: -26.0909786, 15.7664175, -25.9911900, 15.7351284, -41.8261070, 41.7576065
23: -19.4299660, 19.4035301, -19.3465309, 19.3673515, -38.7973175, 38.7500610
24: -27.7054996, 18.1313629, -27.6399117, 18.0923386, -45.7978363, 45.7712746
25: -21.5763779, 21.4746456, -21.5056629, 21.4180088, -42.9943848, 42.9803085
26: -35.3748589, 25.5560551, -35.2137604, 25.4996185, -60.8744774, 60.7698135
27: -26.3622742, 17.0585709, -26.2846737, 17.0032692, -43.3655434, 43.3432465
28: -20.2278519, 20.4911461, -20.1645527, 20.4452915, -40.6731415, 40.6557007
29: -24.4684944, 15.3805704, -24.3425236, 15.3582954, -39.8267899, 39.7230949
30: -26.8193779, 19.5096664, -26.7305107, 19.4742737, -46.2936516, 46.2401772
31: -28.5734367, 19.9665051, -28.4896889, 19.9144802, -48.4879150, 48.4561920
32: -32.0298309, 15.0345726, -31.9668522, 15.0107651, -46.9077759, 46.8674164
33: -52.5227814, 17.3337631, -52.4516945, 17.2290077, -68.7058868, 68.7453003
34: -45.5822105, 5.8276091, -45.5382614, 5.7614422, -49.8491898, 49.8904343
35: -41.1327438, 14.1943903, -41.0994263, 14.1202202, -54.2479553, 54.3010635
36: -36.7542915, 17.8912239, -36.7113037, 17.8614731, -54.6064148, 54.5934677
37: -59.3504143, 8.3617964, -59.2500153, 8.3389196, -67.6058197, 67.5321808
38: -45.8973656, 17.6458702, -45.8422623, 17.6041241, -63.5014877, 63.4881325
39: -55.1658592, 18.2839661, -55.0755463, 18.1698208, -73.2260895, 73.2519684
40: -44.8398895, 6.0860071, -44.7635651, 6.0355549, -50.8022003, 50.7756844
41: -35.4980431, 16.9570236, -35.4568596, 16.9155140, -52.3656921, 52.3668594
42: -23.7852211, 15.0001745, -23.7459793, 14.9754477, -38.7606697, 38.7461548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 701

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3204124, upper bound: 24.2633831
time: 183.25 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2844175, upper bound: 24.2853355
time: 68.36 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 253.89 seconds
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 253.89
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3002564
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 253.89
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3196501
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 253.89
Output dim: 20, lower bound: -24.2844175, upper bound: 24.3010221
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 253.89
Output dim: 20, lower bound: -24.2844175, upper bound: 24.3204129
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 253.89
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2836641
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 253.89
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2845896
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 253.89
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2633831
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 253.89
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2633831
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 253.89
Output dim: 20, lower bound: -24.3204124, upper bound: 24.2633831
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 253.89
Output dim: 20, lower bound: -24.2844175, upper bound: 24.2853355

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -42.9865990, 19.3739090, -42.9534531, 19.3447723, -62.3313713, 62.3273621
1: -21.2800941, 18.2209797, -21.2529640, 18.1703548, -39.4504471, 39.4739456
2: -15.0304623, 20.2472076, -15.0194464, 20.1877594, -35.2182236, 35.2666550
3: -20.4158669, 22.5249786, -20.3898067, 22.4010410, -42.8169098, 42.9147873
4: -24.0784492, 20.1029129, -24.0589142, 19.9906120, -44.0690613, 44.1618271
5: -19.1645393, 21.6698990, -19.1242256, 21.5787354, -40.7432747, 40.7941246
6: -33.2484589, 15.8695164, -33.2116470, 15.8768406, -49.1252975, 49.0811615
7: -24.6211319, 19.6494942, -24.5892982, 19.5986176, -44.2197495, 44.2387924
8: -27.8546906, 26.9377308, -27.8604736, 26.8753281, -54.7300186, 54.7982025
9: -24.0235710, 23.0993786, -23.9865284, 23.0058498, -47.0294189, 47.0859070
10: -32.4246445, 25.0440025, -32.3214798, 24.9577370, -57.3823814, 57.3654823
11: -27.1083183, 16.3248539, -26.9661980, 16.3153343, -43.4236526, 43.2910538
12: -32.2975807, 22.4911652, -32.1812210, 22.4824581, -54.4134903, 54.2981262
13: -31.2452984, 30.7859154, -31.2696629, 30.7626877, -62.0079880, 62.0555801
14: -51.5737915, 16.1362762, -51.4020042, 16.1459732, -67.7197647, 67.5382843
15: -26.6921768, 17.8677101, -26.6467953, 17.7692299, -44.4614067, 44.5145035
16: -34.0190926, 18.1732292, -33.9319992, 18.0971165, -52.1162109, 52.1052284
17: -50.8156662, 17.8056316, -50.6387711, 17.8107128, -68.6263809, 68.4444046
18: -35.9219017, 18.0095444, -35.8374100, 18.0525799, -53.9744797, 53.8469543
19: -20.6675167, 14.1208372, -20.5614052, 14.0986080, -34.7661247, 34.6822433
20: -20.6459503, 17.8517609, -20.5659180, 17.8484592, -38.4944077, 38.4176788
21: -26.3841190, 15.5505733, -26.2524376, 15.5267611, -41.9108810, 41.8030090
22: -26.0449677, 15.7252674, -25.9435844, 15.7026958, -41.7476654, 41.6688538
23: -19.3846779, 19.3329926, -19.2455311, 19.2844162, -38.6690941, 38.5785217
24: -27.6525097, 18.0556107, -27.5534058, 18.0305443, -45.6830521, 45.6090164
25: -21.5232334, 21.3844433, -21.4284630, 21.3624287, -42.8856621, 42.8129044
26: -35.2989693, 25.4753361, -35.1414490, 25.4512100, -60.7501793, 60.6167831
27: -26.2804375, 16.9729385, -26.2218037, 16.9908524, -43.2712898, 43.1947403
28: -20.1695213, 20.4149265, -20.0869598, 20.3884125, -40.5579338, 40.5018845
29: -24.4236870, 15.3441830, -24.2941608, 15.3248158, -39.7485046, 39.6383438
30: -26.7888088, 19.4441013, -26.6628189, 19.4132462, -46.2020569, 46.1069183
31: -28.4940987, 19.8831749, -28.4175968, 19.8959160, -48.3900146, 48.3007736
32: -32.0017471, 15.0059938, -31.9727688, 15.0128479, -46.8800125, 46.8446770
33: -52.4499893, 17.2544022, -52.4469299, 17.1774445, -68.5782166, 68.6537018
34: -45.5480156, 5.7977829, -45.5473747, 5.7397547, -49.7862473, 49.8451347
35: -41.0938225, 14.1540031, -41.0790253, 14.0738926, -54.1607056, 54.2204819
36: -36.7198029, 17.8637295, -36.6833839, 17.8141098, -54.5243988, 54.5376129
37: -59.2721062, 8.2812576, -59.2063026, 8.2628412, -67.4494781, 67.4045105
38: -45.8320694, 17.6039963, -45.8058205, 17.5555573, -63.3876266, 63.4098167
39: -55.0781326, 18.2129765, -55.0760193, 18.1242332, -73.0914764, 73.1776123
40: -44.7791443, 6.0497789, -44.7695961, 6.0303268, -50.7346649, 50.7459679
41: -35.4585457, 16.9166946, -35.4473724, 16.8911343, -52.3015289, 52.3154984
42: -23.7697430, 14.9718914, -23.7341156, 14.9617090, -38.7314529, 38.7060089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=207, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=401, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 596

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2907484
time: 69.24 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2505523, upper bound: 24.3192570
time: 66.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -43.0140610, 19.3785343, -43.0524673, 19.4067936, -62.4208527, 62.4309998
1: -21.2981148, 18.2249298, -21.3093910, 18.2142544, -39.5123672, 39.5343208
2: -15.0417662, 20.2495060, -15.0581236, 20.2145309, -35.2562981, 35.3076286
3: -20.4391060, 22.5285797, -20.4654617, 22.4693985, -42.9085045, 42.9940414
4: -24.1022758, 20.1069489, -24.1376839, 20.0505257, -44.1528015, 44.2446327
5: -19.1877174, 21.6735058, -19.2033005, 21.6448135, -40.8325310, 40.8768082
6: -33.2468376, 15.8764133, -33.2170258, 15.8825970, -49.1294327, 49.0934372
7: -24.6377792, 19.6524944, -24.6444855, 19.6420536, -44.2798309, 44.2969818
8: -27.8714123, 26.9421234, -27.9156475, 26.9214153, -54.7928276, 54.8577728
9: -24.0535278, 23.1035995, -24.0846329, 23.0861835, -47.1397095, 47.1882324
10: -32.4614677, 25.0491600, -32.4423294, 25.0502262, -57.5116959, 57.4914894
11: -27.1191006, 16.3455658, -27.0224533, 16.3832664, -43.5023651, 43.3680191
12: -32.2989273, 22.5008087, -32.1929626, 22.5212650, -54.4552307, 54.3254547
13: -31.2670097, 30.7945251, -31.3409233, 30.8208675, -62.0878754, 62.1354485
14: -51.5907936, 16.1413670, -51.4662552, 16.1735039, -67.7642975, 67.6076202
15: -26.7187481, 17.8723621, -26.7368584, 17.8329163, -44.5516663, 44.6092224
16: -34.0449600, 18.1783218, -34.0199814, 18.1590652, -52.2040253, 52.1983032
17: -50.8351212, 17.8188362, -50.7070312, 17.8641472, -68.6992645, 68.5258636
18: -35.9309692, 18.0237522, -35.8875046, 18.1004086, -54.0313797, 53.9112549
19: -20.6742554, 14.1444025, -20.6356773, 14.1687136, -34.8429680, 34.7800789
20: -20.6527138, 17.8738174, -20.6294518, 17.9175034, -38.5702171, 38.5032692
21: -26.3937492, 15.5799379, -26.3220406, 15.6152964, -42.0090446, 41.9019775
22: -26.0519676, 15.7413692, -26.0004101, 15.7531013, -41.8050690, 41.7417793
23: -19.3920479, 19.3681030, -19.3506622, 19.3948574, -38.7869034, 38.7187653
24: -27.6584587, 18.0872974, -27.6451969, 18.1294594, -45.7879181, 45.7324944
25: -21.5291939, 21.4166374, -21.5100918, 21.4642220, -42.9934158, 42.9267273
26: -35.3078613, 25.5053749, -35.2259254, 25.5441914, -60.8520508, 60.7313004
27: -26.2857914, 16.9953423, -26.2937717, 17.0621548, -43.3479462, 43.2891159
28: -20.1751099, 20.4450340, -20.1706886, 20.4841995, -40.6593094, 40.6157227
29: -24.4314156, 15.3586216, -24.3499947, 15.3718758, -39.8032913, 39.7086182
30: -26.7943554, 19.4696827, -26.7321320, 19.5021324, -46.2964859, 46.2018127
31: -28.5021782, 19.9061584, -28.4988937, 19.9675636, -48.4697418, 48.4050522
32: -32.0011978, 15.0116777, -31.9824715, 15.0172548, -46.8842926, 46.8626213
33: -52.4562302, 17.2723980, -52.5012589, 17.2364712, -68.6463928, 68.7413177
34: -45.5483742, 5.8046932, -45.5578423, 5.7634144, -49.8151703, 49.8845978
35: -41.0984650, 14.1703386, -41.1217957, 14.1236315, -54.2180328, 54.2957840
36: -36.7241058, 17.8802528, -36.7242432, 17.8667488, -54.5814209, 54.5952759
37: -59.2804184, 8.3069468, -59.2950325, 8.3448076, -67.5408936, 67.5262833
38: -45.8385086, 17.6225700, -45.8611832, 17.6172085, -63.4557190, 63.4837532
39: -55.0863800, 18.2278099, -55.1342888, 18.1718712, -73.1482239, 73.2579346
40: -44.7825241, 6.0571518, -44.7969894, 6.0386620, -50.7468109, 50.7827682
41: -35.4635239, 16.9248428, -35.4746857, 16.9215393, -52.3371048, 52.3533173
42: -23.7710381, 14.9778204, -23.7499409, 14.9834356, -38.7544746, 38.7277603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=207, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 596

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2840222, upper bound: 24.2915110
time: 73.19 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2840222, upper bound: 24.3200182
time: 78.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -43.0790176, 19.4325657, -42.9874687, 19.3527412, -62.4317589, 62.4200363
1: -21.3248100, 18.2567787, -21.2826920, 18.1824417, -39.5072517, 39.5394707
2: -15.0708828, 20.2857170, -15.0290146, 20.1783333, -35.2492142, 35.3147316
3: -20.4731483, 22.5714054, -20.4314232, 22.4265633, -42.8997116, 43.0028305
4: -24.1503983, 20.1389980, -24.0895386, 20.0185013, -44.1688995, 44.2285385
5: -19.2140846, 21.7130604, -19.1769257, 21.6052799, -40.8193665, 40.8899841
6: -33.2749939, 15.8967819, -33.1888695, 15.8622456, -49.1372375, 49.0856514
7: -24.6651592, 19.6839561, -24.6171150, 19.6105862, -44.2757454, 44.3010712
8: -27.9300232, 26.9897766, -27.8569870, 26.8737907, -54.8038139, 54.8467636
9: -24.0991077, 23.1450005, -24.0389977, 23.0448112, -47.1439209, 47.1839981
10: -32.4933739, 25.0875874, -32.4104004, 25.0118065, -57.5051804, 57.4979858
11: -27.1633682, 16.3883705, -26.9781914, 16.3404408, -43.5038071, 43.3665619
12: -32.3179779, 22.5344162, -32.1738930, 22.4876785, -54.4408264, 54.3397980
13: -31.3467808, 30.8803978, -31.2611465, 30.7349854, -62.0817642, 62.1415443
14: -51.6382904, 16.1818771, -51.4187241, 16.1330261, -67.7713165, 67.6006012
15: -26.7491493, 17.8821754, -26.7064571, 17.8231182, -44.5722656, 44.5886307
16: -34.0693359, 18.2048168, -33.9955978, 18.1325455, -52.2018814, 52.2004166
17: -50.8964195, 17.8797703, -50.6455841, 17.8032265, -68.6996460, 68.5253525
18: -36.0093002, 18.0997086, -35.8091965, 18.0244064, -54.0337067, 53.9089050
19: -20.7147598, 14.1723700, -20.5951576, 14.1407290, -34.8554878, 34.7675285
20: -20.7040367, 17.9241371, -20.5780926, 17.8671646, -38.5712013, 38.5022278
21: -26.4415436, 15.6216564, -26.2742081, 15.5735741, -42.0151176, 41.8958664
22: -26.0909786, 15.7664175, -25.9614162, 15.7280235, -41.8190002, 41.7278328
23: -19.4299660, 19.4035301, -19.3127480, 19.3593979, -38.7893639, 38.7162781
24: -27.7054996, 18.1313629, -27.5981808, 18.0853672, -45.7908669, 45.7295456
25: -21.5763779, 21.4746456, -21.4629230, 21.4061947, -42.9825745, 42.9375687
26: -35.3748589, 25.5560551, -35.1589127, 25.4935074, -60.8683662, 60.7149658
27: -26.3622742, 17.0585709, -26.2173252, 16.9989357, -43.3612099, 43.2758942
28: -20.2278519, 20.4911461, -20.1179352, 20.4380569, -40.6659088, 40.6090813
29: -24.4684944, 15.3805704, -24.3129120, 15.3499041, -39.8183975, 39.6934814
30: -26.8193779, 19.5096664, -26.7071037, 19.4621296, -46.2815094, 46.2167702
31: -28.5734367, 19.9665051, -28.4276505, 19.9072075, -48.4806442, 48.3941574
32: -32.0298309, 15.0345726, -31.9538040, 14.9943628, -46.8912201, 46.8557434
33: -52.5227814, 17.3337631, -52.4346886, 17.1750851, -68.6537323, 68.7339096
34: -45.5822105, 5.8276091, -45.5239754, 5.7404938, -49.8289108, 49.8708878
35: -41.1327438, 14.1943903, -41.0875435, 14.0996189, -54.2272263, 54.2866173
36: -36.7542915, 17.8912239, -36.6940460, 17.8557568, -54.6007309, 54.5759850
37: -59.3504143, 8.3617964, -59.2251053, 8.2899818, -67.5576935, 67.5096130
38: -45.8973656, 17.6458702, -45.8023682, 17.5938835, -63.4912491, 63.4482384
39: -55.1658592, 18.2839661, -55.0548248, 18.1156750, -73.1721344, 73.2339096
40: -44.8398895, 6.0860071, -44.7395515, 6.0098162, -50.7764282, 50.7530861
41: -35.4980431, 16.9570236, -35.4400673, 16.8893776, -52.3395233, 52.3508682
42: -23.7852211, 15.0001745, -23.7357731, 14.9610996, -38.7463226, 38.7359467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=207, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 873

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 596

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3200178, upper bound: 24.2555322
time: 70.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3200178, upper bound: 24.2840222
time: 78.46 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 151.06 seconds
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 151.06
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2907484
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 151.06
Output dim: 20, lower bound: -24.2505523, upper bound: 24.3192570
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 151.06
Output dim: 20, lower bound: -24.2840222, upper bound: 24.2915110
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 151.06
Output dim: 20, lower bound: -24.2840222, upper bound: 24.3200182
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 151.06
Output dim: 20, lower bound: -24.3200178, upper bound: 24.2555322
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 151.06
Output dim: 20, lower bound: -24.3200178, upper bound: 24.2840222

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -42.9941101, 19.4044418, -42.9524078, 19.3441639, -62.3382721, 62.3568497
1: -21.2813110, 18.2447014, -21.2524643, 18.1697960, -39.4511070, 39.4971657
2: -15.0332117, 20.2629356, -15.0189199, 20.1873665, -35.2205772, 35.2818565
3: -20.4174385, 22.5621071, -20.3891754, 22.4005394, -42.8179779, 42.9512825
4: -24.0813313, 20.1420517, -24.0581932, 19.9902725, -44.0716019, 44.2002449
5: -19.1671906, 21.7014656, -19.1234055, 21.5783520, -40.7455444, 40.8248711
6: -33.2765541, 15.8730431, -33.2112732, 15.8760624, -49.1526184, 49.0843163
7: -24.6237869, 19.6562805, -24.5888481, 19.5981712, -44.2219582, 44.2451286
8: -27.8587246, 26.9514389, -27.8600769, 26.8747787, -54.7335052, 54.8115158
9: -24.0263748, 23.1488190, -23.9856720, 23.0054855, -47.0318604, 47.1344910
10: -32.4324913, 25.1018410, -32.3203392, 24.9571877, -57.3896790, 57.4221802
11: -27.1640987, 16.3252106, -26.9655590, 16.3144455, -43.4785461, 43.2907715
12: -32.2927094, 22.4943943, -32.1780128, 22.4817696, -54.4078827, 54.2986069
13: -31.2477684, 30.8531723, -31.2681217, 30.7620506, -62.0098190, 62.1212921
14: -51.5818520, 16.1619644, -51.4004021, 16.1455784, -67.7274323, 67.5623627
15: -26.6960125, 17.9268608, -26.6456852, 17.7687702, -44.4647827, 44.5725479
16: -34.0284348, 18.1874695, -33.9315414, 18.0964108, -52.1248474, 52.1190109
17: -50.8226624, 17.8267288, -50.6378593, 17.8102684, -68.6329346, 68.4645844
18: -35.9513626, 18.0143661, -35.8366318, 18.0519657, -54.0033264, 53.8509979
19: -20.7204666, 14.1215801, -20.5609207, 14.0978508, -34.8183174, 34.6825027
20: -20.6807899, 17.8519211, -20.5651760, 17.8476982, -38.5284882, 38.4170990
21: -26.4519501, 15.5511560, -26.2517166, 15.5255928, -41.9775429, 41.8028717
22: -26.0664024, 15.7277651, -25.9428787, 15.7022972, -41.7686996, 41.6706429
23: -19.4411011, 19.3340645, -19.2448521, 19.2834206, -38.7245216, 38.5789185
24: -27.7080021, 18.0554314, -27.5527725, 18.0296516, -45.7376556, 45.6082039
25: -21.5595360, 21.3868790, -21.4279232, 21.3616028, -42.9211388, 42.8148041
26: -35.3350983, 25.4782944, -35.1403465, 25.4502945, -60.7853928, 60.6186409
27: -26.3336163, 16.9749870, -26.2211761, 16.9900379, -43.3236542, 43.1961632
28: -20.2130337, 20.4165688, -20.0864716, 20.3874798, -40.6005135, 40.5030403
29: -24.4543991, 15.3437395, -24.2935448, 15.3243341, -39.7787323, 39.6372833
30: -26.8505821, 19.4471626, -26.6622581, 19.4121437, -46.2627258, 46.1094208
31: -28.5550823, 19.8842850, -28.4168854, 19.8950081, -48.4500885, 48.3011703
32: -32.0068436, 15.0090752, -31.9705505, 15.0126286, -46.8861084, 46.8455963
33: -52.4934311, 17.2558498, -52.4463272, 17.1764126, -68.6253357, 68.6480255
34: -45.5489082, 5.8019104, -45.5452499, 5.7393923, -49.7948074, 49.8406448
35: -41.1078873, 14.1562729, -41.0785866, 14.0734339, -54.1800995, 54.2156982
36: -36.7206573, 17.8650398, -36.6797256, 17.8139076, -54.5251617, 54.5352898
37: -59.3305550, 8.2819252, -59.2055511, 8.2617188, -67.5087891, 67.4012299
38: -45.8358459, 17.6064949, -45.8019485, 17.5552864, -63.3911324, 63.4084435
39: -55.0961876, 18.2127304, -55.0752869, 18.1229591, -73.1064606, 73.1752014
40: -44.8122559, 6.0518837, -44.7690430, 6.0291729, -50.7675171, 50.7458420
41: -35.4820709, 16.9214859, -35.4469681, 16.8903599, -52.3252563, 52.3189392
42: -23.7972603, 14.9755936, -23.7337685, 14.9608850, -38.7581444, 38.7093620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=207, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=401, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2239222, upper bound: 24.3162264
time: 62.27 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2475471, upper bound: 24.3162088
time: 77.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -43.0215683, 19.4090691, -43.0513992, 19.4062157, -62.4277840, 62.4604683
1: -21.2993126, 18.2486305, -21.3088951, 18.2137051, -39.5130157, 39.5575256
2: -15.0445461, 20.2652512, -15.0576124, 20.2141399, -35.2586861, 35.3228645
3: -20.4406548, 22.5657158, -20.4648514, 22.4688950, -42.9095497, 43.0305672
4: -24.1051769, 20.1460705, -24.1369514, 20.0501804, -44.1553574, 44.2830200
5: -19.1903553, 21.7050400, -19.2024994, 21.6444340, -40.8347893, 40.9075394
6: -33.2749634, 15.8799305, -33.2166290, 15.8818083, -49.1567726, 49.0965576
7: -24.6404209, 19.6592560, -24.6440163, 19.6416073, -44.2820282, 44.3032722
8: -27.8754234, 26.9559155, -27.9152374, 26.9208946, -54.7963181, 54.8711548
9: -24.0563087, 23.1530285, -24.0837955, 23.0858078, -47.1421165, 47.2368240
10: -32.4692955, 25.1069965, -32.4412003, 25.0496864, -57.5189819, 57.5481949
11: -27.1748753, 16.3459053, -27.0218372, 16.3823910, -43.5572662, 43.3677444
12: -32.2940979, 22.5040779, -32.1897736, 22.5205612, -54.4496765, 54.3259277
13: -31.2695160, 30.8618011, -31.3393707, 30.8202362, -62.0897522, 62.2011719
14: -51.5988884, 16.1671181, -51.4646378, 16.1731148, -67.7720032, 67.6317596
15: -26.7225990, 17.9315224, -26.7357502, 17.8324413, -44.5550385, 44.6672745
16: -34.0543175, 18.1925755, -34.0195122, 18.1583385, -52.2126541, 52.2120895
17: -50.8420906, 17.8399467, -50.7061081, 17.8637314, -68.7058258, 68.5460510
18: -35.9604416, 18.0285263, -35.8867302, 18.0998077, -54.0602493, 53.9152565
19: -20.7271976, 14.1451330, -20.6352043, 14.1679554, -34.8951530, 34.7803383
20: -20.6875687, 17.8739700, -20.6286869, 17.9167213, -38.6042900, 38.5026550
21: -26.4615498, 15.5805092, -26.3213120, 15.6141024, -42.0756531, 41.9018211
22: -26.0734272, 15.7438688, -25.9997215, 15.7527218, -41.8261490, 41.7435913
23: -19.4484844, 19.3691597, -19.3499794, 19.3938560, -38.8423386, 38.7191391
24: -27.7139378, 18.0870571, -27.6445694, 18.1285782, -45.8425140, 45.7316284
25: -21.5655003, 21.4191017, -21.5095482, 21.4633942, -43.0288925, 42.9286499
26: -35.3440018, 25.5083561, -35.2248383, 25.5432720, -60.8872757, 60.7331924
27: -26.3389320, 16.9973984, -26.2931633, 17.0613384, -43.4002686, 43.2905617
28: -20.2186356, 20.4466801, -20.1701679, 20.4832458, -40.7018814, 40.6168480
29: -24.4621391, 15.3581953, -24.3493786, 15.3713942, -39.8335342, 39.7075729
30: -26.8561306, 19.4727573, -26.7315540, 19.5010204, -46.3571510, 46.2043114
31: -28.5631790, 19.9072495, -28.4981937, 19.9666634, -48.5298424, 48.4054413
32: -32.0062943, 15.0147753, -31.9802361, 15.0170422, -46.8903961, 46.8635559
33: -52.4996605, 17.2738743, -52.5006752, 17.2354488, -68.6935349, 68.7356796
34: -45.5492935, 5.8088017, -45.5556870, 5.7630062, -49.8237152, 49.8801270
35: -41.1125641, 14.1726513, -41.1213608, 14.1231899, -54.2374496, 54.2910194
36: -36.7249718, 17.8815613, -36.7205582, 17.8665466, -54.5821686, 54.5929184
37: -59.3388824, 8.3075714, -59.2942810, 8.3436890, -67.6001892, 67.5230789
38: -45.8423500, 17.6250515, -45.8573380, 17.6169605, -63.4593124, 63.4823914
39: -55.1044044, 18.2275391, -55.1335373, 18.1705627, -73.1631775, 73.2554779
40: -44.8156357, 6.0592651, -44.7964287, 6.0375166, -50.7796631, 50.7826767
41: -35.4870682, 16.9296246, -35.4742203, 16.9207840, -52.3608322, 52.3567848
42: -23.7985630, 14.9815693, -23.7495899, 14.9826021, -38.7811661, 38.7311592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=207, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2239222, upper bound: 24.3169546
time: 71.22 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2809326, upper bound: 24.3169400
time: 62.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -43.0357895, 19.4252338, -42.9731293, 19.3503075, -62.3860970, 62.3983612
1: -21.3002167, 18.2498856, -21.2745285, 18.1801300, -39.4803467, 39.5244141
2: -15.0499964, 20.2813702, -15.0220966, 20.1768723, -35.2268677, 35.3034668
3: -20.4435577, 22.5655479, -20.4216156, 22.4246140, -42.8681717, 42.9871635
4: -24.1082611, 20.1312599, -24.0755844, 20.0159130, -44.1241760, 44.2068443
5: -19.1795864, 21.7071915, -19.1654892, 21.6033287, -40.7829132, 40.8726807
6: -33.2667046, 15.8653469, -33.1861458, 15.8517179, -49.1184235, 49.0514908
7: -24.6514664, 19.6795597, -24.6126060, 19.6090984, -44.2605667, 44.2921677
8: -27.9139824, 26.9828205, -27.8516483, 26.8714485, -54.7854309, 54.8344688
9: -24.0545769, 23.1378651, -24.0242748, 23.0424404, -47.0970154, 47.1621399
10: -32.4404678, 25.0763149, -32.3928452, 25.0081005, -57.4485703, 57.4691620
11: -27.1462574, 16.3396339, -26.9725208, 16.3242970, -43.4705544, 43.3121567
12: -32.3082962, 22.5150337, -32.1705208, 22.4812965, -54.4244843, 54.3169022
13: -31.2756920, 30.8643837, -31.2376232, 30.7296753, -62.0053673, 62.1020050
14: -51.5805130, 16.1704540, -51.3995361, 16.1292572, -67.7097702, 67.5699921
15: -26.6966667, 17.8718987, -26.6890697, 17.8197136, -44.5163803, 44.5609665
16: -34.0571136, 18.1949463, -33.9914894, 18.1292801, -52.1863937, 52.1864357
17: -50.8543739, 17.8691139, -50.6316986, 17.7996655, -68.6540375, 68.5008087
18: -35.9975052, 18.0721531, -35.8052902, 18.0152588, -54.0127640, 53.8774414
19: -20.7018013, 14.1272640, -20.5908585, 14.1258049, -34.8276062, 34.7181244
20: -20.6909962, 17.8835335, -20.5737877, 17.8537045, -38.5447006, 38.4573212
21: -26.4220943, 15.5548086, -26.2677631, 15.5514469, -41.9735413, 41.8225708
22: -26.0798988, 15.7525311, -25.9577293, 15.7234058, -41.8033066, 41.7102585
23: -19.4142437, 19.3450241, -19.3075180, 19.3400497, -38.7542953, 38.6525421
24: -27.6917725, 18.0806427, -27.5936413, 18.0685883, -45.7603607, 45.6742859
25: -21.5647697, 21.4311314, -21.4590759, 21.3917828, -42.9565506, 42.8902054
26: -35.3596687, 25.5150032, -35.1538658, 25.4798794, -60.8395462, 60.6688690
27: -26.3505783, 17.0103378, -26.2134609, 16.9829731, -43.3335495, 43.2238007
28: -20.2165184, 20.4397964, -20.1141777, 20.4210510, -40.6375694, 40.5539742
29: -24.4555664, 15.3595409, -24.3086262, 15.3429461, -39.7985115, 39.6681671
30: -26.8061962, 19.4489746, -26.7027359, 19.4420338, -46.2482300, 46.1517105
31: -28.5593719, 19.9192352, -28.4229355, 19.8915768, -48.4509506, 48.3421707
32: -32.0188675, 15.0251093, -31.9501934, 14.9912167, -46.8768539, 46.8421936
33: -52.5098228, 17.2860947, -52.4303818, 17.1592636, -68.6213150, 68.6746140
34: -45.5708351, 5.8170996, -45.5202141, 5.7369814, -49.8114243, 49.8478661
35: -41.1214600, 14.1790066, -41.0838089, 14.0944204, -54.2075958, 54.2609863
36: -36.7382889, 17.8821182, -36.6887054, 17.8527641, -54.5816803, 54.5615463
37: -59.3353539, 8.3040886, -59.2200699, 8.2708139, -67.5220947, 67.4436111
38: -45.8777504, 17.6378174, -45.7957726, 17.5912304, -63.4689789, 63.4335899
39: -55.1477470, 18.2743759, -55.0488091, 18.1124916, -73.1484375, 73.2164612
40: -44.8295746, 6.0530872, -44.7361145, 5.9988880, -50.7543335, 50.7148590
41: -35.4879456, 16.9264526, -35.4367142, 16.8792343, -52.3187256, 52.3159103
42: -23.7768211, 14.9659748, -23.7330036, 14.9497366, -38.7265587, 38.6989784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=207, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2934162, upper bound: 24.2525407
time: 73.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3169398, upper bound: 24.2525290
time: 74.15 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -43.0865288, 19.4631081, -42.9864159, 19.3521500, -62.4386787, 62.4495239
1: -21.3260174, 18.2804699, -21.2821922, 18.1818466, -39.5078659, 39.5626602
2: -15.0736504, 20.3014565, -15.0285139, 20.1779308, -35.2515793, 35.3299713
3: -20.4747086, 22.6085548, -20.4308167, 22.4260750, -42.9007835, 43.0393715
4: -24.1532974, 20.1781330, -24.0887871, 20.0181236, -44.1714211, 44.2669220
5: -19.2167454, 21.7445927, -19.1761227, 21.6048985, -40.8216438, 40.9207153
6: -33.3031120, 15.9002991, -33.1884842, 15.8614635, -49.1645737, 49.0887833
7: -24.6678181, 19.6907387, -24.6166458, 19.6101303, -44.2779465, 44.3073845
8: -27.9340897, 27.0035267, -27.8565788, 26.8732433, -54.8073349, 54.8601074
9: -24.1018906, 23.1944294, -24.0381622, 23.0444145, -47.1463051, 47.2325897
10: -32.5012436, 25.1454277, -32.4092331, 25.0112686, -57.5125122, 57.5546608
11: -27.2191353, 16.3887138, -26.9775848, 16.3395729, -43.5587082, 43.3662987
12: -32.3131027, 22.5376816, -32.1707153, 22.4869843, -54.4352875, 54.3403244
13: -31.3492680, 30.9476852, -31.2596302, 30.7343636, -62.0836334, 62.2073135
14: -51.6463623, 16.2076283, -51.4171028, 16.1326084, -67.7789688, 67.6247330
15: -26.7529945, 17.9413280, -26.7053452, 17.8226643, -44.5756607, 44.6466751
16: -34.0786591, 18.2190666, -33.9951477, 18.1318436, -52.2105026, 52.2142143
17: -50.9034119, 17.9008751, -50.6447372, 17.8028069, -68.7062225, 68.5456085
18: -36.0387573, 18.1045017, -35.8084221, 18.0237923, -54.0625496, 53.9129257
19: -20.7677078, 14.1731014, -20.5946712, 14.1399832, -34.9076920, 34.7677727
20: -20.7388935, 17.9242840, -20.5773430, 17.8663769, -38.6052704, 38.5016251
21: -26.5093536, 15.6222124, -26.2734890, 15.5723877, -42.0817413, 41.8957024
22: -26.1124096, 15.7689257, -25.9607048, 15.7276516, -41.8400612, 41.7296295
23: -19.4863834, 19.4045811, -19.3120766, 19.3584099, -38.8447952, 38.7166595
24: -27.7609501, 18.1311684, -27.5975780, 18.0844650, -45.8454132, 45.7287445
25: -21.6126900, 21.4770927, -21.4623604, 21.4053688, -43.0180588, 42.9394531
26: -35.4109879, 25.5590134, -35.1578331, 25.4925709, -60.9035568, 60.7168465
27: -26.4154358, 17.0606174, -26.2167053, 16.9981194, -43.4135551, 43.2773209
28: -20.2713890, 20.4927788, -20.1174335, 20.4371185, -40.7085075, 40.6102142
29: -24.4992447, 15.3801222, -24.3122883, 15.3494282, -39.8486710, 39.6924095
30: -26.8811626, 19.5127602, -26.7065506, 19.4610348, -46.3421974, 46.2193108
31: -28.6344395, 19.9675865, -28.4269485, 19.9062805, -48.5407181, 48.3945351
32: -32.0349541, 15.0377111, -31.9515648, 14.9941549, -46.8972855, 46.8566742
33: -52.5661812, 17.3352470, -52.4340973, 17.1740685, -68.7008133, 68.7283173
34: -45.5831337, 5.8317366, -45.5218430, 5.7400932, -49.8374634, 49.8664017
35: -41.1467934, 14.1966906, -41.0870857, 14.0991898, -54.2465973, 54.2818604
36: -36.7551727, 17.8925400, -36.6903534, 17.8555374, -54.6014557, 54.5736465
37: -59.4088707, 8.3624964, -59.2243385, 8.2888498, -67.6170044, 67.5064468
38: -45.9011917, 17.6483879, -45.7984924, 17.5936317, -63.4948235, 63.4468803
39: -55.1838722, 18.2836838, -55.0540619, 18.1143970, -73.1871185, 73.2315445
40: -44.8730164, 6.0880976, -44.7390022, 6.0086784, -50.8093033, 50.7529602
41: -35.5215607, 16.9618149, -35.4396515, 16.8885689, -52.3632050, 52.3543434
42: -23.8127365, 15.0039177, -23.7354164, 14.9602518, -38.7729874, 38.7393341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=207, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2934162, upper bound: 24.2809520
time: 77.33 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3169398, upper bound: 24.2809329
time: 79.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 159.30 seconds
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 159.30
Output dim: 20, lower bound: -24.2239222, upper bound: 24.3162264
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 159.30
Output dim: 20, lower bound: -24.2475471, upper bound: 24.3162088
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 159.30
Output dim: 20, lower bound: -24.2239222, upper bound: 24.3169546
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 159.30
Output dim: 20, lower bound: -24.2809326, upper bound: 24.3169400
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 159.30
Output dim: 20, lower bound: -24.2934162, upper bound: 24.2525407
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 159.30
Output dim: 20, lower bound: -24.3169398, upper bound: 24.2525290
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 159.30
Output dim: 20, lower bound: -24.2934162, upper bound: 24.2809520
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 159.30
Output dim: 20, lower bound: -24.3169398, upper bound: 24.2809329

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -42.9941101, 19.4044418, -42.9192009, 19.3347511, -62.3288612, 62.3236427
1: -21.2813110, 18.2447014, -21.2404270, 18.1639023, -39.4452133, 39.4851303
2: -15.0332117, 20.2629356, -15.0014391, 20.1815300, -35.2147408, 35.2643738
3: -20.4174385, 22.5621071, -20.3639259, 22.3912582, -42.8086967, 42.9260330
4: -24.0813313, 20.1420517, -24.0329094, 19.9839325, -44.0652618, 44.1749611
5: -19.1671906, 21.7014656, -19.1015568, 21.5694046, -40.7365952, 40.8030243
6: -33.2765541, 15.8730431, -33.2014961, 15.8581467, -49.1347008, 49.0745392
7: -24.6237869, 19.6562805, -24.5698738, 19.5886917, -44.2124786, 44.2261543
8: -27.8587246, 26.9514389, -27.8368378, 26.8652134, -54.7239380, 54.7882767
9: -24.0263748, 23.1488190, -23.9618130, 22.9963169, -47.0226898, 47.1106339
10: -32.4324913, 25.1018410, -32.3030548, 24.9456234, -57.3781128, 57.4048958
11: -27.1640987, 16.3252106, -26.9499798, 16.2845287, -43.4486275, 43.2751923
12: -32.2927094, 22.4943943, -32.1662598, 22.4187431, -54.3433228, 54.2868729
13: -31.2477684, 30.8531723, -31.2547855, 30.7215157, -61.9692841, 62.1079559
14: -51.5818520, 16.1619644, -51.3801346, 16.0822182, -67.6640701, 67.5420990
15: -26.6960125, 17.9268608, -26.6047916, 17.7589264, -44.4549408, 44.5316544
16: -34.0284348, 18.1874695, -33.9099312, 18.0839939, -52.1124268, 52.0974007
17: -50.8226624, 17.8267288, -50.6161919, 17.7433128, -68.5659790, 68.4429169
18: -35.9513626, 18.0143661, -35.8229027, 18.0416260, -53.9929886, 53.8372688
19: -20.7204666, 14.1215801, -20.5496788, 14.0902786, -34.8107452, 34.6712570
20: -20.6807899, 17.8519211, -20.5529747, 17.8165436, -38.4973335, 38.4048958
21: -26.4519501, 15.5511560, -26.2387447, 15.4985905, -41.9505386, 41.7899017
22: -26.0664024, 15.7277651, -25.9278355, 15.6868734, -41.7532768, 41.6556015
23: -19.4411011, 19.3340645, -19.2359009, 19.2755470, -38.7166481, 38.5699654
24: -27.7080021, 18.0554314, -27.5381107, 18.0236988, -45.7317009, 45.5935440
25: -21.5595360, 21.3868790, -21.4143085, 21.3376122, -42.8971481, 42.8011856
26: -35.3350983, 25.4782944, -35.1238098, 25.4171028, -60.7522011, 60.6021042
27: -26.3336163, 16.9749870, -26.2042713, 16.9841785, -43.3177948, 43.1792603
28: -20.2130337, 20.4165688, -20.0760212, 20.3735619, -40.5865936, 40.4925919
29: -24.4543991, 15.3437395, -24.2816963, 15.3059692, -39.7603683, 39.6254349
30: -26.8505821, 19.4471626, -26.6502686, 19.3841572, -46.2347412, 46.0974312
31: -28.5550823, 19.8842850, -28.4026833, 19.8849850, -48.4400673, 48.2869682
32: -32.0068436, 15.0090752, -31.9615822, 14.9908600, -46.8642426, 46.8366470
33: -52.4934311, 17.2558498, -52.4153175, 17.1638832, -68.6123962, 68.6150055
34: -45.5489082, 5.8019104, -45.5191345, 5.7295828, -49.7846985, 49.8092537
35: -41.1078873, 14.1562729, -41.0640793, 14.0671425, -54.1726074, 54.1985626
36: -36.7206573, 17.8650398, -36.6703033, 17.7983894, -54.5095825, 54.5258331
37: -59.3305550, 8.2819252, -59.1860733, 8.2541265, -67.5003662, 67.3806763
38: -45.8358459, 17.6064949, -45.7874603, 17.5376892, -63.3735352, 63.3939552
39: -55.0961876, 18.2127304, -55.0581551, 18.1167259, -73.0997925, 73.1570816
40: -44.8122559, 6.0518837, -44.7412338, 6.0190792, -50.7574768, 50.7178612
41: -35.4820709, 16.9214859, -35.4278908, 16.8817596, -52.3165894, 52.2999458
42: -23.7972603, 14.9755936, -23.7240372, 14.9476995, -38.7449608, 38.6996307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=206, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1689

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2239222, upper bound: 24.2926474
time: 73.11 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2239222, upper bound: 24.3162088
time: 69.25 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -42.9868240, 19.4023819, -42.9630966, 19.3982086, -62.3850327, 62.3654785
1: -21.2770042, 18.2429314, -21.2605267, 18.2057800, -39.4827843, 39.5034561
2: -15.0278902, 20.2612591, -15.0222092, 20.2424107, -35.2703018, 35.2834702
3: -20.4139576, 22.5590210, -20.3904305, 22.4836273, -42.8975830, 42.9494514
4: -24.0747299, 20.1403484, -24.0576897, 20.0765438, -44.1512756, 44.1980362
5: -19.1622334, 21.6989861, -19.1232948, 21.6502838, -40.8125153, 40.8222809
6: -33.2733307, 15.8667068, -33.2591324, 15.8876038, -49.1609344, 49.1258392
7: -24.6181736, 19.6529655, -24.6000004, 19.6211433, -44.2393188, 44.2529678
8: -27.8518219, 26.9486160, -27.8604355, 26.9408531, -54.7926750, 54.8090515
9: -24.0209198, 23.1462536, -23.9879131, 23.0503235, -47.0712433, 47.1341667
10: -32.4278336, 25.0989304, -32.3624496, 24.9870186, -57.4148521, 57.4613800
11: -27.1601734, 16.3141174, -27.1008186, 16.2989006, -43.4590759, 43.4149361
12: -32.2892914, 22.4846840, -32.3495483, 22.4839325, -54.4041595, 54.4592438
13: -31.2442379, 30.8452492, -31.3383980, 30.7777214, -62.0219574, 62.1836472
14: -51.5759315, 16.1531830, -51.6053429, 16.1483402, -67.7242737, 67.7585297
15: -26.6850491, 17.9245281, -26.6516666, 17.8437443, -44.5287933, 44.5761948
16: -34.0205536, 18.1831074, -33.9620399, 18.1427631, -52.1633148, 52.1451492
17: -50.8181839, 17.8173370, -50.8806648, 17.8154049, -68.6335907, 68.6979980
18: -35.9478989, 18.0120983, -35.8881989, 18.0641327, -54.0120316, 53.9002991
19: -20.7175961, 14.1183577, -20.6233406, 14.0983543, -34.8159485, 34.7416992
20: -20.6779060, 17.8469429, -20.6295605, 17.8488350, -38.5267410, 38.4765015
21: -26.4485893, 15.5473814, -26.3629456, 15.5338564, -41.9824448, 41.9103279
22: -26.0610237, 15.7214165, -25.9931755, 15.7102652, -41.7712898, 41.7145920
23: -19.4388504, 19.3312531, -19.2941151, 19.2890701, -38.7279205, 38.6253662
24: -27.7017956, 18.0523434, -27.5668736, 18.0312366, -45.7330322, 45.6192169
25: -21.5549316, 21.3815022, -21.4751282, 21.3724556, -42.9273872, 42.8566284
26: -35.3305664, 25.4710579, -35.2370911, 25.4488792, -60.7794456, 60.7081490
27: -26.3248940, 16.9734020, -26.2283955, 17.0180473, -43.3429413, 43.2017975
28: -20.2107239, 20.4131889, -20.1145115, 20.3892365, -40.5999603, 40.5277023
29: -24.4492798, 15.3390570, -24.3671436, 15.3256292, -39.7749100, 39.7061996
30: -26.8467655, 19.4354420, -26.7188797, 19.4048157, -46.2515793, 46.1543198
31: -28.5511265, 19.8781662, -28.4654503, 19.8933372, -48.4444656, 48.3436165
32: -32.0038223, 15.0035725, -32.0061264, 15.0118637, -46.8818741, 46.8762779
33: -52.4871025, 17.2522831, -52.4678116, 17.2407188, -68.6817474, 68.6630249
34: -45.5453300, 5.7988901, -45.5602074, 5.8319998, -49.8921967, 49.8497162
35: -41.1044922, 14.1541271, -41.0937004, 14.1153336, -54.2181854, 54.2240181
36: -36.7176743, 17.8625450, -36.7261314, 17.8279762, -54.5362778, 54.5790787
37: -59.3236313, 8.2795849, -59.2396851, 8.2891645, -67.5265656, 67.4302216
38: -45.8313904, 17.6019039, -45.8668823, 17.5741348, -63.4055252, 63.4687881
39: -55.0909576, 18.2106056, -55.1069107, 18.1683006, -73.1474609, 73.2036514
40: -44.8035736, 6.0479183, -44.7840424, 6.1196499, -50.8475876, 50.7548294
41: -35.4746933, 16.9185791, -35.4556808, 16.9379025, -52.3637085, 52.3234634
42: -23.7950058, 14.9710150, -23.7647896, 14.9671612, -38.7621689, 38.7358055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=206, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=401, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 532

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2463233, upper bound: 24.2892081
time: 82.30 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2463233, upper bound: 24.3149843
time: 71.06 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -43.0215683, 19.4090691, -43.0182343, 19.3967590, -62.4183273, 62.4273033
1: -21.2993126, 18.2486305, -21.2968407, 18.2078171, -39.5071297, 39.5454712
2: -15.0445461, 20.2652512, -15.0401487, 20.2083111, -35.2528572, 35.3054008
3: -20.4406548, 22.5657158, -20.4395943, 22.4596138, -42.9002686, 43.0053101
4: -24.1051769, 20.1460705, -24.1117020, 20.0438728, -44.1490479, 44.2577744
5: -19.1903553, 21.7050400, -19.1806526, 21.6354599, -40.8258133, 40.8856926
6: -33.2749634, 15.8799305, -33.2068405, 15.8639002, -49.1388626, 49.0867691
7: -24.6404209, 19.6592560, -24.6250763, 19.6320972, -44.2725182, 44.2843323
8: -27.8754234, 26.9559155, -27.8920593, 26.9113083, -54.7867317, 54.8479767
9: -24.0563087, 23.1530285, -24.0599060, 23.0766144, -47.1329231, 47.2129364
10: -32.4692955, 25.1069965, -32.4239044, 25.0381336, -57.5074310, 57.5308990
11: -27.1748753, 16.3459053, -27.0062256, 16.3524628, -43.5273361, 43.3521309
12: -32.2940979, 22.5040779, -32.1780128, 22.4575768, -54.3851318, 54.3142319
13: -31.2695160, 30.8618011, -31.3260098, 30.7797146, -62.0492325, 62.1878128
14: -51.5988884, 16.1671181, -51.4443436, 16.1097584, -67.7086487, 67.6114655
15: -26.7225990, 17.9315224, -26.6948509, 17.8225708, -44.5451698, 44.6263733
16: -34.0543175, 18.1925755, -33.9978943, 18.1459599, -52.2002792, 52.1904678
17: -50.8420906, 17.8399467, -50.6844025, 17.7968082, -68.6389008, 68.5243530
18: -35.9604416, 18.0285263, -35.8730202, 18.0894604, -54.0499039, 53.9015465
19: -20.7271976, 14.1451330, -20.6239548, 14.1604004, -34.8875961, 34.7690887
20: -20.6875687, 17.8739700, -20.6165085, 17.8855743, -38.5731430, 38.4904785
21: -26.4615498, 15.5805092, -26.3083630, 15.5870962, -42.0486450, 41.8888702
22: -26.0734272, 15.7438688, -25.9846611, 15.7372990, -41.8107262, 41.7285309
23: -19.4484844, 19.3691597, -19.3410339, 19.3859558, -38.8344421, 38.7101936
24: -27.7139378, 18.0870571, -27.6298943, 18.1226120, -45.8365479, 45.7169495
25: -21.5655003, 21.4191017, -21.4959335, 21.4394054, -43.0049057, 42.9150352
26: -35.3440018, 25.5083561, -35.2083054, 25.5100861, -60.8540878, 60.7166595
27: -26.3389320, 16.9973984, -26.2762489, 17.0554790, -43.3944092, 43.2736473
28: -20.2186356, 20.4466801, -20.1597366, 20.4693260, -40.6879616, 40.6064148
29: -24.4621391, 15.3581953, -24.3375053, 15.3530350, -39.8151741, 39.6957016
30: -26.8561306, 19.4727573, -26.7195759, 19.4730587, -46.3291893, 46.1923332
31: -28.5631790, 19.9072495, -28.4840164, 19.9566193, -48.5197983, 48.3912659
32: -32.0062943, 15.0147753, -31.9712448, 14.9952879, -46.8685455, 46.8545837
33: -52.4996605, 17.2738743, -52.4696350, 17.2228928, -68.6806183, 68.7026138
34: -45.5492935, 5.8088017, -45.5295410, 5.7532196, -49.8135986, 49.8487511
35: -41.1125641, 14.1726513, -41.1068344, 14.1169062, -54.2299805, 54.2738647
36: -36.7249718, 17.8815613, -36.7111206, 17.8510151, -54.5666199, 54.5834656
37: -59.3388824, 8.3075714, -59.2748184, 8.3360395, -67.5917969, 67.5024948
38: -45.8423500, 17.6250515, -45.8428917, 17.5993271, -63.4416771, 63.4679413
39: -55.1044044, 18.2275391, -55.1164093, 18.1643715, -73.1565857, 73.2373962
40: -44.8156357, 6.0592651, -44.7685966, 6.0273933, -50.7696075, 50.7546921
41: -35.4870682, 16.9296246, -35.4551392, 16.9122124, -52.3521729, 52.3377914
42: -23.7985630, 14.9815693, -23.7398643, 14.9693880, -38.7679520, 38.7214355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=206, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1664

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1689

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2574281, upper bound: 24.2934163
time: 75.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2574281, upper bound: 24.3169404
time: 64.23 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -43.0142860, 19.4069729, -43.0621300, 19.4602203, -62.4745064, 62.4691010
1: -21.2950020, 18.2468739, -21.3169632, 18.2497063, -39.5447083, 39.5638351
2: -15.0392017, 20.2635460, -15.0609131, 20.2691956, -35.3083954, 35.3244591
3: -20.4372025, 22.5626392, -20.4660759, 22.5519753, -42.9891777, 43.0287170
4: -24.0985718, 20.1443520, -24.1364365, 20.1364555, -44.2350273, 44.2807884
5: -19.1854057, 21.7025833, -19.2023773, 21.7163334, -40.9017410, 40.9049606
6: -33.2717018, 15.8736057, -33.2644806, 15.8933468, -49.1650467, 49.1380844
7: -24.6348057, 19.6559563, -24.6552162, 19.6645794, -44.2993851, 44.3111725
8: -27.8685703, 26.9530869, -27.9156170, 26.9869690, -54.8555374, 54.8687057
9: -24.0508690, 23.1504765, -24.0860138, 23.1306496, -47.1815186, 47.2364883
10: -32.4646606, 25.1040802, -32.4833031, 25.0795193, -57.5441818, 57.5873833
11: -27.1709671, 16.3348198, -27.1569519, 16.3668499, -43.5378189, 43.4917717
12: -32.2906799, 22.4943275, -32.3613243, 22.5227356, -54.4459457, 54.4865723
13: -31.2659702, 30.8538551, -31.4096260, 30.8358803, -62.1018524, 62.2634811
14: -51.5929642, 16.1583252, -51.6695557, 16.1758690, -67.7688293, 67.8278809
15: -26.7116070, 17.9291515, -26.7417278, 17.9074192, -44.6190262, 44.6708794
16: -34.0464439, 18.1882133, -34.0500107, 18.2046928, -52.2511368, 52.2382240
17: -50.8376160, 17.8305435, -50.9488487, 17.8689327, -68.7065506, 68.7793884
18: -35.9569778, 18.0262299, -35.9383011, 18.1119366, -54.0689163, 53.9645309
19: -20.7243214, 14.1419277, -20.6976109, 14.1684637, -34.8927841, 34.8395386
20: -20.6846867, 17.8689919, -20.6930904, 17.9179153, -38.6026001, 38.5620804
21: -26.4582214, 15.5767231, -26.4324989, 15.6224337, -42.0806541, 42.0092239
22: -26.0680332, 15.7375240, -26.0500164, 15.7607136, -41.8287468, 41.7875404
23: -19.4462166, 19.3663521, -19.3992348, 19.3994942, -38.8457108, 38.7655869
24: -27.7077293, 18.0839920, -27.6586781, 18.1301613, -45.8378906, 45.7426682
25: -21.5608902, 21.4136963, -21.5567551, 21.4742470, -43.0351372, 42.9704514
26: -35.3394699, 25.5011063, -35.3215866, 25.5418262, -60.8812943, 60.8226929
27: -26.3302364, 16.9958076, -26.3003349, 17.0893097, -43.4195480, 43.2961426
28: -20.2163200, 20.4432926, -20.1982212, 20.4850006, -40.7013206, 40.6415138
29: -24.4569931, 15.3534956, -24.4229603, 15.3726959, -39.8296890, 39.7764549
30: -26.8523350, 19.4610710, -26.7882004, 19.4937305, -46.3460655, 46.2492714
31: -28.5592003, 19.9011459, -28.5467949, 19.9649620, -48.5241623, 48.4479408
32: -32.0032539, 15.0092478, -32.0157928, 15.0162735, -46.8861694, 46.8942261
33: -52.4933472, 17.2703094, -52.5221672, 17.2997055, -68.7499008, 68.7506485
34: -45.5456886, 5.8057947, -45.5706367, 5.8556347, -49.9210892, 49.8891525
35: -41.1091728, 14.1705036, -41.1364822, 14.1650743, -54.2754974, 54.2993393
36: -36.7219810, 17.8790493, -36.7669449, 17.8806419, -54.5932922, 54.6366806
37: -59.3319321, 8.3052864, -59.3284225, 8.3711576, -67.6179962, 67.5520401
38: -45.8379135, 17.6204681, -45.9222870, 17.6358185, -63.4737320, 63.5427551
39: -55.0991859, 18.2254181, -55.1651649, 18.2159348, -73.2041626, 73.2839813
40: -44.8069382, 6.0552473, -44.8114548, 6.1280050, -50.8596954, 50.7917023
41: -35.4796600, 16.9267311, -35.4829140, 16.9683304, -52.3992844, 52.3612709
42: -23.7962914, 14.9769745, -23.7806187, 14.9888592, -38.7851486, 38.7575912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=206, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1664

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 532

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2226805, upper bound: 24.2899443
time: 74.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2226805, upper bound: 24.2892077
time: 95.54 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 172.67 seconds
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 172.67
Output dim: 20, lower bound: -24.2239222, upper bound: 24.2926474
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 172.67
Output dim: 20, lower bound: -24.2239222, upper bound: 24.3162088
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 172.67
Output dim: 20, lower bound: -24.2463233, upper bound: 24.2892081
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 172.67
Output dim: 20, lower bound: -24.2463233, upper bound: 24.3149843
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 172.67
Output dim: 20, lower bound: -24.2574281, upper bound: 24.2934163
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 172.67
Output dim: 20, lower bound: -24.2574281, upper bound: 24.3169404
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 172.67
Output dim: 20, lower bound: -24.2226805, upper bound: 24.2899443
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 172.67
Output dim: 20, lower bound: -24.2226805, upper bound: 24.2892077
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 172.67
Output dim: 20, lower bound: -24.3169398, upper bound: 24.2525290
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 172.67
Output dim: 20, lower bound: -24.3169398, upper bound: 24.2809329

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 80.77 + 3577.74 = 3658.52 seconds
