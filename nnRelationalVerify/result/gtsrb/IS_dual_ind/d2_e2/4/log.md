## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 3600 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.24 + 80.84 = 83.08 seconds
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 701

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2934227, upper bound: 24.3293512
time: 65.90 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2934227, upper bound: 24.3294822
time: 62.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 128.16 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 128.16
Output dim: 20, lower bound: -24.2934227, upper bound: 24.3293512
IS_A2, status: Status.UNKNOWN, split count: 1, time: 128.16
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

Time for backsubstitution: 1.83 seconds

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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2592121, upper bound: 24.3279231
time: 83.53 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2592121, upper bound: 24.3279231
time: 67.40 seconds

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

Time for backsubstitution: 1.76 seconds

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
time: 76.85 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2592121, upper bound: 24.3288701
time: 72.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 151.36 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 151.36
Output dim: 20, lower bound: -24.2592121, upper bound: 24.3279231
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 151.36
Output dim: 20, lower bound: -24.2592121, upper bound: 24.3279231
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 151.36
Output dim: 20, lower bound: -24.2952408, upper bound: 24.3280538
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 151.36
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

Time for backsubstitution: 1.85 seconds

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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2984972
time: 76.49 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3196501
time: 73.81 seconds

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

Time for backsubstitution: 1.85 seconds

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
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2993177
time: 81.82 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3204129
time: 76.66 seconds

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

Time for backsubstitution: 1.86 seconds

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
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2986522
time: 74.03 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3198122
time: 73.18 seconds

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

Time for backsubstitution: 1.77 seconds

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
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2994783
time: 81.31 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3205725
time: 78.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 161.96 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 161.96
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2984972
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 161.96
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3196501
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 161.96
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2993177
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 161.96
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3204129
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 161.96
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2986522
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 161.96
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3198122
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 161.96
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2994783
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 161.96
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3205725

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -42.9193649, 19.3375320, -42.8916321, 19.3227348, -62.2420998, 62.2291641
1: -21.2385082, 18.1719170, -21.2247581, 18.1560402, -39.3945465, 39.3966751
2: -14.9722204, 20.1693230, -14.9842806, 20.1731853, -35.1454048, 35.1536026
3: -20.3465691, 22.4104519, -20.3455238, 22.3819122, -42.7284813, 42.7559738
4: -24.0165329, 20.0068436, -24.0053253, 19.9783840, -43.9949188, 44.0121689
5: -19.1026745, 21.5910721, -19.0902176, 21.5601349, -40.6628113, 40.6812897
6: -33.1927338, 15.8434811, -33.1957626, 15.8638191, -49.0565529, 49.0392456
7: -24.5484085, 19.5985146, -24.5547333, 19.5839977, -44.1324081, 44.1532478
8: -27.7964134, 26.8585262, -27.7980347, 26.8548088, -54.6512222, 54.6565628
9: -23.9737091, 23.0296402, -23.9431725, 22.9874840, -46.9611931, 46.9728127
10: -32.3587379, 24.9864922, -32.2941475, 24.9358425, -57.2945786, 57.2806396
11: -26.9507999, 16.2736111, -26.9473724, 16.2692566, -43.2200546, 43.2209854
12: -32.1617012, 22.3652916, -32.1655273, 22.4371109, -54.2529907, 54.1827927
13: -31.2052307, 30.7042122, -31.1945362, 30.7254486, -61.9306793, 61.8987503
14: -51.3624496, 16.0048103, -51.3622589, 16.0986481, -67.4610977, 67.3670654
15: -26.6501389, 17.8059845, -26.6200600, 17.7639046, -44.4140434, 44.4260445
16: -33.9416733, 18.1175308, -33.9124184, 18.0834255, -52.0251007, 52.0299492
17: -50.5943184, 17.6925926, -50.5806808, 17.7638092, -68.3581238, 68.2732697
18: -35.7819443, 17.9636364, -35.8027725, 17.9779510, -53.7598953, 53.7664108
19: -20.5713463, 14.1036949, -20.5406075, 14.0704069, -34.6417542, 34.6443024
20: -20.5519314, 17.7883472, -20.5420685, 17.7890015, -38.3409348, 38.3304138
21: -26.2444172, 15.4922848, -26.2292595, 15.4770012, -41.7214203, 41.7215424
22: -25.9367161, 15.6742659, -25.9224224, 15.6729946, -41.6097107, 41.5966873
23: -19.2918682, 19.2955704, -19.2280483, 19.2479630, -38.5398331, 38.5236206
24: -27.5779724, 18.0244408, -27.5320129, 17.9875774, -45.5655518, 45.5564537
25: -21.4424095, 21.3301926, -21.4080925, 21.3017387, -42.7441483, 42.7382851
26: -35.1297646, 25.3433952, -35.1088867, 25.3749371, -60.5046997, 60.4522820
27: -26.1921825, 16.9644623, -26.1876602, 16.9328957, -43.1250763, 43.1521225
28: -20.0931168, 20.3803215, -20.0623322, 20.3406544, -40.4337692, 40.4426537
29: -24.2874260, 15.2917194, -24.2748413, 15.2978821, -39.5853081, 39.5665588
30: -26.6870098, 19.3789139, -26.6508980, 19.3680782, -46.0550880, 46.0298119
31: -28.3953667, 19.8601112, -28.3837395, 19.8387985, -48.2341652, 48.2438507
32: -31.9529896, 14.9735842, -31.9508171, 14.9978476, -46.8215942, 46.7952461
33: -52.3623199, 17.1381264, -52.3761520, 17.1490688, -68.4127350, 68.4146957
34: -45.5063972, 5.7187815, -45.5175171, 5.7295427, -49.6838455, 49.6856728
35: -41.0575104, 14.0752401, -41.0495110, 14.0634851, -54.0571747, 54.0609589
36: -36.6770477, 17.8320007, -36.6636543, 17.8051567, -54.4729996, 54.4864349
37: -59.1968346, 8.2468405, -59.1491661, 8.2404013, -67.3608093, 67.3171539
38: -45.7757721, 17.5604630, -45.7716827, 17.5382843, -63.3140564, 63.3321457
39: -54.9897156, 18.0891514, -54.9967957, 18.1028709, -72.9648895, 72.9567413
40: -44.7299652, 5.9958048, -44.7251625, 6.0189371, -50.6667709, 50.6387787
41: -35.4107285, 16.8696404, -35.4192505, 16.8747482, -52.2325821, 52.2348175
42: -23.7315083, 14.9366913, -23.7243710, 14.9455786, -38.6770859, 38.6610641

Time for backsubstitution: 1.87 seconds

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

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2791302
time: 77.22 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2984972
time: 81.82 seconds

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

Time for backsubstitution: 1.76 seconds

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
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3002564
time: 138.20 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3196501
time: 70.39 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -42.9468460, 19.3421440, -42.9906616, 19.3847771, -62.3316231, 62.3328056
1: -21.2565422, 18.1758461, -21.2811775, 18.1999454, -39.4564896, 39.4570236
2: -14.9835491, 20.1716213, -15.0229816, 20.1999702, -35.1835175, 35.1946030
3: -20.3697910, 22.4140892, -20.4211750, 22.4502563, -42.8200455, 42.8352661
4: -24.0403881, 20.0108986, -24.0841026, 20.0382824, -44.0786705, 44.0950012
5: -19.1258717, 21.5946503, -19.1693096, 21.6262131, -40.7520828, 40.7639618
6: -33.1910744, 15.8503847, -33.2011452, 15.8695354, -49.0606079, 49.0515289
7: -24.5650749, 19.6014881, -24.6099262, 19.6274090, -44.1924820, 44.2114143
8: -27.8131237, 26.8629913, -27.8531914, 26.9009037, -54.7140274, 54.7161827
9: -24.0036449, 23.0338593, -24.0412807, 23.0678196, -47.0714645, 47.0751419
10: -32.3955612, 24.9916382, -32.4149742, 25.0283699, -57.4239311, 57.4066124
11: -26.9615784, 16.2943039, -27.0036545, 16.3372059, -43.2987823, 43.2979584
12: -32.1630402, 22.3749466, -32.1773033, 22.4758892, -54.2947235, 54.2101212
13: -31.2269917, 30.7128448, -31.2657986, 30.7836342, -62.0106277, 61.9786453
14: -51.3794289, 16.0099449, -51.4265366, 16.1261692, -67.5056000, 67.4364777
15: -26.6767235, 17.8106289, -26.7101097, 17.8275776, -44.5043030, 44.5207367
16: -33.9675446, 18.1226425, -34.0003700, 18.1453476, -52.1128922, 52.1230125
17: -50.6137733, 17.7058220, -50.6489296, 17.8171940, -68.4309692, 68.3547516
18: -35.7910500, 17.9778061, -35.8528290, 18.0257721, -53.8168221, 53.8306351
19: -20.5780907, 14.1272621, -20.6148720, 14.1405134, -34.7186050, 34.7421341
20: -20.5587006, 17.8103905, -20.6055946, 17.8580551, -38.4167557, 38.4159851
21: -26.2540245, 15.5216265, -26.2988701, 15.5655384, -41.8195648, 41.8204956
22: -25.9437294, 15.6903524, -25.9792690, 15.7234402, -41.6671677, 41.6696205
23: -19.2992439, 19.3306656, -19.3331680, 19.3583965, -38.6576385, 38.6638336
24: -27.5838928, 18.0561123, -27.6238041, 18.0865192, -45.6704102, 45.6799164
25: -21.4483852, 21.3623753, -21.4897594, 21.4035530, -42.8519363, 42.8521347
26: -35.1387138, 25.3734398, -35.1933823, 25.4679222, -60.6066360, 60.5668221
27: -26.1975002, 16.9868546, -26.2596455, 17.0041981, -43.2016983, 43.2464981
28: -20.0987225, 20.4104233, -20.1460667, 20.4364204, -40.5351410, 40.5564880
29: -24.2951851, 15.3061657, -24.3306580, 15.3449411, -39.6401253, 39.6368256
30: -26.6925716, 19.4044876, -26.7202320, 19.4569511, -46.1495209, 46.1247177
31: -28.4034767, 19.8830872, -28.4650574, 19.9104309, -48.3139076, 48.3481445
32: -31.9523907, 14.9792643, -31.9605141, 15.0022497, -46.8258438, 46.8132095
33: -52.3685379, 17.1560822, -52.4304543, 17.2080727, -68.4809113, 68.5022736
34: -45.5067520, 5.7256813, -45.5279770, 5.7532034, -49.7127914, 49.7251701
35: -41.0621758, 14.0915947, -41.0922928, 14.1132364, -54.1144791, 54.1363068
36: -36.6813507, 17.8485107, -36.7045288, 17.8578072, -54.5299911, 54.5440941
37: -59.2051544, 8.2725334, -59.2378654, 8.3223276, -67.4522095, 67.4389572
38: -45.7822113, 17.5790348, -45.8269920, 17.5999546, -63.3821640, 63.4060287
39: -54.9979134, 18.1039619, -55.0550690, 18.1504936, -73.0215149, 73.0369644
40: -44.7333374, 6.0031900, -44.7525444, 6.0272703, -50.6788864, 50.6755905
41: -35.4156952, 16.8778114, -35.4465485, 16.9051743, -52.2681503, 52.2726860
42: -23.7327919, 14.9426394, -23.7402000, 14.9672709, -38.7000618, 38.6828384

Time for backsubstitution: 1.86 seconds

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
type: B, layer: 1, pos: 701

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2844175, upper bound: 24.2799507
time: 104.72 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2844175, upper bound: 24.2993177
time: 90.92 seconds

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

Time for backsubstitution: 1.84 seconds

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
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2844175, upper bound: 24.3010221
time: 72.77 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2844175, upper bound: 24.3204129
time: 70.40 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -42.9843864, 19.3915710, -42.8955193, 19.3372116, -62.3215981, 62.2870903
1: -21.2652397, 18.2037697, -21.2261734, 18.1639900, -39.4292297, 39.4299431
2: -15.0013504, 20.2055435, -14.9862185, 20.1821804, -35.1835327, 35.1917610
3: -20.3806000, 22.4532814, -20.3465233, 22.3928394, -42.7734375, 42.7998047
4: -24.0647202, 20.0388947, -24.0084763, 19.9867687, -44.0514908, 44.0473709
5: -19.1290817, 21.6305962, -19.0916615, 21.5702133, -40.6992950, 40.7222595
6: -33.2208557, 15.8638496, -33.2003899, 15.8671846, -49.0880394, 49.0642395
7: -24.5758152, 19.6299419, -24.5564423, 19.5912704, -44.1670837, 44.1863861
8: -27.8550987, 26.9061565, -27.8004704, 26.8669720, -54.7220688, 54.7066269
9: -24.0193081, 23.0710163, -23.9468117, 22.9983616, -47.0176697, 47.0178299
10: -32.3906708, 25.0248775, -32.2974510, 24.9443474, -57.3350182, 57.3223267
11: -26.9950218, 16.3164349, -26.9595680, 16.2716274, -43.2666473, 43.2760010
12: -32.1807556, 22.3988838, -32.1680908, 22.4410553, -54.2760620, 54.2211838
13: -31.2850189, 30.7900810, -31.1963272, 30.7480354, -62.0330544, 61.9864082
14: -51.4099655, 16.0452690, -51.3680267, 16.1090164, -67.5189819, 67.4132996
15: -26.6805382, 17.8157883, -26.6235561, 17.7646484, -44.4451866, 44.4393463
16: -33.9660454, 18.1440029, -33.9161987, 18.0891914, -52.0552368, 52.0602036
17: -50.6556931, 17.7535095, -50.5843735, 17.7797413, -68.4354324, 68.3378830
18: -35.8602333, 18.0396500, -35.8225479, 17.9791603, -53.8393936, 53.8621979
19: -20.6118603, 14.1316910, -20.5504036, 14.0708580, -34.6827164, 34.6820946
20: -20.6032639, 17.8386803, -20.5554199, 17.7905731, -38.3938370, 38.3941002
21: -26.2922306, 15.5340099, -26.2412758, 15.4786005, -41.7708321, 41.7752838
22: -25.9757004, 15.6993475, -25.9308949, 15.6745682, -41.6502686, 41.6302414
23: -19.3297729, 19.3310356, -19.2378025, 19.2499542, -38.5797272, 38.5688400
24: -27.6249752, 18.0685482, -27.5440083, 17.9889297, -45.6139069, 45.6125565
25: -21.4895897, 21.3882065, -21.4205265, 21.3047256, -42.7943153, 42.8087311
26: -35.1967697, 25.3940887, -35.1248665, 25.3759537, -60.5727234, 60.5189552
27: -26.2686462, 17.0276814, -26.2071972, 16.9336853, -43.2023315, 43.2348785
28: -20.1458454, 20.4264660, -20.0758476, 20.3423080, -40.4881516, 40.5023117
29: -24.3245106, 15.3136902, -24.2830162, 15.3000183, -39.6245270, 39.5967064
30: -26.7120266, 19.4189129, -26.6573925, 19.3711491, -46.0831757, 46.0763054
31: -28.4666004, 19.9204807, -28.4019470, 19.8404388, -48.3070374, 48.3224258
32: -31.9816704, 14.9964981, -31.9541340, 15.0024290, -46.8562698, 46.8201523
33: -52.4288750, 17.1994743, -52.3809280, 17.1649055, -68.4955902, 68.4735184
34: -45.5402374, 5.7417030, -45.5213661, 5.7340622, -49.7246552, 49.7150993
35: -41.0918007, 14.0992756, -41.0526009, 14.0678558, -54.0945511, 54.0895462
36: -36.7072411, 17.8429432, -36.6675110, 17.8065891, -54.5046921, 54.5013733
37: -59.2667770, 8.3016243, -59.1559372, 8.2543297, -67.4445419, 67.3745804
38: -45.8345833, 17.5837822, -45.7821732, 17.5410309, -63.3756142, 63.3659554
39: -55.0692024, 18.1453362, -55.0023994, 18.1188374, -73.0608368, 73.0158691
40: -44.7873993, 6.0246325, -44.7317772, 6.0263348, -50.7329865, 50.6730042
41: -35.4452972, 16.9018288, -35.4237671, 16.8824768, -52.2749939, 52.2707214
42: -23.7456856, 14.9590340, -23.7269745, 14.9492455, -38.6949310, 38.6860085

Time for backsubstitution: 1.77 seconds

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

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2625571
time: 63.47 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2634713
time: 73.07 seconds

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

Time for backsubstitution: 1.84 seconds

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
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2836641
time: 78.88 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2845896
time: 61.95 seconds

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

Time for backsubstitution: 1.86 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 701

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2633831
time: 75.61 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2633831
time: 80.67 seconds

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

Time for backsubstitution: 1.77 seconds

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
time: 189.32 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2844175, upper bound: 24.2853355
time: 70.99 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 262.20 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2791302
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2984972
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3002564
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.2509442, upper bound: 24.3196501
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.2844175, upper bound: 24.2799507
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.2844175, upper bound: 24.2993177
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.2844175, upper bound: 24.3010221
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.2844175, upper bound: 24.3204129
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2625571
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2634713
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2836641
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2845896
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2633831
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2633831
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.3204124, upper bound: 24.2633831
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 262.20
Output dim: 20, lower bound: -24.2844175, upper bound: 24.2853355

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -42.9193649, 19.3375320, -42.8795547, 19.2877769, -62.2071419, 62.2170868
1: -21.2385082, 18.1719170, -21.2200928, 18.1371212, -39.3756294, 39.3920097
2: -14.9722204, 20.1693230, -14.9783707, 20.1500263, -35.1222458, 35.1476936
3: -20.3465691, 22.4104519, -20.3408127, 22.3552551, -42.7018242, 42.7512665
4: -24.0165329, 20.0068436, -23.9967690, 19.9563980, -43.9729309, 44.0036125
5: -19.1026745, 21.5910721, -19.0843182, 21.5363960, -40.6390686, 40.6753922
6: -33.1927338, 15.8434811, -33.1802330, 15.8551292, -49.0478630, 49.0237122
7: -24.5484085, 19.5985146, -24.5487576, 19.5657310, -44.1141396, 44.1472702
8: -27.7964134, 26.8585262, -27.7903900, 26.8255444, -54.6219559, 54.6489182
9: -23.9737091, 23.0296402, -23.9335575, 22.9610424, -46.9347534, 46.9631958
10: -32.3587379, 24.9864922, -32.2850075, 24.9149742, -57.2737122, 57.2714996
11: -26.9507999, 16.2736111, -26.9175682, 16.2626839, -43.2134857, 43.1911774
12: -32.1617012, 22.3652916, -32.1582794, 22.4207191, -54.2365570, 54.1753159
13: -31.2052307, 30.7042122, -31.1874599, 30.6712856, -61.8765182, 61.8916702
14: -51.3624496, 16.0048103, -51.3450317, 16.0742245, -67.4366760, 67.3498383
15: -26.6501389, 17.8059845, -26.6108685, 17.7560272, -44.4061661, 44.4168549
16: -33.9416733, 18.1175308, -33.9011002, 18.0687180, -52.0103912, 52.0186310
17: -50.5943184, 17.6925926, -50.5695190, 17.7252121, -68.3195343, 68.2621155
18: -35.7819443, 17.9636364, -35.7548981, 17.9726067, -53.7545509, 53.7185364
19: -20.5713463, 14.1036949, -20.5163269, 14.0680637, -34.6394119, 34.6200218
20: -20.5519314, 17.7883472, -20.5095215, 17.7836494, -38.3355789, 38.2978668
21: -26.2444172, 15.4922848, -26.1992893, 15.4718428, -41.7162590, 41.6915741
22: -25.9367161, 15.6742659, -25.9011345, 15.6674843, -41.6042023, 41.5754013
23: -19.2918682, 19.2955704, -19.2040310, 19.2419987, -38.5338669, 38.4996033
24: -27.5779724, 18.0244408, -27.5022888, 17.9819641, -45.5599365, 45.5267296
25: -21.4424095, 21.3301926, -21.3778000, 21.2929192, -42.7353287, 42.7079926
26: -35.1297646, 25.3433952, -35.0700073, 25.3698483, -60.4996109, 60.4134026
27: -26.1921825, 16.9644623, -26.1398735, 16.9293404, -43.1215210, 43.1043358
28: -20.0931168, 20.3803215, -20.0292625, 20.3350754, -40.4281921, 40.4095840
29: -24.2874260, 15.2917194, -24.2534218, 15.2916250, -39.5790520, 39.5451431
30: -26.6870098, 19.3789139, -26.6339970, 19.3589916, -46.0460014, 46.0129089
31: -28.3953667, 19.8601112, -28.3399029, 19.8331432, -48.2285080, 48.2000122
32: -31.9529896, 14.9735842, -31.9411068, 14.9860125, -46.8096313, 46.7853432
33: -52.3623199, 17.1381264, -52.3639145, 17.1110191, -68.3762436, 68.4023438
34: -45.5063972, 5.7187815, -45.5070953, 5.7130909, -49.6680908, 49.6729965
35: -41.0575104, 14.0752401, -41.0407333, 14.0472794, -54.0408783, 54.0513916
36: -36.6770477, 17.8320007, -36.6502838, 17.8008976, -54.4687195, 54.4729156
37: -59.1968346, 8.2468405, -59.1310501, 8.2054119, -67.3265457, 67.2987976
38: -45.7757721, 17.5604630, -45.7423325, 17.5308132, -63.3065872, 63.3027954
39: -54.9897156, 18.0891514, -54.9817162, 18.0647354, -72.9267731, 72.9418259
40: -44.7299652, 5.9958048, -44.7077789, 6.0005913, -50.6483765, 50.6212769
41: -35.4107285, 16.8696404, -35.4069977, 16.8562832, -52.2141418, 52.2224655
42: -23.7315083, 14.9366913, -23.7167683, 14.9348984, -38.6664047, 38.6534576

Time for backsubstitution: 1.85 seconds

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
type: A, layer: 1, pos: 596

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2501062
time: 66.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2787512
time: 77.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -42.9193649, 19.3375320, -42.9445724, 19.3418369, -62.2612000, 62.2821045
1: -21.2385082, 18.1719170, -21.2468147, 18.1689587, -39.4074669, 39.4187317
2: -14.9722204, 20.1693230, -15.0074940, 20.1862297, -35.1584511, 35.1768188
3: -20.3465691, 22.4104519, -20.3748398, 22.3980732, -42.7446442, 42.7852936
4: -24.0165329, 20.0068436, -24.0449142, 19.9884319, -44.0049667, 44.0517578
5: -19.1026745, 21.5910721, -19.1106892, 21.5758934, -40.6785660, 40.7017593
6: -33.1927338, 15.8434811, -33.2083588, 15.8754644, -49.0681992, 49.0518417
7: -24.5484085, 19.5985146, -24.5761757, 19.5971909, -44.1455994, 44.1746902
8: -27.7964134, 26.8585262, -27.8490295, 26.8731499, -54.6695633, 54.7075577
9: -23.9737091, 23.0296402, -23.9791908, 23.0024090, -46.9761200, 47.0088310
10: -32.3587379, 24.9864922, -32.3169403, 24.9533825, -57.3121185, 57.3034325
11: -26.9507999, 16.2736111, -26.9618149, 16.3055115, -43.2563095, 43.2354279
12: -32.1617012, 22.3652916, -32.1773453, 22.4542637, -54.2699432, 54.1943436
13: -31.2052307, 30.7042122, -31.2672291, 30.7571182, -61.9623489, 61.9714432
14: -51.3624496, 16.0048103, -51.3925934, 16.1147003, -67.4771500, 67.3973999
15: -26.6501389, 17.8059845, -26.6412773, 17.7658329, -44.4159698, 44.4472618
16: -33.9416733, 18.1175308, -33.9254990, 18.0951996, -52.0368729, 52.0430298
17: -50.5943184, 17.6925926, -50.6309052, 17.7861137, -68.3804321, 68.3235016
18: -35.7819443, 17.9636364, -35.8331528, 18.0486336, -53.8305779, 53.7967911
19: -20.5713463, 14.1036949, -20.5568390, 14.0960522, -34.6673965, 34.6605339
20: -20.5519314, 17.7883472, -20.5608864, 17.8339939, -38.3859253, 38.3492355
21: -26.2444172, 15.4922848, -26.2471161, 15.5135956, -41.7580109, 41.7394028
22: -25.9367161, 15.6742659, -25.9401417, 15.6925659, -41.6292801, 41.6144066
23: -19.2918682, 19.2955704, -19.2419243, 19.2774734, -38.5693436, 38.5374947
24: -27.5779724, 18.0244408, -27.5492897, 18.0260544, -45.6040268, 45.5737305
25: -21.4424095, 21.3301926, -21.4250011, 21.3509407, -42.7933502, 42.7551956
26: -35.1297646, 25.3433952, -35.1370010, 25.4205132, -60.5502777, 60.4803963
27: -26.1921825, 16.9644623, -26.2163258, 16.9925842, -43.1847687, 43.1807861
28: -20.0931168, 20.3803215, -20.0819912, 20.3812027, -40.4743195, 40.4623108
29: -24.2874260, 15.2917194, -24.2905045, 15.3135939, -39.6010208, 39.5822220
30: -26.6870098, 19.3789139, -26.6590290, 19.3990021, -46.0860138, 46.0379410
31: -28.3953667, 19.8601112, -28.4111290, 19.8935127, -48.2888794, 48.2712402
32: -31.9529896, 14.9735842, -31.9697952, 15.0089197, -46.8328171, 46.8154144
33: -52.3623199, 17.1381264, -52.4304810, 17.1723614, -68.4360733, 68.4694824
34: -45.5063972, 5.7187815, -45.5409393, 5.7359886, -49.6906204, 49.7092705
35: -41.0575104, 14.0752401, -41.0750389, 14.0712738, -54.0646057, 54.0842972
36: -36.6770477, 17.8320007, -36.6804733, 17.8118706, -54.4797592, 54.5032043
37: -59.1968346, 8.2468405, -59.2009964, 8.2601728, -67.3797073, 67.3687363
38: -45.7757721, 17.5604630, -45.8011093, 17.5541191, -63.3298912, 63.3615723
39: -54.9897156, 18.0891514, -55.0612106, 18.1209049, -72.9828949, 73.0218735
40: -44.7299652, 5.9958048, -44.7651711, 6.0293846, -50.6775513, 50.6801071
41: -35.4107285, 16.8696404, -35.4415817, 16.8884888, -52.2463531, 52.2571945
42: -23.7315083, 14.9366913, -23.7309494, 14.9572325, -38.6887398, 38.6676407

Time for backsubstitution: 1.85 seconds

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
type: A, layer: 1, pos: 596

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2694849
time: 123.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2981164
time: 165.37 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -42.9865990, 19.3739090, -42.8884506, 19.2907028, -62.2773018, 62.2623596
1: -21.2800941, 18.2209797, -21.2262402, 18.1385307, -39.4186249, 39.4472198
2: -15.0304623, 20.2472076, -14.9903259, 20.1515388, -35.1819992, 35.2375336
3: -20.4158669, 22.5249786, -20.3557892, 22.3582268, -42.7740936, 42.8807678
4: -24.0784492, 20.1029129, -24.0107689, 19.9585819, -44.0370331, 44.1136818
5: -19.1645393, 21.6698990, -19.0978451, 21.5392227, -40.7037621, 40.7677460
6: -33.2484589, 15.8695164, -33.1835022, 15.8565102, -49.1049690, 49.0530167
7: -24.6211319, 19.6494942, -24.5619125, 19.5671692, -44.1883011, 44.2114067
8: -27.8546906, 26.9377308, -27.8017921, 26.8277435, -54.6824341, 54.7395248
9: -24.0235710, 23.0993786, -23.9408836, 22.9644871, -46.9880600, 47.0402603
10: -32.4246445, 25.0440025, -32.2895279, 24.9193172, -57.3439636, 57.3335304
11: -27.1083183, 16.3248539, -26.9219151, 16.2724876, -43.3808060, 43.2467690
12: -32.2975807, 22.4911652, -32.1621170, 22.4488945, -54.3801346, 54.2790909
13: -31.2452984, 30.7859154, -31.1898956, 30.6768112, -61.9221115, 61.9758110
14: -51.5737915, 16.1362762, -51.3544540, 16.1055241, -67.6793137, 67.4907303
15: -26.6921768, 17.8677101, -26.6164036, 17.7594357, -44.4516144, 44.4841156
16: -34.0190926, 18.1732292, -33.9076309, 18.0706291, -52.0897217, 52.0808601
17: -50.8156662, 17.8056316, -50.5773697, 17.7498322, -68.5654984, 68.3830032
18: -35.9219017, 18.0095444, -35.7591553, 17.9765511, -53.8984528, 53.7686996
19: -20.6675167, 14.1208372, -20.5208912, 14.0706120, -34.7381287, 34.6417274
20: -20.6459503, 17.8517609, -20.5145760, 17.7981262, -38.4440765, 38.3663368
21: -26.3841190, 15.5505733, -26.2046242, 15.4850340, -41.8691521, 41.7551956
22: -26.0449677, 15.7252674, -25.9045792, 15.6776161, -41.7225838, 41.6298447
23: -19.3846779, 19.3329926, -19.2076092, 19.2489777, -38.6336555, 38.5406036
24: -27.6525097, 18.0556107, -27.5063992, 17.9864464, -45.6389542, 45.5620117
25: -21.5232334, 21.3844433, -21.3812714, 21.3043804, -42.8276138, 42.7657166
26: -35.2989693, 25.4753361, -35.0744019, 25.4005318, -60.6995010, 60.5497360
27: -26.2804375, 16.9729385, -26.1453190, 16.9276352, -43.2080727, 43.1182556
28: -20.1695213, 20.4149265, -20.0342121, 20.3422852, -40.5118065, 40.4491386
29: -24.4236870, 15.3441830, -24.2570992, 15.3028336, -39.7265205, 39.6012802
30: -26.7888088, 19.4441013, -26.6377983, 19.3732376, -46.1620483, 46.0819016
31: -28.4940987, 19.8831749, -28.3463497, 19.8355560, -48.3296547, 48.2295227
32: -32.0017471, 15.0059938, -31.9440956, 14.9899626, -46.8568344, 46.8145599
33: -52.4499893, 17.2544022, -52.3803558, 17.1161423, -68.5184097, 68.5865097
34: -45.5480156, 5.7977829, -45.5135269, 5.7168493, -49.7637329, 49.8088493
35: -41.0938225, 14.1540031, -41.0447693, 14.0498724, -54.1369553, 54.1875763
36: -36.7198029, 17.8637295, -36.6531906, 17.8031101, -54.5134277, 54.5073471
37: -59.2721062, 8.2812576, -59.1363564, 8.2080660, -67.3962708, 67.3345337
38: -45.8320694, 17.6039963, -45.7470245, 17.5322533, -63.3643227, 63.3510208
39: -55.0781326, 18.2129765, -54.9965553, 18.0681038, -73.0353851, 73.0975266
40: -44.7791443, 6.0497789, -44.7121849, 6.0015020, -50.7054749, 50.6871223
41: -35.4585457, 16.9166946, -35.4127960, 16.8589573, -52.2692642, 52.2807846
42: -23.7697430, 14.9718914, -23.7199402, 14.9394169, -38.7091599, 38.6918335

Time for backsubstitution: 1.87 seconds

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

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2713728
time: 93.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2998678
time: 74.53 seconds

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

Time for backsubstitution: 1.76 seconds

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
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2907484
time: 71.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2505523, upper bound: 24.3192570
time: 68.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -42.9468460, 19.3421440, -42.9785995, 19.3498154, -62.2966614, 62.3207436
1: -21.2565422, 18.1758461, -21.2765179, 18.1810341, -39.4375763, 39.4523621
2: -14.9835491, 20.1716213, -15.0170870, 20.1767902, -35.1603394, 35.1887093
3: -20.3697910, 22.4140892, -20.4164696, 22.4235916, -42.7933807, 42.8305588
4: -24.0403881, 20.0108986, -24.0755043, 20.0163212, -44.0567093, 44.0864029
5: -19.1258717, 21.5946503, -19.1633720, 21.6024513, -40.7283249, 40.7580223
6: -33.1910744, 15.8503847, -33.1855736, 15.8608513, -49.0519257, 49.0359573
7: -24.5650749, 19.6014881, -24.6039505, 19.6091499, -44.1742249, 44.2054367
8: -27.8131237, 26.8629913, -27.8455696, 26.8715897, -54.6847153, 54.7085609
9: -24.0036449, 23.0338593, -24.0316811, 23.0413647, -47.0450096, 47.0655403
10: -32.3955612, 24.9916382, -32.4058685, 25.0074615, -57.4030228, 57.3975067
11: -26.9615784, 16.2943039, -26.9738197, 16.3306351, -43.2922134, 43.2681236
12: -32.1630402, 22.3749466, -32.1699905, 22.4594612, -54.2783508, 54.2026367
13: -31.2269917, 30.7128448, -31.2587090, 30.7294731, -61.9564667, 61.9715538
14: -51.3794289, 16.0099449, -51.4092903, 16.1017532, -67.4811859, 67.4192352
15: -26.6767235, 17.8106289, -26.7009068, 17.8197098, -44.4964333, 44.5115356
16: -33.9675446, 18.1226425, -33.9890823, 18.1306477, -52.0981903, 52.1117249
17: -50.6137733, 17.7058220, -50.6377563, 17.7786331, -68.3924103, 68.3435822
18: -35.7910500, 17.9778061, -35.8049812, 18.0204697, -53.8115196, 53.7827873
19: -20.5780907, 14.1272621, -20.5905933, 14.1381798, -34.7162704, 34.7178574
20: -20.5587006, 17.8103905, -20.5730495, 17.8526783, -38.4113770, 38.3834381
21: -26.2540245, 15.5216265, -26.2688961, 15.5603838, -41.8144073, 41.7905235
22: -25.9437294, 15.6903524, -25.9579735, 15.7179031, -41.6616325, 41.6483269
23: -19.2992439, 19.3306656, -19.3091621, 19.3524323, -38.6516762, 38.6398277
24: -27.5838928, 18.0561123, -27.5940838, 18.0808868, -45.6647797, 45.6501961
25: -21.4483852, 21.3623753, -21.4594479, 21.3947220, -42.8431091, 42.8218231
26: -35.1387138, 25.3734398, -35.1544952, 25.4628277, -60.6015396, 60.5279350
27: -26.1975002, 16.9868546, -26.2118702, 17.0006638, -43.1981659, 43.1987228
28: -20.0987225, 20.4104233, -20.1129646, 20.4308395, -40.5295639, 40.5233879
29: -24.2951851, 15.3061657, -24.3092499, 15.3386946, -39.6338806, 39.6154175
30: -26.6925716, 19.4044876, -26.7033138, 19.4478626, -46.1404343, 46.1078033
31: -28.4034767, 19.8830872, -28.4212055, 19.9048119, -48.3082886, 48.3042908
32: -31.9523907, 14.9792643, -31.9508152, 14.9904308, -46.8139114, 46.8033142
33: -52.3685379, 17.1560822, -52.4182167, 17.1699753, -68.4444427, 68.4899063
34: -45.5067520, 5.7256813, -45.5175438, 5.7367353, -49.6970367, 49.7124939
35: -41.0621758, 14.0915947, -41.0835266, 14.0970116, -54.0982132, 54.1267014
36: -36.6813507, 17.8485107, -36.6911469, 17.8535480, -54.5257339, 54.5305710
37: -59.2051544, 8.2725334, -59.2197876, 8.2873230, -67.4178772, 67.4205933
38: -45.7822113, 17.5790348, -45.7976685, 17.5924454, -63.3746567, 63.3767014
39: -54.9979134, 18.1039619, -55.0399742, 18.1123257, -72.9834137, 73.0220337
40: -44.7333374, 6.0031900, -44.7351227, 6.0089054, -50.6604996, 50.6580734
41: -35.4156952, 16.8778114, -35.4342766, 16.8867073, -52.2496796, 52.2603149
42: -23.7327919, 14.9426394, -23.7325974, 14.9565954, -38.6893883, 38.6752357

Time for backsubstitution: 1.86 seconds

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
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 657
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
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
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
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 873

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 596

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2840222, upper bound: 24.2509132
time: 82.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2840222, upper bound: 24.2795630
time: 71.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -42.9468460, 19.3421440, -43.0435791, 19.4038601, -62.3507080, 62.3857231
1: -21.2565422, 18.1758461, -21.3032188, 18.2128658, -39.4694061, 39.4790649
2: -14.9835491, 20.1716213, -15.0461941, 20.2130222, -35.1965714, 35.2178154
3: -20.3697910, 22.4140892, -20.4505234, 22.4664211, -42.8362122, 42.8646126
4: -24.0403881, 20.0108986, -24.1236858, 20.0483627, -44.0887527, 44.1345825
5: -19.1258717, 21.5946503, -19.1897774, 21.6419659, -40.7678375, 40.7844276
6: -33.1910744, 15.8503847, -33.2137375, 15.8811989, -49.0722733, 49.0641212
7: -24.5650749, 19.6014881, -24.6313324, 19.6406193, -44.2056961, 44.2328186
8: -27.8131237, 26.8629913, -27.9042587, 26.9192219, -54.7323456, 54.7672501
9: -24.0036449, 23.0338593, -24.0772781, 23.0827522, -47.0863953, 47.1111374
10: -32.3955612, 24.9916382, -32.4378052, 25.0458908, -57.4414520, 57.4294434
11: -26.9615784, 16.2943039, -27.0180836, 16.3734608, -43.3350372, 43.3123856
12: -32.1630402, 22.3749466, -32.1890869, 22.4930687, -54.3116913, 54.2216492
13: -31.2269917, 30.7128448, -31.3384686, 30.8153172, -62.0423088, 62.0513153
14: -51.3794289, 16.0099449, -51.4568253, 16.1422138, -67.5216446, 67.4667664
15: -26.6767235, 17.8106289, -26.7313194, 17.8295193, -44.5062408, 44.5419464
16: -33.9675446, 18.1226425, -34.0134506, 18.1571350, -52.1246796, 52.1360931
17: -50.6137733, 17.7058220, -50.6991310, 17.8395691, -68.4533386, 68.4049530
18: -35.7910500, 17.9778061, -35.8832855, 18.0964737, -53.8875237, 53.8610916
19: -20.5780907, 14.1272621, -20.6311169, 14.1661577, -34.7442474, 34.7583771
20: -20.5587006, 17.8103905, -20.6244011, 17.9030190, -38.4617195, 38.4347916
21: -26.2540245, 15.5216265, -26.3167152, 15.6020918, -41.8561172, 41.8383408
22: -25.9437294, 15.6903524, -25.9969635, 15.7429676, -41.6866989, 41.6873169
23: -19.2992439, 19.3306656, -19.3470650, 19.3878841, -38.6871262, 38.6777306
24: -27.5838928, 18.0561123, -27.6410770, 18.1249962, -45.7088890, 45.6971893
25: -21.4483852, 21.3623753, -21.5066319, 21.4527588, -42.9011459, 42.8690071
26: -35.1387138, 25.3734398, -35.2214966, 25.5135078, -60.6522217, 60.5949364
27: -26.1975002, 16.9868546, -26.2883358, 17.0638866, -43.2613869, 43.2751923
28: -20.0987225, 20.4104233, -20.1656990, 20.4769821, -40.5757065, 40.5761223
29: -24.2951851, 15.3061657, -24.3463211, 15.3606501, -39.6558342, 39.6524887
30: -26.6925716, 19.4044876, -26.7283363, 19.4878693, -46.1804428, 46.1328239
31: -28.4034767, 19.8830872, -28.4924431, 19.9651470, -48.3686218, 48.3755302
32: -31.9523907, 14.9792643, -31.9794960, 15.0133228, -46.8371048, 46.8334045
33: -52.3685379, 17.1560822, -52.4848404, 17.2313786, -68.5042419, 68.5571213
34: -45.5067520, 5.7256813, -45.5513878, 5.7596159, -49.7195282, 49.7487640
35: -41.0621758, 14.0915947, -41.1178055, 14.1210604, -54.1219635, 54.1596413
36: -36.6813507, 17.8485107, -36.7213211, 17.8645287, -54.5367279, 54.5608597
37: -59.2051544, 8.2725334, -59.2897339, 8.3421335, -67.4710846, 67.4905624
38: -45.7822113, 17.5790348, -45.8564911, 17.6157665, -63.3979797, 63.4355240
39: -54.9979134, 18.1039619, -55.1194687, 18.1684914, -73.0395966, 73.1020966
40: -44.7333374, 6.0031900, -44.7925644, 6.0377073, -50.6896744, 50.7169037
41: -35.4156952, 16.8778114, -35.4688644, 16.9188843, -52.2819824, 52.2950363
42: -23.7327919, 14.9426394, -23.7467728, 14.9789200, -38.7117119, 38.6894112

Time for backsubstitution: 1.87 seconds

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
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 657
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
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
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
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 873

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 596

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2840222, upper bound: 24.2702915
time: 95.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2840222, upper bound: 24.2989283
time: 61.10 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 158.42 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 158.42
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2501062
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 158.42
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2787512
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 158.42
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2694849
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 158.42
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2981164
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 158.42
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2713728
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 158.42
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2998678
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 158.42
Output dim: 20, lower bound: -24.2505523, upper bound: 24.2907484
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 158.42
Output dim: 20, lower bound: -24.2505523, upper bound: 24.3192570
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 158.42
Output dim: 20, lower bound: -24.2840222, upper bound: 24.2509132
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 158.42
Output dim: 20, lower bound: -24.2840222, upper bound: 24.2795630
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 158.42
Output dim: 20, lower bound: -24.2840222, upper bound: 24.2702915
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 158.42
Output dim: 20, lower bound: -24.2840222, upper bound: 24.2989283
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 158.42
Output dim: 20, lower bound: -24.2844175, upper bound: 24.3010221
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 158.42
Output dim: 20, lower bound: -24.2844175, upper bound: 24.3204129
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 158.42
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2625571
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 158.42
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2634713
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 158.42
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2836641
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 158.42
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2845896
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 158.42
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2633831
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 158.42
Output dim: 20, lower bound: -24.2509442, upper bound: 24.2633831
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 158.42
Output dim: 20, lower bound: -24.3204124, upper bound: 24.2633831
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 158.42
Output dim: 20, lower bound: -24.2844175, upper bound: 24.2853355

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 83.08 + 3533.84 = 3616.92 seconds
