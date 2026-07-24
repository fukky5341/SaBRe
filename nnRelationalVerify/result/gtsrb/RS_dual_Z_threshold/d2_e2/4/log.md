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
execution time: IAR + RelationalAnalysis = 2.80 + 77.54 = 80.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 20, lower bound: -24.3304293, upper bound: 24.3304293

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 627

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3298167, upper bound: 24.2982539
time: 81.45 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2982539, upper bound: 24.3298167
time: 82.13 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 163.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 163.70
Output dim: 20, lower bound: -24.3298167, upper bound: 24.2982539
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 163.70
Output dim: 20, lower bound: -24.2982539, upper bound: 24.3298167

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3527679, 54.3520050
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8747787, 46.8744202
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6616669, 68.6601639
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8469849, 49.8446350
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2379913, 54.2363739
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5737915, 54.5737610
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5347595, 67.5340347
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1541748, 73.1535416
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7577820, 50.7573700
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3378983, 52.3376694
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1557

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3074820, upper bound: 24.2982239
time: 86.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3297867, upper bound: 24.2759238
time: 73.13 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3520050, 54.3527603
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8744202, 46.8747826
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6601715, 68.6616745
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8446350, 49.8469887
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2363739, 54.2379875
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5737610, 54.5737915
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5340271, 67.5347519
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1535492, 73.1541748
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7573700, 50.7577820
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3376694, 52.3378944
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1557

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2759238, upper bound: 24.3297867
time: 60.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2982239, upper bound: 24.3074820
time: 67.25 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 130.39 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 130.39
Output dim: 20, lower bound: -24.3074820, upper bound: 24.2982239
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 130.39
Output dim: 20, lower bound: -24.3297867, upper bound: 24.2759238
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 130.39
Output dim: 20, lower bound: -24.2759238, upper bound: 24.3297867
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 130.39
Output dim: 20, lower bound: -24.2982239, upper bound: 24.3074820

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3571320, 54.3574219
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8768845, 46.8770332
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6704025, 68.6709900
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8606796, 49.8615913
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2473602, 54.2479973
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739441, 54.5739517
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5389328, 67.5392075
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1578674, 73.1581192
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7601776, 50.7603302
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3391876, 52.3392792
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 533

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2970994, upper bound: 24.2723604
time: 68.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2929334, upper bound: 24.2833226
time: 72.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3581848, 54.3563843
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8773880, 46.8765297
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6724930, 68.6688995
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8639450, 49.8583260
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2496185, 54.2457581
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739746, 54.5739136
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5399246, 67.5382080
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1587524, 73.1572418
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7607269, 50.7597580
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3394928, 52.3389702
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 533

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3148098, upper bound: 24.2614305
time: 71.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.3038747, upper bound: 24.2655957
time: 54.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3563843, 54.3581772
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8765259, 46.8773956
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6689072, 68.6724930
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8583298, 49.8639450
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2457581, 54.2496109
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739136, 54.5739822
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5382004, 67.5399246
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1572418, 73.1587524
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7597504, 50.7607346
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3389740, 52.3395042
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 533

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2655957, upper bound: 24.3038747
time: 87.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2614305, upper bound: 24.3148098
time: 73.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3574219, 54.3571320
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8770294, 46.8768921
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6709824, 68.6704102
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8615952, 49.8606796
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2480011, 54.2473717
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739594, 54.5739441
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5392075, 67.5389252
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1581116, 73.1578674
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7603302, 50.7601700
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3392792, 52.3391953
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 533

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2833226, upper bound: 24.2929334
time: 67.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2723604, upper bound: 24.2970994
time: 103.21 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 172.91 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 172.91
Output dim: 20, lower bound: -24.2970994, upper bound: 24.2723604
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 172.91
Output dim: 20, lower bound: -24.2929334, upper bound: 24.2833226
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 172.91
Output dim: 20, lower bound: -24.3148098, upper bound: 24.2614305
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 172.91
Output dim: 20, lower bound: -24.3038747, upper bound: 24.2655957
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 172.91
Output dim: 20, lower bound: -24.2655957, upper bound: 24.3038747
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 172.91
Output dim: 20, lower bound: -24.2614305, upper bound: 24.3148098
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 172.91
Output dim: 20, lower bound: -24.2833226, upper bound: 24.2929334
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 172.91
Output dim: 20, lower bound: -24.2723604, upper bound: 24.2970994

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3580170, 54.3560791
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8773193, 46.8763885
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6722031, 68.6683121
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8634720, 49.8573990
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2492981, 54.2451248
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739594, 54.5738907
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5398026, 67.5379333
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1586304, 73.1569901
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7606506, 50.7595901
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3394623, 52.3388824
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 603

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3143212, upper bound: 24.2402795
time: 68.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2936989, upper bound: 24.2609407
time: 64.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3563843, 54.3580246
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8765259, 46.8773193
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6689072, 68.6721878
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8583298, 49.8634720
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2457581, 54.2492943
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739136, 54.5739670
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5382004, 67.5397873
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1572418, 73.1586227
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7597504, 50.7606506
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3389740, 52.3394585
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 603

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2609407, upper bound: 24.2936989
time: 206.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2402795, upper bound: 24.3143212
time: 64.79 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 274.04 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 274.04
Output dim: 20, lower bound: -24.3143212, upper bound: 24.2402795
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 274.04
Output dim: 20, lower bound: -24.2936989, upper bound: 24.2609407
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 274.04
Output dim: 20, lower bound: -24.2609407, upper bound: 24.2936989
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 274.04
Output dim: 20, lower bound: -24.2402795, upper bound: 24.3143212

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3588257, 54.3566589
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8777237, 46.8766708
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6726685, 68.6683197
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8660202, 49.8591995
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2501373, 54.2454567
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5740051, 54.5739288
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5405731, 67.5384903
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1593018, 73.1574554
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7610550, 50.7598648
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3397064, 52.3390579
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3111869, upper bound: 24.2134309
time: 75.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2875001, upper bound: 24.2371559
time: 67.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3569489, 54.3588333
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8768082, 46.8777237
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6688995, 68.6726761
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8601303, 49.8660202
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2460785, 54.2501373
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739441, 54.5740089
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5387878, 67.5405731
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1577148, 73.1592865
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7600174, 50.7610474
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3391418, 52.3397064
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2371559, upper bound: 24.2875001
time: 88.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2134309, upper bound: 24.3111869
time: 76.76 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 167.43 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 167.43
Output dim: 20, lower bound: -24.3111869, upper bound: 24.2134309
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 167.43
Output dim: 20, lower bound: -24.2875001, upper bound: 24.2371559
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 167.43
Output dim: 20, lower bound: -24.2371559, upper bound: 24.2875001
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 167.43
Output dim: 20, lower bound: -24.2134309, upper bound: 24.3111869

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3578491, 54.3560486
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8772583, 46.8763962
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6715698, 68.6679916
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8646622, 49.8590508
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2489853, 54.2451439
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739746, 54.5739059
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5399323, 67.5382233
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1588745, 73.1573639
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7607574, 50.7597961
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3395538, 52.3390198
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 604

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3107622, upper bound: 24.1880829
time: 70.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2477223, upper bound: 24.2123214
time: 71.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3563538, 54.3588333
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8765335, 46.8777237
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6685791, 68.6726761
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8599701, 49.8660202
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2457809, 54.2501373
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739136, 54.5740089
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5385132, 67.5405731
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1576233, 73.1592865
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7599640, 50.7610474
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3391113, 52.3397064
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 604

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2123215, upper bound: 24.2792232
time: 74.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.1880829, upper bound: 24.3107622
time: 83.92 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 160.44 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 160.44
Output dim: 20, lower bound: -24.3107622, upper bound: 24.1880829
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 160.44
Output dim: 20, lower bound: -24.2477223, upper bound: 24.2123214
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 160.44
Output dim: 20, lower bound: -24.2123215, upper bound: 24.2792232
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 160.44
Output dim: 20, lower bound: -24.1880829, upper bound: 24.3107622

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3585663, 54.3566437
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8776093, 46.8766785
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6714478, 68.6675797
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8669662, 49.8609085
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2493896, 54.2452316
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739975, 54.5739288
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5405655, 67.5387115
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1591492, 73.1575241
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7610779, 50.7600212
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3397827, 52.3392029
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 571

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3103974, upper bound: 24.1601056
time: 64.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2846879, upper bound: 24.1876705
time: 62.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3569489, 54.3595657
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8768158, 46.8780670
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6681671, 68.6725845
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8618393, 49.8683243
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2458649, 54.2505226
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739212, 54.5740433
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5389938, 67.5411987
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1577759, 73.1596069
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7601776, 50.7613640
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3392792, 52.3399277
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 571

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.1876705, upper bound: 24.2846879
time: 67.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.1601056, upper bound: 24.3103974
time: 82.33 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 152.52 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 152.52
Output dim: 20, lower bound: -24.3103974, upper bound: 24.1601056
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 152.52
Output dim: 20, lower bound: -24.2846879, upper bound: 24.1876705
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 152.52
Output dim: 20, lower bound: -24.1876705, upper bound: 24.2846879
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 152.52
Output dim: 20, lower bound: -24.1601056, upper bound: 24.3103974

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3633499, 54.3623886
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8798828, 46.8794250
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6587448, 68.6568451
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8750916, 49.8721085
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2348328, 54.2327805
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5741348, 54.5741119
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5355377, 67.5346298
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1557007, 73.1549072
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7581482, 50.7576141
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3404541, 52.3401718
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 605

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3097817, upper bound: 24.1347649
time: 67.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2797150, upper bound: 24.1586808
time: 76.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3626938, 54.3643341
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8795700, 46.8803406
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6574173, 68.6598434
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8730240, 49.8764496
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2334137, 54.2359467
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5741043, 54.5741882
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5349121, 67.5361862
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1551666, 73.1561584
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7577820, 50.7584229
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3402557, 52.3406067
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 605

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.1586808, upper bound: 24.2797150
time: 77.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.1347649, upper bound: 24.3097817
time: 68.45 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 148.03 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 148.03
Output dim: 20, lower bound: -24.3097817, upper bound: 24.1347649
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 148.03
Output dim: 20, lower bound: -24.2797150, upper bound: 24.1586808
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 148.03
Output dim: 20, lower bound: -24.1586808, upper bound: 24.2797150
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 148.03
Output dim: 20, lower bound: -24.1347649, upper bound: 24.3097817

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3637238, 54.3627319
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8797989, 46.8793182
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6566544, 68.6546631
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8773041, 49.8741837
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2327042, 54.2305679
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5741119, 54.5740738
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5352325, 67.5342865
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1548767, 73.1540527
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7579193, 50.7573776
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3403625, 52.3400574
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 595

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3093877, upper bound: 24.1048267
time: 63.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2804351, upper bound: 24.1343627
time: 89.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3630371, 54.3647156
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8794556, 46.8802567
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6552658, 68.6577606
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8750992, 49.8786583
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2311935, 54.2338409
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5740814, 54.5741577
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5345612, 67.5358734
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1543121, 73.1553040
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7575378, 50.7582016
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3401489, 52.3405075
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 595

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.1343627, upper bound: 24.2804351
time: 81.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.1048267, upper bound: 24.3093877
time: 89.92 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 173.29 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 173.29
Output dim: 20, lower bound: -24.3093877, upper bound: 24.1048267
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 173.29
Output dim: 20, lower bound: -24.2804351, upper bound: 24.1343627
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 173.29
Output dim: 20, lower bound: -24.1343627, upper bound: 24.2804351
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 173.29
Output dim: 20, lower bound: -24.1048267, upper bound: 24.3093877

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3564758, 54.3562012
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8769226, 46.8768005
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6667023, 68.6661682
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8891830, 49.8883438
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2319946, 54.2314224
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5738831, 54.5738831
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5402756, 67.5400162
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1489563, 73.1487350
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7599945, 50.7598381
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3413773, 52.3412933
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 628

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3089368, upper bound: 24.0854468
time: 69.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2842305, upper bound: 24.1036931
time: 80.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3565063, 54.3574600
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8769379, 46.8773804
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6667480, 68.6678162
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8892670, 49.8905373
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2320557, 54.2331467
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5738831, 54.5739212
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5403061, 67.5409164
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1489868, 73.1494064
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7600098, 50.7602577
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3413925, 52.3415260
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 628

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.1036931, upper bound: 24.2842305
time: 72.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.0854468, upper bound: 24.3089368
time: 77.51 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 151.95 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 151.95
Output dim: 20, lower bound: -24.3089368, upper bound: 24.0854468
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 151.95
Output dim: 20, lower bound: -24.2842305, upper bound: 24.1036931
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 151.95
Output dim: 20, lower bound: -24.1036931, upper bound: 24.2842305
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 151.95
Output dim: 20, lower bound: -24.0854468, upper bound: 24.3089368

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.0157013, 19.4076633, -43.0157013, 19.4076633, -62.4233627, 62.4233627
1: -21.2954788, 18.2136002, -21.2954788, 18.2136002, -39.5090790, 39.5090790
2: -15.0419035, 20.2144260, -15.0419035, 20.2144260, -35.2563286, 35.2563286
3: -20.4475021, 22.4683876, -20.4475021, 22.4683876, -42.9158897, 42.9158897
4: -24.1081982, 20.0543518, -24.1081982, 20.0543518, -44.1625519, 44.1625519
5: -19.1927910, 21.6424351, -19.1927910, 21.6424351, -40.8352280, 40.8352280
6: -33.2374649, 15.8808441, -33.2374649, 15.8808441, -49.1183090, 49.1183090
7: -24.6325951, 19.6405907, -24.6325951, 19.6405907, -44.2731857, 44.2731857
8: -27.8742237, 26.9200745, -27.8742237, 26.9200745, -54.7942963, 54.7942963
9: -24.0641670, 23.0866547, -24.0641670, 23.0866547, -47.1508217, 47.1508217
10: -32.4324951, 25.0483704, -32.4324951, 25.0483704, -57.4808655, 57.4808655
11: -27.0268803, 16.3602276, -27.0268803, 16.3602276, -43.3871078, 43.3871078
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3560486, 54.3553162
13: -31.2875786, 30.8193016, -31.2875786, 30.8193016, -62.1068802, 62.1068802
14: -51.4522629, 16.1790085, -51.4522629, 16.1790085, -67.6312714, 67.6312714
15: -26.7306042, 17.8387070, -26.7306042, 17.8387070, -44.5693130, 44.5693130
16: -34.0222969, 18.1589680, -34.0222969, 18.1589680, -52.1812668, 52.1812668
17: -50.6689758, 17.8677597, -50.6689758, 17.8677597, -68.5367355, 68.5367355
18: -35.8837280, 18.0544815, -35.8837280, 18.0544815, -53.9382095, 53.9382095
19: -20.6342335, 14.1514587, -20.6342335, 14.1514587, -34.7856903, 34.7856903
20: -20.6297398, 17.8814735, -20.6297398, 17.8814735, -38.5112152, 38.5112152
21: -26.3227425, 15.5873795, -26.3227425, 15.5873795, -41.9101219, 41.9101219
22: -25.9980488, 15.7434025, -25.9980488, 15.7434025, -41.7414513, 41.7414513
23: -19.3518600, 19.3774853, -19.3518600, 19.3774853, -38.7293472, 38.7293472
24: -27.6459732, 18.1058502, -27.6459732, 18.1058502, -45.7518234, 45.7518234
25: -21.5115223, 21.4272537, -21.5115223, 21.4272537, -42.9387741, 42.9387741
26: -35.2216568, 25.5124054, -35.2216568, 25.5124054, -60.7340622, 60.7340622
27: -26.2918262, 17.0165272, -26.2918262, 17.0165272, -43.3083534, 43.3083534
28: -20.1694546, 20.4531860, -20.1694546, 20.4531860, -40.6226425, 40.6226425
29: -24.3495750, 15.3643093, -24.3495750, 15.3643093, -39.7138824, 39.7138824
30: -26.7367630, 19.4845238, -26.7367630, 19.4845238, -46.2212868, 46.2212868
31: -28.4963379, 19.9270668, -28.4963379, 19.9270668, -48.4234047, 48.4234047
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8768845, 46.8765411
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6665344, 68.6650772
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8889313, 49.8866386
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2318268, 54.2302628
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5738831, 54.5738525
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5401840, 67.5394821
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1488037, 73.1481934
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7599335, 50.7595444
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3413467, 52.3411331
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 872

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 572

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3086115, upper bound: 24.0593098
time: 68.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2744182, upper bound: 24.0843645
time: 215.49 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 286.53 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 286.53
Output dim: 20, lower bound: -24.3086115, upper bound: 24.0593098
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 286.53
Output dim: 20, lower bound: -24.2744182, upper bound: 24.0843645
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 286.53
Output dim: 20, lower bound: -24.0854468, upper bound: 24.3089368

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 80.34 + 3586.91 = 3667.25 seconds
