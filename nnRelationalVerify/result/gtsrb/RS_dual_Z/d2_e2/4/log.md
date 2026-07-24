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
execution time: IAR + RelationalAnalysis = 2.24 + 81.55 = 83.79 seconds
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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 627

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3298167, upper bound: 24.2982539
time: 84.90 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2982539, upper bound: 24.3298167
time: 85.90 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 170.93 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 170.93
Output dim: 20, lower bound: -24.3298167, upper bound: 24.2982539
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 170.93
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

Time for backsubstitution: 1.87 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1557

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3074820, upper bound: 24.2982239
time: 90.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3297867, upper bound: 24.2759238
time: 76.31 seconds

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

Time for backsubstitution: 1.76 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1557

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2759238, upper bound: 24.3297867
time: 63.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2982239, upper bound: 24.3074820
time: 70.22 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 135.62 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 135.62
Output dim: 20, lower bound: -24.3074820, upper bound: 24.2982239
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 135.62
Output dim: 20, lower bound: -24.3297867, upper bound: 24.2759238
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 135.62
Output dim: 20, lower bound: -24.2759238, upper bound: 24.3297867
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 135.62
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

Time for backsubstitution: 1.74 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 533

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2970994, upper bound: 24.2723604
time: 71.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2929334, upper bound: 24.2833226
time: 75.71 seconds

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

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 533

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3148098, upper bound: 24.2614305
time: 74.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3038747, upper bound: 24.2655957
time: 57.38 seconds

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

Time for backsubstitution: 1.82 seconds

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
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2655957, upper bound: 24.3038747
time: 92.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2614305, upper bound: 24.3148098
time: 77.43 seconds

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

Time for backsubstitution: 1.83 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 533

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2833226, upper bound: 24.2929334
time: 71.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2723604, upper bound: 24.2970994
time: 107.41 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 180.34 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 180.34
Output dim: 20, lower bound: -24.2970994, upper bound: 24.2723604
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 180.34
Output dim: 20, lower bound: -24.2929334, upper bound: 24.2833226
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 180.34
Output dim: 20, lower bound: -24.3148098, upper bound: 24.2614305
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 180.34
Output dim: 20, lower bound: -24.3038747, upper bound: 24.2655957
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 180.34
Output dim: 20, lower bound: -24.2655957, upper bound: 24.3038747
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 180.34
Output dim: 20, lower bound: -24.2614305, upper bound: 24.3148098
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 180.34
Output dim: 20, lower bound: -24.2833226, upper bound: 24.2929334
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 180.34
Output dim: 20, lower bound: -24.2723604, upper bound: 24.2970994

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3569794, 54.3570633
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8768158, 46.8768616
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6700974, 68.6702805
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8602066, 49.8604851
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2470551, 54.2472458
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739288, 54.5739288
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5387955, 67.5388794
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1577454, 73.1578217
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7600861, 50.7601318
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3391571, 52.3391724
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

Time for backsubstitution: 1.82 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 603

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2966091, upper bound: 24.2511888
time: 74.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2759813, upper bound: 24.2718727
time: 79.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3571320, 54.3572693
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8768845, 46.8769608
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6704025, 68.6706924
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8606796, 49.8611221
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2473602, 54.2476807
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739441, 54.5739365
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5389328, 67.5390701
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1578674, 73.1579971
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7601776, 50.7602386
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3391876, 52.3392334
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

Time for backsubstitution: 1.74 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2924432, upper bound: 24.2621645
time: 81.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2718171, upper bound: 24.2828319
time: 71.04 seconds

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

Time for backsubstitution: 1.74 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 603

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3143212, upper bound: 24.2402795
time: 71.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2936989, upper bound: 24.2609407
time: 67.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3581848, 54.3562241
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8773880, 46.8764572
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6724930, 68.6686020
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8639450, 49.8578568
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2496185, 54.2454376
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739746, 54.5738983
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5399246, 67.5380783
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1587524, 73.1571121
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7607269, 50.7596741
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3394928, 52.3389244
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

Time for backsubstitution: 1.74 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 603

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3033846, upper bound: 24.2444410
time: 80.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2827437, upper bound: 24.2651061
time: 74.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3562317, 54.3578186
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8764572, 46.8772240
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6686020, 68.6717834
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8578568, 49.8628387
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2454376, 54.2488594
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5738983, 54.5739594
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5380630, 67.5395966
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1571045, 73.1584549
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7596741, 50.7605362
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3389130, 52.3394012
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

Time for backsubstitution: 1.78 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2651061, upper bound: 24.2827437
time: 71.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2444410, upper bound: 24.3033846
time: 70.08 seconds

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

Time for backsubstitution: 1.80 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 603

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2609407, upper bound: 24.2936989
time: 217.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2402795, upper bound: 24.3143212
time: 67.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3572693, 54.3568344
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8769608, 46.8767471
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6706772, 68.6698151
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8611221, 49.8597488
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2476807, 54.2467422
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739288, 54.5739212
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5390701, 67.5386505
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1579895, 73.1576233
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7602386, 50.7600021
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3392334, 52.3391037
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

Time for backsubstitution: 1.85 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2828319, upper bound: 24.2718171
time: 73.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2621645, upper bound: 24.2924432
time: 78.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3574219, 54.3569794
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8770294, 46.8768196
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6709824, 68.6701050
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8615952, 49.8602066
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2480011, 54.2470551
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739594, 54.5739288
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5392075, 67.5387955
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1581116, 73.1577454
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7603302, 50.7600784
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3392792, 52.3391495
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

Time for backsubstitution: 1.75 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 603

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2511888, upper bound: 24.2759812
time: 77.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2511888, upper bound: 24.2966091
time: 61.95 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 140.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.2966091, upper bound: 24.2511888
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.2759813, upper bound: 24.2718727
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.2924432, upper bound: 24.2621645
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.2718171, upper bound: 24.2828319
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.3143212, upper bound: 24.2402795
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.2936989, upper bound: 24.2609407
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.3033846, upper bound: 24.2444410
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.2827437, upper bound: 24.2651061
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.2651061, upper bound: 24.2827437
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.2444410, upper bound: 24.3033846
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.2609407, upper bound: 24.2936989
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.2402795, upper bound: 24.3143212
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.2828319, upper bound: 24.2718171
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.2621645, upper bound: 24.2924432
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.2511888, upper bound: 24.2759812
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 140.84
Output dim: 20, lower bound: -24.2511888, upper bound: 24.2966091

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3577881, 54.3576431
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8772202, 46.8771477
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6705933, 68.6702957
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8627548, 49.8622894
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2478943, 54.2475739
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739746, 54.5739670
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5395813, 67.5394287
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1584167, 73.1582794
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7604752, 50.7603989
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3394012, 52.3393555
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

Time for backsubstitution: 1.76 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2934236, upper bound: 24.2243403
time: 67.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2697748, upper bound: 24.2480897
time: 100.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3575439, 54.3578796
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8771057, 46.8772659
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6701050, 68.6707687
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8620071, 49.8630371
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2473907, 54.2480888
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739594, 54.5739746
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5393524, 67.5396576
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1582031, 73.1584854
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7603531, 50.7605286
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3393250, 52.3394241
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

Time for backsubstitution: 1.83 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2728701, upper bound: 24.2450453
time: 74.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2491029, upper bound: 24.2686826
time: 77.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3579407, 54.3578415
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8772888, 46.8772430
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6708679, 68.6706924
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8632278, 49.8629227
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2482147, 54.2480087
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739746, 54.5739746
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5397339, 67.5396271
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1585388, 73.1584549
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7605667, 50.7605133
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3394318, 52.3394165
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

Time for backsubstitution: 1.83 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2892559, upper bound: 24.2353031
time: 81.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2656132, upper bound: 24.2590891
time: 73.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3576965, 54.3580780
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8771744, 46.8773613
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6703949, 68.6711731
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8624802, 49.8636703
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2476959, 54.2485237
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5739746, 54.5739822
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5395050, 67.5398560
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1583252, 73.1586609
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7604446, 50.7606430
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3393707, 52.3394852
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

Time for backsubstitution: 1.83 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2686981, upper bound: 24.2560119
time: 62.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2449376, upper bound: 24.2796859
time: 66.97 seconds

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

Time for backsubstitution: 1.72 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3111869, upper bound: 24.2134309
time: 78.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2875001, upper bound: 24.2371559
time: 71.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3585968, 54.3568954
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8776093, 46.8767891
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6721954, 68.6688004
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8652725, 49.8599472
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2496185, 54.2459717
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5740051, 54.5739365
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5403442, 67.5387192
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1590881, 73.1576538
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7609177, 50.7599945
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3396454, 52.3391304
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

Time for backsubstitution: 1.75 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2906117, upper bound: 24.2341248
time: 82.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2491029, upper bound: 24.2577371
time: 74.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3589783, 54.3567963
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8777924, 46.8767433
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6729584, 68.6686096
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8664932, 49.8596573
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2504578, 54.2457695
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5740051, 54.5739365
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5407257, 67.5386276
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1594238, 73.1575775
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7611313, 50.7599411
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3397522, 52.3391037
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

Time for backsubstitution: 1.86 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3002036, upper bound: 24.2175929
time: 71.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2765454, upper bound: 24.2413283
time: 64.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3587341, 54.3570404
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8776779, 46.8768578
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6724854, 68.6690903
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8657455, 49.8604050
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2499542, 54.2462845
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5740051, 54.5739441
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5404968, 67.5388565
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1592102, 73.1577759
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7609940, 50.7600708
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3396759, 52.3391724
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2796378, upper bound: 24.2382881
time: 61.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2558606, upper bound: 24.2619111
time: 65.83 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 129.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.2934236, upper bound: 24.2243403
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.2697748, upper bound: 24.2480897
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.2728701, upper bound: 24.2450453
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.2491029, upper bound: 24.2686826
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.2892559, upper bound: 24.2353031
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.2656132, upper bound: 24.2590891
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.2686981, upper bound: 24.2560119
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.2449376, upper bound: 24.2796859
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.3111869, upper bound: 24.2134309
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.2875001, upper bound: 24.2371559
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.2906117, upper bound: 24.2341248
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.2491029, upper bound: 24.2577371
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.3002036, upper bound: 24.2175929
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.2765454, upper bound: 24.2413283
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.2796378, upper bound: 24.2382881
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 129.40
Output dim: 20, lower bound: -24.2558606, upper bound: 24.2619111
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 129.40
Output dim: 20, lower bound: -24.2651061, upper bound: 24.2827437
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 129.40
Output dim: 20, lower bound: -24.2444410, upper bound: 24.3033846
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 129.40
Output dim: 20, lower bound: -24.2609407, upper bound: 24.2936989
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 129.40
Output dim: 20, lower bound: -24.2402795, upper bound: 24.3143212
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 129.40
Output dim: 20, lower bound: -24.2828319, upper bound: 24.2718171
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 129.40
Output dim: 20, lower bound: -24.2621645, upper bound: 24.2924432
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 129.40
Output dim: 20, lower bound: -24.2511888, upper bound: 24.2759812
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 129.40
Output dim: 20, lower bound: -24.2511888, upper bound: 24.2966091

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 83.79 + 3634.34 = 3718.14 seconds
