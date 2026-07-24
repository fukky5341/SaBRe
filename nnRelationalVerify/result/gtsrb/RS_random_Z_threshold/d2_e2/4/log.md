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
execution time: IAR + RelationalAnalysis = 2.82 + 79.49 = 82.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 20, lower bound: -24.3304293, upper bound: 24.3304293

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1728

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 638

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3301122, upper bound: 24.3047669
time: 69.53 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3047669, upper bound: 24.3301122
time: 93.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 163.07 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 163.07
Output dim: 20, lower bound: -24.3301122, upper bound: 24.3047669
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 163.07
Output dim: 20, lower bound: -24.3047669, upper bound: 24.3301122

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3472900, 54.3465729
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8725052, 46.8721542
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6508331, 68.6493988
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8284302, 49.8261757
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2253418, 54.2237968
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735626, 54.5735359
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5291290, 67.5284348
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1499786, 73.1493759
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7565460, 50.7561531
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3373260, 52.3371124
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 720

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 675

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3186881, upper bound: 24.2978118
time: 74.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3231114, upper bound: 24.2934045
time: 78.05 seconds

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3465729, 54.3472900
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8721542, 46.8725052
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6493988, 68.6508331
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8261719, 49.8284264
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2238007, 54.2253418
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735321, 54.5735626
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5284271, 67.5291214
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1493683, 73.1499786
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7561493, 50.7565422
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3371124, 52.3373260
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 537

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 734

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2838867, upper bound: 24.3296118
time: 71.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3042550, upper bound: 24.3093124
time: 63.90 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 137.70 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 137.70
Output dim: 20, lower bound: -24.3186881, upper bound: 24.2978118
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 137.70
Output dim: 20, lower bound: -24.3231114, upper bound: 24.2934045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 137.70
Output dim: 20, lower bound: -24.2838867, upper bound: 24.3296118
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 137.70
Output dim: 20, lower bound: -24.3042550, upper bound: 24.3093124

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3473206, 54.3465195
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8725052, 46.8721237
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6508865, 68.6492996
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8285065, 49.8260307
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2253571, 54.2236710
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735550, 54.5735359
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5291290, 67.5283737
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1499939, 73.1493378
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7565613, 50.7561302
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3373413, 52.3371010
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 559

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3184942, upper bound: 24.2962567
time: 67.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3170756, upper bound: 24.2976023
time: 62.20 seconds

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3472290, 54.3465729
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8724670, 46.8721542
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6507339, 68.6493988
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8282776, 49.8261757
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2252045, 54.2237968
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735550, 54.5735359
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5290680, 67.5284348
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1499329, 73.1493759
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7565155, 50.7561531
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3373108, 52.3371124
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1686

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3222261, upper bound: 24.2771028
time: 90.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3067581, upper bound: 24.2925198
time: 121.28 seconds

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3467865, 54.3476410
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8722382, 46.8726501
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6498260, 68.6515503
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8268280, 49.8295288
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2242432, 54.2260895
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735397, 54.5735779
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5286407, 67.5294571
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1495667, 73.1502914
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7562714, 50.7567368
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3371811, 52.3374329
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 543

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2791674, upper bound: 24.3248937
time: 87.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2791674, upper bound: 24.3248937
time: 73.51 seconds

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3469238, 54.3475037
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8723068, 46.8725815
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6501160, 68.6512680
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8272705, 49.8290863
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2245483, 54.2257843
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735397, 54.5735741
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5287628, 67.5293274
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1496887, 73.1501694
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7563477, 50.7566605
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3372116, 52.3373947
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 703

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1449

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.3036314, upper bound: 24.3020189
time: 69.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2969422, upper bound: 24.3086914
time: 71.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 143.89 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 143.89
Output dim: 20, lower bound: -24.3184942, upper bound: 24.2962567
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 143.89
Output dim: 20, lower bound: -24.3170756, upper bound: 24.2976023
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 143.89
Output dim: 20, lower bound: -24.3222261, upper bound: 24.2771028
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 143.89
Output dim: 20, lower bound: -24.3067581, upper bound: 24.2925198
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 143.89
Output dim: 20, lower bound: -24.2791674, upper bound: 24.3248937
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 143.89
Output dim: 20, lower bound: -24.2791674, upper bound: 24.3248937
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 143.89
Output dim: 20, lower bound: -24.3036314, upper bound: 24.3020189
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 143.89
Output dim: 20, lower bound: -24.2969422, upper bound: 24.3086914

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3460922, 54.3455887
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8718948, 46.8716545
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6483612, 68.6473618
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8245239, 49.8229675
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2226410, 54.2215767
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735092, 54.5735016
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5279083, 67.5274429
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1489105, 73.1484833
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7558517, 50.7555923
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3369446, 52.3367920
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 767

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 597

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3174156, upper bound: 24.2885931
time: 66.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3109209, upper bound: 24.2951428
time: 66.76 seconds

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3463669, 54.3452988
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8720398, 46.8715134
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6489410, 68.6467743
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8254395, 49.8220520
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2232666, 54.2209473
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735397, 54.5734940
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5281830, 67.5271606
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1491547, 73.1482391
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7560196, 50.7554321
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3370209, 52.3367043
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 528

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3077854, upper bound: 24.2727892
time: 81.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2923065, upper bound: 24.2884817
time: 111.34 seconds

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3474197, 54.3468094
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8724442, 46.8721542
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6499939, 68.6487732
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8271332, 49.8251915
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2244339, 54.2231140
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735626, 54.5735321
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5287018, 67.5281372
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1496124, 73.1490936
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7563171, 50.7559738
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3372192, 52.3370361
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 561

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 872

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3133829, upper bound: 24.2769423
time: 69.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2979541, upper bound: 24.2682479
time: 70.93 seconds

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3474808, 54.3467560
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8724670, 46.8721313
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6501007, 68.6486740
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8272858, 49.8250389
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2245255, 54.2230072
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735626, 54.5735321
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5287323, 67.5280838
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1496582, 73.1490555
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7563477, 50.7559433
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3372192, 52.3370209
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1584

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 607

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3064803, upper bound: 24.2875596
time: 64.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2765412, upper bound: 24.2922446
time: 72.94 seconds

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3466339, 54.3473740
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8721848, 46.8725395
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6493835, 68.6508713
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8261642, 49.8284760
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2238922, 54.2254753
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735550, 54.5735855
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5283966, 67.5291061
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1495361, 73.1501541
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7561035, 50.7565041
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3371506, 52.3373718
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1449

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2668790, upper bound: 24.3241826
time: 76.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2778744, upper bound: 24.3079291
time: 62.05 seconds

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3467865, 54.3474960
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8722382, 46.8725967
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6498260, 68.6511154
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8268280, 49.8288574
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2242432, 54.2257385
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735397, 54.5735855
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5286407, 67.5292206
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1495667, 73.1502609
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7562714, 50.7565727
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3371811, 52.3374100
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 561

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 938

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2658613, upper bound: 24.3246577
time: 88.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2789319, upper bound: 24.3115788
time: 82.45 seconds

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3472137, 54.3477020
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8724747, 46.8727150
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6484680, 68.6494751
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8257294, 49.8272781
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2229767, 54.2240372
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735474, 54.5735703
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5281525, 67.5286255
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1491852, 73.1496048
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7561035, 50.7563705
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3371277, 52.3372726
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 767

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 671

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2969267, upper bound: 24.3084819
time: 77.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2967326, upper bound: 24.3086773
time: 66.05 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 145.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 145.62
Output dim: 20, lower bound: -24.3174156, upper bound: 24.2885931
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 145.62
Output dim: 20, lower bound: -24.3109209, upper bound: 24.2951428
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 145.62
Output dim: 20, lower bound: -24.3077854, upper bound: 24.2727892
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 145.62
Output dim: 20, lower bound: -24.2923065, upper bound: 24.2884817
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 145.62
Output dim: 20, lower bound: -24.3133829, upper bound: 24.2769423
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 145.62
Output dim: 20, lower bound: -24.2979541, upper bound: 24.2682479
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 145.62
Output dim: 20, lower bound: -24.3064803, upper bound: 24.2875596
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 145.62
Output dim: 20, lower bound: -24.2765412, upper bound: 24.2922446
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 145.62
Output dim: 20, lower bound: -24.2668790, upper bound: 24.3241826
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 145.62
Output dim: 20, lower bound: -24.2778744, upper bound: 24.3079291
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 145.62
Output dim: 20, lower bound: -24.2658613, upper bound: 24.3246577
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 145.62
Output dim: 20, lower bound: -24.2789319, upper bound: 24.3115788
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 145.62
Output dim: 20, lower bound: -24.2969267, upper bound: 24.3084819
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 145.62
Output dim: 20, lower bound: -24.2967326, upper bound: 24.3086773

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3454895, 54.3448944
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8716049, 46.8713226
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6470947, 68.6458969
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8227005, 49.8208199
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2213135, 54.2200356
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5734940, 54.5734711
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5272980, 67.5267334
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1483765, 73.1478729
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7555389, 50.7552109
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3367767, 52.3366013
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 604

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2973664, upper bound: 24.2884630
time: 69.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3172843, upper bound: 24.2685091
time: 79.45 seconds

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3453979, 54.3449936
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8715591, 46.8713684
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6468811, 68.6460953
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8223801, 49.8211365
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2210999, 54.2202530
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5734940, 54.5734787
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5272064, 67.5268326
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1482849, 73.1479492
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7554779, 50.7552643
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3367462, 52.3366318
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1786

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 606

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3106995, upper bound: 24.2817566
time: 84.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2978350, upper bound: 24.2949212
time: 70.89 seconds

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3458328, 54.3443375
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8717957, 46.8710709
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6496124, 68.6466217
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8259583, 49.8212891
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2239838, 54.2207909
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735321, 54.5734711
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5279694, 67.5265503
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1493835, 73.1481171
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7560272, 50.7552109
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3370819, 52.3366241
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 553

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2902937, upper bound: 24.2707573
time: 138.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.3057822, upper bound: 24.2707519
time: 91.31 seconds

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3477783, 54.3469696
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8732529, 46.8728867
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6533890, 68.6518250
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8284149, 49.8259506
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2244720, 54.2227898
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5736008, 54.5735931
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5289307, 67.5281830
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1510620, 73.1504059
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7571106, 50.7566757
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3376999, 52.3374863
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 539

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 999

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3117822, upper bound: 24.2720098
time: 78.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3084482, upper bound: 24.2753468
time: 78.03 seconds

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3477020, 54.3469696
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8725815, 46.8722305
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6505585, 68.6491089
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8271866, 49.8248901
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2250443, 54.2234802
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735779, 54.5735474
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5289841, 67.5282898
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1498566, 73.1492386
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7564468, 50.7560425
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3373032, 52.3370781
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 856

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1696

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.3030020, upper bound: 24.2870057
time: 72.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.3059136, upper bound: 24.2841145
time: 66.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3467636, 54.3477325
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8722229, 46.8727036
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6498566, 68.6518250
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8266907, 49.8297539
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2243958, 54.2265015
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735550, 54.5736008
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5285416, 67.5294724
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1495514, 73.1503754
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7559357, 50.7564774
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3371582, 52.3374519
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 604

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1622

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2660891, upper bound: 24.3224377
time: 77.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2651373, upper bound: 24.3233959
time: 76.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3469925, 54.3474960
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8723450, 46.8725891
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6503448, 68.6513443
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8274384, 49.8290100
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2249146, 54.2259903
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5735703, 54.5735893
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5287857, 67.5292511
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1497498, 73.1501770
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7560730, 50.7563477
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3372345, 52.3373795
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 767

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2707223, upper bound: 24.3077143
time: 74.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2667178, upper bound: 24.3006987
time: 66.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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
12: -32.1978645, 22.5199432, -32.1978645, 22.5199432, -54.3452301, 54.3454666
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
32: -31.9933815, 15.0148306, -31.9933815, 15.0148306, -46.8711929, 46.8713226
33: -52.4594231, 17.2362213, -52.4594231, 17.2362213, -68.6447754, 68.6450958
34: -45.5578613, 5.7682419, -45.5578613, 5.7682419, -49.8157349, 49.8162689
35: -41.1074944, 14.1278067, -41.1074944, 14.1278067, -54.2184601, 54.2189407
36: -36.7172966, 17.8656311, -36.7172966, 17.8656311, -54.5734634, 54.5734787
37: -59.2570114, 8.3517532, -59.2570114, 8.3517532, -67.5252838, 67.5254211
38: -45.8504524, 17.6144238, -45.8504524, 17.6144238, -63.4648743, 63.4648743
39: -55.0834885, 18.1757660, -55.0834885, 18.1757660, -73.1474152, 73.1477051
40: -44.7912865, 6.0409641, -44.7912865, 6.0409641, -50.7542419, 50.7542953
41: -35.4632111, 16.9205074, -35.4632111, 16.9205074, -52.3361511, 52.3362350
42: -23.7635994, 14.9810286, -23.7635994, 14.9810286, -38.7446289, 38.7446289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 604

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1670

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2648850, upper bound: 24.3130938
time: 119.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2543888, upper bound: 24.3236926
time: 77.93 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 199.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.2973664, upper bound: 24.2884630
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.3172843, upper bound: 24.2685091
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.3106995, upper bound: 24.2817566
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.2978350, upper bound: 24.2949212
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.2902937, upper bound: 24.2707573
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.3057822, upper bound: 24.2707519
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.3117822, upper bound: 24.2720098
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.3084482, upper bound: 24.2753468
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.3030020, upper bound: 24.2870057
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.3059136, upper bound: 24.2841145
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.2660891, upper bound: 24.3224377
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.2651373, upper bound: 24.3233959
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.2707223, upper bound: 24.3077143
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.2667178, upper bound: 24.3006987
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.2648850, upper bound: 24.3130938
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 199.56
Output dim: 20, lower bound: -24.2543888, upper bound: 24.3236926
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 199.56
Output dim: 20, lower bound: -24.2789319, upper bound: 24.3115788
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 199.56
Output dim: 20, lower bound: -24.2969267, upper bound: 24.3084819
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 199.56
Output dim: 20, lower bound: -24.2967326, upper bound: 24.3086773

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 82.31 + 3520.82 = 3603.14 seconds
