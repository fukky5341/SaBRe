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
execution time: IAR + RelationalAnalysis = 2.80 + 77.95 = 80.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 20, lower bound: -24.3304293, upper bound: 24.3304293

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 999
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 1016

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 701

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3293512, upper bound: 24.2934227
time: 78.94 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3293512, upper bound: 24.2934227
time: 69.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 148.34 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 148.34
Output dim: 20, lower bound: -24.3293512, upper bound: 24.2934227
IS_B2, status: Status.UNKNOWN, split count: 1, time: 148.34
Output dim: 20, lower bound: -24.3293512, upper bound: 24.2934227

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -43.0101089, 19.3916149, -42.9980545, 19.3566475, -62.3667564, 62.3896713
1: -21.2933483, 18.2048092, -21.2886810, 18.1858768, -39.4792252, 39.4934921
2: -15.0391407, 20.2038364, -15.0332346, 20.1806679, -35.2198105, 35.2370720
3: -20.4452705, 22.4561901, -20.4405804, 22.4295311, -42.8748016, 42.8967705
4: -24.1042747, 20.0442600, -24.0957203, 20.0222893, -44.1265640, 44.1399803
5: -19.1899796, 21.6313553, -19.1840897, 21.6075783, -40.7975578, 40.8154449
6: -33.2301941, 15.8768749, -33.2146492, 15.8682251, -49.0984192, 49.0915222
7: -24.6297913, 19.6322060, -24.6238289, 19.6139297, -44.2437210, 44.2560349
8: -27.8707047, 26.9066505, -27.8630905, 26.8773613, -54.7480659, 54.7697411
9: -24.0597572, 23.0745430, -24.0501404, 23.0480461, -47.1078033, 47.1246834
10: -32.4282990, 25.0380859, -32.4192085, 25.0171795, -57.4454803, 57.4572945
11: -27.0131931, 16.3571815, -26.9833603, 16.3506088, -43.3638000, 43.3405418
12: -32.1944962, 22.5122185, -32.1872330, 22.4957886, -54.3225021, 54.3314209
13: -31.2842903, 30.7945290, -31.2771702, 30.7403297, -62.0246201, 62.0717010
14: -51.4442368, 16.1676521, -51.4270172, 16.1432285, -67.5874634, 67.5946655
15: -26.7263508, 17.8349266, -26.7171726, 17.8270683, -44.5534210, 44.5521011
16: -34.0170593, 18.1521931, -34.0057716, 18.1374760, -52.1545334, 52.1579666
17: -50.6638336, 17.8500824, -50.6526375, 17.8114700, -68.4753036, 68.5027161
18: -35.8617668, 18.0520477, -35.8139191, 18.0467224, -53.9084892, 53.8659668
19: -20.6230965, 14.1503820, -20.5988197, 14.1480522, -34.7711487, 34.7492027
20: -20.6147938, 17.8790016, -20.5822525, 17.8736401, -38.4884338, 38.4612541
21: -26.3089733, 15.5850210, -26.2789974, 15.5798597, -41.8888321, 41.8640175
22: -25.9882565, 15.7408447, -25.9669437, 15.7352953, -41.7235527, 41.7077866
23: -19.3408356, 19.3747444, -19.3168240, 19.3687897, -38.7096252, 38.6915665
24: -27.6323433, 18.1032200, -27.6026077, 18.0975971, -45.7299423, 45.7058258
25: -21.4976349, 21.4232178, -21.4673271, 21.4144058, -42.9120407, 42.8905449
26: -35.2037964, 25.5099831, -35.1649246, 25.5048714, -60.7086678, 60.6749077
27: -26.2699203, 17.0148983, -26.2221298, 17.0113602, -43.2812805, 43.2370300
28: -20.1542683, 20.4506187, -20.1211700, 20.4450417, -40.5993118, 40.5717888
29: -24.3397388, 15.3614359, -24.3183022, 15.3551989, -39.6949387, 39.6797371
30: -26.7290020, 19.4803524, -26.7121010, 19.4712696, -46.2002716, 46.1924515
31: -28.4762287, 19.9244614, -28.4323883, 19.9188385, -48.3950653, 48.3568497
32: -31.9888992, 15.0093899, -31.9791927, 14.9975691, -46.8516769, 46.8537064
33: -52.4538345, 17.2187176, -52.4415588, 17.1806717, -68.5960312, 68.6200714
34: -45.5530777, 5.7606459, -45.5426445, 5.7441483, -49.8079300, 49.8110161
35: -41.1034889, 14.1202526, -41.0947037, 14.1040659, -54.2029953, 54.2096786
36: -36.7111435, 17.8636436, -36.6977501, 17.8594055, -54.5612335, 54.5519981
37: -59.2487183, 8.3345594, -59.2306404, 8.2995472, -67.4719543, 67.4879227
38: -45.8368378, 17.6109543, -45.8074799, 17.6034508, -63.4402885, 63.4184341
39: -55.0765305, 18.1582184, -55.0614586, 18.1200790, -73.0890503, 73.1122665
40: -44.7833214, 6.0325623, -44.7659149, 6.0142040, -50.7223053, 50.7231293
41: -35.4575882, 16.9117889, -35.4453201, 16.8933392, -52.3041229, 52.3101730
42: -23.7600937, 14.9759102, -23.7524757, 14.9652224, -38.7253151, 38.7283859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1722

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3139601, upper bound: 24.2920722
time: 62.99 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3139601, upper bound: 24.2921034
time: 66.84 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -43.0140152, 19.4061089, -43.0630379, 19.4106846, -62.4246979, 62.4691467
1: -21.2947617, 18.2127457, -21.3153839, 18.2177410, -39.5125046, 39.5281296
2: -15.0410824, 20.2128677, -15.0623693, 20.2168884, -35.2579727, 35.2752380
3: -20.4462624, 22.4670982, -20.4746132, 22.4723587, -42.9186211, 42.9417114
4: -24.1074314, 20.0526562, -24.1438599, 20.0543308, -44.1617622, 44.1965179
5: -19.1914234, 21.6414337, -19.2104492, 21.6470985, -40.8385239, 40.8518829
6: -33.2348480, 15.8802576, -33.2428055, 15.8885746, -49.1234207, 49.1230621
7: -24.6315155, 19.6394768, -24.6512070, 19.6453876, -44.2769012, 44.2906837
8: -27.8731613, 26.9187813, -27.9217548, 26.9249916, -54.7981529, 54.8405380
9: -24.0633545, 23.0853882, -24.0957565, 23.0894451, -47.1528015, 47.1811447
10: -32.4316330, 25.0465660, -32.4511261, 25.0555954, -57.4872284, 57.4976921
11: -27.0253887, 16.3595486, -27.0276184, 16.3934441, -43.4188309, 43.3871689
12: -32.1970482, 22.5161304, -32.2062988, 22.5293770, -54.3608780, 54.3544846
13: -31.2860928, 30.8170853, -31.3569469, 30.8262177, -62.1123123, 62.1740341
14: -51.4500122, 16.1779861, -51.4745560, 16.1837139, -67.6337280, 67.6525421
15: -26.7298317, 17.8357048, -26.7475834, 17.8368702, -44.5667038, 44.5832901
16: -34.0208359, 18.1579685, -34.0301323, 18.1639843, -52.1848221, 52.1881027
17: -50.6675339, 17.8659916, -50.7140083, 17.8724022, -68.5399323, 68.5800018
18: -35.8815346, 18.0532475, -35.8922119, 18.1227360, -54.0042725, 53.9454575
19: -20.6329002, 14.1508331, -20.6393280, 14.1760340, -34.8089333, 34.7901611
20: -20.6281300, 17.8805504, -20.6336060, 17.9239655, -38.5520935, 38.5141563
21: -26.3209705, 15.5866184, -26.3268147, 15.6215849, -41.9425545, 41.9134331
22: -25.9967175, 15.7424202, -26.0059528, 15.7603989, -41.7571182, 41.7483749
23: -19.3506012, 19.3767242, -19.3547363, 19.4042435, -38.7548447, 38.7314606
24: -27.6443424, 18.1045551, -27.6496067, 18.1416950, -45.7860374, 45.7541618
25: -21.5100727, 21.4262199, -21.5145111, 21.4724426, -42.9825134, 42.9407310
26: -35.2197876, 25.5109768, -35.2319336, 25.5555611, -60.7753487, 60.7429123
27: -26.2894745, 17.0156937, -26.2985764, 17.0745926, -43.3640671, 43.3142700
28: -20.1677952, 20.4522743, -20.1738987, 20.4911766, -40.6589737, 40.6261749
29: -24.3479176, 15.3635778, -24.3553963, 15.3771677, -39.7250862, 39.7189751
30: -26.7355251, 19.4834213, -26.7371330, 19.5112724, -46.2467957, 46.2205544
31: -28.4944172, 19.9261055, -28.5036201, 19.9791641, -48.4735794, 48.4297256
32: -31.9922237, 15.0139732, -32.0078583, 15.0204887, -46.8766022, 46.8884277
33: -52.4585991, 17.2345524, -52.5081711, 17.2420654, -68.6548157, 68.7029495
34: -45.5569229, 5.7651205, -45.5764961, 5.7670898, -49.8373337, 49.8518143
35: -41.1066132, 14.1246262, -41.1290054, 14.1280861, -54.2315674, 54.2470856
36: -36.7150192, 17.8650799, -36.7279663, 17.8703651, -54.5761871, 54.5837021
37: -59.2555351, 8.3485098, -59.3005981, 8.3543768, -67.5293732, 67.5717087
38: -45.8473587, 17.6136971, -45.8663101, 17.6267815, -63.4741402, 63.4800072
39: -55.0821991, 18.1741619, -55.1409378, 18.1762276, -73.1482849, 73.2082062
40: -44.7899323, 6.0399103, -44.8233414, 6.0430117, -50.7565460, 50.7893906
41: -35.4620743, 16.9195004, -35.4799042, 16.9255314, -52.3399963, 52.3526230
42: -23.7626953, 14.9795971, -23.7666626, 14.9875546, -38.7502518, 38.7462616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1722

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3140629, upper bound: 24.3281348
time: 82.78 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3140629, upper bound: 24.3281627
time: 77.05 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 162.10 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 162.10
Output dim: 20, lower bound: -24.3139601, upper bound: 24.2920722
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 162.10
Output dim: 20, lower bound: -24.3139601, upper bound: 24.2921034
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 162.10
Output dim: 20, lower bound: -24.3140629, upper bound: 24.3281348
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 162.10
Output dim: 20, lower bound: -24.3140629, upper bound: 24.3281627

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -43.0052414, 19.3849220, -42.9435577, 19.3323517, -62.3375931, 62.3284798
1: -21.2916622, 18.1942024, -21.2553520, 18.1498566, -39.4415207, 39.4495544
2: -15.0372524, 20.1924744, -14.9994059, 20.1430531, -35.1803055, 35.1918793
3: -20.4433479, 22.4464188, -20.4031239, 22.3940754, -42.8374252, 42.8495407
4: -24.1025200, 20.0351963, -24.0710373, 19.9881153, -44.0906372, 44.1062317
5: -19.1879997, 21.6210518, -19.1341076, 21.5690308, -40.7570305, 40.7551575
6: -33.2187653, 15.8745689, -33.1739006, 15.8497849, -49.0685501, 49.0484695
7: -24.6270332, 19.6180477, -24.5669937, 19.5662556, -44.1932907, 44.1850433
8: -27.8689976, 26.8916626, -27.8215218, 26.8258247, -54.6948242, 54.7131844
9: -24.0468807, 23.0710258, -23.9981956, 23.0157604, -47.0626411, 47.0692215
10: -32.4216232, 25.0326233, -32.3874702, 24.9791431, -57.4007645, 57.4200935
11: -27.0046806, 16.3492298, -26.9341488, 16.3175430, -43.3222237, 43.2833786
12: -32.1816788, 22.5093269, -32.1417313, 22.4328270, -54.2465363, 54.2830658
13: -31.2780933, 30.7862720, -31.2460175, 30.7085800, -61.9866714, 62.0322876
14: -51.4391632, 16.1574478, -51.3590813, 16.1085396, -67.5476990, 67.5165253
15: -26.7154636, 17.8290997, -26.6686802, 17.7849350, -44.5003967, 44.4977798
16: -34.0100708, 18.1487465, -33.9637604, 18.1118011, -52.1218719, 52.1125069
17: -50.6594391, 17.8396988, -50.5880089, 17.7708435, -68.4302826, 68.4277039
18: -35.8553162, 18.0484924, -35.7871704, 18.0257912, -53.8811073, 53.8356628
19: -20.6173611, 14.1481895, -20.5695534, 14.1375256, -34.7548866, 34.7177429
20: -20.6109200, 17.8731880, -20.5533848, 17.8510513, -38.4619713, 38.4265747
21: -26.3039837, 15.5795946, -26.2443619, 15.5589828, -41.8629684, 41.8239555
22: -25.9826698, 15.7364693, -25.9361496, 15.7134256, -41.6960945, 41.6726189
23: -19.3357487, 19.3687820, -19.2821407, 19.3443680, -38.6801147, 38.6509247
24: -27.6281338, 18.0937977, -27.5694122, 18.0606976, -45.6888313, 45.6632080
25: -21.4941425, 21.4126472, -21.4393291, 21.3745918, -42.8687363, 42.8519745
26: -35.1928978, 25.5065231, -35.1216812, 25.4529266, -60.6458244, 60.6282043
27: -26.2634182, 17.0072155, -26.1919937, 16.9836464, -43.2470627, 43.1992111
28: -20.1484871, 20.4479961, -20.0922985, 20.4314346, -40.5799217, 40.5402946
29: -24.3332710, 15.3591633, -24.2795162, 15.3414631, -39.6747360, 39.6386795
30: -26.7240295, 19.4700050, -26.6696186, 19.4312134, -46.1552429, 46.1396255
31: -28.4704437, 19.9183426, -28.3888226, 19.8926086, -48.3630524, 48.3071671
32: -31.9745312, 15.0067139, -31.9257622, 14.9730339, -46.8140564, 46.7969093
33: -52.4408875, 17.2155380, -52.3947258, 17.1383362, -68.5383224, 68.5674667
34: -45.5394440, 5.7583866, -45.4975395, 5.7121334, -49.7613449, 49.7623978
35: -41.0916138, 14.1189022, -41.0540085, 14.0810843, -54.1678925, 54.1663284
36: -36.6975441, 17.8622437, -36.6498642, 17.8299789, -54.5181046, 54.5025673
37: -59.2256165, 8.3324242, -59.1487350, 8.2286053, -67.3771667, 67.4011002
38: -45.8313828, 17.6079102, -45.7850800, 17.5833282, -63.4147110, 63.3929901
39: -55.0663681, 18.1550198, -55.0211983, 18.0806561, -73.0413208, 73.0691299
40: -44.7616196, 6.0304070, -44.6876297, 5.9498014, -50.6389999, 50.6422958
41: -35.4376488, 16.9095001, -35.3788071, 16.8555927, -52.2469101, 52.2415047
42: -23.7523499, 14.9732828, -23.7240486, 14.9438953, -38.6962433, 38.6973305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=207, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_B1_B1_B1

### Relational analysis result of IS_B1_B1_B1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2797199, upper bound: 24.2906565
time: 58.62 seconds

## Relational analysis of IS_B1_B1_B2

### Relational analysis result of IS_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3133490, upper bound: 24.2914601
time: 69.70 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -43.0094604, 19.3900719, -42.9961205, 19.3522682, -62.3617287, 62.3861923
1: -21.2930374, 18.2038040, -21.2878075, 18.1829605, -39.4759979, 39.4916115
2: -15.0388870, 20.2027607, -15.0324707, 20.1776142, -35.2164993, 35.2352295
3: -20.4450111, 22.4552193, -20.4397812, 22.4267139, -42.8717270, 42.8950005
4: -24.1040401, 20.0431004, -24.0949821, 20.0189838, -44.1230240, 44.1380844
5: -19.1897182, 21.6303844, -19.1833954, 21.6047630, -40.7944794, 40.8137817
6: -33.2277260, 15.8765774, -33.2075729, 15.8673248, -49.0950508, 49.0841522
7: -24.6294289, 19.6310463, -24.6227589, 19.6104698, -44.2398987, 44.2538071
8: -27.8704224, 26.9052410, -27.8622169, 26.8732624, -54.7436829, 54.7674561
9: -24.0587158, 23.0739632, -24.0469894, 23.0465393, -47.1052551, 47.1209526
10: -32.4267006, 25.0373173, -32.4144211, 25.0150833, -57.4417839, 57.4517365
11: -27.0120659, 16.3548889, -26.9801731, 16.3440132, -43.3560791, 43.3350601
12: -32.1933136, 22.5117321, -32.1837349, 22.4944668, -54.3193207, 54.3188171
13: -31.2831421, 30.7930260, -31.2738781, 30.7360115, -62.0191536, 62.0669022
14: -51.4433784, 16.1654472, -51.4244041, 16.1368942, -67.5802765, 67.5898514
15: -26.7245350, 17.8341331, -26.7124310, 17.8247910, -44.5493240, 44.5465622
16: -34.0162048, 18.1515770, -34.0033188, 18.1357536, -52.1519585, 52.1548958
17: -50.6630363, 17.8468323, -50.6503716, 17.8024521, -68.4654846, 68.4972076
18: -35.8593674, 18.0515709, -35.8067589, 18.0454559, -53.9048233, 53.8583298
19: -20.6223793, 14.1496029, -20.5967903, 14.1459007, -34.7682800, 34.7463913
20: -20.6141472, 17.8778801, -20.5803738, 17.8703232, -38.4844704, 38.4582520
21: -26.3081436, 15.5838280, -26.2766171, 15.5766821, -41.8848267, 41.8604431
22: -25.9873848, 15.7403145, -25.9643726, 15.7337980, -41.7211838, 41.7046890
23: -19.3401966, 19.3740044, -19.3150005, 19.3667336, -38.7069321, 38.6890030
24: -27.6316700, 18.1012630, -27.6006546, 18.0917320, -45.7234039, 45.7019196
25: -21.4969673, 21.4200592, -21.4653454, 21.4050217, -42.9019890, 42.8854065
26: -35.2012329, 25.5094395, -35.1572609, 25.5033169, -60.7045517, 60.6667023
27: -26.2690392, 17.0137768, -26.2195530, 17.0088348, -43.2778740, 43.2333298
28: -20.1535740, 20.4494781, -20.1191826, 20.4418659, -40.5954399, 40.5686607
29: -24.3389988, 15.3611364, -24.3162785, 15.3542986, -39.6932983, 39.6774139
30: -26.7282429, 19.4789391, -26.7099247, 19.4675102, -46.1957550, 46.1888657
31: -28.4753132, 19.9228821, -28.4297428, 19.9141693, -48.3894806, 48.3526230
32: -31.9865627, 15.0089970, -31.9722748, 14.9963837, -46.8480835, 46.8439827
33: -52.4517670, 17.2183609, -52.4356232, 17.1795807, -68.5932922, 68.6008987
34: -45.5509720, 5.7603006, -45.5370789, 5.7431898, -49.8042374, 49.7871017
35: -41.1015816, 14.1200428, -41.0892105, 14.1034355, -54.2004776, 54.1919174
36: -36.7096519, 17.8634453, -36.6934090, 17.8586826, -54.5590744, 54.5470619
37: -59.2465057, 8.3342838, -59.2251320, 8.2987766, -67.4689484, 67.4735870
38: -45.8357086, 17.6105309, -45.8042488, 17.6022873, -63.4379959, 63.4147797
39: -55.0749893, 18.1579304, -55.0569077, 18.1191025, -73.0865784, 73.1026001
40: -44.7806168, 6.0320635, -44.7580948, 6.0127974, -50.7181625, 50.7109528
41: -35.4551315, 16.9113693, -35.4381943, 16.8920860, -52.3003693, 52.3006897
42: -23.7580929, 14.9754467, -23.7467442, 14.9638720, -38.7219658, 38.7221909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=207, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 627

## Relational analysis of IS_B1_B2_A1

### Relational analysis result of IS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3125396, upper bound: 24.2578909
time: 72.74 seconds

## Relational analysis of IS_B1_B2_A2

### Relational analysis result of IS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3125396, upper bound: 24.2578909
time: 75.66 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -43.0091362, 19.3994102, -43.0084991, 19.3864136, -62.3955498, 62.4079094
1: -21.2930889, 18.2021675, -21.2820683, 18.1817207, -39.4748077, 39.4842377
2: -15.0391836, 20.2014866, -15.0285215, 20.1792641, -35.2184486, 35.2300072
3: -20.4443340, 22.4573460, -20.4371395, 22.4369240, -42.8812561, 42.8944855
4: -24.1056690, 20.0435753, -24.1192093, 20.0201511, -44.1258202, 44.1627846
5: -19.1894188, 21.6311340, -19.1604958, 21.6085510, -40.7979698, 40.7916298
6: -33.2233963, 15.8779507, -33.2020264, 15.8701267, -49.0935211, 49.0799789
7: -24.6287498, 19.6252975, -24.5944214, 19.5977478, -44.2264977, 44.2197189
8: -27.8714600, 26.9038277, -27.8802299, 26.8734818, -54.7449417, 54.7840576
9: -24.0504875, 23.0818996, -24.0438576, 23.0571384, -47.1076279, 47.1257553
10: -32.4249573, 25.0411339, -32.4194183, 25.0175323, -57.4424896, 57.4605522
11: -27.0168686, 16.3515797, -26.9784184, 16.3603668, -43.3772354, 43.3299980
12: -32.1842117, 22.5132217, -32.1608124, 22.4663467, -54.2849121, 54.3060913
13: -31.2798843, 30.8088493, -31.3257637, 30.7944489, -62.0743332, 62.1346130
14: -51.4449463, 16.1677570, -51.4066238, 16.1490707, -67.5940170, 67.5743790
15: -26.7189751, 17.8298740, -26.6991119, 17.7947311, -44.5137062, 44.5289841
16: -34.0138359, 18.1545124, -33.9881134, 18.1382942, -52.1521301, 52.1426239
17: -50.6631088, 17.8556213, -50.6493912, 17.8317719, -68.4948807, 68.5050125
18: -35.8750763, 18.0497055, -35.8654480, 18.1018124, -53.9768906, 53.9151535
19: -20.6271744, 14.1486473, -20.6100578, 14.1655064, -34.7926788, 34.7587051
20: -20.6242409, 17.8747692, -20.6047440, 17.9013767, -38.5256195, 38.4795151
21: -26.3159962, 15.5811977, -26.2921715, 15.6007061, -41.9167023, 41.8733673
22: -25.9911404, 15.7380238, -25.9751282, 15.7385044, -41.7296448, 41.7131500
23: -19.3455143, 19.3707600, -19.3200455, 19.3798256, -38.7253418, 38.6908035
24: -27.6400928, 18.0951576, -27.6164112, 18.1047897, -45.7448807, 45.7115707
25: -21.5065842, 21.4156189, -21.4864979, 21.4326096, -42.9391937, 42.9021149
26: -35.2088699, 25.5075378, -35.1886978, 25.5036125, -60.7124825, 60.6962357
27: -26.2829723, 17.0080261, -26.2684822, 17.0469093, -43.3298798, 43.2765083
28: -20.1620178, 20.4496346, -20.1450195, 20.4775600, -40.6395798, 40.5946541
29: -24.3414478, 15.3612938, -24.3165855, 15.3634415, -39.7048874, 39.6778793
30: -26.7305431, 19.4730644, -26.6946316, 19.4712009, -46.2017441, 46.1676941
31: -28.4886322, 19.9199982, -28.4600487, 19.9529724, -48.4416046, 48.3800468
32: -31.9778347, 15.0112839, -31.9544449, 14.9959288, -46.8389587, 46.8315926
33: -52.4456596, 17.2313747, -52.4613762, 17.1996803, -68.5971909, 68.6504669
34: -45.5432930, 5.7628717, -45.5314026, 5.7350359, -49.7907257, 49.8031654
35: -41.0947456, 14.1232595, -41.0883064, 14.1051130, -54.1964264, 54.2037506
36: -36.7014389, 17.8636665, -36.6800385, 17.8409653, -54.5330811, 54.5342827
37: -59.2324448, 8.3463612, -59.2187195, 8.2833443, -67.4346313, 67.4849014
38: -45.8419151, 17.6106262, -45.8438988, 17.6066589, -63.4485741, 63.4545250
39: -55.0719833, 18.1709366, -55.1007004, 18.1368256, -73.1005096, 73.1651230
40: -44.7682495, 6.0377750, -44.7450714, 5.9785986, -50.6732483, 50.7086029
41: -35.4421616, 16.9172192, -35.4134293, 16.8877621, -52.2828217, 52.2838936
42: -23.7549629, 14.9769602, -23.7382298, 14.9662104, -38.7211723, 38.7151909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=207, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_B2_B1_B1

### Relational analysis result of IS_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2798201, upper bound: 24.3267113
time: 73.61 seconds

## Relational analysis of IS_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3134519, upper bound: 24.2914601
time: 1122.66 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -43.0133514, 19.4045429, -43.0611267, 19.4063129, -62.4196625, 62.4656677
1: -21.2944565, 18.2117481, -21.3145256, 18.2148399, -39.5092964, 39.5262756
2: -15.0408201, 20.2117977, -15.0615931, 20.2138424, -35.2546616, 35.2733917
3: -20.4460087, 22.4661732, -20.4738522, 22.4695377, -42.9155464, 42.9400253
4: -24.1072063, 20.0514965, -24.1431618, 20.0510502, -44.1582565, 44.1946564
5: -19.1911640, 21.6404724, -19.2097740, 21.6443024, -40.8354645, 40.8502464
6: -33.2323723, 15.8799486, -33.2357979, 15.8876705, -49.1200409, 49.1157455
7: -24.6311417, 19.6383190, -24.6501369, 19.6419430, -44.2730865, 44.2884560
8: -27.8728733, 26.9173794, -27.9209061, 26.9208775, -54.7937508, 54.8382874
9: -24.0622978, 23.0848465, -24.0926361, 23.0879269, -47.1502228, 47.1774826
10: -32.4300232, 25.0458355, -32.4463730, 25.0534897, -57.4835129, 57.4922104
11: -27.0242672, 16.3572502, -27.0244389, 16.3868351, -43.4111023, 43.3816910
12: -32.1958580, 22.5156860, -32.2028122, 22.5280151, -54.3577423, 54.3418503
13: -31.2849503, 30.8155975, -31.3536720, 30.8218956, -62.1068459, 62.1692696
14: -51.4491119, 16.1757851, -51.4719391, 16.1773643, -67.6264801, 67.6477203
15: -26.7280025, 17.8348980, -26.7428322, 17.8345642, -44.5625687, 44.5777283
16: -34.0199814, 18.1573715, -34.0276947, 18.1622410, -52.1822205, 52.1850662
17: -50.6666908, 17.8627357, -50.7117882, 17.8633900, -68.5300827, 68.5745239
18: -35.8791122, 18.0527878, -35.8850861, 18.1214466, -54.0005569, 53.9378738
19: -20.6321793, 14.1500626, -20.6373062, 14.1738844, -34.8060646, 34.7873688
20: -20.6274986, 17.8794270, -20.6317291, 17.9206314, -38.5481300, 38.5111542
21: -26.3201599, 15.5854416, -26.3244400, 15.6183901, -41.9385490, 41.9098816
22: -25.9958420, 15.7418776, -26.0033932, 15.7588615, -41.7547035, 41.7452698
23: -19.3499451, 19.3759995, -19.3529110, 19.4021950, -38.7521400, 38.7289124
24: -27.6436710, 18.1025982, -27.6476688, 18.1358318, -45.7795029, 45.7502670
25: -21.5093937, 21.4230804, -21.5125427, 21.4630585, -42.9724503, 42.9356232
26: -35.2172165, 25.5104485, -35.2242661, 25.5540047, -60.7712212, 60.7347145
27: -26.2886009, 17.0145836, -26.2960129, 17.0720654, -43.3606644, 43.3105965
28: -20.1671066, 20.4511299, -20.1719265, 20.4879875, -40.6550941, 40.6230545
29: -24.3471909, 15.3632736, -24.3533611, 15.3762732, -39.7234650, 39.7166367
30: -26.7347450, 19.4820271, -26.7349453, 19.5075188, -46.2422638, 46.2169724
31: -28.4935074, 19.9245243, -28.5009899, 19.9745197, -48.4680252, 48.4255142
32: -31.9898739, 15.0135927, -32.0009613, 15.0192900, -46.8730240, 46.8787155
33: -52.4565430, 17.2341671, -52.5022392, 17.2409153, -68.6521530, 68.6837616
34: -45.5548401, 5.7648029, -45.5709457, 5.7661314, -49.8336105, 49.8278999
35: -41.1047173, 14.1244164, -41.1234932, 14.1274395, -54.2289963, 54.2293015
36: -36.7135162, 17.8648567, -36.7236214, 17.8696728, -54.5739975, 54.5788116
37: -59.2533455, 8.3482494, -59.2950706, 8.3535776, -67.5263367, 67.5574112
38: -45.8462639, 17.6132622, -45.8630905, 17.6255760, -63.4718399, 63.4763527
39: -55.0806236, 18.1738472, -55.1363983, 18.1753464, -73.1457977, 73.1985321
40: -44.7872658, 6.0394287, -44.8155174, 6.0416260, -50.7524414, 50.7771759
41: -35.4596367, 16.9190636, -35.4727707, 16.9242477, -52.3362579, 52.3430634
42: -23.7607117, 14.9791241, -23.7609367, 14.9862127, -38.7469254, 38.7400589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=207, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 627

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3126516, upper bound: 24.2939189
time: 87.90 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3275511, upper bound: 24.3275507
time: 68.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 158.39 seconds
IS_B1_B1_B1, status: Status.VERIFIED, split count: 3, time: 158.39
Output dim: 20, lower bound: -24.2797199, upper bound: 24.2906565
IS_B1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 158.39
Output dim: 20, lower bound: -24.3133490, upper bound: 24.2914601
IS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 158.39
Output dim: 20, lower bound: -24.3125396, upper bound: 24.2578909
IS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 158.39
Output dim: 20, lower bound: -24.3125396, upper bound: 24.2578909
IS_B2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 158.39
Output dim: 20, lower bound: -24.2798201, upper bound: 24.3267113
IS_B2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 158.39
Output dim: 20, lower bound: -24.3134519, upper bound: 24.2914601
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 158.39
Output dim: 20, lower bound: -24.3126516, upper bound: 24.2939189
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 158.39
Output dim: 20, lower bound: -24.3275511, upper bound: 24.3275507

## BFS IS instance: IS_B1_B1_B2

### Backsubstitution after applying IS history:
0: -43.0033455, 19.3839474, -42.9381523, 19.3295116, -62.3328552, 62.3220978
1: -21.2905693, 18.1935120, -21.2521210, 18.1478310, -39.4384003, 39.4456329
2: -15.0364494, 20.1920128, -14.9970894, 20.1416950, -35.1781464, 35.1891022
3: -20.4417267, 22.4459114, -20.3984222, 22.3926430, -42.8343697, 42.8443336
4: -24.1011429, 20.0341663, -24.0669975, 19.9852257, -44.0863686, 44.1011658
5: -19.1864433, 21.6205635, -19.1295967, 21.5676384, -40.7540817, 40.7501602
6: -33.2105713, 15.8738499, -33.1493988, 15.8478241, -49.0583954, 49.0232468
7: -24.6258507, 19.6176414, -24.5635529, 19.5651093, -44.1909599, 44.1811943
8: -27.8677616, 26.8910465, -27.8178940, 26.8241615, -54.6919250, 54.7089386
9: -24.0448532, 23.0702686, -23.9923687, 23.0135231, -47.0583763, 47.0626373
10: -32.4191513, 25.0316944, -32.3803596, 24.9764690, -57.3956223, 57.4120560
11: -27.0034409, 16.3479042, -26.9306679, 16.3136711, -43.3171120, 43.2785721
12: -32.1778717, 22.5082970, -32.1304512, 22.4300270, -54.2396698, 54.2752762
13: -31.2764130, 30.7852097, -31.2412243, 30.7055779, -61.9819908, 62.0264359
14: -51.4375114, 16.1558418, -51.3543968, 16.1038437, -67.5413513, 67.5102386
15: -26.7138767, 17.8281078, -26.6640644, 17.7821655, -44.4960403, 44.4921722
16: -34.0079269, 18.1479092, -33.9577370, 18.1094208, -52.1173477, 52.1056442
17: -50.6579132, 17.8384514, -50.5837097, 17.7673798, -68.4252930, 68.4221649
18: -35.8544388, 18.0460339, -35.7846375, 18.0189857, -53.8734245, 53.8306732
19: -20.6165657, 14.1466665, -20.5672817, 14.1331158, -34.7496796, 34.7139473
20: -20.6099892, 17.8719444, -20.5507374, 17.8473854, -38.4573746, 38.4226837
21: -26.3028412, 15.5782700, -26.2411613, 15.5553360, -41.8581772, 41.8194313
22: -25.9818115, 15.7347794, -25.9337597, 15.7084885, -41.6903000, 41.6685410
23: -19.3347588, 19.3667831, -19.2793503, 19.3384914, -38.6732483, 38.6461334
24: -27.6271877, 18.0918789, -27.5668297, 18.0550880, -45.6822739, 45.6587067
25: -21.4932652, 21.4107628, -21.4367867, 21.3691139, -42.8623810, 42.8475494
26: -35.1918411, 25.5045395, -35.1186028, 25.4470558, -60.6388969, 60.6231422
27: -26.2624130, 17.0057640, -26.1891918, 16.9794693, -43.2418823, 43.1949539
28: -20.1477737, 20.4463196, -20.0903282, 20.4265213, -40.5742950, 40.5366478
29: -24.3323193, 15.3582058, -24.2768669, 15.3387041, -39.6710243, 39.6350708
30: -26.7228088, 19.4684944, -26.6662006, 19.4267864, -46.1495972, 46.1346970
31: -28.4694977, 19.9168282, -28.3861885, 19.8882961, -48.3577957, 48.3030167
32: -31.9668999, 15.0059795, -31.9029579, 14.9710445, -46.8043518, 46.7754974
33: -52.4397659, 17.2141075, -52.3916473, 17.1343231, -68.5301514, 68.5672302
34: -45.5338898, 5.7575588, -45.4811630, 5.7097645, -49.7521744, 49.7614517
35: -41.0908203, 14.1176348, -41.0519409, 14.0773764, -54.1601562, 54.1675835
36: -36.6969299, 17.8611927, -36.6481094, 17.8269501, -54.5144272, 54.4998550
37: -59.2244606, 8.3306751, -59.1455841, 8.2235231, -67.3695221, 67.3982010
38: -45.8305664, 17.6062927, -45.7827301, 17.5788670, -63.4094315, 63.3890228
39: -55.0651703, 18.1538143, -55.0180283, 18.0771751, -73.0354156, 73.0665436
40: -44.7532349, 6.0295734, -44.6639595, 5.9474669, -50.6285858, 50.6196594
41: -35.4367142, 16.9086304, -35.3761368, 16.8531761, -52.2431183, 52.2385788
42: -23.7470627, 14.9726305, -23.7083664, 14.9420576, -38.6891212, 38.6809959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=206, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 999
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 1016

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_B1_B1_B2_A1

### Relational analysis result of IS_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2714582, upper bound: 24.2620504
time: 73.06 seconds

## Relational analysis of IS_B1_B1_B2_A2

### Relational analysis result of IS_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.3049188, upper bound: 24.2830477
time: 80.33 seconds

## BFS IS instance: IS_B1_B2_A1

### Backsubstitution after applying IS history:
0: -42.9050293, 19.3251705, -42.9667740, 19.3466682, -62.2516975, 62.2919464
1: -21.2333832, 18.1578789, -21.2686996, 18.1783257, -39.4117088, 39.4265785
2: -14.9978695, 20.1746597, -15.0203352, 20.1748543, -35.1727219, 35.1949959
3: -20.3646622, 22.3854218, -20.4149456, 22.4225998, -42.7872620, 42.8003693
4: -24.0212021, 19.9802818, -24.0697861, 20.0139561, -44.0351562, 44.0500679
5: -19.1061382, 21.5629539, -19.1587105, 21.6006966, -40.7068329, 40.7216644
6: -33.1979065, 15.8688774, -33.2010155, 15.8597364, -49.0576439, 49.0698929
7: -24.5707874, 19.5864792, -24.6049271, 19.6070747, -44.1778641, 44.1914062
8: -27.8116417, 26.8574677, -27.8442688, 26.8681946, -54.6798363, 54.7017365
9: -23.9547462, 22.9914055, -24.0150108, 23.0415497, -46.9962959, 47.0064163
10: -32.2987175, 24.9421577, -32.3751488, 25.0089722, -57.3076897, 57.3173065
11: -26.9523067, 16.2830811, -26.9681320, 16.3219872, -43.2742920, 43.2512131
12: -32.1702957, 22.4701653, -32.1786118, 22.4838314, -54.2842255, 54.2702332
13: -31.2071266, 30.7318630, -31.2504787, 30.7263889, -61.9335175, 61.9823418
14: -51.3744278, 16.1332111, -51.4057083, 16.1301308, -67.5045624, 67.5389175
15: -26.6298580, 17.7676868, -26.6842613, 17.8191528, -44.4490128, 44.4519501
16: -33.9222374, 18.0872803, -33.9753036, 18.1298256, -52.0520630, 52.0625839
17: -50.5905037, 17.7899361, -50.6294975, 17.7879868, -68.3784943, 68.4194336
18: -35.8068008, 17.9969139, -35.7967682, 18.0288200, -53.8356209, 53.7936821
19: -20.5458279, 14.0750771, -20.5892487, 14.1208305, -34.6666565, 34.6643257
20: -20.5479450, 17.8051720, -20.5726643, 17.8470039, -38.3949509, 38.3778381
21: -26.2353230, 15.4916677, -26.2658501, 15.5460091, -41.7813339, 41.7575188
22: -25.9281578, 15.6849422, -25.9565296, 15.7160034, -41.6441612, 41.6414719
23: -19.2322559, 19.2576828, -19.3066368, 19.3296165, -38.5618744, 38.5643196
24: -27.5372677, 17.9967327, -27.5937710, 18.0581493, -45.5954170, 45.5905037
25: -21.4127808, 21.3127670, -21.4584885, 21.3709373, -42.7837181, 42.7712555
26: -35.1136703, 25.4105759, -35.1472168, 25.4713001, -60.5849686, 60.5577927
27: -26.1942253, 16.9382992, -26.2132378, 16.9849701, -43.1791954, 43.1515350
28: -20.0678825, 20.3487740, -20.1128845, 20.4100952, -40.4779778, 40.4616585
29: -24.2805214, 15.3113356, -24.3075829, 15.3389130, -39.6194344, 39.6189194
30: -26.6555157, 19.3856525, -26.7031097, 19.4404068, -46.0959244, 46.0887604
31: -28.3913460, 19.8469353, -28.4206944, 19.8896866, -48.2810326, 48.2676315
32: -31.9540634, 15.0025806, -31.9652004, 14.9899578, -46.8087158, 46.8300095
33: -52.3943558, 17.1553612, -52.4282570, 17.1601791, -68.5054550, 68.5245743
34: -45.5241547, 5.7343302, -45.5311890, 5.7354450, -49.7638245, 49.7490158
35: -41.0567207, 14.0665684, -41.0837402, 14.0857868, -54.1263351, 54.1268692
36: -36.6670189, 17.8077621, -36.6884842, 17.8411560, -54.4986725, 54.4863968
37: -59.1546555, 8.2473154, -59.2156639, 8.2713547, -67.3442383, 67.3745270
38: -45.7780228, 17.5444031, -45.7970085, 17.5821037, -63.3601265, 63.3414116
39: -55.0135536, 18.1068001, -55.0475044, 18.1031380, -73.0037231, 73.0400314
40: -44.7295685, 6.0214071, -44.7462921, 6.0046072, -50.6587372, 50.6883850
41: -35.4251709, 16.8785286, -35.4322815, 16.8830414, -52.2595978, 52.2612762
42: -23.7266045, 14.9519062, -23.7401848, 14.9572620, -38.6838684, 38.6920929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=207, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_B1_B2_A1_B1

### Relational analysis result of IS_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2831689, upper bound: 24.2496256
time: 83.05 seconds

## Relational analysis of IS_B1_B2_A1_B2

### Relational analysis result of IS_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3183296, upper bound: 24.2496256
time: 98.12 seconds

## BFS IS instance: IS_B1_B2_A2

### Backsubstitution after applying IS history:
0: -43.0040398, 19.3871975, -42.9942551, 19.3512840, -62.3553238, 62.3814545
1: -21.2898102, 18.2017670, -21.2867203, 18.1822701, -39.4720802, 39.4884872
2: -15.0365829, 20.2014484, -15.0316753, 20.1771564, -35.2137375, 35.2331238
3: -20.4403133, 22.4537563, -20.4381886, 22.4262085, -42.8665237, 42.8919449
4: -24.0999641, 20.0402145, -24.0935936, 20.0179482, -44.1179123, 44.1338081
5: -19.1852207, 21.6290245, -19.1818581, 21.6042767, -40.7894974, 40.8108826
6: -33.2032661, 15.8746052, -33.1993866, 15.8666096, -49.0698776, 49.0739899
7: -24.6259842, 19.6298923, -24.6215591, 19.6100578, -44.2360420, 44.2514496
8: -27.8667965, 26.9035168, -27.8609753, 26.8726311, -54.7394257, 54.7644920
9: -24.0528641, 23.0717411, -24.0449734, 23.0457649, -47.0986290, 47.1167145
10: -32.4195900, 25.0346375, -32.4119720, 25.0141258, -57.4337158, 57.4466095
11: -27.0085678, 16.3510323, -26.9789352, 16.3426876, -43.3512573, 43.3299675
12: -32.1820602, 22.5089455, -32.1799698, 22.4934807, -54.3115845, 54.3119507
13: -31.2783661, 30.7900200, -31.2722282, 30.7349815, -62.0133476, 62.0622482
14: -51.4386940, 16.1607475, -51.4227371, 16.1352806, -67.5739746, 67.5834808
15: -26.7199097, 17.8313675, -26.7108536, 17.8237801, -44.5436897, 44.5422211
16: -34.0102005, 18.1491947, -34.0011902, 18.1349144, -52.1451149, 52.1503830
17: -50.6587677, 17.8433647, -50.6489372, 17.8012199, -68.4599915, 68.4923019
18: -35.8568802, 18.0447502, -35.8058777, 18.0430069, -53.8998871, 53.8506279
19: -20.6200943, 14.1451979, -20.5959854, 14.1443949, -34.7644882, 34.7411842
20: -20.6114883, 17.8742104, -20.5794563, 17.8690701, -38.4805603, 38.4536667
21: -26.3049278, 15.5801964, -26.2754745, 15.5753736, -41.8803024, 41.8556709
22: -25.9850063, 15.7353630, -25.9635544, 15.7320929, -41.7170982, 41.6989174
23: -19.3373947, 19.3681107, -19.3140221, 19.3647385, -38.7021332, 38.6821327
24: -27.6290741, 18.0956497, -27.5997162, 18.0898094, -45.7188835, 45.6953659
25: -21.4944458, 21.4145851, -21.4644508, 21.4031563, -42.8976021, 42.8790359
26: -35.1981735, 25.5035439, -35.1561737, 25.5013371, -60.6995087, 60.6597176
27: -26.2662354, 17.0096016, -26.2185535, 17.0073776, -43.2736130, 43.2281570
28: -20.1516075, 20.4445457, -20.1185055, 20.4401894, -40.5917969, 40.5630493
29: -24.3363323, 15.3583813, -24.3153267, 15.3533506, -39.6896820, 39.6737061
30: -26.7247982, 19.4745293, -26.7086868, 19.4660034, -46.1908035, 46.1832161
31: -28.4726467, 19.9185753, -28.4288139, 19.9126511, -48.3852997, 48.3473892
32: -31.9637737, 15.0070009, -31.9646301, 14.9956779, -46.8266602, 46.8342743
33: -52.4487076, 17.2143250, -52.4345093, 17.1782169, -68.5930328, 68.5927505
34: -45.5346222, 5.7579746, -45.5315475, 5.7423534, -49.8032990, 49.7779274
35: -41.0994987, 14.1163263, -41.0884285, 14.1021385, -54.2016907, 54.1842384
36: -36.7078896, 17.8603859, -36.6927681, 17.8576546, -54.5563431, 54.5433884
37: -59.2433586, 8.3292046, -59.2239876, 8.2970448, -67.4660721, 67.4659271
38: -45.8333549, 17.6060829, -45.8034592, 17.6006756, -63.4340286, 63.4095421
39: -55.0718193, 18.1544380, -55.0557060, 18.1179028, -73.0839844, 73.0967178
40: -44.7569618, 6.0297108, -44.7496643, 6.0119801, -50.6955414, 50.7005386
41: -35.4524612, 16.9089489, -35.4372559, 16.8912163, -52.2974854, 52.2968674
42: -23.7424393, 14.9735975, -23.7414818, 14.9632139, -38.7056541, 38.7150803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=207, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_B1_B2_A2_B1

### Relational analysis result of IS_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2980279, upper bound: 24.2830828
time: 73.16 seconds

## Relational analysis of IS_B1_B2_A2_B2

### Relational analysis result of IS_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3190792, upper bound: 24.2830828
time: 71.34 seconds

## BFS IS instance: IS_B2_B1_B1

### Backsubstitution after applying IS history:
0: -42.9797935, 19.3938103, -42.9041023, 19.3215179, -62.3013115, 62.2979126
1: -21.2739525, 18.1975060, -21.2224045, 18.1357803, -39.4097328, 39.4199104
2: -15.0270681, 20.1987247, -14.9875126, 20.1511345, -35.1782036, 35.1862373
3: -20.4194832, 22.4532051, -20.3568020, 22.3670940, -42.7865753, 42.8100052
4: -24.0804520, 20.0385303, -24.0363522, 19.9573364, -44.0377884, 44.0748825
5: -19.1647053, 21.6270618, -19.0768776, 21.5411072, -40.7058105, 40.7039413
6: -33.2168121, 15.8703337, -33.1721764, 15.8624401, -49.0792542, 49.0425110
7: -24.6109161, 19.6219215, -24.5358162, 19.5531654, -44.1640816, 44.1577377
8: -27.8535156, 26.8987370, -27.8214130, 26.8256493, -54.6791649, 54.7201500
9: -24.0185013, 23.0769100, -23.9398880, 22.9745674, -46.9930687, 47.0167999
10: -32.3856659, 25.0350208, -32.2914581, 24.9223785, -57.3080444, 57.3264771
11: -27.0048370, 16.3295479, -26.9186287, 16.2885666, -43.2934036, 43.2481766
12: -32.1790543, 22.5025749, -32.1377907, 22.4247704, -54.2363129, 54.2710037
13: -31.2564697, 30.7991676, -31.2497711, 30.7332516, -61.9897232, 62.0489388
14: -51.4262466, 16.1610527, -51.3377609, 16.1167908, -67.5430374, 67.4988098
15: -26.6908073, 17.8242397, -26.6044331, 17.7282944, -44.4191017, 44.4286728
16: -33.9858322, 18.1485767, -33.8941193, 18.0739784, -52.0598106, 52.0426941
17: -50.6421280, 17.8411865, -50.5768318, 17.7748451, -68.4169769, 68.4180145
18: -35.8650818, 18.0330715, -35.8128395, 18.0471401, -53.9122238, 53.8459091
19: -20.6196327, 14.1235771, -20.5335197, 14.0909958, -34.7106285, 34.6570969
20: -20.6165600, 17.8514442, -20.5385742, 17.8287029, -38.4452629, 38.3900185
21: -26.3052368, 15.5505257, -26.2193756, 15.5085344, -41.8137703, 41.7699013
22: -25.9832554, 15.7202663, -25.9159260, 15.6831522, -41.6664085, 41.6361923
23: -19.3371391, 19.3336830, -19.2120800, 19.2635117, -38.6006508, 38.5457611
24: -27.6332626, 18.0615749, -27.5220013, 18.0002499, -45.6335144, 45.5835762
25: -21.4996948, 21.3815651, -21.4023323, 21.3253326, -42.8250275, 42.7838974
26: -35.1988716, 25.4754734, -35.1011276, 25.4047508, -60.6036224, 60.5765991
27: -26.2766266, 16.9841537, -26.1936836, 16.9713974, -43.2480240, 43.1778374
28: -20.1556931, 20.4178505, -20.0593491, 20.3768425, -40.5325356, 40.4771996
29: -24.3327541, 15.3459063, -24.2580872, 15.3136187, -39.6463737, 39.6039925
30: -26.7237263, 19.4459782, -26.6219006, 19.3779030, -46.1016312, 46.0678787
31: -28.4795837, 19.8955040, -28.3760967, 19.8770180, -48.3566017, 48.2715988
32: -31.9707603, 15.0048656, -31.9219589, 14.9895105, -46.8249741, 46.7922287
33: -52.4383202, 17.2119942, -52.4039688, 17.1367092, -68.5207672, 68.5625839
34: -45.5373802, 5.7551212, -45.5045662, 5.7090845, -49.7526093, 49.7627220
35: -41.0892792, 14.1056175, -41.0434761, 14.0516396, -54.1313934, 54.1296463
36: -36.6965179, 17.8461151, -36.6374512, 17.7852898, -54.4723816, 54.4739380
37: -59.2229462, 8.3189392, -59.1268044, 8.1963377, -67.3355408, 67.3601913
38: -45.8346748, 17.5904846, -45.7861977, 17.5405426, -63.3752174, 63.3766823
39: -55.0625725, 18.1549110, -55.0392761, 18.0857162, -73.0379333, 73.0823441
40: -44.7564545, 6.0295763, -44.6939774, 5.9679384, -50.6506958, 50.6490784
41: -35.4362411, 16.9081726, -35.3834496, 16.8549385, -52.2434082, 52.2431717
42: -23.7483921, 14.9703579, -23.7067413, 14.9426804, -38.6910706, 38.6771011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=206, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=401, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_B2_B1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2505146, upper bound: 24.3184613
time: 69.90 seconds

## Relational analysis of IS_B2_B1_B1_B2

### Relational analysis result of IS_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2715856, upper bound: 24.3184613
time: 75.93 seconds

## BFS IS instance: IS_B2_B1_B2

### Backsubstitution after applying IS history:
0: -43.0072403, 19.3984165, -43.0031319, 19.3835335, -62.3907738, 62.4015503
1: -21.2919712, 18.2014427, -21.2788181, 18.1796818, -39.4716530, 39.4802628
2: -15.0383787, 20.2010117, -15.0261974, 20.1779366, -35.2163162, 35.2272110
3: -20.4426994, 22.4568367, -20.4324570, 22.4354534, -42.8781509, 42.8892937
4: -24.1042881, 20.0425625, -24.1151333, 20.0172634, -44.1215515, 44.1576958
5: -19.1878757, 21.6306496, -19.1559715, 21.6071815, -40.7950592, 40.7866211
6: -33.2152214, 15.8772449, -33.1775551, 15.8681545, -49.0833740, 49.0548019
7: -24.6275711, 19.6249123, -24.5909920, 19.5966015, -44.2241745, 44.2159042
8: -27.8702087, 26.9032040, -27.8765755, 26.8717632, -54.7419739, 54.7797775
9: -24.0484390, 23.0811272, -24.0380096, 23.0548935, -47.1033325, 47.1191368
10: -32.4224930, 25.0401993, -32.4123154, 25.0148811, -57.4373741, 57.4525146
11: -27.0156250, 16.3502502, -26.9749088, 16.3565178, -43.3721428, 43.3251572
12: -32.1804276, 22.5122108, -32.1495438, 22.4635582, -54.2780228, 54.2983475
13: -31.2782021, 30.8077927, -31.3210125, 30.7914391, -62.0696411, 62.1288071
14: -51.4432793, 16.1661568, -51.4019775, 16.1443081, -67.5875854, 67.5681305
15: -26.7173824, 17.8288975, -26.6944714, 17.7919674, -44.5093498, 44.5233688
16: -34.0117073, 18.1536732, -33.9821281, 18.1359024, -52.1476097, 52.1358032
17: -50.6615753, 17.8544025, -50.6450691, 17.8283234, -68.4898987, 68.4994736
18: -35.8741837, 18.0472374, -35.8629494, 18.0949421, -53.9691238, 53.9101868
19: -20.6263657, 14.1471233, -20.6077881, 14.1611147, -34.7874794, 34.7549133
20: -20.6233292, 17.8735199, -20.6020889, 17.8977146, -38.5210419, 38.4756088
21: -26.3148575, 15.5798779, -26.2889786, 15.5970449, -41.9119034, 41.8688583
22: -25.9902878, 15.7363358, -25.9727592, 15.7335634, -41.7238503, 41.7090950
23: -19.3445187, 19.3687878, -19.3172264, 19.3739510, -38.7184677, 38.6860123
24: -27.6391697, 18.0932083, -27.6138077, 18.0991859, -45.7383575, 45.7070160
25: -21.5056725, 21.4137726, -21.4839706, 21.4271317, -42.9328041, 42.8977432
26: -35.2077980, 25.5055084, -35.1856308, 25.4977093, -60.7055054, 60.6911392
27: -26.2819691, 17.0065842, -26.2656994, 17.0427094, -43.3246765, 43.2722855
28: -20.1613235, 20.4479618, -20.1430759, 20.4726200, -40.6339417, 40.5910378
29: -24.3405037, 15.3603363, -24.3139267, 15.3606815, -39.7011871, 39.6742630
30: -26.7292976, 19.4715595, -26.6912289, 19.4668007, -46.1960983, 46.1627884
31: -28.4877090, 19.9184837, -28.4573975, 19.9486504, -48.4363594, 48.3758812
32: -31.9702187, 15.0105581, -31.9316578, 14.9939270, -46.8292694, 46.8101807
33: -52.4445343, 17.2300186, -52.4582939, 17.1957054, -68.5889587, 68.6501694
34: -45.5377502, 5.7620449, -45.5150146, 5.7327299, -49.7815552, 49.8022041
35: -41.0939522, 14.1219740, -41.0862350, 14.1014166, -54.1887360, 54.2049828
36: -36.7007980, 17.8626328, -36.6783142, 17.8379307, -54.5293579, 54.5315399
37: -59.2312927, 8.3446207, -59.2155724, 8.2782841, -67.4269180, 67.4819870
38: -45.8410950, 17.6090508, -45.8415527, 17.6021996, -63.4432945, 63.4506035
39: -55.0708084, 18.1697197, -55.0975304, 18.1333370, -73.0946350, 73.1625443
40: -44.7598267, 6.0369558, -44.7213898, 5.9762630, -50.6628265, 50.6859436
41: -35.4412155, 16.9163513, -35.4107399, 16.8853455, -52.2790222, 52.2810287
42: -23.7496834, 14.9763079, -23.7225571, 14.9643679, -38.7140503, 38.6988640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=206, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 999
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 1016

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_B2_B1_B2_B1

### Relational analysis result of IS_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2841086, upper bound: 24.3192026
time: 75.52 seconds

## Relational analysis of IS_B2_B1_B2_B2

### Relational analysis result of IS_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2505146, upper bound: 24.3192026
time: 71.71 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -42.9089127, 19.3396549, -43.0317764, 19.4007282, -62.3096390, 62.3714294
1: -21.2347946, 18.1658020, -21.2954025, 18.2101765, -39.4449692, 39.4612045
2: -14.9997931, 20.1836910, -15.0494547, 20.2110634, -35.2108574, 35.2331467
3: -20.3656368, 22.3963699, -20.4489746, 22.4653816, -42.8310165, 42.8453445
4: -24.0243454, 19.9886665, -24.1179352, 20.0459976, -44.0703430, 44.1066017
5: -19.1075859, 21.5730247, -19.1850433, 21.6402149, -40.7478027, 40.7580681
6: -33.2025299, 15.8722630, -33.2292366, 15.8800488, -49.0825806, 49.1015015
7: -24.5725174, 19.5937424, -24.6323242, 19.6385422, -44.2110596, 44.2260666
8: -27.8140678, 26.8696041, -27.9029388, 26.9158020, -54.7298698, 54.7725449
9: -23.9583569, 23.0022812, -24.0606365, 23.0829182, -47.0412750, 47.0629196
10: -32.3020515, 24.9506531, -32.4070969, 25.0473824, -57.3494339, 57.3577499
11: -26.9644947, 16.2854500, -27.0124016, 16.3648109, -43.3293076, 43.2978516
12: -32.1728516, 22.4741230, -32.1976776, 22.5174332, -54.3225708, 54.2932968
13: -31.2089310, 30.7544174, -31.3302650, 30.8122272, -62.0211563, 62.0846825
14: -51.3802109, 16.1435738, -51.4532623, 16.1706009, -67.5508118, 67.5968323
15: -26.6333618, 17.7684555, -26.7146664, 17.8289471, -44.4623108, 44.4831238
16: -33.9260178, 18.0930500, -33.9996986, 18.1563072, -52.0823250, 52.0927505
17: -50.5941620, 17.8059082, -50.6908226, 17.8489456, -68.4431076, 68.4967346
18: -35.8265533, 17.9981422, -35.8750877, 18.1048260, -53.9313812, 53.8732300
19: -20.5556450, 14.0755405, -20.6297569, 14.1488161, -34.7044601, 34.7052994
20: -20.5612984, 17.8067322, -20.6240444, 17.8973427, -38.4586411, 38.4307785
21: -26.2473335, 15.4932709, -26.3136902, 15.5877266, -41.8350601, 41.8069611
22: -25.9366302, 15.6865129, -25.9955177, 15.7410917, -41.6777229, 41.6820297
23: -19.2420197, 19.2596931, -19.3445301, 19.3650761, -38.6070938, 38.6042252
24: -27.5492859, 17.9980602, -27.6408081, 18.1022682, -45.6515541, 45.6388702
25: -21.4252090, 21.3157692, -21.5056610, 21.4289818, -42.8541908, 42.8214302
26: -35.1296539, 25.4115524, -35.2142715, 25.5219803, -60.6516342, 60.6258240
27: -26.2137966, 16.9390945, -26.2897015, 17.0482140, -43.2620087, 43.2287979
28: -20.0814247, 20.3504295, -20.1656303, 20.4562035, -40.5376282, 40.5160599
29: -24.2887211, 15.3134651, -24.3446655, 15.3608723, -39.6495934, 39.6581306
30: -26.6620140, 19.3887177, -26.7281227, 19.4804115, -46.1424255, 46.1168404
31: -28.4095421, 19.8485603, -28.4919319, 19.9500484, -48.3595886, 48.3404922
32: -31.9573936, 15.0071411, -31.9938965, 15.0128765, -46.8336182, 46.8647079
33: -52.3991852, 17.1711636, -52.4948349, 17.2215195, -68.5642700, 68.6074066
34: -45.5280113, 5.7388229, -45.5650597, 5.7583857, -49.7932281, 49.7897949
35: -41.0598526, 14.0709448, -41.1180153, 14.1098070, -54.1548996, 54.1642494
36: -36.6709023, 17.8091640, -36.7186852, 17.8521271, -54.5136414, 54.5181274
37: -59.1614227, 8.2612076, -59.2855911, 8.3261681, -67.4016113, 67.4582901
38: -45.7885284, 17.5471497, -45.8558388, 17.6054192, -63.3939476, 63.4029884
39: -55.0191574, 18.1227264, -55.1269836, 18.1592941, -73.0628967, 73.1359787
40: -44.7361794, 6.0287714, -44.8037415, 6.0334167, -50.6929779, 50.7546234
41: -35.4296570, 16.8862305, -35.4668503, 16.9152336, -52.2955017, 52.3037109
42: -23.7292061, 14.9555912, -23.7543716, 14.9796038, -38.7088089, 38.7099609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=207, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1019
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_B2_B2_A1_B1

### Relational analysis result of IS_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 20, lower bound: -24.2833059, upper bound: 24.2496256
time: 126.60 seconds

## Relational analysis of IS_B2_B2_A1_B2

### Relational analysis result of IS_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3184934, upper bound: 24.2857491
time: 269.83 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -43.0079651, 19.4016914, -43.0592346, 19.4053402, -62.4133072, 62.4609261
1: -21.2912140, 18.2096958, -21.3134136, 18.2140999, -39.5053139, 39.5231094
2: -15.0385122, 20.2104645, -15.0607920, 20.2133465, -35.2518578, 35.2712555
3: -20.4412975, 22.4647026, -20.4722176, 22.4690380, -42.9103355, 42.9369202
4: -24.1031075, 20.0486050, -24.1417694, 20.0500412, -44.1531487, 44.1903763
5: -19.1866608, 21.6390839, -19.2082195, 21.6438065, -40.8304672, 40.8473053
6: -33.2079010, 15.8779917, -33.2276230, 15.8869438, -49.0948448, 49.1056137
7: -24.6276779, 19.6371803, -24.6489677, 19.6415443, -44.2692223, 44.2861481
8: -27.8692665, 26.9156628, -27.9196663, 26.9202633, -54.7895279, 54.8353271
9: -24.0564499, 23.0825920, -24.0905819, 23.0871372, -47.1435852, 47.1731720
10: -32.4229279, 25.0431767, -32.4439163, 25.0525417, -57.4754715, 57.4870911
11: -27.0207577, 16.3533897, -27.0231991, 16.3855057, -43.4062653, 43.3765869
12: -32.1846085, 22.5128746, -32.1990662, 22.5270100, -54.3499603, 54.3350372
13: -31.2801762, 30.8125954, -31.3520031, 30.8208466, -62.1010208, 62.1645966
14: -51.4444466, 16.1710835, -51.4703140, 16.1757755, -67.6202240, 67.6413956
15: -26.7233906, 17.8321495, -26.7412376, 17.8335915, -44.5569839, 44.5733871
16: -34.0139694, 18.1549702, -34.0255737, 18.1614227, -52.1753922, 52.1805420
17: -50.6624489, 17.8593140, -50.7103157, 17.8621559, -68.5246048, 68.5696259
18: -35.8766289, 18.0459824, -35.8841896, 18.1190128, -53.9956436, 53.9301720
19: -20.6299171, 14.1456594, -20.6365108, 14.1723614, -34.8022766, 34.7821693
20: -20.6248283, 17.8757629, -20.6308079, 17.9193993, -38.5442276, 38.5065689
21: -26.3169670, 15.5817966, -26.3233128, 15.6170731, -41.9340401, 41.9051094
22: -25.9934597, 15.7369356, -26.0025444, 15.7571745, -41.7506332, 41.7394791
23: -19.3471489, 19.3701286, -19.3519135, 19.4001789, -38.7473297, 38.7220421
24: -27.6410713, 18.0969963, -27.6467228, 18.1339073, -45.7749786, 45.7437210
25: -21.5068626, 21.4175911, -21.5116615, 21.4611702, -42.9680328, 42.9292526
26: -35.2141609, 25.5045547, -35.2231827, 25.5519924, -60.7661514, 60.7277374
27: -26.2857990, 17.0104027, -26.2950172, 17.0706177, -43.3564148, 43.3054199
28: -20.1651382, 20.4461975, -20.1712379, 20.4863243, -40.6514626, 40.6174355
29: -24.3445282, 15.3605270, -24.3523941, 15.3753195, -39.7198486, 39.7129211
30: -26.7313290, 19.4776154, -26.7336979, 19.5060101, -46.2373390, 46.2113113
31: -28.4908524, 19.9202271, -28.5000420, 19.9730301, -48.4638824, 48.4202690
32: -31.9671059, 15.0115576, -31.9933205, 15.0185604, -46.8516006, 46.8690262
33: -52.4534683, 17.2301636, -52.5010834, 17.2395668, -68.6518860, 68.6755905
34: -45.5384750, 5.7624311, -45.5653915, 5.7652874, -49.8327026, 49.8187180
35: -41.1026421, 14.1206999, -41.1226997, 14.1261654, -54.2302246, 54.2216492
36: -36.7117844, 17.8618279, -36.7229691, 17.8686180, -54.5712738, 54.5750961
37: -59.2501831, 8.3431597, -59.2939224, 8.3518515, -67.5234680, 67.5496979
38: -45.8439026, 17.6087933, -45.8622894, 17.6239815, -63.4678841, 63.4710846
39: -55.0774498, 18.1703682, -55.1352119, 18.1741180, -73.1432190, 73.1926575
40: -44.7635727, 6.0370626, -44.8071136, 6.0407705, -50.7298126, 50.7667618
41: -35.4569626, 16.9166527, -35.4718437, 16.9233818, -52.3333740, 52.3393021
42: -23.7450333, 14.9772959, -23.7556553, 14.9855423, -38.7305756, 38.7329521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=207, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1019
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_B2_B2_A2_B1

### Relational analysis result of IS_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.2981821, upper bound: 24.3192387
time: 64.13 seconds

## Relational analysis of IS_B2_B2_A2_B2

### Relational analysis result of IS_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 20, lower bound: -24.3192390, upper bound: 24.3192387
time: 262.51 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 328.93 seconds
IS_B1_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 328.93
Output dim: 20, lower bound: -24.2714582, upper bound: 24.2620504
IS_B1_B1_B2_A2, status: Status.VERIFIED, split count: 4, time: 328.93
Output dim: 20, lower bound: -24.3049188, upper bound: 24.2830477
IS_B1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 328.93
Output dim: 20, lower bound: -24.2831689, upper bound: 24.2496256
IS_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 328.93
Output dim: 20, lower bound: -24.3183296, upper bound: 24.2496256
IS_B1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 328.93
Output dim: 20, lower bound: -24.2980279, upper bound: 24.2830828
IS_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 328.93
Output dim: 20, lower bound: -24.3190792, upper bound: 24.2830828
IS_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 328.93
Output dim: 20, lower bound: -24.2505146, upper bound: 24.3184613
IS_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 328.93
Output dim: 20, lower bound: -24.2715856, upper bound: 24.3184613
IS_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 328.93
Output dim: 20, lower bound: -24.2841086, upper bound: 24.3192026
IS_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 328.93
Output dim: 20, lower bound: -24.2505146, upper bound: 24.3192026
IS_B2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 328.93
Output dim: 20, lower bound: -24.2833059, upper bound: 24.2496256
IS_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 328.93
Output dim: 20, lower bound: -24.3184934, upper bound: 24.2857491
IS_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 328.93
Output dim: 20, lower bound: -24.2981821, upper bound: 24.3192387
IS_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 328.93
Output dim: 20, lower bound: -24.3192390, upper bound: 24.3192387

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 80.74 + 3591.93 = 3672.67 seconds
