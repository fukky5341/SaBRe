## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 1800 seconds
Split limit: 100
Threshold: 16.9840546443


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8782654, 40.8782654)
1: (-9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559)
2: (-8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9239731, 20.9239769)
3: (-10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8181686, 25.8181686)
4: (-16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792)
5: (-9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1498642, 25.1498642)
6: (-35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8194046, 31.8193970)
7: (-9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9592056, 23.9592056)
8: (-19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903)
9: (-6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2592163, 32.2592087)
10: (-5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7812958, 34.7812958)
11: (-11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073)
12: (-18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2787781, 29.2787781)
13: (-17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2632294, 44.2632370)
14: (-26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0472717, 38.0472641)
15: (-19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754)
16: (-15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9491196, 27.9491234)
17: (-28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5215454, 28.5215492)
18: (-29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5299454, 26.5299492)
19: (-16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0911179, 22.0911140)
20: (-11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8184700, 23.8184662)
21: (-15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804)
22: (-17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8492355, 22.8492355)
23: (-15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370)
24: (-27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2671509, 31.2671509)
25: (-17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3104019, 27.3104019)
26: (-22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9395638, 32.9395676)
27: (-28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5806236, 28.5806198)
28: (-15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4072495, 27.4072495)
29: (-15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7416039, 24.7416039)
30: (-17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4017715, 32.4017715)
31: (-26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1401215, 33.1401215)
32: (-26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5745087, 30.5745087)
33: (-64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5728149, 44.5728149)
34: (-51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2178955, 27.2178917)
35: (-46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1330948, 34.1330948)
36: (-40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3619690, 36.3619614)
37: (-67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5450439, 35.5450439)
38: (-50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7702560, 40.7702713)
39: (-61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2532501, 48.2532501)
40: (-52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1750259, 24.1750259)
41: (-37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7464905, 30.7464905)
42: (-24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0553589, 23.0553551)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.84 + 63.63 = 66.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 10, lower bound: -17.0010558, upper bound: 17.0010557

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1564

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9993466, upper bound: 16.9623909
time: 47.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.0002102, upper bound: 17.0002098
time: 47.09 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 94.74 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 94.74
Output dim: 10, lower bound: -16.9993466, upper bound: 16.9623909
IS_A2, status: Status.UNKNOWN, split count: 1, time: 94.74
Output dim: 10, lower bound: -17.0002102, upper bound: 17.0002098

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -34.3852196, 6.5745077, -34.4034462, 6.5861893, -40.8372498, 40.8434372
1: -9.2987766, 13.6536131, -9.3152952, 13.6671705, -22.9659462, 22.9689083
2: -8.3087730, 13.4466639, -8.3198986, 13.4592590, -20.8995209, 20.8980484
3: -10.0288115, 16.7872429, -10.0546503, 16.8112984, -25.7644119, 25.7659302
4: -16.3709831, 14.4108257, -16.3914833, 14.4261303, -30.7971134, 30.8023090
5: -9.3567572, 15.9651079, -9.3835449, 15.9881306, -25.0940781, 25.0975342
6: -35.7904205, -0.9015503, -35.8222847, -0.8511529, -31.7402420, 31.7216721
7: -9.4112701, 16.1623840, -9.4296608, 16.1767731, -23.9132004, 23.9178200
8: -19.9310188, 18.3484612, -19.9820404, 18.3859482, -38.3169670, 38.3305016
9: -6.8475161, 27.9836826, -6.8839760, 28.0039711, -32.1916504, 32.2072525
10: -5.5711994, 31.7818947, -5.6337700, 31.8239288, -34.6586456, 34.6808167
11: -11.3388042, 15.0849648, -11.3446274, 15.0938663, -26.4326706, 26.4295921
12: -18.3864994, 16.2626057, -18.4185486, 16.3058205, -29.1996155, 29.1901932
13: -17.8873215, 28.9236317, -17.9291267, 28.9772224, -44.1700439, 44.1564865
14: -26.7239399, 15.5393410, -26.7751350, 15.5596256, -37.9480667, 37.9810944
15: -19.5269566, 11.8423719, -19.5774155, 11.8780289, -31.4049854, 31.4197884
16: -15.5209970, 13.0481405, -15.5380878, 13.0680561, -27.9095001, 27.9068604
17: -28.7265930, 8.6970053, -28.7502747, 8.7061672, -28.4746323, 28.4886169
18: -29.0574150, 3.1644001, -29.0707684, 3.1780195, -26.4953918, 26.4979134
19: -16.0631542, 6.7907219, -16.0788116, 6.8057032, -22.0537987, 22.0541153
20: -11.9537697, 12.3916349, -11.9673557, 12.3997078, -23.7924881, 23.7972794
21: -15.2515068, 12.3087444, -15.2733593, 12.3166981, -27.5682049, 27.5821037
22: -17.0644379, 6.9263749, -17.0850697, 6.9337978, -22.8060684, 22.8189583
23: -15.3862858, 11.1584425, -15.3945961, 11.1688128, -26.5550995, 26.5530396
24: -27.3929234, 6.4464364, -27.4079628, 6.4541411, -31.2381744, 31.2447815
25: -17.4178047, 10.7028713, -17.4346504, 10.7107868, -27.2782745, 27.2853699
26: -22.1063423, 12.8245373, -22.1396999, 12.8476467, -32.8726540, 32.8829193
27: -28.2639923, 5.2255058, -28.2778568, 5.2384920, -28.5422058, 28.5451508
28: -15.4914751, 13.1583757, -15.5022097, 13.1724987, -27.3831787, 27.3794327
29: -15.6615028, 9.5225925, -15.6756725, 9.5292521, -24.7071533, 24.7152405
30: -17.5093880, 16.4467373, -17.5233059, 16.4547997, -32.3662033, 32.3778152
31: -26.9335251, 7.0970469, -26.9616261, 7.1208615, -33.0774994, 33.0790176
32: -26.1386547, 7.1094370, -26.1723461, 7.1531353, -30.4959641, 30.4850159
33: -64.3026810, -14.4641151, -64.3561554, -14.3875685, -44.4482422, 44.4238968
34: -51.9462433, -16.7221718, -51.9698486, -16.6875515, -27.1578445, 27.1475830
35: -46.4327774, -8.3781071, -46.4745407, -8.3142653, -34.0327606, 34.0101776
36: -40.0931473, -2.2847481, -40.1377640, -2.2204366, -36.2530670, 36.2327347
37: -67.2661896, -23.8482552, -67.3050690, -23.7980938, -35.4551544, 35.4408875
38: -50.5414963, -7.2445922, -50.5865326, -7.1739564, -40.6624603, 40.6361465
39: -61.2592621, -10.4151115, -61.3213387, -10.3302393, -48.1084290, 48.0839996
40: -52.4493217, -22.0749149, -52.4651871, -22.0463066, -24.1385574, 24.1203918
41: -37.7031403, -1.5480814, -37.7306480, -1.5024347, -30.6771317, 30.6593399
42: -24.2100830, 0.5762053, -24.2170372, 0.5917473, -23.0319633, 23.0143318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=269, inp2_unstable=270, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1564

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9967339, upper bound: 16.9271504
time: 52.85 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9967339, upper bound: 16.9597774
time: 49.61 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -34.4134674, 6.5880604, -34.4143524, 6.5907965, -40.8718948, 40.8753510
1: -9.3178473, 13.6794739, -9.3181057, 13.6802464, -22.9980927, 22.9975796
2: -8.3227625, 13.4660702, -8.3231239, 13.4667025, -20.9219513, 20.9213753
3: -10.0579014, 16.8324852, -10.0585823, 16.8331757, -25.8156052, 25.8092194
4: -16.3952675, 14.4425449, -16.3958035, 14.4432201, -30.8384876, 30.8383484
5: -9.3889294, 16.0068626, -9.3895130, 16.0075665, -25.1474075, 25.1449051
6: -35.8632965, -0.8498759, -35.8644333, -0.8487191, -31.8035507, 31.8150482
7: -9.4349556, 16.1925735, -9.4354839, 16.1932583, -23.9582214, 23.9560471
8: -19.9849148, 18.4293022, -19.9857807, 18.4305134, -38.4154282, 38.4150848
9: -6.8887806, 28.0284061, -6.8894010, 28.0291843, -32.2566223, 32.2478790
10: -5.6392727, 31.8735123, -5.6398735, 31.8752441, -34.7772522, 34.7674332
11: -11.3498001, 15.0975227, -11.3501873, 15.0984669, -26.4482670, 26.4477100
12: -18.4523792, 16.3117580, -18.4533310, 16.3128052, -29.2684402, 29.2748489
13: -17.9700565, 28.9804993, -17.9711876, 28.9812222, -44.2587891, 44.2597656
14: -26.7854118, 15.5869160, -26.7868347, 15.5877571, -38.0423203, 38.0383606
15: -19.5817413, 11.9235439, -19.5824318, 11.9248133, -31.5065536, 31.5059757
16: -15.5440865, 13.0791044, -15.5445614, 13.0803795, -27.9457169, 27.9391098
17: -28.7647400, 8.7083817, -28.7659187, 8.7091131, -28.5153275, 28.5180664
18: -29.0731888, 3.1841769, -29.0735970, 3.1850204, -26.5281982, 26.5244446
19: -16.0956059, 6.8071012, -16.0962029, 6.8073673, -22.0872345, 22.0895233
20: -11.9739380, 12.4017401, -11.9748592, 12.4018993, -23.8141098, 23.8166771
21: -15.2920513, 12.3187313, -15.2930374, 12.3187847, -27.6108360, 27.6117687
22: -17.0984669, 6.9359183, -17.0999527, 6.9360523, -22.8411407, 22.8467102
23: -15.3995419, 11.1703224, -15.3998480, 11.1713009, -26.5708427, 26.5701714
24: -27.4155293, 6.4568286, -27.4169788, 6.4570370, -31.2623062, 31.2639389
25: -17.4433327, 10.7125473, -17.4440670, 10.7127209, -27.3064270, 27.3090668
26: -22.1441021, 12.8676252, -22.1452637, 12.8682775, -32.9358139, 32.9313202
27: -28.2825508, 5.2491531, -28.2837811, 5.2494845, -28.5780945, 28.5775948
28: -15.5079832, 13.1768284, -15.5088253, 13.1770449, -27.4035721, 27.4051895
29: -15.6823320, 9.5331688, -15.6839819, 9.5335217, -24.7382736, 24.7384872
30: -17.5280037, 16.4607582, -17.5290318, 16.4609623, -32.4032974, 32.3975830
31: -26.9890347, 7.1230626, -26.9898548, 7.1235552, -33.1364212, 33.1394806
32: -26.2096786, 7.1545634, -26.2112236, 7.1554704, -30.5654602, 30.5701218
33: -64.4166489, -14.3845940, -64.4182434, -14.3836908, -44.5669708, 44.5682297
34: -51.9954758, -16.6837158, -51.9962158, -16.6832886, -27.2124023, 27.2154541
35: -46.5251923, -8.3121977, -46.5265808, -8.3117390, -34.1251144, 34.1296539
36: -40.1953125, -2.2183237, -40.1970139, -2.2178688, -36.3541260, 36.3583145
37: -67.3476257, -23.7957478, -67.3490143, -23.7951145, -35.5395126, 35.5432434
38: -50.6426163, -7.1710563, -50.6440544, -7.1703210, -40.7628708, 40.7659378
39: -61.3936844, -10.3278160, -61.3957787, -10.3272038, -48.2465515, 48.2481537
40: -52.4788284, -22.0445595, -52.4796829, -22.0438423, -24.1701355, 24.1761818
41: -37.7646484, -1.4997807, -37.7656097, -1.4987335, -30.7389069, 30.7425079
42: -24.2238941, 0.5958681, -24.2241669, 0.5965223, -23.0504150, 23.0614204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=269, inp2_unstable=270, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1564

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9975993, upper bound: 16.9649595
time: 161.69 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9967339, upper bound: 16.9597774
time: 68.25 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 232.22 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 232.22
Output dim: 10, lower bound: -16.9967339, upper bound: 16.9271504
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 232.22
Output dim: 10, lower bound: -16.9967339, upper bound: 16.9597774
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 232.22
Output dim: 10, lower bound: -16.9975993, upper bound: 16.9649595
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 232.22
Output dim: 10, lower bound: -16.9967339, upper bound: 16.9597774

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -34.3726540, 6.5724001, -34.3768501, 6.5817509, -40.8128510, 40.8120956
1: -9.2922802, 13.6525221, -9.3015013, 13.6649084, -22.9571877, 22.9540234
2: -8.3041344, 13.4455729, -8.3100882, 13.4570026, -20.8895950, 20.8860588
3: -10.0154552, 16.7862186, -10.0263386, 16.8092041, -25.7374954, 25.7365036
4: -16.3573132, 14.4097118, -16.3625240, 14.4238281, -30.7811413, 30.7722359
5: -9.3428669, 15.9642305, -9.3541069, 15.9863644, -25.0684738, 25.0669479
6: -35.7881699, -0.9031696, -35.8176270, -0.8545632, -31.7259445, 31.7090149
7: -9.4051762, 16.1617432, -9.4167223, 16.1755562, -23.9040375, 23.9043198
8: -19.9239922, 18.3472977, -19.9671249, 18.3836498, -38.3076401, 38.3144226
9: -6.8274088, 27.9825153, -6.8413186, 28.0015888, -32.1525192, 32.1633530
10: -5.5493884, 31.7802467, -5.5875926, 31.8205185, -34.6046143, 34.6328278
11: -11.3366337, 15.0694427, -11.3401356, 15.0609789, -26.3976135, 26.4095783
12: -18.3854065, 16.2595215, -18.4162483, 16.2993088, -29.1911240, 29.1803513
13: -17.8747406, 28.9203415, -17.9024773, 28.9703236, -44.1115570, 44.1270370
14: -26.7202854, 15.5373449, -26.7674179, 15.5555305, -37.9356003, 37.9694977
15: -19.5113220, 11.8405180, -19.5443382, 11.8741484, -31.3854713, 31.3848572
16: -15.5083313, 13.0456648, -15.5113087, 13.0628395, -27.8916550, 27.8781128
17: -28.7237682, 8.6928072, -28.7444305, 8.6975803, -28.4593506, 28.4776459
18: -29.0564003, 3.1592627, -29.0687160, 3.1674113, -26.4844360, 26.4825516
19: -16.0615807, 6.7779508, -16.0754929, 6.7786479, -22.0251923, 22.0165482
20: -11.9509602, 12.3808966, -11.9614201, 12.3769283, -23.7673454, 23.7572479
21: -15.2487440, 12.2941446, -15.2675772, 12.2861500, -27.5348930, 27.5617218
22: -17.0626068, 6.9198799, -17.0812149, 6.9200115, -22.7879028, 22.8031998
23: -15.3847694, 11.1374245, -15.3914280, 11.1242437, -26.5090141, 26.5288525
24: -27.3908157, 6.4268255, -27.4035645, 6.4125228, -31.1949921, 31.1864624
25: -17.4156876, 10.6858349, -17.4302158, 10.6746798, -27.2398682, 27.2584076
26: -22.1033535, 12.8115864, -22.1334171, 12.8207111, -32.8392792, 32.8284302
27: -28.2621746, 5.2105613, -28.2741222, 5.2067819, -28.5086708, 28.4692307
28: -15.4895449, 13.1411095, -15.4980774, 13.1359138, -27.3446579, 27.3378792
29: -15.6595745, 9.5145416, -15.6716051, 9.5121803, -24.6894226, 24.7021103
30: -17.5068169, 16.4307327, -17.5177803, 16.4208946, -32.3296661, 32.3417664
31: -26.9313698, 7.0842004, -26.9571342, 7.0935874, -33.0460052, 33.0431290
32: -26.1362457, 7.1079197, -26.1672611, 7.1499000, -30.4855499, 30.4772491
33: -64.3004150, -14.4735708, -64.3513489, -14.4076042, -44.4257202, 44.3917694
34: -51.9447670, -16.7246494, -51.9668655, -16.6927090, -27.1505890, 27.1369629
35: -46.4316177, -8.3860083, -46.4720268, -8.3310566, -34.0142746, 33.9901505
36: -40.0919266, -2.2934537, -40.1352425, -2.2388234, -36.2336578, 36.2170715
37: -67.2647552, -23.8628845, -67.3020935, -23.8291607, -35.4222031, 35.3981476
38: -50.5399475, -7.2526741, -50.5833054, -7.1909690, -40.6437759, 40.6182480
39: -61.2565994, -10.4208784, -61.3157387, -10.3421993, -48.0934143, 48.0692749
40: -52.4464569, -22.0773258, -52.4592361, -22.0515385, -24.1287155, 24.1283112
41: -37.7012787, -1.5548086, -37.7268066, -1.5165892, -30.6615448, 30.6438408
42: -24.2083549, 0.5710917, -24.2133675, 0.5809765, -23.0250282, 23.0040207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=269, inp2_unstable=269, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=291, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1564

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 658

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9662393, upper bound: 16.9263825
time: 56.05 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9959645, upper bound: 16.9263825
time: 48.00 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -34.3824196, 6.5729237, -34.4177780, 6.6546993, -40.9026871, 40.8521194
1: -9.2972641, 13.6529303, -9.3165913, 13.7040043, -23.0012684, 22.9695206
2: -8.3076267, 13.4452457, -8.3255720, 13.4896040, -20.9408875, 20.8908501
3: -10.0261621, 16.7863998, -10.0580864, 16.8973713, -25.8488159, 25.7645187
4: -16.3686752, 14.4092970, -16.3956642, 14.5071182, -30.8757935, 30.8049622
5: -9.3543167, 15.9643631, -9.3894501, 16.0770607, -25.1790237, 25.0980988
6: -35.7841415, -0.9021001, -35.8277054, -0.8454881, -31.7194366, 31.7470932
7: -9.4092207, 16.1613007, -9.4352283, 16.2118931, -23.9491882, 23.9173050
8: -19.9290466, 18.3478546, -19.9883881, 18.4250717, -38.3541183, 38.3362427
9: -6.8438854, 27.9828224, -6.8901253, 28.1171074, -32.3005676, 32.2067947
10: -5.5674706, 31.7802277, -5.6436501, 31.9534569, -34.7841339, 34.6807251
11: -11.3370447, 15.0818367, -11.4331226, 15.0958681, -26.4329128, 26.5149593
12: -18.3834991, 16.2615795, -18.4333839, 16.3228951, -29.1954727, 29.2196198
13: -17.8840141, 28.9218979, -17.9350109, 29.0448456, -44.2365265, 44.1587296
14: -26.7214241, 15.5374298, -26.7929783, 15.5786171, -37.9745026, 37.9949112
15: -19.5242405, 11.8406029, -19.5795879, 11.9683247, -31.4925652, 31.4201908
16: -15.5180473, 13.0470352, -15.5494699, 13.1325321, -27.9701462, 27.9145775
17: -28.7248306, 8.6951504, -28.7766914, 8.7169437, -28.4832001, 28.5101509
18: -29.0569935, 3.1565056, -29.1105614, 3.1786461, -26.4985847, 26.5395279
19: -16.0623245, 6.7882452, -16.1573963, 6.8044825, -22.0482330, 22.1331024
20: -11.9525194, 12.3896751, -12.0326214, 12.3996401, -23.7904968, 23.8634338
21: -15.2501554, 12.3051605, -15.3670120, 12.3149815, -27.5651360, 27.6721725
22: -17.0629177, 6.9235630, -17.1370659, 6.9344630, -22.8007507, 22.8739777
23: -15.3856802, 11.1546764, -15.5132217, 11.1686888, -26.5543690, 26.6678982
24: -27.3919678, 6.4426842, -27.5237141, 6.4539108, -31.2308578, 31.3582535
25: -17.4169979, 10.6990185, -17.5329361, 10.7102165, -27.2747421, 27.3793488
26: -22.1049824, 12.8189440, -22.2218590, 12.8509989, -32.8699532, 32.9677048
27: -28.2621269, 5.2231226, -28.3738937, 5.2406569, -28.5341797, 28.6390305
28: -15.4903593, 13.1554594, -15.5997667, 13.1734753, -27.3775558, 27.4733734
29: -15.6596594, 9.5205975, -15.7393131, 9.5315266, -24.7039604, 24.7797966
30: -17.5080681, 16.4439011, -17.6168556, 16.4570446, -32.3638306, 32.4695663
31: -26.9323788, 7.0941763, -27.0508690, 7.1220102, -33.0761414, 33.1677170
32: -26.1269264, 7.1091328, -26.1737080, 7.1596556, -30.4873962, 30.5069351
33: -64.3013077, -14.4669218, -64.4116058, -14.3825054, -44.4524841, 44.4791641
34: -51.9435883, -16.7230034, -51.9772949, -16.6802368, -27.1614723, 27.1595535
35: -46.4321213, -8.3798923, -46.5217896, -8.3098564, -34.0349884, 34.0555573
36: -40.0902939, -2.2861404, -40.1819496, -2.2179856, -36.2497559, 36.2781754
37: -67.2645035, -23.8512897, -67.3975296, -23.7927017, -35.4503098, 35.5289154
38: -50.5390739, -7.2462072, -50.6466675, -7.1620951, -40.6673126, 40.6964493
39: -61.2579460, -10.4166641, -61.3731308, -10.3285341, -48.1054993, 48.1349945
40: -52.4357071, -22.0755291, -52.4598236, -22.0342369, -24.1602554, 24.1190300
41: -37.6989250, -1.5496244, -37.7633057, -1.4947891, -30.6772232, 30.6918030
42: -24.2044296, 0.5750542, -24.2376709, 0.5990181, -23.0223694, 23.0504570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=269, inp2_unstable=269, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1564

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 658

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9662393, upper bound: 16.9590100
time: 39.79 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9959645, upper bound: 16.9590100
time: 900.65 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -34.4008942, 6.5859709, -34.3877411, 6.5863676, -40.8474808, 40.8440094
1: -9.3113232, 13.6784163, -9.3043251, 13.6779461, -22.9892693, 22.9827423
2: -8.3181400, 13.4649467, -8.3133307, 13.4644823, -20.9120178, 20.9093513
3: -10.0445461, 16.8314400, -10.0302448, 16.8310509, -25.7886887, 25.7798309
4: -16.3816109, 14.4414692, -16.3668289, 14.4409227, -30.8225327, 30.8082981
5: -9.3750391, 16.0060425, -9.3600473, 16.0058022, -25.1218719, 25.1142731
6: -35.8610497, -0.8514910, -35.8597984, -0.8521233, -31.7892075, 31.8024063
7: -9.4288826, 16.1919708, -9.4225979, 16.1920204, -23.9490433, 23.9425468
8: -19.9778404, 18.4281826, -19.9708652, 18.4282150, -38.4060555, 38.3990479
9: -6.8686528, 28.0272732, -6.8467755, 28.0268097, -32.2174911, 32.2039566
10: -5.6175146, 31.8719063, -5.5936899, 31.8718414, -34.7231903, 34.7194061
11: -11.3476372, 15.0820007, -11.3456335, 15.0655947, -26.4132309, 26.4276352
12: -18.4512768, 16.3086243, -18.4509926, 16.3063869, -29.2599030, 29.2650146
13: -17.9574642, 28.9771690, -17.9445496, 28.9742699, -44.2002869, 44.2303391
14: -26.7817535, 15.5849857, -26.7791405, 15.5835953, -38.0299606, 38.0268021
15: -19.5661163, 11.9216890, -19.5493431, 11.9209061, -31.4870224, 31.4710312
16: -15.5313997, 13.0766144, -15.5177383, 13.0751705, -27.9278488, 27.9103737
17: -28.7619934, 8.7042284, -28.7600880, 8.7004833, -28.5000000, 28.5070915
18: -29.0721207, 3.1790428, -29.0715027, 3.1744633, -26.5172234, 26.5090828
19: -16.0940247, 6.7943368, -16.0928898, 6.7803240, -22.0586319, 22.0519676
20: -11.9711285, 12.3910217, -11.9689293, 12.3791199, -23.7889786, 23.7766342
21: -15.2892933, 12.3041477, -15.2872686, 12.2882595, -27.5775528, 27.5914154
22: -17.0966358, 6.9294138, -17.0960960, 6.9222779, -22.8229828, 22.8309441
23: -15.3980265, 11.1492844, -15.3966980, 11.1267262, -26.5247536, 26.5459824
24: -27.4134636, 6.4371963, -27.4126015, 6.4154048, -31.2191391, 31.2056732
25: -17.4412441, 10.6954918, -17.4396515, 10.6765795, -27.2680206, 27.2821503
26: -22.1411228, 12.8546705, -22.1389904, 12.8413286, -32.9024773, 32.8768158
27: -28.2807961, 5.2342186, -28.2800217, 5.2177992, -28.5445251, 28.5016937
28: -15.5060558, 13.1595497, -15.5047073, 13.1404610, -27.3650513, 27.3635979
29: -15.6803970, 9.5251217, -15.6799583, 9.5164471, -24.7205963, 24.7253761
30: -17.5253868, 16.4447670, -17.5235138, 16.4270592, -32.3667068, 32.3615189
31: -26.9868507, 7.1101727, -26.9853859, 7.0962386, -33.1049347, 33.1036072
32: -26.2073040, 7.1530237, -26.2061729, 7.1522250, -30.5550308, 30.5623627
33: -64.4143372, -14.3939629, -64.4134216, -14.4037142, -44.5444946, 44.5360947
34: -51.9940033, -16.6861496, -51.9932327, -16.6884193, -27.2051697, 27.2048531
35: -46.5240364, -8.3201418, -46.5241089, -8.3285561, -34.1066284, 34.1096420
36: -40.1940842, -2.2270389, -40.1945038, -2.2362514, -36.3346786, 36.3426514
37: -67.3461838, -23.8104801, -67.3460922, -23.8262062, -35.5065918, 35.5005035
38: -50.6410789, -7.1791129, -50.6408539, -7.1874027, -40.7442245, 40.7480469
39: -61.3909912, -10.3335590, -61.3902168, -10.3391733, -48.2314301, 48.2334747
40: -52.4759827, -22.0470314, -52.4737625, -22.0490742, -24.1603165, 24.1841125
41: -37.7627563, -1.5064774, -37.7617645, -1.5128746, -30.7233505, 30.7269821
42: -24.2221527, 0.5907621, -24.2205143, 0.5857468, -23.0434837, 23.0511131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=269, inp2_unstable=269, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=291, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1564

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 658

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9662393, upper bound: 16.9641894
time: 64.15 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9968301, upper bound: 16.9641894
time: 45.98 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -34.4106865, 6.5864592, -34.4286652, 6.6593275, -40.9373627, 40.8840637
1: -9.3163223, 13.6787930, -9.3194275, 13.7170744, -23.0333977, 22.9982204
2: -8.3216200, 13.4646492, -8.3288174, 13.4970608, -20.9632874, 20.9141655
3: -10.0552197, 16.8316193, -10.0620022, 16.9192429, -25.8999710, 25.8078308
4: -16.3929386, 14.4410305, -16.3999939, 14.5242338, -30.9171715, 30.8410244
5: -9.3865175, 16.0061455, -9.3954182, 16.0964699, -25.2323761, 25.1454391
6: -35.8570251, -0.8504219, -35.8698273, -0.8430314, -31.7826691, 31.8404846
7: -9.4329462, 16.1915321, -9.4410410, 16.2283707, -23.9942245, 23.9555130
8: -19.9829540, 18.4287357, -19.9920959, 18.4696484, -38.4526024, 38.4208298
9: -6.8851070, 28.0275631, -6.8955650, 28.1423187, -32.3654938, 32.2474365
10: -5.6355667, 31.8718491, -5.6497459, 32.0047531, -34.9027710, 34.7673340
11: -11.3480358, 15.0944061, -11.4386187, 15.1004667, -26.4485016, 26.5330238
12: -18.4493980, 16.3107262, -18.4681435, 16.3299103, -29.2642593, 29.3042679
13: -17.9667511, 28.9787750, -17.9770851, 29.0488548, -44.3252716, 44.2619934
14: -26.7829285, 15.5849953, -26.8047028, 15.6067600, -38.0688095, 38.0522003
15: -19.5790138, 11.9217873, -19.5845985, 12.0150547, -31.5940685, 31.5063858
16: -15.5411167, 13.0779762, -15.5558949, 13.1448612, -28.0064011, 27.9468079
17: -28.7629871, 8.7065735, -28.7923088, 8.7198772, -28.5238342, 28.5395889
18: -29.0727310, 3.1763000, -29.1133804, 3.1856565, -26.5314026, 26.5660400
19: -16.0947838, 6.8046312, -16.1747665, 6.8061604, -22.0816841, 22.1685257
20: -11.9726429, 12.3997974, -12.0401125, 12.4018459, -23.8121033, 23.8828049
21: -15.2907200, 12.3151579, -15.3867054, 12.3170700, -27.6077900, 27.7018623
22: -17.0969086, 6.9330864, -17.1519737, 6.9367194, -22.8358002, 22.9017296
23: -15.3989639, 11.1665573, -15.5184584, 11.1711903, -26.5701542, 26.6850166
24: -27.4146061, 6.4530826, -27.5327339, 6.4567866, -31.2550507, 31.3774414
25: -17.4425488, 10.7086887, -17.5423813, 10.7121382, -27.3028870, 27.4030571
26: -22.1427402, 12.8620510, -22.2274323, 12.8716030, -32.9331284, 33.0161057
27: -28.2807007, 5.2467685, -28.3797913, 5.2516475, -28.5700836, 28.6714821
28: -15.5069017, 13.1739216, -15.6063824, 13.1780357, -27.3979721, 27.4991226
29: -15.6804457, 9.5311565, -15.7476416, 9.5357962, -24.7351074, 24.8030586
30: -17.5266705, 16.4579124, -17.6226044, 16.4631996, -32.4008789, 32.4893494
31: -26.9878178, 7.1201549, -27.0790825, 7.1246915, -33.1351013, 33.2281876
32: -26.1979847, 7.1542540, -26.2125835, 7.1619964, -30.5569229, 30.5920868
33: -64.4152374, -14.3874063, -64.4736938, -14.3786764, -44.5712433, 44.6234818
34: -51.9928627, -16.6845112, -52.0037079, -16.6759338, -27.2160339, 27.2274017
35: -46.5245018, -8.3140078, -46.5738678, -8.3073483, -34.1273651, 34.1750183
36: -40.1924744, -2.2197747, -40.2412186, -2.2154250, -36.3507919, 36.4038620
37: -67.3458405, -23.7988491, -67.4415512, -23.7897644, -35.5346298, 35.6312790
38: -50.6401787, -7.1726646, -50.7041855, -7.1585402, -40.7677612, 40.8262482
39: -61.3923302, -10.3293943, -61.4475632, -10.3255396, -48.2435608, 48.2992096
40: -52.4652405, -22.0451813, -52.4743080, -22.0317612, -24.1918564, 24.1748390
41: -37.7603989, -1.5013571, -37.7983093, -1.4910870, -30.7390289, 30.7749557
42: -24.2182465, 0.5947285, -24.2447910, 0.6037951, -23.0408287, 23.0975456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=269, inp2_unstable=269, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1564

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 658

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9662393, upper bound: 16.9968296
time: 48.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9968301, upper bound: 16.9968296
time: 60.50 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 111.18 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 111.18
Output dim: 10, lower bound: -16.9662393, upper bound: 16.9263825
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 111.18
Output dim: 10, lower bound: -16.9959645, upper bound: 16.9263825
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 111.18
Output dim: 10, lower bound: -16.9662393, upper bound: 16.9590100
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 111.18
Output dim: 10, lower bound: -16.9959645, upper bound: 16.9590100
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 111.18
Output dim: 10, lower bound: -16.9662393, upper bound: 16.9641894
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 111.18
Output dim: 10, lower bound: -16.9968301, upper bound: 16.9641894
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 111.18
Output dim: 10, lower bound: -16.9662393, upper bound: 16.9968296
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 111.18
Output dim: 10, lower bound: -16.9968301, upper bound: 16.9968296

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -34.3780212, 6.6219711, -34.3746223, 6.5808592, -40.8133698, 40.8587189
1: -9.2939568, 13.6796274, -9.3006325, 13.6643000, -22.9582558, 22.9802589
2: -8.3065767, 13.4644442, -8.3093090, 13.4564085, -20.8839111, 20.9095230
3: -10.0178690, 16.8232765, -10.0250025, 16.8083076, -25.7353516, 25.7720795
4: -16.3593655, 14.4594488, -16.3607864, 14.4225245, -30.7818909, 30.8202362
5: -9.3461761, 16.0052071, -9.3524418, 15.9853096, -25.0674820, 25.1063004
6: -35.7933426, -0.9007840, -35.8148537, -0.8555813, -31.7401810, 31.7112198
7: -9.4104948, 16.1909599, -9.4156294, 16.1748638, -23.9033203, 23.9352226
8: -19.9290619, 18.3786831, -19.9656944, 18.3828888, -38.3119507, 38.3443756
9: -6.8293338, 28.0370789, -6.8391571, 28.0010452, -32.1474304, 32.2148743
10: -5.5499926, 31.8565941, -5.5845060, 31.8191528, -34.5942154, 34.7059708
11: -11.3669481, 15.0726261, -11.3388710, 15.0595961, -26.4265442, 26.4114971
12: -18.4138088, 16.2667046, -18.4152260, 16.2983513, -29.2189789, 29.1841888
13: -17.9056320, 28.9252357, -17.9006367, 28.9684639, -44.1407318, 44.1297379
14: -26.7419987, 15.5405006, -26.7648010, 15.5548353, -37.9603271, 37.9812775
15: -19.5175533, 11.8914490, -19.5426941, 11.8727541, -31.3903084, 31.4341431
16: -15.5101242, 13.0907955, -15.5091858, 13.0615263, -27.8906631, 27.9213409
17: -28.7743435, 8.6925402, -28.7424736, 8.6950264, -28.5068359, 28.4687157
18: -29.0576248, 3.1698842, -29.0681992, 3.1653776, -26.4826126, 26.4944839
19: -16.0840340, 6.7779245, -16.0747051, 6.7775526, -22.0474777, 22.0142899
20: -11.9706888, 12.3818102, -11.9604063, 12.3763390, -23.7909012, 23.7550011
21: -15.2690001, 12.2947969, -15.2661810, 12.2844419, -27.5534420, 27.5609779
22: -17.0846062, 6.9214344, -17.0792465, 6.9191027, -22.8156052, 22.7964401
23: -15.4304886, 11.1386166, -15.3906031, 11.1223888, -26.5528774, 26.5292206
24: -27.4228134, 6.4291902, -27.4019432, 6.4111395, -31.2259598, 31.1828003
25: -17.4523849, 10.6878071, -17.4288979, 10.6729250, -27.2746887, 27.2575531
26: -22.1452045, 12.8128147, -22.1324348, 12.8177910, -32.8839531, 32.8253479
27: -28.2943840, 5.2104335, -28.2725716, 5.2047505, -28.5421600, 28.4633102
28: -15.5380726, 13.1434908, -15.4971581, 13.1342773, -27.3915100, 27.3347931
29: -15.6903934, 9.5156841, -15.6690207, 9.5113935, -24.7203522, 24.6988220
30: -17.5402946, 16.4338112, -17.5161037, 16.4195786, -32.3619461, 32.3367691
31: -26.9447708, 7.0855412, -26.9561691, 7.0924158, -33.0664825, 33.0376511
32: -26.1437721, 7.1144266, -26.1651001, 7.1492553, -30.5040817, 30.4812012
33: -64.3173141, -14.4677219, -64.3508453, -14.4086065, -44.4536743, 44.3894806
34: -51.9511108, -16.7192039, -51.9664955, -16.6936150, -27.1636887, 27.1410141
35: -46.4632492, -8.3839579, -46.4716110, -8.3325062, -34.0450592, 33.9906235
36: -40.1514740, -2.2926707, -40.1343346, -2.2402520, -36.2929611, 36.2138290
37: -67.3031464, -23.8605289, -67.3005676, -23.8306561, -35.4589233, 35.3923569
38: -50.5856934, -7.2507491, -50.5827713, -7.1924958, -40.6898499, 40.6165161
39: -61.2783203, -10.4178677, -61.3142433, -10.3433304, -48.1180573, 48.0702209
40: -52.4469681, -22.0496826, -52.4559479, -22.0520821, -24.1226883, 24.1560211
41: -37.7226639, -1.5525694, -37.7260780, -1.5183201, -30.6858139, 30.6451645
42: -24.2188053, 0.5751028, -24.2120857, 0.5804720, -23.0454750, 22.9984169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=268, inp2_unstable=269, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=291, inp2_unstable=290, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1564

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 575

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9391962, upper bound: 16.9243072
time: 67.17 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9948631, upper bound: 16.9244055
time: 50.05 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 119.49 seconds
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 119.49
Output dim: 10, lower bound: -16.9391962, upper bound: 16.9243072
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 119.49
Output dim: 10, lower bound: -16.9948631, upper bound: 16.9244055
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 119.49
Output dim: 10, lower bound: -16.9959645, upper bound: 16.9590100
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 119.49
Output dim: 10, lower bound: -16.9968301, upper bound: 16.9641894
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 119.49
Output dim: 10, lower bound: -16.9662393, upper bound: 16.9968296
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 119.49
Output dim: 10, lower bound: -16.9968301, upper bound: 16.9968296

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 66.47 + 1824.05 = 1890.52 seconds
