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
execution time: IAR + RelationalAnalysis = 2.90 + 63.37 = 66.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 10, lower bound: -17.0010558, upper bound: 17.0010557

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1021
type: B, layer: 1, pos: 1021
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1546
type: B, layer: 1, pos: 1546
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 627

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9658054, upper bound: 16.9984449
time: 48.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9984447, upper bound: 16.9984449
time: 131.20 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 180.22 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 180.22
Output dim: 10, lower bound: -16.9658054, upper bound: 16.9984449
IS_A2, status: Status.UNKNOWN, split count: 1, time: 180.22
Output dim: 10, lower bound: -16.9984447, upper bound: 16.9984449

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -34.3889809, 6.5903082, -34.4029999, 6.5926514, -40.8469162, 40.8538132
1: -9.3047352, 13.6790009, -9.3119774, 13.6801672, -22.9849014, 22.9909782
2: -8.3138332, 13.4654274, -8.3189945, 13.4665871, -20.9119415, 20.9140549
3: -10.0311680, 16.8320007, -10.0461025, 16.8331108, -25.7887421, 25.7912521
4: -16.3675766, 14.4418364, -16.3828335, 14.4430504, -30.8106270, 30.8246689
5: -9.3608332, 16.0067062, -9.3764019, 16.0076389, -25.1192245, 25.1242523
6: -35.8613434, -0.8505621, -35.8637390, -0.8487654, -31.8067245, 31.8050613
7: -9.4232979, 16.1929245, -9.4300995, 16.1935768, -23.9456787, 23.9500351
8: -19.9720249, 18.4298553, -19.9798622, 18.4310341, -38.4030609, 38.4097176
9: -6.8476863, 28.0278111, -6.8701687, 28.0290833, -32.2152863, 32.2200546
10: -5.5945582, 31.8741684, -5.6189375, 31.8759174, -34.7332687, 34.7272873
11: -11.3462677, 15.0669689, -11.3486156, 15.0843315, -26.4305992, 26.4155846
12: -18.4522572, 16.3079014, -18.4535103, 16.3112755, -29.2689285, 29.2702332
13: -17.9461365, 28.9753418, -17.9601936, 28.9789467, -44.2337036, 44.2046204
14: -26.7811432, 15.5846806, -26.7851677, 15.5868416, -38.0356979, 38.0348892
15: -19.5503235, 11.9226685, -19.5677872, 11.9246664, -31.4749908, 31.4904556
16: -15.5184317, 13.0768967, -15.5325708, 13.0796337, -27.9203873, 27.9312668
17: -28.7616081, 8.7014980, -28.7647095, 8.7059422, -28.5105515, 28.5062332
18: -29.0720692, 3.1758180, -29.0730629, 3.1812258, -26.5145798, 26.5189819
19: -16.0937366, 6.7807498, -16.0954514, 6.7950211, -22.0535812, 22.0625229
20: -11.9702168, 12.3793306, -11.9733286, 12.3913841, -23.7784233, 23.7933121
21: -15.2886877, 12.2883854, -15.2916946, 12.3043346, -27.5930214, 27.5800800
22: -17.0983124, 6.9224768, -17.1002960, 6.9297686, -22.8334808, 22.8310776
23: -15.3971481, 11.1280613, -15.3987970, 11.1515980, -26.5487461, 26.5268593
24: -27.4145412, 6.4156785, -27.4168739, 6.4376817, -31.2088547, 31.2239914
25: -17.4406433, 10.6768208, -17.4429855, 10.6958761, -27.2834625, 27.2719803
26: -22.1405487, 12.8422050, -22.1438599, 12.8561583, -32.8850517, 32.9061661
27: -28.2816772, 5.2182455, -28.2835960, 5.2349930, -28.5047150, 28.5470734
28: -15.5057812, 13.1407824, -15.5079508, 13.1601048, -27.3656693, 27.3686790
29: -15.6822596, 9.5169449, -15.6843386, 9.5259647, -24.7284927, 24.7239037
30: -17.5249519, 16.4273396, -17.5278339, 16.4453011, -32.3657227, 32.3651581
31: -26.9865589, 7.0969243, -26.9888897, 7.1113706, -33.1042633, 33.1085892
32: -26.2082863, 7.1534610, -26.2108994, 7.1551690, -30.5667496, 30.5640755
33: -64.4155731, -14.4025154, -64.4180222, -14.3919468, -44.5406952, 44.5502777
34: -51.9942780, -16.6878128, -51.9958344, -16.6851330, -27.2072906, 27.2106323
35: -46.5259514, -8.3279858, -46.5272827, -8.3190794, -34.1130981, 34.1146469
36: -40.1968307, -2.2355814, -40.1981354, -2.2259011, -36.3462677, 36.3424759
37: -67.3480530, -23.8252888, -67.3496246, -23.8088856, -35.5023041, 35.5120926
38: -50.6427345, -7.1864958, -50.6443939, -7.1775012, -40.7523117, 40.7515945
39: -61.3931236, -10.3382444, -61.3959808, -10.3320351, -48.2384949, 48.2380981
40: -52.4749870, -22.0481339, -52.4780655, -22.0454025, -24.1829758, 24.1652145
41: -37.7631874, -1.5114603, -37.7651176, -1.5039845, -30.7309952, 30.7309341
42: -24.2209167, 0.5866337, -24.2228279, 0.5922985, -23.0450363, 23.0484161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=269, inp2_unstable=270, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=291, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1021
type: A, layer: 1, pos: 1021
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1546
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 658

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9352826, upper bound: 16.9976760
time: 233.17 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9352826, upper bound: 16.9976760
time: 57.49 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -34.4299164, 6.6632814, -34.4127922, 6.5931568, -40.8869400, 40.9436646
1: -9.3198318, 13.7180796, -9.3169804, 13.6805563, -23.0003891, 23.0350609
2: -8.3293085, 13.4980192, -8.3224878, 13.4662390, -20.9167786, 20.9653244
3: -10.0629406, 16.9201965, -10.0568209, 16.8332710, -25.8167725, 25.9025497
4: -16.4007092, 14.5251484, -16.3941994, 14.4426193, -30.8433285, 30.9193478
5: -9.3962183, 16.0974293, -9.3878632, 16.0077591, -25.1503983, 25.2347870
6: -35.8713760, -0.8414450, -35.8597221, -0.8476710, -31.8447952, 31.7985382
7: -9.4417801, 16.2292786, -9.4341698, 16.1931076, -23.9586639, 23.9951782
8: -19.9932518, 18.4712906, -19.9849625, 18.4315853, -38.4248352, 38.4562531
9: -6.8964624, 28.1433754, -6.8866324, 28.0293732, -32.2587357, 32.3680573
10: -5.6505957, 32.0070953, -5.6370096, 31.8758698, -34.7811813, 34.9068375
11: -11.4392557, 15.1018562, -11.3490105, 15.0967093, -26.5359650, 26.4508667
12: -18.4693794, 16.3314514, -18.4515572, 16.3133163, -29.3081970, 29.2745972
13: -17.9787064, 29.0498505, -17.9694366, 28.9805641, -44.2654419, 44.3296661
14: -26.8067017, 15.6077938, -26.7863731, 15.5868416, -38.0611496, 38.0737305
15: -19.5855789, 12.0168152, -19.5806980, 11.9247818, -31.5103607, 31.5975132
16: -15.5565920, 13.1466265, -15.5423040, 13.0809984, -27.9567871, 28.0097885
17: -28.7938766, 8.7209015, -28.7656746, 8.7082253, -28.5430679, 28.5300598
18: -29.1139488, 3.1869669, -29.0736847, 3.1784811, -26.5715370, 26.5331688
19: -16.1756058, 6.8065519, -16.0962067, 6.8052983, -22.1701126, 22.0855637
20: -12.0414238, 12.4020576, -11.9748726, 12.4001665, -23.8845749, 23.8164444
21: -15.3881016, 12.3171959, -15.2931156, 12.3153534, -27.7034550, 27.6103115
22: -17.1541653, 6.9369454, -17.1005859, 6.9334517, -22.9042892, 22.8438950
23: -15.5189028, 11.1725388, -15.3997059, 11.1688576, -26.6877594, 26.5722446
24: -27.5347118, 6.4570818, -27.4180145, 6.4535317, -31.3806763, 31.2599030
25: -17.5433903, 10.7123756, -17.4442787, 10.7090836, -27.4043579, 27.3068390
26: -22.2290382, 12.8724718, -22.1454697, 12.8635120, -33.0243301, 32.9368439
27: -28.3814507, 5.2520986, -28.2835236, 5.2475152, -28.6745262, 28.5726318
28: -15.6074371, 13.1783504, -15.5087414, 13.1744595, -27.5011826, 27.4016647
29: -15.7499132, 9.5362968, -15.6843977, 9.5320015, -24.8061676, 24.7384109
30: -17.6240578, 16.4634819, -17.5291080, 16.4584408, -32.4935455, 32.3993835
31: -27.0802612, 7.1253519, -26.9898567, 7.1213384, -33.2288589, 33.1387787
32: -26.2146950, 7.1632447, -26.2016182, 7.1563849, -30.5964050, 30.5659790
33: -64.4758301, -14.3774557, -64.4189758, -14.3852882, -44.6280518, 44.5770569
34: -52.0047379, -16.6753349, -51.9946632, -16.6834908, -27.2298355, 27.2215195
35: -46.5757446, -8.3067017, -46.5277596, -8.3129578, -34.1785049, 34.1354141
36: -40.2435532, -2.2147403, -40.1965103, -2.2186112, -36.4074326, 36.3585663
37: -67.4435272, -23.7888584, -67.3493195, -23.7972832, -35.6330795, 35.5402069
38: -50.7061386, -7.1575298, -50.6435547, -7.1710491, -40.8305588, 40.7751465
39: -61.4504547, -10.3246288, -61.3973770, -10.3278732, -48.3042145, 48.2502136
40: -52.4755249, -22.0308666, -52.4673576, -22.0435543, -24.1736908, 24.1967430
41: -37.7996521, -1.4896297, -37.7627831, -1.4988317, -30.7789307, 30.7465591
42: -24.2452049, 0.6046667, -24.2188969, 0.5962868, -23.0914879, 23.0457764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=269, inp2_unstable=270, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1021
type: B, layer: 1, pos: 1021
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1546
type: A, layer: 1, pos: 1546
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 658

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9976757, upper bound: 16.9679429
time: 49.04 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9976757, upper bound: 16.9976758
time: 58.26 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 109.62 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 109.62
Output dim: 10, lower bound: -16.9352826, upper bound: 16.9976760
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 109.62
Output dim: 10, lower bound: -16.9352826, upper bound: 16.9976760
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 109.62
Output dim: 10, lower bound: -16.9976757, upper bound: 16.9679429
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 109.62
Output dim: 10, lower bound: -16.9976757, upper bound: 16.9976758

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -34.3541718, 6.5832930, -34.3825340, 6.5884838, -40.8055344, 40.8245163
1: -9.2904243, 13.6754313, -9.3035583, 13.6780987, -22.9685230, 22.9789886
2: -8.3055763, 13.4614382, -8.3141136, 13.4642477, -20.8956451, 20.9013672
3: -10.0038338, 16.8286209, -10.0300274, 16.8311005, -25.7593689, 25.7715378
4: -16.3319855, 14.4376202, -16.3619080, 14.4405432, -30.7725296, 30.7995281
5: -9.3281727, 16.0036469, -9.3571749, 16.0058231, -25.0847778, 25.1019211
6: -35.8527794, -0.8559456, -35.8586655, -0.8519626, -31.7937546, 31.7931442
7: -9.4064131, 16.1899681, -9.4201546, 16.1918087, -23.9224243, 23.9340782
8: -19.9493713, 18.4252892, -19.9665451, 18.4283924, -38.3777618, 38.3918343
9: -6.8030429, 28.0245857, -6.8439364, 28.0271664, -32.1677399, 32.1892929
10: -5.5370007, 31.8679218, -5.5850978, 31.8722305, -34.6715317, 34.6867065
11: -11.3388405, 15.0421314, -11.3442240, 15.0697079, -26.4085484, 26.3863564
12: -18.4486847, 16.2903786, -18.4514313, 16.3008194, -29.2540894, 29.2491989
13: -17.9377480, 28.9598579, -17.9552250, 28.9698048, -44.2156677, 44.1834641
14: -26.7684746, 15.5786886, -26.7776852, 15.5832558, -38.0133820, 38.0134277
15: -19.5093651, 11.9155245, -19.5437126, 11.9204445, -31.4298096, 31.4592361
16: -15.4886246, 13.0681839, -15.5150127, 13.0745106, -27.8849945, 27.9046707
17: -28.7549973, 8.6625147, -28.7608032, 8.6830120, -28.4796143, 28.4617615
18: -29.0690765, 3.1684313, -29.0713253, 3.1768732, -26.5063896, 26.5098495
19: -16.0893536, 6.7649221, -16.0928764, 6.7856960, -22.0392227, 22.0427437
20: -11.9618416, 12.3690014, -11.9683971, 12.3852119, -23.7639923, 23.7768402
21: -15.2803974, 12.2796822, -15.2868509, 12.2991629, -27.5795593, 27.5665321
22: -17.0896759, 6.9137158, -17.0952415, 6.9245319, -22.8177567, 22.8134384
23: -15.3932800, 11.0897408, -15.3965149, 11.1290503, -26.5223312, 26.4862556
24: -27.4081688, 6.3892798, -27.4130669, 6.4221406, -31.1876221, 31.1941833
25: -17.4341583, 10.6453629, -17.4391365, 10.6773624, -27.2590790, 27.2369003
26: -22.1300163, 12.8233404, -22.1376514, 12.8450546, -32.8638649, 32.8799896
27: -28.2736969, 5.2040935, -28.2789173, 5.2266440, -28.4872055, 28.5261269
28: -15.5000420, 13.1068783, -15.5045834, 13.1401730, -27.3403473, 27.3313446
29: -15.6733961, 9.5075026, -15.6791401, 9.5203762, -24.7133255, 24.7081032
30: -17.5166245, 16.4012222, -17.5229301, 16.4299011, -32.3421936, 32.3336487
31: -26.9816704, 7.0894408, -26.9859982, 7.1069546, -33.0908127, 33.0918121
32: -26.1994667, 7.1487041, -26.2057171, 7.1523643, -30.5519180, 30.5511856
33: -64.4105759, -14.4188004, -64.4151840, -14.4015512, -44.5224609, 44.5249710
34: -51.9909706, -16.6961441, -51.9938965, -16.6900368, -27.1976776, 27.1989899
35: -46.5224648, -8.3520784, -46.5252075, -8.3332949, -34.0951538, 34.0876617
36: -40.1914139, -2.2674675, -40.1949730, -2.2446589, -36.3223038, 36.3074417
37: -67.3433838, -23.8538513, -67.3468399, -23.8256798, -35.4806061, 35.4806519
38: -50.6374359, -7.2107716, -50.6413460, -7.1919303, -40.7324142, 40.7238846
39: -61.3869934, -10.3557901, -61.3924332, -10.3423471, -48.2220306, 48.2166901
40: -52.4527512, -22.0550461, -52.4649887, -22.0494537, -24.1552887, 24.1452675
41: -37.7573929, -1.5213900, -37.7617760, -1.5098448, -30.7186813, 30.7169991
42: -24.2162628, 0.5791664, -24.2200394, 0.5879002, -23.0331726, 23.0333099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=268, inp2_unstable=270, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=291, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1021
type: A, layer: 1, pos: 1021
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1546
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 904

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9241549, upper bound: 16.9795529
time: 40.64 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9346401, upper bound: 16.9970228
time: 41.93 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -34.3943634, 6.6398830, -34.4007759, 6.5916967, -40.8475418, 40.9004288
1: -9.3063850, 13.7060671, -9.3111134, 13.6795540, -22.9859390, 23.0171814
2: -8.3162823, 13.4842844, -8.3181973, 13.4659653, -20.9062271, 20.9375267
3: -10.0335732, 16.8690720, -10.0447836, 16.8322029, -25.7866364, 25.8268204
4: -16.3696022, 14.4915628, -16.3810959, 14.4417667, -30.8113689, 30.8726578
5: -9.3641558, 16.0477047, -9.3747454, 16.0065804, -25.1182098, 25.1636505
6: -35.8665352, -0.8481379, -35.8609390, -0.8497677, -31.8209000, 31.8072357
7: -9.4286070, 16.2221527, -9.4290123, 16.1928787, -23.9449615, 23.9809418
8: -19.9770718, 18.4612064, -19.9784241, 18.4302464, -38.4073181, 38.4396286
9: -6.8495846, 28.0823536, -6.8679800, 28.0285568, -32.2101974, 32.2715912
10: -5.5951352, 31.9504890, -5.6158590, 31.8745098, -34.7228546, 34.8004227
11: -11.3765926, 15.0701609, -11.3473778, 15.0829239, -26.4595165, 26.4175377
12: -18.4807224, 16.3150673, -18.4524899, 16.3102646, -29.2968063, 29.2740173
13: -17.9769821, 28.9802151, -17.9583588, 28.9770584, -44.2628784, 44.2073059
14: -26.8028393, 15.5878515, -26.7825603, 15.5861177, -38.0604248, 38.0465164
15: -19.5565491, 11.9735918, -19.5661736, 11.9233208, -31.4798698, 31.5397644
16: -15.5202427, 13.1220331, -15.5304241, 13.0783262, -27.9193878, 27.9745522
17: -28.8121910, 8.7012482, -28.7628288, 8.7033796, -28.5580597, 28.4973030
18: -29.0732536, 3.1865063, -29.0725765, 3.1791668, -26.5127602, 26.5309219
19: -16.1161957, 6.7806950, -16.0946693, 6.7939024, -22.0758514, 22.0602531
20: -11.9899645, 12.3802509, -11.9723158, 12.3907890, -23.8019485, 23.7910500
21: -15.3089342, 12.2890511, -15.2903214, 12.3026161, -27.6115494, 27.5793724
22: -17.1203079, 6.9240465, -17.0983276, 6.9288678, -22.8612099, 22.8243141
23: -15.4428520, 11.1292591, -15.3979712, 11.1497307, -26.5925827, 26.5272293
24: -27.4465294, 6.4180694, -27.4151955, 6.4363041, -31.2398453, 31.2202835
25: -17.4773674, 10.6787958, -17.4416637, 10.6941681, -27.3183060, 27.2711258
26: -22.1823502, 12.8434696, -22.1428852, 12.8532295, -32.9296341, 32.9030838
27: -28.3138084, 5.2181239, -28.2820511, 5.2329350, -28.5382080, 28.5411873
28: -15.5543156, 13.1431170, -15.5070171, 13.1584702, -27.4125443, 27.3656158
29: -15.7130318, 9.5181046, -15.6817188, 9.5251675, -24.7593918, 24.7206192
30: -17.5583935, 16.4304352, -17.5261307, 16.4439564, -32.3980331, 32.3601761
31: -26.9999657, 7.0982666, -26.9879074, 7.1102333, -33.1247253, 33.1031570
32: -26.2158165, 7.1599698, -26.2086906, 7.1545076, -30.5852509, 30.5679855
33: -64.4324951, -14.3966503, -64.4175720, -14.3929291, -44.5686340, 44.5480270
34: -52.0006104, -16.6824150, -51.9954491, -16.6860504, -27.2203598, 27.2146530
35: -46.5576019, -8.3259220, -46.5268478, -8.3205166, -34.1438751, 34.1150436
36: -40.2564163, -2.2347927, -40.1972542, -2.2273431, -36.4055481, 36.3392563
37: -67.3864746, -23.8228168, -67.3480988, -23.8103447, -35.5390549, 35.5063248
38: -50.6885071, -7.1845112, -50.6438675, -7.1789908, -40.7984390, 40.7498703
39: -61.4148064, -10.3352947, -61.3945389, -10.3331795, -48.2631989, 48.2390442
40: -52.4754524, -22.0203953, -52.4748001, -22.0459919, -24.1769638, 24.1929092
41: -37.7845573, -1.5092010, -37.7643700, -1.5057297, -30.7552032, 30.7322350
42: -24.2313404, 0.5906296, -24.2215385, 0.5917702, -23.0654755, 23.0428085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=268, inp2_unstable=270, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=290, inp2_unstable=291, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1021
type: B, layer: 1, pos: 1021
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1546
type: A, layer: 1, pos: 1546
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 904

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9241549, upper bound: 16.9795529
time: 54.53 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9643841, upper bound: 16.9970228
time: 54.47 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -34.4094582, 6.6591425, -34.3780060, 6.5861168, -40.8577118, 40.9023285
1: -9.3113861, 13.7160368, -9.3026810, 13.6770058, -22.9883919, 23.0187187
2: -8.3244114, 13.4956732, -8.3142185, 13.4622641, -20.9041061, 20.9490089
3: -10.0468245, 16.9182129, -10.0294476, 16.8298721, -25.7970505, 25.8731995
4: -16.3797531, 14.5225983, -16.3586369, 14.4383554, -30.8181076, 30.8812351
5: -9.3769560, 16.0955982, -9.3552227, 16.0046749, -25.1280060, 25.2003632
6: -35.8663025, -0.8446460, -35.8511505, -0.8530793, -31.8329086, 31.7855148
7: -9.4317837, 16.2275238, -9.4173012, 16.1901646, -23.9427032, 23.9719315
8: -19.9799385, 18.4686165, -19.9623165, 18.4270287, -38.4069672, 38.4309311
9: -6.8702083, 28.1414986, -6.8419867, 28.0261555, -32.2279663, 32.3205643
10: -5.6167784, 32.0034294, -5.5794821, 31.8696365, -34.7406158, 34.8451309
11: -11.4348307, 15.0872402, -11.3416214, 15.0718689, -26.5066986, 26.4288616
12: -18.4673538, 16.3210545, -18.4480534, 16.2957878, -29.2871704, 29.2596970
13: -17.9737206, 29.0407925, -17.9610786, 28.9650555, -44.2441101, 44.3117905
14: -26.7992020, 15.6042099, -26.7736931, 15.5808582, -38.0397263, 38.0514755
15: -19.5615044, 12.0126057, -19.5397434, 11.9176416, -31.4791451, 31.5523491
16: -15.5390205, 13.1414909, -15.5124559, 13.0722580, -27.9301987, 27.9744110
17: -28.7899666, 8.6980124, -28.7590847, 8.6693125, -28.4985580, 28.4990997
18: -29.1121883, 3.1825957, -29.0706635, 3.1710958, -26.5623627, 26.5249252
19: -16.1730423, 6.7972355, -16.0918388, 6.7894993, -22.1503410, 22.0712280
20: -12.0364971, 12.3958969, -11.9664946, 12.3898182, -23.8681412, 23.8019981
21: -15.3832521, 12.3120327, -15.2848606, 12.3066406, -27.6898918, 27.5968933
22: -17.1491394, 6.9316907, -17.0919762, 6.9246674, -22.8866653, 22.8281784
23: -15.5166349, 11.1499672, -15.3958616, 11.1305552, -26.6471901, 26.5458298
24: -27.5309849, 6.4415388, -27.4116306, 6.4271655, -31.3508606, 31.2385788
25: -17.5395813, 10.6937809, -17.4377899, 10.6776237, -27.3693542, 27.2824669
26: -22.2228527, 12.8613319, -22.1349602, 12.8446312, -32.9981384, 32.9156952
27: -28.3767891, 5.2437882, -28.2755737, 5.2333713, -28.6535683, 28.5550919
28: -15.6041031, 13.1583862, -15.5030394, 13.1405783, -27.4638214, 27.3762894
29: -15.7447586, 9.5306969, -15.6755781, 9.5225658, -24.7903709, 24.7232857
30: -17.6191406, 16.4481239, -17.5207558, 16.4323025, -32.4620514, 32.3758240
31: -27.0774097, 7.1209240, -26.9849720, 7.1138463, -33.2120438, 33.1253586
32: -26.2095070, 7.1604509, -26.1928730, 7.1516600, -30.5836258, 30.5511551
33: -64.4728851, -14.3869991, -64.4140167, -14.4015522, -44.6027069, 44.5587387
34: -52.0028076, -16.6802654, -51.9913406, -16.6917973, -27.2182159, 27.2119064
35: -46.5736847, -8.3209352, -46.5242691, -8.3370466, -34.1515198, 34.1174088
36: -40.2403870, -2.2335119, -40.1911240, -2.2504816, -36.3724136, 36.3346481
37: -67.4408188, -23.8056335, -67.3446960, -23.8257885, -35.6016541, 35.5184860
38: -50.7029877, -7.1720357, -50.6381989, -7.1953506, -40.8027954, 40.7551727
39: -61.4468842, -10.3349934, -61.3912659, -10.3453989, -48.2827911, 48.2337799
40: -52.4624519, -22.0349426, -52.4451218, -22.0505142, -24.1537628, 24.1690483
41: -37.7963181, -1.4954844, -37.7570381, -1.5088282, -30.7650681, 30.7343369
42: -24.2424393, 0.6002522, -24.2142563, 0.5887909, -23.0763931, 23.0339127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=269, inp2_unstable=269, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1021
type: B, layer: 1, pos: 1021
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1546
type: A, layer: 1, pos: 1546
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9795526, upper bound: 16.9567470
time: 45.80 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9970226, upper bound: 16.9672934
time: 54.99 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -34.4276695, 6.6623125, -34.4181671, 6.6427364, -40.9335556, 40.9442978
1: -9.3189392, 13.7174969, -9.3186455, 13.7076550, -23.0265942, 23.0361423
2: -8.3285427, 13.4973879, -8.3249454, 13.4851112, -20.9402618, 20.9596138
3: -10.0615988, 16.9192867, -10.0591908, 16.8703213, -25.8523331, 25.9004135
4: -16.3989658, 14.5238752, -16.3962517, 14.4923182, -30.8912849, 30.9201279
5: -9.3945827, 16.0963631, -9.3911896, 16.0487366, -25.1898117, 25.2338257
6: -35.8685684, -0.8424835, -35.8649178, -0.8452539, -31.8469620, 31.8126984
7: -9.4407063, 16.2285881, -9.4394941, 16.2223415, -23.9895706, 23.9944687
8: -19.9918518, 18.4704857, -19.9900150, 18.4629517, -38.4548035, 38.4605026
9: -6.8942838, 28.1428642, -6.8885136, 28.0838623, -32.3102798, 32.3629761
10: -5.6475315, 32.0057449, -5.6375914, 31.9522171, -34.8543396, 34.8964310
11: -11.4379778, 15.1004543, -11.3793631, 15.0998955, -26.5378723, 26.4798164
12: -18.4684525, 16.3304634, -18.4800644, 16.3205185, -29.3119965, 29.3024597
13: -17.9768677, 29.0480194, -18.0002937, 28.9854679, -44.2681580, 44.3588562
14: -26.8040504, 15.6070900, -26.8080654, 15.5900450, -38.0728302, 38.0983887
15: -19.5839157, 12.0154791, -19.5869083, 11.9757128, -31.5596275, 31.6023865
16: -15.5545063, 13.1452885, -15.5440969, 13.1261215, -28.0000763, 28.0088272
17: -28.7919426, 8.7183361, -28.8162460, 8.7079773, -28.5341339, 28.5775795
18: -29.1134224, 3.1849108, -29.0748863, 3.1891279, -26.5834160, 26.5313568
19: -16.1748524, 6.8054619, -16.1186676, 6.8052754, -22.1678352, 22.1078568
20: -12.0403709, 12.4014673, -11.9946041, 12.4010563, -23.8823395, 23.8400269
21: -15.3867044, 12.3154678, -15.3133869, 12.3160334, -27.7027378, 27.6288548
22: -17.1522026, 6.9360547, -17.1226044, 6.9349980, -22.8975372, 22.8716507
23: -15.5180721, 11.1706657, -15.4454403, 11.1700859, -26.6881580, 26.6161060
24: -27.5330830, 6.4557319, -27.4500237, 6.4559617, -31.3769684, 31.2908783
25: -17.5420570, 10.7106819, -17.4810066, 10.7110500, -27.4035645, 27.3417206
26: -22.2280960, 12.8695488, -22.1872749, 12.8646860, -33.0212364, 32.9814453
27: -28.3798752, 5.2500696, -28.3156986, 5.2473965, -28.6686325, 28.6061172
28: -15.6065140, 13.1767187, -15.5572538, 13.1767969, -27.4981079, 27.4485016
29: -15.7473221, 9.5355320, -15.7151985, 9.5331745, -24.8028526, 24.7693634
30: -17.6223450, 16.4621639, -17.5625763, 16.4615307, -32.4885483, 32.4316864
31: -27.0793152, 7.1242642, -27.0033226, 7.1226735, -33.2234039, 33.1592865
32: -26.2124996, 7.1626129, -26.2092018, 7.1629019, -30.6003723, 30.5844040
33: -64.4753418, -14.3784895, -64.4359589, -14.3794489, -44.6257782, 44.6049805
34: -52.0043755, -16.6762848, -52.0009842, -16.6781120, -27.2338867, 27.2346039
35: -46.5753136, -8.3081684, -46.5593605, -8.3109159, -34.1789627, 34.1662064
36: -40.2426605, -2.2161489, -40.2560921, -2.2178154, -36.4042206, 36.4179306
37: -67.4419785, -23.7903137, -67.3877411, -23.7948151, -35.6273270, 35.5769196
38: -50.7055511, -7.1590309, -50.6892891, -7.1691151, -40.8288040, 40.8212204
39: -61.4489594, -10.3257370, -61.4191437, -10.3248911, -48.3051300, 48.2749023
40: -52.4722519, -22.0314102, -52.4678879, -22.0158882, -24.2014236, 24.1907082
41: -37.7989120, -1.4913816, -37.7842102, -1.4966230, -30.7802582, 30.7708244
42: -24.2439384, 0.6041508, -24.2293377, 0.6002789, -23.0858841, 23.0661926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=269, inp2_unstable=269, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=291, inp2_unstable=291, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1021
type: B, layer: 1, pos: 1021
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1546
type: A, layer: 1, pos: 1546
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9795526, upper bound: 16.9864750
time: 60.72 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9970226, upper bound: 16.9970228
time: 46.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 109.51 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 109.51
Output dim: 10, lower bound: -16.9241549, upper bound: 16.9795529
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 109.51
Output dim: 10, lower bound: -16.9346401, upper bound: 16.9970228
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 109.51
Output dim: 10, lower bound: -16.9241549, upper bound: 16.9795529
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 109.51
Output dim: 10, lower bound: -16.9643841, upper bound: 16.9970228
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 109.51
Output dim: 10, lower bound: -16.9795526, upper bound: 16.9567470
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 109.51
Output dim: 10, lower bound: -16.9970226, upper bound: 16.9672934
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 109.51
Output dim: 10, lower bound: -16.9795526, upper bound: 16.9864750
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 109.51
Output dim: 10, lower bound: -16.9970226, upper bound: 16.9970228

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -34.3537903, 6.5822897, -34.3818779, 6.5868206, -40.8007812, 40.8280411
1: -9.2903957, 13.6747627, -9.3035221, 13.6771507, -22.9675465, 22.9782848
2: -8.3055496, 13.4608841, -8.3140821, 13.4632874, -20.8828812, 20.9007568
3: -10.0037918, 16.8280582, -10.0299416, 16.8301163, -25.7565918, 25.7749786
4: -16.3319359, 14.4370594, -16.3618221, 14.4396534, -30.7715893, 30.7988815
5: -9.3281202, 16.0029984, -9.3570976, 16.0049210, -25.0813828, 25.1079941
6: -35.8518372, -0.8568254, -35.8571167, -0.8534942, -31.7929306, 31.7923431
7: -9.4063377, 16.1894512, -9.4199810, 16.1909142, -23.9070892, 23.9334450
8: -19.9492931, 18.4247398, -19.9663773, 18.4274178, -38.3767090, 38.3911171
9: -6.8018484, 28.0244751, -6.8419571, 28.0269699, -32.1651382, 32.1890640
10: -5.5354509, 31.8677063, -5.5824709, 31.8718548, -34.6674347, 34.6926727
11: -11.3382645, 15.0411682, -11.3434143, 15.0680780, -26.4063416, 26.3845825
12: -18.4478149, 16.2902927, -18.4498425, 16.3007183, -29.2530441, 29.1969147
13: -17.9355526, 28.9596481, -17.9517899, 28.9695091, -44.2058105, 44.1851654
14: -26.7675438, 15.5785828, -26.7761421, 15.5831184, -38.0113907, 37.9837265
15: -19.5086327, 11.9151154, -19.5424004, 11.9197035, -31.4283371, 31.4575157
16: -15.4876165, 13.0671291, -15.5132971, 13.0727005, -27.8828888, 27.9072227
17: -28.7542801, 8.6624212, -28.7594986, 8.6828632, -28.4753265, 28.4233513
18: -29.0662003, 3.1674128, -29.0663567, 3.1752033, -26.5081787, 26.5076485
19: -16.0890656, 6.7646599, -16.0923920, 6.7853174, -22.0517464, 22.0370712
20: -11.9614830, 12.3684597, -11.9677706, 12.3842630, -23.7772064, 23.7710533
21: -15.2800188, 12.2796326, -15.2861738, 12.2991323, -27.5791512, 27.5658073
22: -17.0887260, 6.9136000, -17.0936356, 6.9243469, -22.8174057, 22.8119469
23: -15.3930807, 11.0888557, -15.3962250, 11.1275291, -26.5206108, 26.4850807
24: -27.4080658, 6.3882890, -27.4129181, 6.4204922, -31.2041779, 31.1879807
25: -17.4338455, 10.6452789, -17.4385681, 10.6772060, -27.2585831, 27.2317543
26: -22.1288300, 12.8232355, -22.1356564, 12.8448048, -32.8763885, 32.8730087
27: -28.2734222, 5.2022982, -28.2784595, 5.2235951, -28.5081253, 28.5207558
28: -15.4995337, 13.1068249, -15.5036726, 13.1400986, -27.3509750, 27.3264465
29: -15.6728115, 9.5074139, -15.6781168, 9.5202541, -24.7126236, 24.7012024
30: -17.5160065, 16.4007359, -17.5217819, 16.4290752, -32.3490372, 32.3304291
31: -26.9813004, 7.0886655, -26.9854069, 7.1056247, -33.0976791, 33.0883484
32: -26.1986504, 7.1480408, -26.2043304, 7.1512785, -30.5533218, 30.5491943
33: -64.4104767, -14.4191322, -64.4150085, -14.4021769, -44.5277405, 44.5186844
34: -51.9908333, -16.6964912, -51.9936523, -16.6906185, -27.1942482, 27.1980858
35: -46.5216141, -8.3523951, -46.5237350, -8.3337784, -34.0940018, 34.0861130
36: -40.1902084, -2.2679591, -40.1930313, -2.2454448, -36.3246002, 36.3036804
37: -67.3423462, -23.8539600, -67.3450394, -23.8258514, -35.4936295, 35.4741287
38: -50.6365891, -7.2114973, -50.6397972, -7.1932244, -40.7384796, 40.7203140
39: -61.3868027, -10.3560638, -61.3921204, -10.3427610, -48.2235718, 48.2080688
40: -52.4526672, -22.0554218, -52.4648209, -22.0499821, -24.1638107, 24.1401520
41: -37.7571983, -1.5216904, -37.7613754, -1.5104027, -30.7340088, 30.7123184
42: -24.2160149, 0.5790319, -24.2196426, 0.5876837, -23.0340729, 23.0321922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=268, inp2_unstable=269, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=291, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1021
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1546
type: A, layer: 1, pos: 1546
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 903

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.8863592, upper bound: 16.9737139
time: 48.52 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9341697, upper bound: 16.9965516
time: 55.26 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -34.3939667, 6.6389246, -34.4001465, 6.5900707, -40.8428192, 40.9039841
1: -9.3063564, 13.7054338, -9.3110619, 13.6786318, -22.9849892, 23.0164948
2: -8.3162622, 13.4837294, -8.3181591, 13.4649944, -20.8934860, 20.9369431
3: -10.0335178, 16.8685074, -10.0446835, 16.8312073, -25.7838135, 25.8302002
4: -16.3695564, 14.4910583, -16.3810081, 14.4408693, -30.8104248, 30.8720665
5: -9.3641014, 16.0470352, -9.3746471, 16.0056705, -25.1148682, 25.1696854
6: -35.8655701, -0.8490205, -35.8593292, -0.8512869, -31.8201218, 31.8064346
7: -9.4285202, 16.2216091, -9.4288788, 16.1919842, -23.9296265, 23.9802475
8: -19.9770012, 18.4606762, -19.9782753, 18.4292965, -38.4062958, 38.4389496
9: -6.8484015, 28.0822449, -6.8659844, 28.0283813, -32.2075806, 32.2713623
10: -5.5935926, 31.9502716, -5.6132460, 31.8741837, -34.7187500, 34.8064117
11: -11.3760529, 15.0692120, -11.3465710, 15.0812922, -26.4573441, 26.4157829
12: -18.4797974, 16.3149776, -18.4508991, 16.3101482, -29.2957840, 29.2217712
13: -17.9747810, 28.9800129, -17.9548836, 28.9767036, -44.2529755, 44.2090302
14: -26.8019161, 15.5877934, -26.7810059, 15.5859804, -38.0584259, 38.0168610
15: -19.5557652, 11.9731703, -19.5648479, 11.9225931, -31.4783592, 31.5380173
16: -15.5192518, 13.1209822, -15.5287485, 13.0765133, -27.9172974, 27.9770813
17: -28.8114166, 8.7011585, -28.7615051, 8.7032375, -28.5537949, 28.4589195
18: -29.0704117, 3.1854753, -29.0676498, 3.1774726, -26.5145264, 26.5287247
19: -16.1159267, 6.7804656, -16.0941925, 6.7935152, -22.0883942, 22.0545883
20: -11.9895954, 12.3797092, -11.9716568, 12.3898392, -23.8152008, 23.7852669
21: -15.3085423, 12.2890301, -15.2896290, 12.3025646, -27.6111069, 27.5786591
22: -17.1193657, 6.9239388, -17.0967140, 6.9286938, -22.8608093, 22.8228531
23: -15.4426699, 11.1283731, -15.3976698, 11.1482162, -26.5908852, 26.5260429
24: -27.4464111, 6.4171619, -27.4150372, 6.4346161, -31.2564011, 31.2140884
25: -17.4770145, 10.6787052, -17.4410419, 10.6940374, -27.3177643, 27.2659454
26: -22.1811619, 12.8433666, -22.1408768, 12.8529968, -32.9422188, 32.8961563
27: -28.3135777, 5.2163577, -28.2816391, 5.2298841, -28.5591469, 28.5358353
28: -15.5537891, 13.1430779, -15.5061340, 13.1583710, -27.4231949, 27.3606873
29: -15.7124500, 9.5180302, -15.6806803, 9.5250626, -24.7586670, 24.7137413
30: -17.5577393, 16.4299717, -17.5250340, 16.4431267, -32.4048691, 32.3569336
31: -26.9996052, 7.0974770, -26.9872952, 7.1089163, -33.1315994, 33.0996933
32: -26.2150383, 7.1593146, -26.2073174, 7.1534209, -30.5866241, 30.5659943
33: -64.4324112, -14.3970156, -64.4174042, -14.3935795, -44.5739746, 44.5417175
34: -52.0004768, -16.6827602, -51.9952011, -16.6866608, -27.2169495, 27.2137527
35: -46.5567360, -8.3262138, -46.5253944, -8.3210411, -34.1427841, 34.1135101
36: -40.2552147, -2.2353315, -40.1953087, -2.2281275, -36.4078751, 36.3355637
37: -67.3854218, -23.8229198, -67.3463135, -23.8105297, -35.5520706, 35.4998245
38: -50.6876221, -7.1852584, -50.6423988, -7.1802382, -40.8044739, 40.7463531
39: -61.4146347, -10.3355541, -61.3942070, -10.3335905, -48.2647247, 48.2304382
40: -52.4753113, -22.0207710, -52.4746246, -22.0464783, -24.1855011, 24.1878242
41: -37.7843895, -1.5095177, -37.7640038, -1.5062399, -30.7705002, 30.7275543
42: -24.2311211, 0.5905027, -24.2211494, 0.5915842, -23.0663300, 23.0417099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=268, inp2_unstable=269, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=290, inp2_unstable=291, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1021
type: B, layer: 1, pos: 1021
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1546
type: A, layer: 1, pos: 1546
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 903

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9161089, upper bound: 16.9737139
time: 56.88 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9639128, upper bound: 16.9965516
time: 56.01 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -34.4087677, 6.6574516, -34.3776169, 6.5851402, -40.8612289, 40.8975754
1: -9.3113394, 13.7150812, -9.3026695, 13.6763535, -22.9876938, 23.0177498
2: -8.3243732, 13.4947376, -8.3141956, 13.4617014, -20.9034805, 20.9362488
3: -10.0467272, 16.9172401, -10.0294161, 16.8292999, -25.8004532, 25.8704071
4: -16.3796768, 14.5217381, -16.3585892, 14.4378309, -30.8175087, 30.8803272
5: -9.3768835, 16.0946732, -9.3551559, 16.0040359, -25.1340561, 25.1969833
6: -35.8647652, -0.8461680, -35.8501663, -0.8539643, -31.8321152, 31.7847366
7: -9.4316492, 16.2266350, -9.4172230, 16.1896210, -23.9420242, 23.9566345
8: -19.9797840, 18.4677029, -19.9622173, 18.4265118, -38.4062958, 38.4299202
9: -6.8682156, 28.1412773, -6.8408108, 28.0260391, -32.2277451, 32.3179321
10: -5.6141548, 32.0030403, -5.5779552, 31.8694077, -34.7465973, 34.8410034
11: -11.4340668, 15.0856085, -11.3410606, 15.0709209, -26.5049877, 26.4266701
12: -18.4657192, 16.3209572, -18.4471397, 16.2957325, -29.2348862, 29.2586975
13: -17.9702721, 29.0404243, -17.9588470, 28.9648800, -44.2458344, 44.3019104
14: -26.7976818, 15.6040907, -26.7728081, 15.5807981, -38.0100250, 38.0494995
15: -19.5601845, 12.0118914, -19.5390015, 11.9172421, -31.4774265, 31.5508919
16: -15.5373402, 13.1396904, -15.5114841, 13.0712357, -27.9327087, 27.9723282
17: -28.7887020, 8.6978512, -28.7583160, 8.6692371, -28.4601517, 28.4948158
18: -29.1072063, 3.1809387, -29.0677986, 3.1700964, -26.5601654, 26.5267143
19: -16.1725559, 6.7968502, -16.0915546, 6.7892699, -22.1446838, 22.0837631
20: -12.0358543, 12.3949471, -11.9661493, 12.3892775, -23.8623581, 23.8152237
21: -15.3825827, 12.3120031, -15.2844687, 12.3066235, -27.6892052, 27.5964718
22: -17.1475258, 6.9315090, -17.0910263, 6.9245672, -22.8852158, 22.8277817
23: -15.5163441, 11.1484470, -15.3956995, 11.1296749, -26.6460190, 26.5441475
24: -27.5308132, 6.4398847, -27.4115334, 6.4261994, -31.3446732, 31.2551346
25: -17.5389862, 10.6936474, -17.4374485, 10.6775532, -27.3641739, 27.2820053
26: -22.2208481, 12.8611193, -22.1337738, 12.8445215, -32.9911652, 32.9282074
27: -28.3763046, 5.2407384, -28.2753143, 5.2316093, -28.6482239, 28.5759850
28: -15.6032066, 13.1582985, -15.5025177, 13.1405258, -27.4589081, 27.3869362
29: -15.7437334, 9.5305691, -15.6749802, 9.5225048, -24.7834778, 24.7225342
30: -17.6180267, 16.4472828, -17.5201302, 16.4318390, -32.4587784, 32.3826523
31: -27.0767345, 7.1196089, -26.9846249, 7.1130681, -33.2086029, 33.1322327
32: -26.2081623, 7.1593618, -26.1920376, 7.1510062, -30.5816269, 30.5525284
33: -64.4727325, -14.3876915, -64.4139175, -14.4019537, -44.5963898, 44.5640488
34: -52.0025177, -16.6808777, -51.9912186, -16.6921749, -27.2172737, 27.2084808
35: -46.5722160, -8.3213978, -46.5233765, -8.3373318, -34.1499786, 34.1163101
36: -40.2384415, -2.2343130, -40.1899147, -2.2509913, -36.3686523, 36.3369293
37: -67.4389801, -23.8058319, -67.3436356, -23.8259220, -35.5951309, 35.5315323
38: -50.7015457, -7.1732903, -50.6373138, -7.1960907, -40.7992325, 40.7612534
39: -61.4465599, -10.3353901, -61.3910828, -10.3456383, -48.2742310, 48.2352600
40: -52.4622498, -22.0354862, -52.4450226, -22.0508595, -24.1486664, 24.1776047
41: -37.7959709, -1.4960146, -37.7568779, -1.5091314, -30.7603531, 30.7495880
42: -24.2420502, 0.6000538, -24.2140312, 0.5886717, -23.0752563, 23.0348015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=268, inp2_unstable=269, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1021
type: B, layer: 1, pos: 1021
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1546
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 903

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9737136, upper bound: 16.9189597
time: 108.42 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9965513, upper bound: 16.9668223
time: 428.94 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 539.66 seconds
IS_A1_A1_B2_A1, status: Status.VERIFIED, split count: 4, time: 539.66
Output dim: 10, lower bound: -16.8863592, upper bound: 16.9737139
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 539.66
Output dim: 10, lower bound: -16.9341697, upper bound: 16.9965516
IS_A1_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 539.66
Output dim: 10, lower bound: -16.9161089, upper bound: 16.9737139
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 539.66
Output dim: 10, lower bound: -16.9639128, upper bound: 16.9965516
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 539.66
Output dim: 10, lower bound: -16.9737136, upper bound: 16.9189597
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 539.66
Output dim: 10, lower bound: -16.9965513, upper bound: 16.9668223
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 539.66
Output dim: 10, lower bound: -16.9795526, upper bound: 16.9864750
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 539.66
Output dim: 10, lower bound: -16.9970226, upper bound: 16.9970228

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 66.27 + 1752.61 = 1818.88 seconds
