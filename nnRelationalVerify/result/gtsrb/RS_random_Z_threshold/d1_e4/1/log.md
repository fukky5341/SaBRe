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
execution time: IAR + RelationalAnalysis = 2.78 + 62.16 = 64.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 10, lower bound: -17.0010558, upper bound: 17.0010557

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 554

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1446

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.0005066, upper bound: 16.9976122
time: 59.63 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9976122, upper bound: 17.0005066
time: 54.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 113.99 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 113.99
Output dim: 10, lower bound: -17.0005066, upper bound: 16.9976122
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 113.99
Output dim: 10, lower bound: -16.9976122, upper bound: 17.0005066

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8804016, 40.8806992
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9251099, 20.9251747
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8131104, 25.8125992
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1470566, 25.1466446
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8200684, 31.8201599
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9571609, 23.9567757
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2531204, 32.2522430
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7770004, 34.7768173
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2711868, 29.2719269
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2609253, 44.2610321
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0503769, 38.0503311
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9459076, 27.9454422
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5193634, 28.5202866
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5298347, 26.5298271
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0906601, 22.0906525
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8207130, 23.8200760
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8505783, 22.8504257
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2690506, 31.2689972
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3118820, 27.3117828
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9407997, 32.9407501
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5807953, 28.5807648
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4087448, 27.4086838
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7411346, 24.7411232
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4042282, 32.4032593
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1403046, 33.1402893
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5737305, 30.5738182
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5739746, 44.5741043
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2193260, 27.2194710
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1321869, 34.1322632
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3576050, 36.3580627
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5353546, 35.5364304
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7648621, 40.7658234
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2504883, 48.2508545
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1734161, 24.1737213
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7410202, 30.7415161
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0556145, 23.0556221

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 517

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.0004098, upper bound: 16.9940433
time: 46.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9940433, upper bound: 16.9975162
time: 54.44 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8807068, 40.8803940
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9251785, 20.9251099
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8125992, 25.8131104
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1466446, 25.1470566
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8201599, 31.8200684
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9567719, 23.9571648
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2522507, 32.2531204
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7768173, 34.7770004
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2719193, 29.2711945
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2610168, 44.2609177
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0503311, 38.0503845
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9454422, 27.9459076
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5202866, 28.5193634
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5298347, 26.5298309
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0906525, 22.0906601
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8200722, 23.8207054
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8504257, 22.8505745
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2689972, 31.2690506
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3117828, 27.3118820
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9407539, 32.9408035
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5807648, 28.5807953
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4086838, 27.4087448
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7411194, 24.7411385
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4032593, 32.4042206
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1402893, 33.1403046
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5738144, 30.5737305
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5740967, 44.5739746
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2194710, 27.2193222
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1322632, 34.1321869
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3580627, 36.3575974
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5364227, 35.5353470
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7658081, 40.7648697
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2508545, 48.2504883
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1737213, 24.1734161
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7415161, 30.7410202
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0556297, 23.0556107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 723

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9797686, upper bound: 16.9999474
time: 47.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9970530, upper bound: 16.9826625
time: 44.94 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 94.63 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 94.63
Output dim: 10, lower bound: -17.0004098, upper bound: 16.9940433
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 94.63
Output dim: 10, lower bound: -16.9940433, upper bound: 16.9975162
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 94.63
Output dim: 10, lower bound: -16.9797686, upper bound: 16.9999474
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 94.63
Output dim: 10, lower bound: -16.9970530, upper bound: 16.9826625

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8784332, 40.8788834
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9252701, 20.9253197
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8131714, 25.8126602
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1477280, 25.1472931
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8169098, 31.8169479
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9561920, 23.9557266
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2536316, 32.2528229
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7793808, 34.7794647
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2669144, 29.2680740
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2615814, 44.2618637
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0478439, 38.0479584
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9471283, 27.9465485
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5183105, 28.5192871
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5318832, 26.5316772
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0913544, 22.0912209
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8187408, 23.8180122
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8500900, 22.8499336
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2720184, 31.2715302
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3106232, 27.3105278
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9362221, 32.9356766
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5834312, 28.5829277
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4053268, 27.4050140
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7410965, 24.7410812
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.3975296, 32.3961487
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1399078, 33.1399002
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5688858, 30.5691605
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5745697, 44.5746841
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2206726, 27.2208290
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1343994, 34.1345062
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3586349, 36.3590164
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5366669, 35.5377655
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7642746, 40.7652359
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2507172, 48.2510834
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1707840, 24.1711273
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7395248, 30.7400055
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0519295, 23.0520401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1713

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 538

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9943485, upper bound: 16.9929757
time: 45.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9993378, upper bound: 16.9879834
time: 52.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8785706, 40.8787460
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9252625, 20.9253311
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8131714, 25.8126678
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1477127, 25.1473160
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8168640, 31.8170013
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9561234, 23.9558029
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2536926, 32.2527542
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7796555, 34.7791901
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2673416, 29.2676468
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2617340, 44.2617035
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0480118, 38.0477905
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9470062, 27.9466667
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5183640, 28.5192413
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5316849, 26.5318832
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0912323, 22.0913429
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8186493, 23.8181038
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8500824, 22.8499451
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2715836, 31.2719650
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3106232, 27.3105240
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9357262, 32.9361725
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5829582, 28.5833969
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4050751, 27.4052658
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7410965, 24.7410774
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.3971100, 32.3965607
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1399078, 33.1398926
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5690765, 30.5689697
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5745544, 44.5746918
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2206802, 27.2208176
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1344299, 34.1344757
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3585587, 36.3591003
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5366821, 35.5377579
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7642899, 40.7652283
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2507172, 48.2510834
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1708221, 24.1710892
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7395096, 30.7400169
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0520287, 23.0519409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1659

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1629

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9957695, upper bound: 16.9967974
time: 48.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9962183, upper bound: 16.9963486
time: 55.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8760376, 40.8763123
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9245453, 20.9245224
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8108292, 25.8115540
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1418152, 25.1428223
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8235703, 31.8238678
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9541626, 23.9548340
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2494431, 32.2506638
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7738419, 34.7743378
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2727890, 29.2721481
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2604065, 44.2604218
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0503998, 38.0504532
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9430771, 27.9435539
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5180511, 28.5173759
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5313110, 26.5312386
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0886993, 22.0884628
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8195076, 23.8200760
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8472481, 22.8469925
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2658386, 31.2655029
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3077927, 27.3073845
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9382706, 32.9379959
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5780106, 28.5777054
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4052277, 27.4048500
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7399788, 24.7398643
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4015579, 32.4024277
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1394806, 33.1394119
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5758820, 30.5760155
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5696411, 44.5689011
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2136192, 27.2126694
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1253204, 34.1243057
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3540192, 36.3529739
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5327835, 35.5311813
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7621765, 40.7607422
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2452087, 48.2440491
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1752853, 24.1750793
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7408295, 30.7402954
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0580254, 23.0583191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1558

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1544

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9795212, upper bound: 16.9996962
time: 84.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9795212, upper bound: 16.9996962
time: 85.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8766174, 40.8757248
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9245834, 20.9244843
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8110428, 25.8113403
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1424103, 25.1422272
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8239594, 31.8234787
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9544373, 23.9545517
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2497787, 32.2503204
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7741547, 34.7740326
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2728806, 29.2720566
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2605286, 44.2602768
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0503998, 38.0504532
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9430923, 27.9435463
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5183029, 28.5171242
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5312347, 26.5313148
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0884552, 22.0887070
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8194466, 23.8201370
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8468437, 22.8474007
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2654419, 31.2658920
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3072891, 27.3078880
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9379501, 32.9383163
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5776749, 28.5780373
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4047928, 27.4052887
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7398567, 24.7399940
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4014587, 32.4025269
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1394043, 33.1394958
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5761032, 30.5757942
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5690308, 44.5695114
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2128181, 27.2134628
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1243744, 34.1252518
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3534393, 36.3535461
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5322647, 35.5317001
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7616882, 40.7612228
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2444153, 48.2448425
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1753845, 24.1749802
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7407913, 30.7403297
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0583305, 23.0580177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1021

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9969678, upper bound: 16.9679202
time: 55.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9823095, upper bound: 16.9825773
time: 55.87 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 113.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 113.12
Output dim: 10, lower bound: -16.9943485, upper bound: 16.9929757
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 113.12
Output dim: 10, lower bound: -16.9993378, upper bound: 16.9879834
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 113.12
Output dim: 10, lower bound: -16.9957695, upper bound: 16.9967974
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 113.12
Output dim: 10, lower bound: -16.9962183, upper bound: 16.9963486
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 113.12
Output dim: 10, lower bound: -16.9795212, upper bound: 16.9996962
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 113.12
Output dim: 10, lower bound: -16.9795212, upper bound: 16.9996962
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 113.12
Output dim: 10, lower bound: -16.9969678, upper bound: 16.9679202
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 113.12
Output dim: 10, lower bound: -16.9823095, upper bound: 16.9825773

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8796768, 40.8803864
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9250793, 20.9251175
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8127899, 25.8122711
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1478119, 25.1473694
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8122330, 31.8120499
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9562607, 23.9557877
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2519226, 32.2511597
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7767410, 34.7770157
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2660828, 29.2671661
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2620087, 44.2621765
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0397339, 38.0394287
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9490891, 27.9487000
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5098801, 28.5102692
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5324936, 26.5326118
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0911636, 22.0910416
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8185425, 23.8177567
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8499298, 22.8498535
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2686005, 31.2681732
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3091888, 27.3091125
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9320908, 32.9317398
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5804062, 28.5800476
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4041977, 27.4039116
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7406921, 24.7406578
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.3941498, 32.3924866
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1383667, 33.1385422
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5626984, 30.5628967
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5750122, 44.5751877
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2222633, 27.2225304
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1346283, 34.1347885
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3576126, 36.3580551
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5409546, 35.5426483
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7643814, 40.7653503
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2502899, 48.2507477
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1715088, 24.1720657
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7385025, 30.7390442
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0519867, 23.0520973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 528

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9943395, upper bound: 16.9921606
time: 45.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9935334, upper bound: 16.9929667
time: 52.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8799362, 40.8801193
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9250641, 20.9251251
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8127823, 25.8122787
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1477966, 25.1473770
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8120117, 31.8122711
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9562454, 23.9557953
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2519531, 32.2511292
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7769318, 34.7768173
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2660065, 29.2672424
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2619171, 44.2622757
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0393219, 38.0398483
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9492798, 27.9485054
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5092926, 28.5108643
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5328217, 26.5322838
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0911713, 22.0910339
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8184814, 23.8178177
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8500137, 22.8497696
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2686691, 31.2681122
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3092041, 27.3090973
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9322891, 32.9315414
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5805435, 28.5799103
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4042206, 27.4038849
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7406693, 24.7406807
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.3938751, 32.3927612
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1385498, 33.1383591
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5626221, 30.5629768
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5750732, 44.5751190
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2223701, 27.2224236
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1346893, 34.1347275
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3576736, 36.3579865
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5415497, 35.5420532
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7643814, 40.7653351
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2503815, 48.2506561
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1717224, 24.1718483
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7385559, 30.7389908
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0519867, 23.0521011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 721

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1566

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9746634, upper bound: 16.9869838
time: 56.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9983388, upper bound: 16.9633136
time: 43.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8769836, 40.8768692
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9223709, 20.9219933
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8081207, 25.8071365
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1414108, 25.1402817
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8172989, 31.8175659
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9489136, 23.9476280
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2467804, 32.2450790
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7821426, 34.7812195
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2706451, 29.2709198
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2570343, 44.2565689
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0479126, 38.0476608
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9440994, 27.9434204
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5197296, 28.5205994
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5260239, 26.5266953
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0874557, 22.0879669
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8193703, 23.8187943
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8430710, 22.8437576
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2671585, 31.2679367
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3048172, 27.3052902
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9310684, 32.9318924
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5791092, 28.5799026
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4008179, 27.4014015
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7378159, 24.7381973
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.3998642, 32.3993301
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1349869, 33.1355743
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5681686, 30.5680962
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5852203, 44.5866852
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2032700, 27.2050934
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1408539, 34.1425552
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3701706, 36.3716888
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5376587, 35.5401917
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7772522, 40.7795563
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2635651, 48.2649384
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1794739, 24.1805687
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7416611, 30.7425499
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0522423, 23.0519714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1647

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1463

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9948482, upper bound: 16.9959743
time: 56.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9949372, upper bound: 16.9958320
time: 80.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8766785, 40.8771667
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9219208, 20.9224434
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8076324, 25.8076172
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1406860, 25.1410141
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8174286, 31.8174438
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9479523, 23.9486046
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2460175, 32.2458344
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7816849, 34.7816772
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2706146, 29.2709579
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2566071, 44.2569962
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0478821, 38.0476913
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9437637, 27.9437561
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5197220, 28.5206070
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5264893, 26.5262299
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0878525, 22.0875664
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8193398, 23.8188286
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8438950, 22.8429337
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2675552, 31.2675476
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3053894, 27.3047218
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9314423, 32.9315109
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5794678, 28.5795517
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4012070, 27.4010086
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7382126, 24.7378006
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.3998795, 32.3993149
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1355972, 33.1349716
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5681992, 30.5680618
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5865479, 44.5853577
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2049561, 27.2034111
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1425171, 34.1408997
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3711472, 36.3707199
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5391083, 35.5387344
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7786102, 40.7781982
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2645721, 48.2639313
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1802979, 24.1797409
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7420425, 30.7421646
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0520515, 23.0521584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1543

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1571

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9847978, upper bound: 16.9959528
time: 51.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9958224, upper bound: 16.9849254
time: 58.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8756790, 40.8753357
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9241791, 20.9242439
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8112564, 25.8112946
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1425171, 25.1426849
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8184052, 31.8161087
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9541855, 23.9548111
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2494507, 32.2506256
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7742081, 34.7742004
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2713394, 29.2699738
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2598114, 44.2600250
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0472412, 38.0482330
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9429398, 27.9433556
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5169144, 28.5166245
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5310974, 26.5314674
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0870667, 22.0877647
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8191948, 23.8199615
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8450966, 22.8456879
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2636414, 31.2643967
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3065643, 27.3067818
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9367676, 32.9376907
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5777206, 28.5776978
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4046936, 27.4052200
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7397652, 24.7400322
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4012756, 32.4028168
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1385880, 33.1390686
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5724869, 30.5708313
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5681458, 44.5666504
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2120056, 27.2102547
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1247025, 34.1243515
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3539200, 36.3528976
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5318756, 35.5298309
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7622223, 40.7601929
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2450562, 48.2439423
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1709290, 24.1685333
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7378998, 30.7359009
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0527077, 23.0503235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 954

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 561

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9791820, upper bound: 16.9821533
time: 52.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9619600, upper bound: 16.9993572
time: 51.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8750687, 40.8759460
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9242706, 20.9241524
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8105698, 25.8119812
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1416702, 25.1435242
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8158035, 31.8187027
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9541397, 23.9548492
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2494202, 32.2506714
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7737045, 34.7747040
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2706146, 29.2707062
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2600098, 44.2598343
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0481720, 38.0473022
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9428787, 27.9434128
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5172958, 28.5162392
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5315399, 26.5310249
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0880051, 22.0868301
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8193932, 23.8197670
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8459435, 22.8448410
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2647324, 31.2633057
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3071899, 27.3061523
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9379578, 32.9364929
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5780029, 28.5774155
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4055939, 27.4043198
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7401466, 24.7396507
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4019470, 32.4021530
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1391373, 33.1385117
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5706940, 30.5726242
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5673981, 44.5674057
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2112045, 27.2110596
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1253738, 34.1236801
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3539352, 36.3528900
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5314331, 35.5302887
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7616119, 40.7607956
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2451019, 48.2439117
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1687393, 24.1707306
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7364273, 30.7373695
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0500374, 23.0530014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1557

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 538

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9733244, upper bound: 16.9984987
time: 44.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9783216, upper bound: 16.9934983
time: 43.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8707809, 40.8690491
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9244843, 20.9243736
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8050995, 25.8045502
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1311646, 25.1293945
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8293076, 31.8279800
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9470673, 23.9463081
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2416077, 32.2411118
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7689590, 34.7680969
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2748489, 29.2738113
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2637177, 44.2625961
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0418701, 38.0407181
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9352264, 27.9345856
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5051880, 28.5021515
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5305672, 26.5305061
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0812912, 22.0824318
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8208466, 23.8215942
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8402100, 22.8416290
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2690964, 31.2705460
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3012009, 27.3025589
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9326439, 32.9336929
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5820427, 28.5833817
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.3989563, 27.4001770
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7387619, 24.7390594
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4053726, 32.4053955
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1332932, 33.1341476
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5752106, 30.5750046
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5511017, 44.5537796
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.1956444, 27.1984558
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1062469, 34.1092834
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3401260, 36.3419571
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5108643, 35.5126190
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7422333, 40.7439423
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2221985, 48.2256165
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1664734, 24.1670227
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7284012, 30.7293968
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0592155, 23.0587654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1787

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 673

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9737631, upper bound: 16.9677168
time: 53.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9967647, upper bound: 16.9619991
time: 54.90 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 110.81 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 110.81
Output dim: 10, lower bound: -16.9943395, upper bound: 16.9921606
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 110.81
Output dim: 10, lower bound: -16.9935334, upper bound: 16.9929667
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 110.81
Output dim: 10, lower bound: -16.9746634, upper bound: 16.9869838
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 110.81
Output dim: 10, lower bound: -16.9983388, upper bound: 16.9633136
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 110.81
Output dim: 10, lower bound: -16.9948482, upper bound: 16.9959743
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 110.81
Output dim: 10, lower bound: -16.9949372, upper bound: 16.9958320
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 110.81
Output dim: 10, lower bound: -16.9847978, upper bound: 16.9959528
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 110.81
Output dim: 10, lower bound: -16.9958224, upper bound: 16.9849254
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 110.81
Output dim: 10, lower bound: -16.9791820, upper bound: 16.9821533
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 110.81
Output dim: 10, lower bound: -16.9619600, upper bound: 16.9993572
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 110.81
Output dim: 10, lower bound: -16.9733244, upper bound: 16.9984987
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 110.81
Output dim: 10, lower bound: -16.9783216, upper bound: 16.9934983
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 110.81
Output dim: 10, lower bound: -16.9737631, upper bound: 16.9677168
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 110.81
Output dim: 10, lower bound: -16.9967647, upper bound: 16.9619991

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8767090, 40.8776398
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9175186, 20.9181213
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8129196, 25.8124771
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1420517, 25.1421280
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8111115, 31.8109970
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9492493, 23.9493637
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2538147, 32.2534943
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7770691, 34.7773666
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2583771, 29.2586136
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2598267, 44.2595139
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0223770, 38.0201797
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9490051, 27.9486313
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4770737, 28.4742775
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5326042, 26.5327187
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0862198, 22.0869751
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8220673, 23.8217888
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8475494, 22.8471794
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2700729, 31.2695847
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3051376, 27.3040886
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9328842, 32.9326477
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5770111, 28.5770569
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.3994370, 27.3987503
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7411499, 24.7406311
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.3800659, 32.3771515
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1319504, 33.1328125
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5560913, 30.5567856
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5783539, 44.5780411
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2317619, 27.2308426
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1420898, 34.1416931
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3581772, 36.3585892
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5451050, 35.5461807
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7562408, 40.7582245
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2462311, 48.2472076
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1677170, 24.1678162
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7301331, 30.7316093
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0521164, 23.0523148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 738

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 533

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9939530, upper bound: 16.9851805
time: 51.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9856713, upper bound: 16.9917256
time: 45.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8769226, 40.8774185
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9180832, 20.9175644
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8130035, 25.8123932
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1425705, 25.1416092
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8111877, 31.8109283
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9498367, 23.9487762
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2542725, 32.2530365
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7770844, 34.7773514
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2575378, 29.2594604
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2593384, 44.2600021
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0204849, 38.0220642
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9490128, 27.9486237
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4738998, 28.4774628
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5326042, 26.5327225
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0870972, 22.0860977
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8225708, 23.8212852
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8472519, 22.8474808
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2700119, 31.2696457
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3041687, 27.3050575
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9329987, 32.9325409
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5774155, 28.5766525
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.3990402, 27.3991508
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7406616, 24.7411194
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.3788147, 32.3784027
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1326218, 33.1321411
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5565872, 30.5562859
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5778656, 44.5785370
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2305717, 27.2320328
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1415253, 34.1422577
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3581467, 36.3586197
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5444946, 35.5467911
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7572479, 40.7572250
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2467499, 48.2466888
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1672592, 24.1682739
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7310715, 30.7306709
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0522003, 23.0522270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1714

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 971

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9928161, upper bound: 16.9927840
time: 46.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9933508, upper bound: 16.9922431
time: 57.68 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 106.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 106.39
Output dim: 10, lower bound: -16.9939530, upper bound: 16.9851805
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 106.39
Output dim: 10, lower bound: -16.9856713, upper bound: 16.9917256
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 106.39
Output dim: 10, lower bound: -16.9928161, upper bound: 16.9927840
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 106.39
Output dim: 10, lower bound: -16.9933508, upper bound: 16.9922431
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 106.39
Output dim: 10, lower bound: -16.9746634, upper bound: 16.9869838
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 106.39
Output dim: 10, lower bound: -16.9983388, upper bound: 16.9633136
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 106.39
Output dim: 10, lower bound: -16.9948482, upper bound: 16.9959743
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 106.39
Output dim: 10, lower bound: -16.9949372, upper bound: 16.9958320
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 106.39
Output dim: 10, lower bound: -16.9847978, upper bound: 16.9959528
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 106.39
Output dim: 10, lower bound: -16.9958224, upper bound: 16.9849254
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 106.39
Output dim: 10, lower bound: -16.9619600, upper bound: 16.9993572
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 106.39
Output dim: 10, lower bound: -16.9733244, upper bound: 16.9984987
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 106.39
Output dim: 10, lower bound: -16.9783216, upper bound: 16.9934983
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 106.39
Output dim: 10, lower bound: -16.9967647, upper bound: 16.9619991

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 64.94 + 1772.16 = 1837.09 seconds
