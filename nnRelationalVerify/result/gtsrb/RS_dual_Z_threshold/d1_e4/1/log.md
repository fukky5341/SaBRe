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
execution time: IAR + RelationalAnalysis = 2.74 + 61.74 = 64.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 10, lower bound: -17.0010558, upper bound: 17.0010557

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9887035, upper bound: 17.0005581
time: 59.97 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.0005581, upper bound: 16.9887035
time: 51.13 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 111.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 111.22
Output dim: 10, lower bound: -16.9887035, upper bound: 17.0005581
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 111.22
Output dim: 10, lower bound: -17.0005581, upper bound: 16.9887035

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8774185, 40.8773575
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9207458, 20.9204407
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8190918, 25.8191147
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1483383, 25.1476822
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8111343, 31.8103409
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9589233, 23.9583130
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2608795, 32.2608948
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7753906, 34.7757645
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2784348, 29.2787933
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2637024, 44.2644730
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0391998, 38.0399170
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9454727, 27.9452362
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5029144, 28.5045204
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5255356, 26.5251236
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0865250, 22.0859337
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8261032, 23.8249474
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8501930, 22.8503494
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2667084, 31.2667236
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3104248, 27.3107147
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9446068, 32.9442673
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5773315, 28.5762482
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4065704, 27.4066162
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7425117, 24.7428055
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.3997879, 32.3999405
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1361771, 33.1356277
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5734482, 30.5733376
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5777740, 44.5782547
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2282143, 27.2292633
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1368103, 34.1370621
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3598938, 36.3598022
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5507050, 35.5510178
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7621460, 40.7610855
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2542725, 48.2542114
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1759567, 24.1760254
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7433624, 30.7430649
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0561256, 23.0560722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9715358, upper bound: 17.0000550
time: 54.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9882074, upper bound: 16.9834274
time: 58.20 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8773575, 40.8774185
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9204407, 20.9207458
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8191147, 25.8190918
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1476822, 25.1483383
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8103409, 31.8111343
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9583130, 23.9589233
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2608948, 32.2608871
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7757645, 34.7753830
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2788010, 29.2784348
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2644806, 44.2637024
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0399170, 38.0391998
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9452362, 27.9454765
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.5045242, 28.5029144
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5251236, 26.5255356
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0859299, 22.0865211
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8249435, 23.8261070
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8503456, 22.8501892
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2667236, 31.2667084
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3107147, 27.3104210
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9442635, 32.9446106
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5762482, 28.5773354
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4066162, 27.4065704
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7428093, 24.7425117
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.3999405, 32.3997879
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1356201, 33.1361771
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5733414, 30.5734482
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5782623, 44.5777740
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2292671, 27.2282104
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1370621, 34.1368103
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3598022, 36.3599014
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5510101, 35.5507050
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7610931, 40.7621384
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2542114, 48.2542725
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1760254, 24.1759567
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7430649, 30.7433624
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0560646, 23.0561256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1555

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9834274, upper bound: 16.9882074
time: 54.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.0000550, upper bound: 16.9715358
time: 49.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 107.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 107.03
Output dim: 10, lower bound: -16.9715358, upper bound: 17.0000550
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 107.03
Output dim: 10, lower bound: -16.9882074, upper bound: 16.9834274
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 107.03
Output dim: 10, lower bound: -16.9834274, upper bound: 16.9882074
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 107.03
Output dim: 10, lower bound: -17.0000550, upper bound: 16.9715358

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8800430, 40.8801575
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9200668, 20.9197731
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8224487, 25.8223038
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1501541, 25.1493759
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8087387, 31.8078918
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9602203, 23.9594269
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2613525, 32.2613449
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7738342, 34.7742538
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2838287, 29.2843475
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2651672, 44.2660065
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0379333, 38.0386963
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9416428, 27.9413300
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4951553, 28.4970284
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5257111, 26.5253181
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0871887, 22.0866127
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8318558, 23.8305206
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8496552, 22.8498039
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2656555, 31.2656631
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3120346, 27.3124008
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9471512, 32.9467773
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5794907, 28.5782509
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4087219, 27.4086952
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7411652, 24.7414551
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4076843, 32.4075928
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1374588, 33.1369934
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5747833, 30.5746231
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5804596, 44.5809097
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2318192, 27.2327805
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1403198, 34.1404343
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3613586, 36.3612061
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5597076, 35.5603485
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7659302, 40.7649384
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2579041, 48.2579193
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1818619, 24.1822014
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7464523, 30.7460480
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0546837, 23.0546455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1554

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9566263, upper bound: 16.9995276
time: 51.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9709960, upper bound: 16.9845588
time: 54.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8802261, 40.8799896
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9200745, 20.9197617
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8222809, 25.8224716
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1500397, 25.1494980
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8086853, 31.8079529
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9600372, 23.9596100
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2613373, 32.2613525
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7738800, 34.7742081
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2839813, 29.2841873
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2652283, 44.2659454
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0379791, 38.0386581
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9415665, 27.9414062
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4954224, 28.4967575
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5257263, 26.5253067
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0872040, 22.0865974
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8316803, 23.8306885
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8496399, 22.8498154
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2656479, 31.2656708
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3121033, 27.3123283
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9471207, 32.9468079
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5793381, 28.5784073
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4086456, 27.4087715
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7411652, 24.7414589
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4074402, 32.4078369
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1375580, 33.1369019
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5747299, 30.5746765
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5804291, 44.5809479
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2317276, 27.2328720
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1401825, 34.1405792
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3613129, 36.3612671
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5600433, 35.5600204
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7659912, 40.7648773
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2579956, 48.2578278
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1821365, 24.1819305
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7463455, 30.7461548
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0546989, 23.0546265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1554

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9730748, upper bound: 16.9828839
time: 49.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9876781, upper bound: 16.9683310
time: 48.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8799820, 40.8802185
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9197617, 20.9200745
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8224716, 25.8222809
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1494980, 25.1500397
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8079529, 31.8086853
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9596024, 23.9600372
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2613525, 32.2613373
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7742081, 34.7738800
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2841873, 29.2839813
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2659454, 44.2652283
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0386658, 38.0379791
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9414062, 27.9415741
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4967575, 28.4954224
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5252991, 26.5257301
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0865936, 22.0872040
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8306961, 23.8316803
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8498154, 22.8496399
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2656708, 31.2656479
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3123245, 27.3121033
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9468079, 32.9471207
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5784073, 28.5793381
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4087753, 27.4086456
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7414551, 24.7411613
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4078369, 32.4074402
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1369019, 33.1375504
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5746765, 30.5747299
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5809479, 44.5804214
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2328720, 27.2317276
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1405792, 34.1401825
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3612671, 36.3613052
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5600128, 35.5600357
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7648773, 40.7659912
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2578278, 48.2579956
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1819305, 24.1821327
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7461548, 30.7463455
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0546227, 23.0546951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1554

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9683311, upper bound: 16.9876780
time: 53.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9828839, upper bound: 16.9730748
time: 45.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8801651, 40.8800507
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9197769, 20.9200668
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8223038, 25.8224487
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1493759, 25.1501541
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8078918, 31.8087387
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9594269, 23.9602203
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2613525, 32.2613449
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7742538, 34.7738342
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2843475, 29.2838287
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2660065, 44.2651672
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0386963, 38.0379410
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9413300, 27.9416504
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4970245, 28.4951515
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5253143, 26.5257149
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0866089, 22.0871849
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8305206, 23.8318481
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8498001, 22.8496552
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2656631, 31.2656555
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3124008, 27.3120346
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9467773, 32.9471512
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5782547, 28.5794945
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4086914, 27.4087219
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7414551, 24.7411652
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4075928, 32.4076843
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1370010, 33.1374588
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5746155, 30.5747833
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5809174, 44.5804672
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2327805, 27.2318192
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1404419, 34.1403198
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3612061, 36.3613663
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5603485, 35.5597153
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7649384, 40.7659302
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2579193, 48.2579041
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1821976, 24.1818657
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7460480, 30.7464523
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0546379, 23.0546799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1554

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9845588, upper bound: 16.9709960
time: 51.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9995276, upper bound: 16.9566263
time: 56.12 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 109.88 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 109.88
Output dim: 10, lower bound: -16.9566263, upper bound: 16.9995276
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 109.88
Output dim: 10, lower bound: -16.9709960, upper bound: 16.9845588
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 109.88
Output dim: 10, lower bound: -16.9730748, upper bound: 16.9828839
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 109.88
Output dim: 10, lower bound: -16.9876781, upper bound: 16.9683310
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 109.88
Output dim: 10, lower bound: -16.9683311, upper bound: 16.9876780
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 109.88
Output dim: 10, lower bound: -16.9828839, upper bound: 16.9730748
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 109.88
Output dim: 10, lower bound: -16.9845588, upper bound: 16.9709960
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 109.88
Output dim: 10, lower bound: -16.9995276, upper bound: 16.9566263

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8811111, 40.8812637
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9202499, 20.9199562
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8228455, 25.8227158
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1506042, 25.1496964
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8024521, 31.8012390
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9613266, 23.9603653
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2609100, 32.2609406
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7690811, 34.7697144
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2843857, 29.2849274
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2658386, 44.2668228
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0390167, 38.0397568
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9402084, 27.9398117
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4943237, 28.4961929
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5244217, 26.5239487
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0876312, 22.0870247
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8348465, 23.8332024
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8496246, 22.8497772
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2652664, 31.2652588
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3117065, 27.3120651
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9490967, 32.9485779
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5753098, 28.5737228
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4099274, 27.4098587
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7411461, 24.7414322
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4107971, 32.4106674
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1371307, 33.1366425
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5751801, 30.5750046
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5821838, 44.5826263
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2342300, 27.2352448
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1424255, 34.1425247
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3601608, 36.3599167
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5632629, 35.5640411
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7637634, 40.7625275
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2602844, 48.2603607
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1829834, 24.1833153
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7476273, 30.7471581
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0531387, 23.0531387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1586

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9401818, upper bound: 16.9989555
time: 60.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9560535, upper bound: 16.9830737
time: 49.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8811417, 40.8812103
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9202499, 20.9199524
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8228607, 25.8227005
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1504745, 25.1498566
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8020859, 31.8015137
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9611664, 23.9605179
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2609406, 32.2609177
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7692871, 34.7695007
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2843628, 29.2849045
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2659912, 44.2666550
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0390015, 38.0397491
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9401245, 27.9398422
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4943085, 28.4961967
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5243454, 26.5240250
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0876007, 22.0870590
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8345261, 23.8335075
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8496246, 22.8497772
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2652435, 31.2652740
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3116989, 27.3120575
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9489517, 32.9486694
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5749664, 28.5740433
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4098892, 27.4098816
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7411461, 24.7414322
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4107590, 32.4106903
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1371002, 33.1366730
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5751724, 30.5750160
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5821838, 44.5826263
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2342377, 27.2351990
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1423798, 34.1425400
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3600845, 36.3599930
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5633850, 35.5639038
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7635193, 40.7626572
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2603149, 48.2602997
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1829453, 24.1833229
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7475510, 30.7472153
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0531769, 23.0531006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1586

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9545474, upper bound: 16.9839866
time: 51.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9704233, upper bound: 16.9680889
time: 43.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8813248, 40.8810425
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9202499, 20.9199448
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8226929, 25.8228683
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1503525, 25.1499710
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8020248, 31.8015823
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9609833, 23.9607086
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2609406, 32.2609253
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7693329, 34.7694550
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2845688, 29.2847519
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2660522, 44.2666016
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0390472, 38.0397110
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9400482, 27.9399223
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4945831, 28.4959297
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5243607, 26.5240097
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0876160, 22.0870438
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8343582, 23.8336906
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8496246, 22.8497925
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2652435, 31.2652817
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3117752, 27.3120041
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9489212, 32.9487000
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5748138, 28.5742188
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4098053, 27.4099693
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7411385, 24.7414398
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4105148, 32.4109344
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1371918, 33.1365738
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5751114, 30.5750732
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5821381, 44.5826645
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2341614, 27.2352905
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1422577, 34.1426773
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3600235, 36.3600540
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5637207, 35.5635757
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7635803, 40.7626266
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2604370, 48.2602081
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1832275, 24.1830521
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7474594, 30.7473259
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0531921, 23.0530853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1586

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9712168, upper bound: 16.9677584
time: 49.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9871059, upper bound: 16.9518824
time: 49.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8810501, 40.8813171
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9199448, 20.9202576
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8228683, 25.8226929
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1499786, 25.1503525
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8015823, 31.8020248
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9607086, 23.9609833
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2609253, 32.2609329
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7694550, 34.7693329
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2847519, 29.2845688
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2666016, 44.2660446
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0397186, 38.0390396
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9399185, 27.9400482
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4959259, 28.4945869
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5240097, 26.5243607
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0870438, 22.0876160
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8336868, 23.8343620
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8497925, 22.8496170
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2652817, 31.2652435
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3120041, 27.3117752
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9487000, 32.9489212
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5742188, 28.5748138
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4099731, 27.4098091
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7414360, 24.7411385
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4109421, 32.4105148
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1365738, 33.1371918
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5750732, 30.5751152
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5826569, 44.5821457
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2352905, 27.2341614
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1426773, 34.1422577
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3600540, 36.3600235
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5635681, 35.5637360
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7626343, 40.7635803
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2602081, 48.2604370
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1830521, 24.1832275
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7473221, 30.7474556
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0530853, 23.0531921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1586

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9401818, upper bound: 16.9871059
time: 55.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9677584, upper bound: 16.9712168
time: 47.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8812027, 40.8811417
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9199524, 20.9202461
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8227005, 25.8228607
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1498566, 25.1504745
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8015137, 31.8020859
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9605179, 23.9611588
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2609253, 32.2609329
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7695007, 34.7692795
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2849045, 29.2843628
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2666626, 44.2659912
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0397491, 38.0390015
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9398422, 27.9401245
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4962006, 28.4943123
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5240250, 26.5243454
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0870590, 22.0876007
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8335037, 23.8345337
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8497772, 22.8496323
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2652740, 31.2652512
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3120575, 27.3117027
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9486694, 32.9489517
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5740509, 28.5749664
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4098816, 27.4098854
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7414360, 24.7411423
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4106903, 32.4107590
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1366653, 33.1371002
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5750198, 30.5751686
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5826263, 44.5821838
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2351990, 27.2342377
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1425400, 34.1423798
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3599930, 36.3600769
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5639038, 35.5633850
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7626648, 40.7635193
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2602997, 48.2603149
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1833191, 24.1829414
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7472153, 30.7475548
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0531006, 23.0531769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1586

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9680889, upper bound: 16.9704233
time: 63.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9839866, upper bound: 16.9545474
time: 48.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8812637, 40.8811035
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9199524, 20.9202461
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8227158, 25.8228455
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1496964, 25.1506042
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.8012390, 31.8024521
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9603729, 23.9613266
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2609406, 32.2609177
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7697144, 34.7690735
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2849274, 29.2843857
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2668152, 44.2658310
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0397491, 38.0390091
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9398041, 27.9402122
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4962006, 28.4943199
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5239487, 26.5244217
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0870209, 22.0876312
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8331985, 23.8348503
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8497772, 22.8496323
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2652588, 31.2652664
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3120651, 27.3117065
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9485779, 32.9490967
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5737305, 28.5753098
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4098587, 27.4099274
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7414360, 24.7411461
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4106674, 32.4107971
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1366425, 33.1371307
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5750046, 30.5751801
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5826263, 44.5821838
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2352448, 27.2342339
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1425247, 34.1424179
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3599167, 36.3601532
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5640259, 35.5632629
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7625122, 40.7637482
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2603607, 48.2602844
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1833115, 24.1829872
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7471542, 30.7476234
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0531387, 23.0531387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1586

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9712168, upper bound: 16.9560535
time: 53.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9989555, upper bound: 16.9401818
time: 52.43 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 107.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 107.80
Output dim: 10, lower bound: -16.9401818, upper bound: 16.9989555
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 107.80
Output dim: 10, lower bound: -16.9560535, upper bound: 16.9830737
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 107.80
Output dim: 10, lower bound: -16.9545474, upper bound: 16.9839866
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 107.80
Output dim: 10, lower bound: -16.9704233, upper bound: 16.9680889
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 107.80
Output dim: 10, lower bound: -16.9712168, upper bound: 16.9677584
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 107.80
Output dim: 10, lower bound: -16.9871059, upper bound: 16.9518824
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 107.80
Output dim: 10, lower bound: -16.9401818, upper bound: 16.9871059
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 107.80
Output dim: 10, lower bound: -16.9677584, upper bound: 16.9712168
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 107.80
Output dim: 10, lower bound: -16.9680889, upper bound: 16.9704233
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 107.80
Output dim: 10, lower bound: -16.9839866, upper bound: 16.9545474
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 107.80
Output dim: 10, lower bound: -16.9712168, upper bound: 16.9560535
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 107.80
Output dim: 10, lower bound: -16.9989555, upper bound: 16.9401818

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8830795, 40.8834381
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9201279, 20.9198608
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8245621, 25.8243713
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1525574, 25.1515579
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.7960358, 31.7939148
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9634018, 23.9624596
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2615967, 32.2616196
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7673416, 34.7680893
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2788162, 29.2785721
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2591553, 44.2600479
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0446014, 38.0445328
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9400864, 27.9396667
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4926758, 28.4939842
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5113907, 26.5110779
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0891800, 22.0887260
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8361740, 23.8344116
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8515434, 22.8516922
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2640457, 31.2642059
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3114243, 27.3115807
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9435806, 32.9431381
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5734787, 28.5719528
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4087830, 27.4087296
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7411041, 24.7413864
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4133377, 32.4129868
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1423264, 33.1419220
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5724335, 30.5722198
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5888519, 44.5895386
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2406693, 27.2417717
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1501694, 34.1502533
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3616257, 36.3613510
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5733566, 35.5744553
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7664337, 40.7651138
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2667999, 48.2671967
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1896210, 24.1901169
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7500000, 30.7496147
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0475502, 23.0474167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1566

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9154909, upper bound: 16.9979608
time: 66.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9391870, upper bound: 16.9742775
time: 52.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8834915, 40.8830185
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9201584, 20.9198265
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8243484, 25.8245926
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1522217, 25.1519318
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.7947083, 31.7951660
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9630814, 23.9627876
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2615967, 32.2616119
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7677078, 34.7677155
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2782135, 29.2791824
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2592773, 44.2599182
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0438080, 38.0452957
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9399033, 27.9397964
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4923706, 28.4942894
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5114822, 26.5109787
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0893173, 22.0885963
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8355789, 23.8350105
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8515282, 22.8516998
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2641907, 31.2640610
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3112869, 27.3117218
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9434814, 32.9431839
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5730438, 28.5723953
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4086838, 27.4088249
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7410965, 24.7413979
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4128342, 32.4134827
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1424789, 33.1417694
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5723343, 30.5723267
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5890503, 44.5893326
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2406921, 27.2417221
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1499863, 34.1504288
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3614578, 36.3615265
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5741501, 35.5736771
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7661591, 40.7653046
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2672882, 48.2667236
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1900330, 24.1896896
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7499161, 30.7497025
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0474663, 23.0474930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1566

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9624252, upper bound: 16.9508875
time: 54.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9861112, upper bound: 16.9271927
time: 56.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8830185, 40.8834915
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9198227, 20.9201622
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8245926, 25.8243484
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1519318, 25.1522217
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.7951660, 31.7947083
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9627762, 23.9630775
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2616119, 32.2616043
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7677231, 34.7677155
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2791824, 29.2782135
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2599182, 44.2592773
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0453033, 38.0438156
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9397964, 27.9399071
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4942932, 28.4923782
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5109787, 26.5114899
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0886002, 22.0893135
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8350143, 23.8355751
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8516960, 22.8515282
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2640610, 31.2641907
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3117218, 27.3112869
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9431839, 32.9434814
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5723953, 28.5730438
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4088211, 27.4086838
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7414017, 24.7410927
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4134827, 32.4128342
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1417694, 33.1424713
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5723267, 30.5723267
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5893402, 44.5890503
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2417297, 27.2406883
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1504288, 34.1499863
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3615341, 36.3614502
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5736618, 35.5741501
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7653046, 40.7661667
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2667236, 48.2672882
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1896896, 24.1900330
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7497025, 30.7499161
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0474968, 23.0474701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1566

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9271927, upper bound: 16.9861112
time: 55.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9508876, upper bound: 16.9624252
time: 39.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -34.4155960, 6.5947161, -34.4155960, 6.5947161, -40.8834457, 40.8830872
1: -9.3184948, 13.6812601, -9.3184948, 13.6812601, -22.9997559, 22.9997559
2: -8.3236542, 13.4676647, -8.3236542, 13.4676647, -20.9198608, 20.9201279
3: -10.0594912, 16.8341255, -10.0594912, 16.8341255, -25.8243790, 25.8245621
4: -16.3965263, 14.4441528, -16.3965263, 14.4441528, -30.8406792, 30.8406792
5: -9.3902941, 16.0085144, -9.3902941, 16.0085144, -25.1515579, 25.1525574
6: -35.8659782, -0.8471365, -35.8659782, -0.8471365, -31.7939148, 31.7960358
7: -9.4362078, 16.1941643, -9.4362078, 16.1941643, -23.9624557, 23.9633980
8: -19.9869308, 18.4321613, -19.9869308, 18.4321613, -38.4190903, 38.4190903
9: -6.8903174, 28.0302124, -6.8903174, 28.0302124, -32.2616272, 32.2615967
10: -5.6407290, 31.8775368, -5.6407290, 31.8775368, -34.7680893, 34.7673416
11: -11.3507690, 15.0998383, -11.3507690, 15.0998383, -26.4506073, 26.4506073
12: -18.4545746, 16.3143616, -18.4545746, 16.3143616, -29.2785721, 29.2788162
13: -17.9727459, 28.9822502, -17.9727459, 28.9822502, -44.2600403, 44.2591476
14: -26.7888527, 15.5888100, -26.7888527, 15.5888100, -38.0445404, 38.0445938
15: -19.5834122, 11.9265633, -19.5834122, 11.9265633, -31.5099754, 31.5099754
16: -15.5452671, 13.0821056, -15.5452671, 13.0821056, -27.9396667, 27.9400864
17: -28.7674770, 8.7101002, -28.7674770, 8.7101002, -28.4939880, 28.4926796
18: -29.0741482, 3.1863661, -29.0741482, 3.1863661, -26.5110779, 26.5113907
19: -16.0970421, 6.8077726, -16.0970421, 6.8077726, -22.0887222, 22.0891838
20: -11.9761553, 12.4021196, -11.9761553, 12.4021196, -23.8344193, 23.8361702
21: -15.2944660, 12.3189144, -15.2944660, 12.3189144, -27.6133804, 27.6133804
22: -17.1021347, 6.9362593, -17.1021347, 6.9362593, -22.8516960, 22.8515396
23: -15.4003029, 11.1726341, -15.4003029, 11.1726341, -26.5729370, 26.5729370
24: -27.4189568, 6.4573197, -27.4189568, 6.4573197, -31.2642059, 31.2640457
25: -17.4450817, 10.7129469, -17.4450817, 10.7129469, -27.3115845, 27.3114243
26: -22.1468601, 12.8691368, -22.1468601, 12.8691368, -32.9431381, 32.9435806
27: -28.2854176, 5.2499361, -28.2854176, 5.2499361, -28.5719528, 28.5734787
28: -15.5098782, 13.1773853, -15.5098782, 13.1773853, -27.4087296, 27.4087830
29: -15.6862774, 9.5340004, -15.6862774, 9.5340004, -24.7413864, 24.7411041
30: -17.5304642, 16.4612694, -17.5304642, 16.4612694, -32.4129944, 32.4133377
31: -26.9910278, 7.1242313, -26.9910278, 7.1242313, -33.1419220, 33.1423187
32: -26.2133141, 7.1567116, -26.2133141, 7.1567116, -30.5722198, 30.5724335
33: -64.4204102, -14.3824625, -64.4204102, -14.3824625, -44.5895386, 44.5888519
34: -51.9972687, -16.6826344, -51.9972687, -16.6826344, -27.2417755, 27.2406693
35: -46.5284424, -8.3111324, -46.5284424, -8.3111324, -34.1502533, 34.1501694
36: -40.1993790, -2.2172003, -40.1993790, -2.2172003, -36.3613510, 36.3616257
37: -67.3510742, -23.7942066, -67.3510742, -23.7942066, -35.5744553, 35.5733643
38: -50.6460075, -7.1694117, -50.6460075, -7.1694117, -40.7651215, 40.7664261
39: -61.3986816, -10.3262959, -61.3986816, -10.3262959, -48.2671967, 48.2667999
40: -52.4808998, -22.0429268, -52.4808998, -22.0429268, -24.1901169, 24.1896210
41: -37.7669983, -1.4972658, -37.7669983, -1.4972658, -30.7496185, 30.7500000
42: -24.2245750, 0.5974169, -24.2245750, 0.5974169, -23.0474129, 23.0475464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=270, inp2_unstable=270, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=292, inp2_unstable=292, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=33, inp2_unstable=33, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1566

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -16.9742775, upper bound: 16.9391869
time: 43.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -16.9979608, upper bound: 16.9154909
time: 43.94 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 89.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 89.26
Output dim: 10, lower bound: -16.9154909, upper bound: 16.9979608
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 89.26
Output dim: 10, lower bound: -16.9391870, upper bound: 16.9742775
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 89.26
Output dim: 10, lower bound: -16.9624252, upper bound: 16.9508875
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 89.26
Output dim: 10, lower bound: -16.9861112, upper bound: 16.9271927
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 89.26
Output dim: 10, lower bound: -16.9271927, upper bound: 16.9861112
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 89.26
Output dim: 10, lower bound: -16.9508876, upper bound: 16.9624252
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 89.26
Output dim: 10, lower bound: -16.9742775, upper bound: 16.9391869
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 89.26
Output dim: 10, lower bound: -16.9979608, upper bound: 16.9154909

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 64.48 + 1812.90 = 1877.38 seconds
