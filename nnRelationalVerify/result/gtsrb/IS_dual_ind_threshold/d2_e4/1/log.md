## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 3600 seconds
Split limit: 100
Threshold: 46.4948137449


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=389, inp2_unstable=389, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=559, inp2_unstable=559, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-84.7672882, 39.3736267, -84.7672882, 39.3736267, -124.1409149, 124.1408997)
1: (-42.5773087, 31.1177921, -42.5773087, 31.1177921, -73.6950912, 73.6950989)
2: (-34.6925201, 35.4725037, -34.6925201, 35.4725037, -70.1650238, 70.1650162)
3: (-46.5430717, 38.9130402, -46.5430717, 38.9130402, -85.4561157, 85.4561157)
4: (-46.9768639, 39.0464897, -46.9768639, 39.0464897, -86.0233536, 86.0233536)
5: (-43.2595673, 41.2553864, -43.2595673, 41.2553864, -84.5149384, 84.5149536)
6: (-61.4064789, 41.3731499, -61.4064789, 41.3731499, -102.7796173, 102.7796249)
7: (-52.2261963, 39.0390739, -52.2261963, 39.0390739, -91.2652740, 91.2652740)
8: (-66.3197632, 48.3093834, -66.3197632, 48.3093834, -114.6291428, 114.6291428)
9: (-44.0484085, 41.3931618, -44.0484085, 41.3931618, -85.4415665, 85.4415588)
10: (-58.5848846, 49.0133591, -58.5848846, 49.0133591, -107.5982361, 107.5982437)
11: (-49.7151680, 36.1760406, -49.7151680, 36.1760406, -85.8912048, 85.8912048)
12: (-66.5686646, 50.4459152, -66.5686646, 50.4459152, -117.0145798, 117.0145798)
13: (-71.8024597, 54.1911697, -71.8024597, 54.1911697, -125.9936295, 125.9936295)
14: (-104.4877167, 36.7296181, -104.4877167, 36.7296181, -141.2173309, 141.2173309)
15: (-50.6970444, 35.7664642, -50.6970444, 35.7664642, -86.4635086, 86.4635086)
16: (-58.7854805, 40.5555725, -58.7854805, 40.5555725, -99.3410416, 99.3410492)
17: (-101.2854004, 34.3224030, -101.2854004, 34.3224030, -135.6078033, 135.6078033)
18: (-58.6059418, 52.6934662, -58.6059418, 52.6934662, -111.2993927, 111.2993927)
19: (-34.6089745, 27.3176270, -34.6089745, 27.3176270, -61.9265976, 61.9266014)
20: (-39.1725731, 32.8310394, -39.1725731, 32.8310394, -72.0036011, 72.0036011)
21: (-46.2914467, 34.8583450, -46.2914467, 34.8583450, -81.1497803, 81.1497879)
22: (-49.6540070, 32.1295013, -49.6540070, 32.1295013, -81.7835083, 81.7835083)
23: (-36.4852295, 37.0147552, -36.4852295, 37.0147552, -73.4999771, 73.4999847)
24: (-48.5414314, 41.0755920, -48.5414314, 41.0755920, -89.6170197, 89.6170197)
25: (-44.3950119, 37.3724670, -44.3950119, 37.3724670, -81.7674713, 81.7674713)
26: (-64.4201508, 54.6686516, -64.4201508, 54.6686516, -119.0887985, 119.0887985)
27: (-47.7461967, 41.8227158, -47.7461967, 41.8227158, -89.5689011, 89.5689087)
28: (-38.2709885, 41.1035767, -38.2709885, 41.1035767, -79.3745651, 79.3745651)
29: (-48.5310211, 29.4440346, -48.5310211, 29.4440346, -77.9750519, 77.9750519)
30: (-50.2593193, 45.3741074, -50.2593193, 45.3741074, -95.6334152, 95.6334229)
31: (-47.6183662, 41.5403900, -47.6183662, 41.5403900, -89.1587524, 89.1587524)
32: (-59.9542313, 35.7897034, -59.9542313, 35.7897034, -95.7439270, 95.7439270)
33: (-85.9071808, 43.8326645, -85.9071808, 43.8326645, -129.7398376, 129.7398376)
34: (-79.8334198, 28.5293617, -79.8334198, 28.5293617, -108.3627777, 108.3627777)
35: (-70.2571030, 37.5221252, -70.2571030, 37.5221252, -107.7792282, 107.7792282)
36: (-71.2104950, 39.3675041, -71.2104950, 39.3675041, -110.5780029, 110.5780029)
37: (-97.7557907, 34.7876053, -97.7557907, 34.7876053, -132.5433960, 132.5433960)
38: (-87.2721405, 42.3223114, -87.2721405, 42.3223114, -129.5944519, 129.5944519)
39: (-96.3795929, 46.4996338, -96.3795929, 46.4996338, -142.8792114, 142.8792114)
40: (-73.8562927, 32.8088455, -73.8562927, 32.8088455, -106.6651382, 106.6651382)
41: (-64.6400299, 44.5428925, -64.6400299, 44.5428925, -109.1828995, 109.1829071)
42: (-48.1290588, 30.1027393, -48.1290588, 30.1027393, -78.2317963, 78.2317963)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.77 + 161.56 = 164.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 19, lower bound: -46.5413551, upper bound: 46.5413551

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 973

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5406916, upper bound: 46.5178504
time: 111.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5406916, upper bound: 46.5406916
time: 119.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 231.44 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 231.44
Output dim: 19, lower bound: -46.5406916, upper bound: 46.5178504
IS_A2, status: Status.UNKNOWN, split count: 1, time: 231.44
Output dim: 19, lower bound: -46.5406916, upper bound: 46.5406916

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -84.6492767, 39.3426781, -84.6958160, 39.3550606, -124.0043259, 124.0384979
1: -42.4727859, 31.0947056, -42.5141602, 31.1038723, -73.5766525, 73.6088562
2: -34.5970230, 35.4494743, -34.6350708, 35.4586563, -70.0556793, 70.0845490
3: -46.3708115, 38.8792458, -46.4397659, 38.8927040, -85.2635193, 85.3190155
4: -46.8696785, 39.0187302, -46.9125671, 39.0297432, -85.8994141, 85.9312973
5: -43.1086731, 41.2242393, -43.1689529, 41.2366486, -84.3453217, 84.3931885
6: -61.3528633, 41.3426895, -61.3739166, 41.3547363, -102.7075958, 102.7166061
7: -52.0585556, 39.0157089, -52.1252518, 39.0250168, -91.0835724, 91.1409607
8: -66.1233978, 48.2781601, -66.2015610, 48.2904930, -114.4138947, 114.4797211
9: -43.9320984, 41.3684387, -43.9786568, 41.3782654, -85.3103638, 85.3470917
10: -58.5204926, 48.9759941, -58.5460396, 48.9908104, -107.5113068, 107.5220337
11: -49.6502838, 36.1101532, -49.6759148, 36.1364365, -85.7866974, 85.7860718
12: -66.5258560, 50.3030434, -66.5428772, 50.3601608, -116.8860016, 116.8459167
13: -71.6618042, 54.1371689, -71.7180862, 54.1586037, -125.8204041, 125.8552551
14: -104.4020767, 36.6405411, -104.4359436, 36.6762161, -141.0782928, 141.0764771
15: -50.6567535, 35.7317047, -50.6726875, 35.7454910, -86.4022446, 86.4043884
16: -58.6653671, 40.5314941, -58.7128143, 40.5410614, -99.2064209, 99.2443008
17: -101.1826248, 34.2303772, -101.2232208, 34.2671280, -135.4497528, 135.4535828
18: -58.5568886, 52.4877777, -58.5763245, 52.5701370, -111.1270294, 111.0641022
19: -34.5682487, 27.2801399, -34.5844307, 27.2946682, -61.8629150, 61.8645706
20: -39.1354103, 32.7994690, -39.1501541, 32.8120232, -71.9474335, 71.9496231
21: -46.2393150, 34.8374138, -46.2599602, 34.8456841, -81.0849915, 81.0973663
22: -49.6101074, 32.0234337, -49.6275406, 32.0656319, -81.6757355, 81.6509705
23: -36.4469833, 36.9062691, -36.4621048, 36.9497223, -73.3967056, 73.3683777
24: -48.5007858, 40.9011536, -48.5169411, 40.9711227, -89.4719086, 89.4180908
25: -44.3555565, 37.2609291, -44.3712654, 37.3055496, -81.6611023, 81.6321869
26: -64.3729477, 54.5179825, -64.3917236, 54.5780029, -118.9509430, 118.9097061
27: -47.7022171, 41.7224350, -47.7196922, 41.7626762, -89.4648895, 89.4421234
28: -38.2368546, 41.0402908, -38.2504120, 41.0655136, -79.3023605, 79.2907028
29: -48.4779778, 29.3491402, -48.4989700, 29.3870773, -77.8650513, 77.8481064
30: -50.2133942, 45.2978973, -50.2316246, 45.3283844, -95.5417786, 95.5295181
31: -47.5685997, 41.4434204, -47.5883636, 41.4821854, -89.0507812, 89.0317764
32: -59.9095917, 35.7518311, -59.9273186, 35.7668610, -95.6764526, 95.6791534
33: -85.8583069, 43.7837372, -85.8777008, 43.8030853, -129.6613922, 129.6614380
34: -79.7960052, 28.3807716, -79.8108368, 28.4402351, -108.2362366, 108.1916046
35: -70.2217102, 37.4225693, -70.2357941, 37.4621086, -107.6838226, 107.6583633
36: -71.1822052, 39.2933083, -71.1934509, 39.3226776, -110.5048828, 110.4867554
37: -97.6895599, 34.6667900, -97.7157898, 34.7151680, -132.4047241, 132.3825684
38: -87.2236786, 42.1930237, -87.2429504, 42.2445145, -129.4681854, 129.4359589
39: -96.3137970, 46.4568520, -96.3399658, 46.4734116, -142.7872009, 142.7968140
40: -73.7969513, 32.7726135, -73.8203735, 32.7869720, -106.5839081, 106.5929871
41: -64.5952377, 44.4952469, -64.6129379, 44.5141449, -109.1093826, 109.1081848
42: -48.0752907, 30.0735588, -48.0963516, 30.0851212, -78.1604156, 78.1699066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=388, inp2_unstable=389, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=559, inp2_unstable=559, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 973

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 595

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5186815, upper bound: 46.5167692
time: 114.64 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5186815, upper bound: 46.5167692
time: 171.68 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -84.8086548, 39.4309540, -84.7595978, 39.3683548, -124.1770096, 124.1905518
1: -42.5865936, 31.1985035, -42.5723724, 31.1139870, -73.7005768, 73.7708740
2: -34.7027969, 35.5708847, -34.6883888, 35.4690437, -70.1718445, 70.2592697
3: -46.5530090, 39.0868797, -46.5364571, 38.9083099, -85.4613037, 85.6233368
4: -46.9841156, 39.1853905, -46.9720993, 39.0413208, -86.0254364, 86.1574860
5: -43.2681808, 41.4189301, -43.2533722, 41.2514954, -84.5196762, 84.6723022
6: -61.4413147, 41.3657684, -61.3859978, 41.3702621, -102.8115768, 102.7517548
7: -52.2298737, 39.1768875, -52.2188988, 39.0353584, -91.2652283, 91.3957825
8: -66.3302155, 48.4535522, -66.3116379, 48.3045654, -114.6347809, 114.7651825
9: -44.0743942, 41.5388641, -44.0428543, 41.3878174, -85.4622040, 85.5817184
10: -58.6106644, 49.0660019, -58.5797997, 49.0072517, -107.6179123, 107.6457977
11: -49.8154182, 36.1938019, -49.7085876, 36.1725464, -85.9879608, 85.9023895
12: -66.6890411, 50.4556541, -66.5626068, 50.4396400, -117.1286774, 117.0182495
13: -71.8119049, 54.2815475, -71.7950134, 54.1853600, -125.9972610, 126.0765610
14: -104.5984802, 36.7426453, -104.4779739, 36.7246284, -141.3231049, 141.2206116
15: -50.7081985, 35.8353500, -50.6932373, 35.7617188, -86.4699173, 86.5285873
16: -58.8380928, 40.6671982, -58.7777252, 40.5509605, -99.3890381, 99.4449158
17: -101.3638611, 34.3531570, -101.2766800, 34.3167725, -135.6806335, 135.6298370
18: -58.7477875, 52.7099304, -58.6000519, 52.6851273, -111.4329147, 111.3099823
19: -34.7210159, 27.3256073, -34.6044502, 27.3141937, -62.0352097, 61.9300575
20: -39.2153778, 32.8420639, -39.1666946, 32.8286743, -72.0440521, 72.0087585
21: -46.3734894, 34.8748169, -46.2858391, 34.8551636, -81.2286530, 81.1606522
22: -49.7896194, 32.1397972, -49.6486244, 32.1247330, -81.9143524, 81.7884140
23: -36.6311684, 37.0212097, -36.4805641, 37.0099373, -73.6411057, 73.5017700
24: -48.7230339, 41.0848541, -48.5359383, 41.0690002, -89.7920151, 89.6207886
25: -44.5355415, 37.3780670, -44.3903160, 37.3670731, -81.9026108, 81.7683868
26: -64.5424957, 54.6886177, -64.4136200, 54.6624832, -119.2049789, 119.1022339
27: -47.8657799, 41.8306732, -47.7395401, 41.8186493, -89.6844330, 89.5702057
28: -38.3789520, 41.1180153, -38.2670593, 41.0998154, -79.4787598, 79.3850708
29: -48.6593781, 29.4546146, -48.5238037, 29.4401283, -78.0995026, 77.9784164
30: -50.3591156, 45.3871155, -50.2530289, 45.3698235, -95.7289276, 95.6401443
31: -47.7751770, 41.5511398, -47.6132965, 41.5356026, -89.3107681, 89.1644363
32: -60.0302238, 35.7978821, -59.9476891, 35.7866020, -95.8168259, 95.7455597
33: -85.9810333, 43.8523407, -85.9015503, 43.8280602, -129.8090973, 129.7538910
34: -79.9697723, 28.5371532, -79.8288574, 28.5226383, -108.4924088, 108.3660049
35: -70.3335495, 37.5349960, -70.2531128, 37.5164795, -107.8500290, 107.7881012
36: -71.2983246, 39.3753967, -71.2046585, 39.3633041, -110.6616135, 110.5800552
37: -97.9239120, 34.8014832, -97.7468414, 34.7823448, -132.7062531, 132.5483246
38: -87.3722534, 42.3429718, -87.2646484, 42.3156853, -129.6879272, 129.6076202
39: -96.4729919, 46.5091171, -96.3727341, 46.4951935, -142.9681549, 142.8818512
40: -73.9070969, 32.8096275, -73.8481064, 32.8056641, -106.7127609, 106.6577301
41: -64.7216949, 44.5562515, -64.6337280, 44.5390053, -109.2606964, 109.1899796
42: -48.1651115, 30.1388016, -48.1215477, 30.1002274, -78.2653351, 78.2603455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=388, inp2_unstable=389, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=561, inp2_unstable=559, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 973

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 595

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5186815, upper bound: 46.5396178
time: 125.20 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5186815, upper bound: 46.5396178
time: 105.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 233.16 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 233.16
Output dim: 19, lower bound: -46.5186815, upper bound: 46.5167692
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 233.16
Output dim: 19, lower bound: -46.5186815, upper bound: 46.5167692
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 233.16
Output dim: 19, lower bound: -46.5186815, upper bound: 46.5396178
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 233.16
Output dim: 19, lower bound: -46.5186815, upper bound: 46.5396178

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -84.5897827, 39.3237457, -84.5771790, 39.2880402, -123.8778152, 123.9009247
1: -42.4528580, 31.0737343, -42.4915466, 31.0487995, -73.5016479, 73.5652771
2: -34.5657806, 35.4348450, -34.5773430, 35.4174576, -69.9832306, 70.0121841
3: -46.3026810, 38.8604622, -46.3201180, 38.8005371, -85.1032104, 85.1805725
4: -46.8110657, 38.9988861, -46.8039017, 38.9534760, -85.7645340, 85.8027878
5: -43.0405426, 41.2082977, -43.0458488, 41.1495514, -84.1900940, 84.2541428
6: -61.3208084, 41.3153534, -61.3023071, 41.2967567, -102.6175537, 102.6176605
7: -52.0349541, 38.9979553, -52.0917664, 38.9833603, -91.0183105, 91.0897217
8: -66.0914764, 48.2492828, -66.1417084, 48.2216148, -114.3130951, 114.3909836
9: -43.8841324, 41.3482399, -43.8917923, 41.2822838, -85.1664124, 85.2400284
10: -58.4786568, 48.9423103, -58.4607124, 48.8787575, -107.3574142, 107.4030151
11: -49.6155014, 36.0330048, -49.5432663, 36.0017014, -85.6172028, 85.5762711
12: -66.4830322, 50.2632904, -66.4649963, 50.2759552, -116.7589722, 116.7282715
13: -71.4995117, 54.0976868, -71.4360428, 53.9618378, -125.4613495, 125.5337296
14: -104.3148804, 36.6094666, -104.2579041, 36.6090355, -140.9239044, 140.8673706
15: -50.5917511, 35.7076035, -50.5466080, 35.6345139, -86.2262650, 86.2542114
16: -58.6351089, 40.4981918, -58.6301079, 40.4617310, -99.0968399, 99.1282959
17: -101.1084290, 34.1999207, -101.0898743, 34.1694412, -135.2778625, 135.2897949
18: -58.5251312, 52.4129333, -58.4548416, 52.4388084, -110.9639359, 110.8677673
19: -34.5393677, 27.2414246, -34.4702187, 27.2275810, -61.7669487, 61.7116432
20: -39.1089249, 32.7694778, -39.0589828, 32.7542953, -71.8632202, 71.8284607
21: -46.2024498, 34.7678223, -46.1141243, 34.7222214, -80.9246674, 80.8819427
22: -49.5764313, 32.0056839, -49.5417213, 32.0422516, -81.6186829, 81.5474014
23: -36.4128609, 36.8157578, -36.3149567, 36.7926636, -73.2055206, 73.1307068
24: -48.4643784, 40.7925797, -48.3588371, 40.7890053, -89.2533875, 89.1514130
25: -44.3280525, 37.2073517, -44.2610779, 37.2069969, -81.5350494, 81.4684296
26: -64.3347168, 54.4572830, -64.2599030, 54.4642982, -118.7990036, 118.7171860
27: -47.6687965, 41.6035309, -47.5792847, 41.5575180, -89.2263107, 89.1828156
28: -38.2122383, 40.9715271, -38.1390762, 40.9371643, -79.1493988, 79.1106033
29: -48.4424553, 29.3170433, -48.4071770, 29.3269749, -77.7694321, 77.7242203
30: -50.1836853, 45.1903038, -50.0848846, 45.1334953, -95.3171692, 95.2751846
31: -47.5269547, 41.3967056, -47.4288445, 41.4005775, -88.9275208, 88.8255463
32: -59.8459091, 35.7317200, -59.8054733, 35.7135391, -95.5594482, 95.5371933
33: -85.8149109, 43.7661057, -85.7669983, 43.7681656, -129.5830688, 129.5331116
34: -79.7577209, 28.3595657, -79.7265320, 28.4058571, -108.1635742, 108.0860977
35: -70.1719055, 37.4129639, -70.1325912, 37.4496689, -107.6215668, 107.5455551
36: -71.0986557, 39.2776031, -71.0348053, 39.2789764, -110.3776321, 110.3124084
37: -97.6359100, 34.6024704, -97.5700073, 34.5988693, -132.2347717, 132.1724701
38: -87.1055603, 42.1742020, -87.0203934, 42.1966209, -129.3021851, 129.1945953
39: -96.2322693, 46.4441681, -96.1740646, 46.4206696, -142.6529236, 142.6182251
40: -73.7576447, 32.7380257, -73.7313690, 32.7187576, -106.4764023, 106.4693909
41: -64.5621338, 44.4435806, -64.5384598, 44.4127808, -108.9749146, 108.9820404
42: -48.0521011, 30.0428867, -48.0391159, 30.0155010, -78.0675964, 78.0820007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=388, inp2_unstable=388, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=559, inp2_unstable=559, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 973

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1623

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4981156, upper bound: 46.4771675
time: 166.15 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4981156, upper bound: 46.4961360
time: 110.35 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -84.6454697, 39.3411484, -84.6886749, 39.3521309, -123.9975891, 124.0298233
1: -42.4711838, 31.0935211, -42.5111160, 31.1016502, -73.5728302, 73.6046371
2: -34.5950813, 35.4471054, -34.6314926, 35.4541397, -70.0492249, 70.0785980
3: -46.3683395, 38.8775787, -46.4351501, 38.8895607, -85.2579041, 85.3127289
4: -46.8675842, 39.0166130, -46.9086609, 39.0257187, -85.8933029, 85.9252777
5: -43.1059418, 41.2228775, -43.1637840, 41.2341003, -84.3400421, 84.3866577
6: -61.3438416, 41.3412323, -61.3571434, 41.3520470, -102.6958847, 102.6983719
7: -52.0563736, 39.0142365, -52.1212006, 39.0221786, -91.0785522, 91.1354370
8: -66.1217957, 48.2761269, -66.1986694, 48.2866783, -114.4084778, 114.4747925
9: -43.9300766, 41.3664551, -43.9748611, 41.3746147, -85.3046875, 85.3413086
10: -58.5181885, 48.9721794, -58.5417328, 48.9835052, -107.5016937, 107.5139084
11: -49.6473618, 36.1080399, -49.6704369, 36.1324768, -85.7798386, 85.7784729
12: -66.5214233, 50.3006020, -66.5345306, 50.3555870, -116.8770142, 116.8351212
13: -71.6571426, 54.1338272, -71.7093277, 54.1522560, -125.8094025, 125.8431473
14: -104.3980026, 36.6376381, -104.4283600, 36.6707382, -141.0687408, 141.0659943
15: -50.6545677, 35.7293777, -50.6685944, 35.7410660, -86.3956299, 86.3979645
16: -58.6622124, 40.5264587, -58.7068443, 40.5315094, -99.1937103, 99.2332916
17: -101.1790237, 34.2282944, -101.2163620, 34.2632103, -135.4422302, 135.4446564
18: -58.5546074, 52.4847260, -58.5720863, 52.5643539, -111.1189575, 111.0568085
19: -34.5651550, 27.2787209, -34.5785179, 27.2920132, -61.8571625, 61.8572350
20: -39.1334267, 32.7977371, -39.1464195, 32.8087883, -71.9422150, 71.9441528
21: -46.2361145, 34.8351746, -46.2539139, 34.8415146, -81.0776291, 81.0890884
22: -49.6064339, 32.0219650, -49.6205368, 32.0628815, -81.6693115, 81.6425018
23: -36.4441757, 36.9037476, -36.4568748, 36.9449844, -73.3891602, 73.3606262
24: -48.4978714, 40.8980064, -48.5114670, 40.9651947, -89.4630432, 89.4094696
25: -44.3533783, 37.2587814, -44.3671341, 37.3015900, -81.6549683, 81.6259155
26: -64.3703308, 54.5148811, -64.3867950, 54.5722656, -118.9425812, 118.9016724
27: -47.6979752, 41.7195282, -47.7116356, 41.7571716, -89.4551468, 89.4311600
28: -38.2336655, 41.0381813, -38.2443428, 41.0615692, -79.2952271, 79.2825241
29: -48.4732857, 29.3479958, -48.4900208, 29.3849525, -77.8582382, 77.8380127
30: -50.2107468, 45.2950821, -50.2266273, 45.3231354, -95.5338745, 95.5217133
31: -47.5653000, 41.4417000, -47.5820885, 41.4789581, -89.0442581, 89.0237885
32: -59.9013214, 35.7505074, -59.9139786, 35.7643280, -95.6656494, 95.6644897
33: -85.8557892, 43.7813911, -85.8729706, 43.7986603, -129.6544342, 129.6543427
34: -79.7870331, 28.3790073, -79.7937927, 28.4369011, -108.2239380, 108.1727982
35: -70.2169037, 37.4216003, -70.2266693, 37.4602814, -107.6771698, 107.6482697
36: -71.1727905, 39.2922363, -71.1754303, 39.3206291, -110.4934082, 110.4676666
37: -97.6856689, 34.6632385, -97.7084732, 34.7084579, -132.3941345, 132.3717041
38: -87.2185669, 42.1911964, -87.2336578, 42.2410812, -129.4596558, 129.4248505
39: -96.3102112, 46.4548416, -96.3332443, 46.4695206, -142.7797241, 142.7880859
40: -73.7939453, 32.7697830, -73.8147430, 32.7815475, -106.5754852, 106.5845261
41: -64.5907974, 44.4930763, -64.6046066, 44.5100517, -109.1008453, 109.0976868
42: -48.0720367, 30.0714378, -48.0902405, 30.0811081, -78.1531448, 78.1616821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=388, inp2_unstable=388, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=559, inp2_unstable=559, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 973

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1623

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4981156, upper bound: 46.4771675
time: 107.20 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4981156, upper bound: 46.4961360
time: 114.31 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -84.7491074, 39.4120331, -84.6409607, 39.3013725, -124.0504761, 124.0529861
1: -42.5666542, 31.1775398, -42.5497589, 31.0589390, -73.6255951, 73.7272949
2: -34.6715317, 35.5562325, -34.6306458, 35.4278564, -70.0993805, 70.1868744
3: -46.4848785, 39.0680923, -46.4167786, 38.8161316, -85.3010101, 85.4848709
4: -46.9255142, 39.1655540, -46.8634262, 38.9650192, -85.8905334, 86.0289764
5: -43.2000313, 41.4030266, -43.1302872, 41.1643829, -84.3644104, 84.5333099
6: -61.4092407, 41.3384399, -61.3143387, 41.3122597, -102.7214966, 102.6527786
7: -52.2062225, 39.1591225, -52.1854668, 38.9936752, -91.1998901, 91.3445892
8: -66.2982407, 48.4246864, -66.2517471, 48.2357178, -114.5339584, 114.6764297
9: -44.0264282, 41.5186958, -43.9559784, 41.2918625, -85.3182831, 85.4746704
10: -58.5688324, 49.0323181, -58.4944572, 48.8952179, -107.4640503, 107.5267792
11: -49.7806931, 36.1166267, -49.5759392, 36.0378113, -85.8184967, 85.6925659
12: -66.6462173, 50.4158211, -66.4846954, 50.3554497, -117.0016632, 116.9005127
13: -71.6496735, 54.2421646, -71.5129929, 53.9887085, -125.6383820, 125.7551575
14: -104.5112991, 36.7115288, -104.2999420, 36.6574554, -141.1687622, 141.0114746
15: -50.6431923, 35.8112450, -50.5671539, 35.6507797, -86.2939758, 86.3783951
16: -58.8078003, 40.6339111, -58.6951180, 40.4716263, -99.2794113, 99.3290253
17: -101.2896347, 34.3226585, -101.1433105, 34.2190857, -135.5087280, 135.4659729
18: -58.7160225, 52.6350327, -58.4785614, 52.5537949, -111.2698059, 111.1135941
19: -34.6921425, 27.2868862, -34.4902573, 27.2471199, -61.9392624, 61.7771454
20: -39.1888924, 32.8120613, -39.0755615, 32.7709389, -71.9598236, 71.8876190
21: -46.3366661, 34.8052254, -46.1400223, 34.7316742, -81.0683441, 80.9452515
22: -49.7559433, 32.1220245, -49.5628128, 32.1013412, -81.8572845, 81.6848373
23: -36.5970993, 36.9307251, -36.3334503, 36.8528862, -73.4499817, 73.2641754
24: -48.6866684, 40.9762840, -48.3778687, 40.8868904, -89.5735550, 89.3541489
25: -44.5080719, 37.3244553, -44.2801208, 37.2685089, -81.7765808, 81.6045761
26: -64.5042801, 54.6279144, -64.2818069, 54.5487747, -119.0530548, 118.9097214
27: -47.8324089, 41.7117424, -47.5991478, 41.6134720, -89.4458771, 89.3108902
28: -38.3544006, 41.0492516, -38.1557693, 40.9714737, -79.3258667, 79.2050171
29: -48.6238441, 29.4225063, -48.4319878, 29.3799992, -78.0038452, 77.8544846
30: -50.3294411, 45.2795181, -50.1063194, 45.1748962, -95.5043259, 95.3858261
31: -47.7335548, 41.5044479, -47.4538307, 41.4539795, -89.1875305, 88.9582825
32: -59.9666138, 35.7777481, -59.8257980, 35.7332916, -95.6999054, 95.6035461
33: -85.9377060, 43.8347435, -85.7908783, 43.7931595, -129.7308502, 129.6256104
34: -79.9315338, 28.5159569, -79.7445679, 28.4883404, -108.4198685, 108.2605133
35: -70.2837524, 37.5254059, -70.1499023, 37.5040855, -107.7878342, 107.6753082
36: -71.2147827, 39.3596649, -71.0459747, 39.3195724, -110.5343475, 110.4056396
37: -97.8702850, 34.7371941, -97.6011200, 34.6660233, -132.5362854, 132.3383179
38: -87.2541656, 42.3242111, -87.0420685, 42.2678146, -129.5219727, 129.3662720
39: -96.3915176, 46.4964447, -96.2067642, 46.4424477, -142.8339691, 142.7032166
40: -73.8677826, 32.7750168, -73.7591400, 32.7374420, -106.6052246, 106.5341492
41: -64.6886063, 44.5045967, -64.5592575, 44.4376297, -109.1262360, 109.0638580
42: -48.1419296, 30.1081562, -48.0643349, 30.0306034, -78.1725159, 78.1724854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=388, inp2_unstable=388, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=561, inp2_unstable=559, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 973

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1623

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4981156, upper bound: 46.4999206
time: 118.17 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4981156, upper bound: 46.5188362
time: 134.78 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -84.8048325, 39.4294434, -84.7524643, 39.3654785, -124.1703110, 124.1819077
1: -42.5849876, 31.1973171, -42.5693130, 31.1117821, -73.6967697, 73.7666245
2: -34.7008591, 35.5685272, -34.6847878, 35.4645538, -70.1654129, 70.2533112
3: -46.5505333, 39.0852127, -46.5318451, 38.9051590, -85.4556808, 85.6170502
4: -46.9820328, 39.1832390, -46.9681931, 39.0372696, -86.0193024, 86.1514282
5: -43.2654190, 41.4175568, -43.2482224, 41.2489243, -84.5143280, 84.6657791
6: -61.4322739, 41.3642883, -61.3691864, 41.3675613, -102.7998352, 102.7334671
7: -52.2276688, 39.1754074, -52.2148590, 39.0325241, -91.2601929, 91.3902664
8: -66.3285980, 48.4515076, -66.3086853, 48.3007507, -114.6293488, 114.7601852
9: -44.0723648, 41.5369110, -44.0390472, 41.3841591, -85.4565277, 85.5759583
10: -58.6083336, 49.0621796, -58.5754738, 48.9999809, -107.6083145, 107.6376495
11: -49.8125000, 36.1916847, -49.7030907, 36.1685715, -85.9810715, 85.8947754
12: -66.6846008, 50.4531555, -66.5542603, 50.4350586, -117.1196594, 117.0074158
13: -71.8072968, 54.2781906, -71.7862244, 54.1790276, -125.9863129, 126.0644073
14: -104.5944672, 36.7397461, -104.4704208, 36.7191620, -141.3136139, 141.2101746
15: -50.7059822, 35.8330078, -50.6891594, 35.7573318, -86.4633179, 86.5221634
16: -58.8349533, 40.6621704, -58.7717514, 40.5414047, -99.3763580, 99.4339218
17: -101.3602219, 34.3510590, -101.2698212, 34.3128281, -135.6730499, 135.6208649
18: -58.7454872, 52.7068405, -58.5958138, 52.6793480, -111.4248352, 111.3026428
19: -34.7178802, 27.3241920, -34.5985489, 27.3115463, -62.0294266, 61.9227409
20: -39.2133942, 32.8403358, -39.1629639, 32.8254395, -72.0388336, 72.0032959
21: -46.3702774, 34.8725739, -46.2798004, 34.8509941, -81.2212677, 81.1523743
22: -49.7859268, 32.1383057, -49.6415939, 32.1219978, -81.9079285, 81.7798996
23: -36.6283875, 37.0186958, -36.4753304, 37.0052185, -73.6336060, 73.4940186
24: -48.7201271, 41.0817108, -48.5304832, 41.0630493, -89.7831726, 89.6121979
25: -44.5333481, 37.3759384, -44.3861885, 37.3630943, -81.8964386, 81.7621307
26: -64.5399170, 54.6855698, -64.4086685, 54.6567001, -119.1966095, 119.0942383
27: -47.8615379, 41.8277283, -47.7314682, 41.8131027, -89.6746368, 89.5591965
28: -38.3757591, 41.1159210, -38.2609901, 41.0958710, -79.4716263, 79.3769073
29: -48.6546707, 29.4534607, -48.5148468, 29.4379883, -78.0926590, 77.9683075
30: -50.3564682, 45.3842926, -50.2480087, 45.3645630, -95.7210312, 95.6323013
31: -47.7718506, 41.5494308, -47.6070290, 41.5323792, -89.3042297, 89.1564636
32: -60.0219765, 35.7965317, -59.9343758, 35.7840767, -95.8060532, 95.7309113
33: -85.9785156, 43.8500023, -85.8968048, 43.8236542, -129.8021698, 129.7467957
34: -79.9608307, 28.5354233, -79.8118057, 28.5193405, -108.4801712, 108.3472290
35: -70.3287354, 37.5340118, -70.2440186, 37.5146523, -107.8433838, 107.7780228
36: -71.2889099, 39.3743248, -71.1866608, 39.3612175, -110.6501160, 110.5609894
37: -97.9200134, 34.7979469, -97.7395630, 34.7756195, -132.6956329, 132.5375061
38: -87.3671417, 42.3411789, -87.2553711, 42.3122635, -129.6794128, 129.5965576
39: -96.4694214, 46.5070877, -96.3660202, 46.4912758, -142.9606781, 142.8731079
40: -73.9040527, 32.8067856, -73.8424530, 32.8002815, -106.7043304, 106.6492386
41: -64.7172775, 44.5540810, -64.6253967, 44.5348969, -109.2521667, 109.1794739
42: -48.1618614, 30.1366863, -48.1153946, 30.0962200, -78.2580795, 78.2520828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=388, inp2_unstable=388, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=561, inp2_unstable=559, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 973

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1623

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4981156, upper bound: 46.4999206
time: 124.82 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4981156, upper bound: 46.5188362
time: 126.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 253.84 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 253.84
Output dim: 19, lower bound: -46.4981156, upper bound: 46.4771675
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 253.84
Output dim: 19, lower bound: -46.4981156, upper bound: 46.4961360
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 253.84
Output dim: 19, lower bound: -46.4981156, upper bound: 46.4771675
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 253.84
Output dim: 19, lower bound: -46.4981156, upper bound: 46.4961360
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 253.84
Output dim: 19, lower bound: -46.4981156, upper bound: 46.4999206
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 253.84
Output dim: 19, lower bound: -46.4981156, upper bound: 46.5188362
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 253.84
Output dim: 19, lower bound: -46.4981156, upper bound: 46.4999206
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 253.84
Output dim: 19, lower bound: -46.4981156, upper bound: 46.5188362

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -84.5582657, 39.2701149, -84.5595932, 39.2584419, -123.8166962, 123.8297119
1: -42.4402122, 31.0419998, -42.4845047, 31.0311623, -73.4713745, 73.5264969
2: -34.5536728, 35.3992805, -34.5705910, 35.3966751, -69.9503326, 69.9698715
3: -46.2853165, 38.8415527, -46.3105087, 38.7899017, -85.0752182, 85.1520615
4: -46.7955017, 38.9339218, -46.7952042, 38.9171524, -85.7126541, 85.7291183
5: -43.0288010, 41.1652069, -43.0393600, 41.1254883, -84.1542892, 84.2045670
6: -61.2300758, 41.2979202, -61.2521896, 41.2870636, -102.5171356, 102.5501099
7: -52.0184631, 38.9505997, -52.0826530, 38.9565887, -90.9750519, 91.0332489
8: -66.0795670, 48.1899414, -66.1350250, 48.1884537, -114.2680206, 114.3249588
9: -43.8176689, 41.3305588, -43.8545723, 41.2724113, -85.0900803, 85.1851273
10: -58.4391479, 48.9109230, -58.4386215, 48.8612823, -107.3004303, 107.3495331
11: -49.5801811, 35.9891281, -49.5235977, 35.9773254, -85.5575104, 85.5127182
12: -66.3522034, 50.2341576, -66.3915100, 50.2598419, -116.6120224, 116.6256714
13: -71.3688354, 54.0726242, -71.3639832, 53.9479828, -125.3168182, 125.4366074
14: -104.2725372, 36.4943771, -104.2341614, 36.5449142, -140.8174438, 140.7285309
15: -50.5672226, 35.6828995, -50.5328636, 35.6207962, -86.1880188, 86.2157593
16: -58.6029282, 40.4562111, -58.6122589, 40.4381638, -99.0410919, 99.0684586
17: -101.0719528, 34.0575104, -101.0693970, 34.0911713, -135.1631165, 135.1269073
18: -58.4920769, 52.2677231, -58.4363976, 52.3573074, -110.8493805, 110.7041092
19: -34.5236397, 27.1892281, -34.4614716, 27.1987419, -61.7223778, 61.6506996
20: -39.0848236, 32.7267456, -39.0456429, 32.7306213, -71.8154449, 71.7723846
21: -46.1786575, 34.7112617, -46.1009102, 34.6906891, -80.8693466, 80.8121719
22: -49.5425758, 31.9639530, -49.5228615, 32.0191040, -81.5616684, 81.4868164
23: -36.3944435, 36.7640762, -36.3047447, 36.7641068, -73.1585464, 73.0688171
24: -48.4283333, 40.6850586, -48.3388596, 40.7297745, -89.1581116, 89.0239105
25: -44.3056526, 37.1331406, -44.2485886, 37.1660461, -81.4716949, 81.3817291
26: -64.2954712, 54.3988075, -64.2380371, 54.4320831, -118.7275467, 118.6368332
27: -47.6320000, 41.5086365, -47.5589142, 41.5049438, -89.1369324, 89.0675507
28: -38.1927910, 40.9100342, -38.1283188, 40.9032669, -79.0960541, 79.0383530
29: -48.4021912, 29.2893353, -48.3847694, 29.3115044, -77.7136993, 77.6741028
30: -50.1507797, 45.0983162, -50.0666275, 45.0818253, -95.2326050, 95.1649399
31: -47.5029526, 41.3190727, -47.4155312, 41.3559265, -88.8588715, 88.7346039
32: -59.6860771, 35.7064095, -59.7165527, 35.6994171, -95.3854904, 95.4229584
33: -85.7354431, 43.7505302, -85.7230835, 43.7594986, -129.4949341, 129.4736023
34: -79.7133942, 28.3483009, -79.7019196, 28.3996201, -108.1130142, 108.0502167
35: -70.1222992, 37.4048615, -70.1046143, 37.4451485, -107.5674438, 107.5094757
36: -71.0215073, 39.2696648, -70.9920197, 39.2745857, -110.2960892, 110.2616882
37: -97.5593185, 34.5878677, -97.5274506, 34.5907211, -132.1500092, 132.1153259
38: -87.0404053, 42.1548615, -86.9839325, 42.1858711, -129.2262726, 129.1387939
39: -96.1273575, 46.4328995, -96.1159821, 46.4144173, -142.5417480, 142.5488892
40: -73.6714020, 32.7259293, -73.6835251, 32.7120209, -106.3834229, 106.4094543
41: -64.4539490, 44.4250107, -64.4772644, 44.4024811, -108.8564224, 108.9022751
42: -47.9278374, 30.0220203, -47.9695969, 30.0038815, -77.9317017, 77.9916153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=387, inp2_unstable=388, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=559, inp2_unstable=559, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 973

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 19, lower bound: -46.4845979, upper bound: 46.4758298
time: 175.15 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4967778, upper bound: 46.4758298
time: 116.15 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -84.6332092, 39.3185234, -84.5682373, 39.2648201, -123.8980255, 123.8867645
1: -42.4797096, 31.0768585, -42.4881439, 31.0376968, -73.5174103, 73.5650024
2: -34.6082535, 35.4503441, -34.5744896, 35.4093971, -70.0176544, 70.0248260
3: -46.3338776, 38.8786736, -46.3130417, 38.7942238, -85.1280975, 85.1917114
4: -46.8715858, 39.0032654, -46.7987099, 38.9332161, -85.8048019, 85.8019714
5: -43.0970192, 41.2295761, -43.0428391, 41.1390762, -84.2360916, 84.2724152
6: -61.3389130, 41.4129791, -61.2917824, 41.2930908, -102.6320038, 102.7047577
7: -52.0982170, 39.0004616, -52.0888100, 38.9743347, -91.0725479, 91.0892715
8: -66.1486359, 48.2596359, -66.1388626, 48.2108688, -114.3595047, 114.3984985
9: -43.9026527, 41.4512558, -43.8823509, 41.2777939, -85.1804504, 85.3336029
10: -58.4955444, 48.9874077, -58.4450874, 48.8729019, -107.3684464, 107.4324875
11: -49.7316437, 36.0462303, -49.5384979, 35.9915390, -85.7231827, 85.5847244
12: -66.4990540, 50.4210968, -66.4543152, 50.2714615, -116.7705154, 116.8754044
13: -71.5247650, 54.3088112, -71.4257126, 53.9557114, -125.4804764, 125.7345123
14: -104.4409943, 36.6076965, -104.2481384, 36.6001549, -141.0411530, 140.8558350
15: -50.6136208, 35.7433472, -50.5376511, 35.6309662, -86.2445831, 86.2809906
16: -58.7482872, 40.5147705, -58.6247635, 40.4534912, -99.2017822, 99.1395340
17: -101.3159561, 34.2026482, -101.0811691, 34.1542816, -135.4702454, 135.2838135
18: -58.7341537, 52.4194832, -58.4446297, 52.4202652, -111.1544189, 110.8641129
19: -34.6347694, 27.2519951, -34.4657669, 27.2237625, -61.8585320, 61.7177620
20: -39.1578903, 32.7739563, -39.0550308, 32.7497253, -71.9076157, 71.8289871
21: -46.2985077, 34.7699318, -46.1097031, 34.7148972, -81.0133972, 80.8796310
22: -49.6604042, 32.0127487, -49.5355072, 32.0322762, -81.6926575, 81.5482559
23: -36.4975357, 36.8251305, -36.3116264, 36.7861099, -73.2836456, 73.1367569
24: -48.6170425, 40.8099899, -48.3518028, 40.7821617, -89.3991852, 89.1617889
25: -44.3914413, 37.2172241, -44.2558632, 37.2001648, -81.5916061, 81.4730835
26: -64.4642792, 54.4588356, -64.2526932, 54.4475441, -118.9118195, 118.7115326
27: -47.8135796, 41.6141586, -47.5723381, 41.5512009, -89.3647766, 89.1864929
28: -38.3009872, 40.9813080, -38.1356277, 40.9312325, -79.2322235, 79.1169281
29: -48.5094681, 29.3291187, -48.3998718, 29.3210506, -77.8305206, 77.7289886
30: -50.3236122, 45.2003708, -50.0801468, 45.1229095, -95.4465179, 95.2805176
31: -47.6377945, 41.4125443, -47.4223213, 41.3943710, -89.0321655, 88.8348541
32: -59.8581200, 35.8969383, -59.7922821, 35.7082214, -95.5663452, 95.6892242
33: -85.8371429, 43.8804016, -85.7554703, 43.7649345, -129.6020813, 129.6358643
34: -79.7819138, 28.3809528, -79.7194824, 28.4005394, -108.1824493, 108.1004333
35: -70.1883392, 37.4491119, -70.1237793, 37.4477043, -107.6360474, 107.5728912
36: -71.1046448, 39.3660240, -71.0193176, 39.2775345, -110.3821793, 110.3853455
37: -97.6544189, 34.6588364, -97.5533905, 34.5963287, -132.2507324, 132.2122192
38: -87.1172943, 42.2579346, -87.0055237, 42.1939354, -129.3112335, 129.2634583
39: -96.2306442, 46.5878944, -96.1512985, 46.4191742, -142.6498108, 142.7391815
40: -73.7660904, 32.8295441, -73.7111816, 32.7164001, -106.4824905, 106.5407257
41: -64.5793228, 44.5421638, -64.5293579, 44.4099655, -108.9892731, 109.0715103
42: -48.0617714, 30.2178764, -48.0279274, 30.0102596, -78.0720291, 78.2458038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=387, inp2_unstable=388, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=560, inp2_unstable=559, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 973

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 19, lower bound: -46.4845979, upper bound: 46.4948029
time: 126.77 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4967778, upper bound: 46.4948029
time: 104.30 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -84.6139374, 39.2874718, -84.6710815, 39.3225555, -123.9364929, 123.9585495
1: -42.4585037, 31.0618057, -42.5040665, 31.0840340, -73.5425339, 73.5658722
2: -34.5829544, 35.4115410, -34.6247482, 35.4333763, -70.0163269, 70.0362854
3: -46.3509903, 38.8586502, -46.4255600, 38.8789673, -85.2299576, 85.2842102
4: -46.8520126, 38.9516296, -46.8999672, 38.9894333, -85.8414307, 85.8515930
5: -43.0941925, 41.1797523, -43.1572952, 41.2100372, -84.3042297, 84.3370514
6: -61.2531128, 41.3237915, -61.3070526, 41.3423386, -102.5954514, 102.6308441
7: -52.0399094, 38.9669113, -52.1120338, 38.9953690, -91.0352783, 91.0789413
8: -66.1098175, 48.2168236, -66.1920013, 48.2534866, -114.3633041, 114.4088287
9: -43.8635941, 41.3488197, -43.9376144, 41.3647308, -85.2283249, 85.2864380
10: -58.4786873, 48.9408188, -58.5196495, 48.9660568, -107.4447403, 107.4604645
11: -49.6121330, 36.0641708, -49.6508331, 36.1081085, -85.7202454, 85.7150040
12: -66.3905945, 50.2714844, -66.4610443, 50.3394928, -116.7300873, 116.7325287
13: -71.5264435, 54.1087341, -71.6372528, 54.1383667, -125.6648102, 125.7459869
14: -104.3557129, 36.5225601, -104.4046097, 36.6065559, -140.9622650, 140.9271698
15: -50.6300125, 35.7047157, -50.6548347, 35.7273369, -86.3573456, 86.3595428
16: -58.6300621, 40.4844971, -58.6890717, 40.5079117, -99.1379700, 99.1735687
17: -101.1425705, 34.0858765, -101.1958618, 34.1849327, -135.3274994, 135.2817383
18: -58.5215759, 52.3394928, -58.5536575, 52.4828339, -111.0044022, 110.8931503
19: -34.5494423, 27.2265339, -34.5698090, 27.2631760, -61.8126183, 61.7963409
20: -39.1093712, 32.7550087, -39.1330948, 32.7851257, -71.8944855, 71.8880997
21: -46.2123642, 34.7786255, -46.2407303, 34.8099823, -81.0223389, 81.0193558
22: -49.5725822, 31.9802170, -49.6016731, 32.0397491, -81.6123352, 81.5818863
23: -36.4258156, 36.8520546, -36.4466705, 36.9164352, -73.3422546, 73.2987213
24: -48.4618530, 40.7905197, -48.4914932, 40.9059525, -89.3678055, 89.2820129
25: -44.3309746, 37.1845779, -44.3546906, 37.2606430, -81.5916138, 81.5392685
26: -64.3311462, 54.4564743, -64.3649445, 54.5400238, -118.8711624, 118.8214188
27: -47.6611900, 41.6246643, -47.6913033, 41.7045898, -89.3657837, 89.3159637
28: -38.2142410, 40.9767075, -38.2335739, 41.0276871, -79.2419281, 79.2102814
29: -48.4330482, 29.3202763, -48.4676361, 29.3694687, -77.8025055, 77.7879105
30: -50.1778870, 45.2031097, -50.2084122, 45.2715034, -95.4493866, 95.4115143
31: -47.5413246, 41.3640556, -47.5687943, 41.4343147, -88.9756393, 88.9328461
32: -59.7414932, 35.7251892, -59.8250504, 35.7501984, -95.4916916, 95.5502319
33: -85.7763672, 43.7658539, -85.8290482, 43.7899933, -129.5663605, 129.5949097
34: -79.7426910, 28.3677292, -79.7691345, 28.4306488, -108.1733398, 108.1368637
35: -70.1672745, 37.4134865, -70.1986694, 37.4557343, -107.6229858, 107.6121521
36: -71.0956650, 39.2843437, -71.1326599, 39.3162384, -110.4119034, 110.4170074
37: -97.6091156, 34.6485863, -97.6658936, 34.7002907, -132.3094025, 132.3144836
38: -87.1534348, 42.1718407, -87.1972198, 42.2303467, -129.3837891, 129.3690491
39: -96.2053528, 46.4435005, -96.2751694, 46.4632683, -142.6686096, 142.7186584
40: -73.7076874, 32.7577248, -73.7668991, 32.7748375, -106.4825134, 106.5246277
41: -64.4826355, 44.4744797, -64.5434418, 44.4997253, -108.9823608, 109.0179214
42: -47.9477615, 30.0505791, -48.0207062, 30.0694656, -78.0172272, 78.0712891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=387, inp2_unstable=388, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=559, inp2_unstable=559, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 973

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5054367, upper bound: 46.4758298
time: 118.74 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5175073, upper bound: 46.4758298
time: 111.04 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -84.6889267, 39.3359680, -84.6797333, 39.3289490, -124.0178757, 124.0157013
1: -42.4980240, 31.0966797, -42.5077286, 31.0905628, -73.5885849, 73.6044006
2: -34.6375580, 35.4626274, -34.6286392, 35.4460678, -70.0836258, 70.0912628
3: -46.3995438, 38.8957901, -46.4280663, 38.8832474, -85.2827911, 85.3238525
4: -46.9280777, 39.0210457, -46.9034729, 39.0054550, -85.9335327, 85.9245148
5: -43.1623917, 41.2440910, -43.1607780, 41.2236366, -84.3860321, 84.4048691
6: -61.3619461, 41.4387894, -61.3466072, 41.3483963, -102.7103424, 102.7854004
7: -52.1196289, 39.0167732, -52.1182289, 39.0131226, -91.1327515, 91.1350021
8: -66.1789246, 48.2864990, -66.1958389, 48.2759094, -114.4548340, 114.4823380
9: -43.9485931, 41.4695053, -43.9654007, 41.3700905, -85.3186798, 85.4349060
10: -58.5350723, 49.0172806, -58.5260963, 48.9776649, -107.5127411, 107.5433807
11: -49.7635612, 36.1212692, -49.6656990, 36.1223221, -85.8858719, 85.7869720
12: -66.5374451, 50.4583359, -66.5238953, 50.3510590, -116.8885040, 116.9822311
13: -71.6823807, 54.3449249, -71.6989746, 54.1460571, -125.8284302, 126.0438919
14: -104.5241089, 36.6358795, -104.4186096, 36.6618652, -141.1859741, 141.0544891
15: -50.6764145, 35.7651405, -50.6596298, 35.7375298, -86.4139404, 86.4247742
16: -58.7753983, 40.5430794, -58.7015305, 40.5232697, -99.2986603, 99.2446136
17: -101.3865051, 34.2310371, -101.2076721, 34.2480316, -135.6345367, 135.4386902
18: -58.7636261, 52.4912567, -58.5619049, 52.5458145, -111.3094406, 111.0531616
19: -34.6605682, 27.2892952, -34.5740852, 27.2881947, -61.9487610, 61.8633766
20: -39.1824150, 32.8022270, -39.1424637, 32.8042221, -71.9866333, 71.9446869
21: -46.3322067, 34.8373032, -46.2494736, 34.8341980, -81.1664047, 81.0867767
22: -49.6904182, 32.0290222, -49.6143227, 32.0529251, -81.7433472, 81.6433411
23: -36.5288849, 36.9131126, -36.4535599, 36.9384499, -73.4673309, 73.3666687
24: -48.6505394, 40.9154549, -48.5044289, 40.9583473, -89.6088867, 89.4198761
25: -44.4167938, 37.2686501, -44.3619385, 37.2947540, -81.7115479, 81.6305847
26: -64.4999313, 54.5165176, -64.3795929, 54.5554428, -119.0553665, 118.8961029
27: -47.8427238, 41.7301636, -47.7047119, 41.7508507, -89.5935745, 89.4348755
28: -38.3224144, 41.0479965, -38.2408524, 41.0556564, -79.3780670, 79.2888489
29: -48.5403442, 29.3600578, -48.4827271, 29.3790340, -77.9193802, 77.8427887
30: -50.3506927, 45.3051453, -50.2218933, 45.3125839, -95.6632767, 95.5270386
31: -47.6761703, 41.4575424, -47.5755730, 41.4727554, -89.1489258, 89.0331116
32: -59.9135323, 35.9156799, -59.9007797, 35.7590027, -95.6725311, 95.8164597
33: -85.8781128, 43.8957214, -85.8614044, 43.7954636, -129.6735840, 129.7571259
34: -79.8112488, 28.4003639, -79.7867279, 28.4315720, -108.2428207, 108.1870880
35: -70.2333755, 37.4577255, -70.2178650, 37.4582863, -107.6916656, 107.6755829
36: -71.1788101, 39.3806953, -71.1599655, 39.3191910, -110.4980011, 110.5406494
37: -97.7042389, 34.7195816, -97.6918793, 34.7059517, -132.4101868, 132.4114532
38: -87.2303162, 42.2748795, -87.2187958, 42.2384644, -129.4687653, 129.4936829
39: -96.3085938, 46.5985451, -96.3104782, 46.4679642, -142.7765350, 142.9090271
40: -73.8023758, 32.8612938, -73.7945557, 32.7792358, -106.5816116, 106.6558456
41: -64.6080017, 44.5916634, -64.5954971, 44.5072479, -109.1152420, 109.1871567
42: -48.0817413, 30.2463894, -48.0790596, 30.0758400, -78.1575775, 78.3254471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=387, inp2_unstable=388, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=560, inp2_unstable=559, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 973

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5054367, upper bound: 46.4948029
time: 138.17 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5175073, upper bound: 46.4948029
time: 106.86 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -84.7175598, 39.3584366, -84.6233215, 39.2717743, -123.9893341, 123.9817505
1: -42.5539780, 31.1458359, -42.5427284, 31.0412979, -73.5952682, 73.6885605
2: -34.6594162, 35.5206833, -34.6239204, 35.4070587, -70.0664749, 70.1446075
3: -46.4675713, 39.0491905, -46.4071808, 38.8055077, -85.2730789, 85.4563675
4: -46.9099579, 39.1005898, -46.8547287, 38.9287033, -85.8386612, 85.9553146
5: -43.1882706, 41.3599548, -43.1237869, 41.1403084, -84.3285675, 84.4837418
6: -61.3185196, 41.3209572, -61.2642326, 41.3025398, -102.6210632, 102.5851898
7: -52.1897240, 39.1117935, -52.1763382, 38.9669075, -91.1566238, 91.2881317
8: -66.2863235, 48.3653297, -66.2450562, 48.2025604, -114.4888840, 114.6103745
9: -43.9599380, 41.5010414, -43.9187546, 41.2819786, -85.2419128, 85.4197998
10: -58.5293465, 49.0009613, -58.4723740, 48.8777275, -107.4070740, 107.4733200
11: -49.7454185, 36.0727806, -49.5563126, 36.0134277, -85.7588501, 85.6290894
12: -66.5153732, 50.3866882, -66.4112091, 50.3393860, -116.8547592, 116.7978973
13: -71.5189362, 54.2171288, -71.4409103, 53.9747963, -125.4937286, 125.6580353
14: -104.4690170, 36.5964661, -104.2761993, 36.5932999, -141.0623169, 140.8726654
15: -50.6186714, 35.7865448, -50.5534058, 35.6370544, -86.2557220, 86.3399506
16: -58.7755775, 40.5919418, -58.6772690, 40.4480400, -99.2236176, 99.2692108
17: -101.2532196, 34.1802750, -101.1228180, 34.1407852, -135.3940125, 135.3031006
18: -58.6830025, 52.4898186, -58.4600906, 52.4723206, -111.1553192, 110.9499054
19: -34.6764221, 27.2347145, -34.4815445, 27.2182961, -61.8947182, 61.7162514
20: -39.1648064, 32.7693176, -39.0622253, 32.7472687, -71.9120636, 71.8315430
21: -46.3128662, 34.7486534, -46.1268158, 34.7001419, -81.0130005, 80.8754578
22: -49.7220535, 32.0802917, -49.5439339, 32.0781860, -81.8002319, 81.6242218
23: -36.5787201, 36.8790131, -36.3232384, 36.8243370, -73.4030533, 73.2022552
24: -48.6506462, 40.8687820, -48.3578911, 40.8276443, -89.4782867, 89.2266693
25: -44.4856834, 37.2502747, -44.2676620, 37.2275620, -81.7132416, 81.5179367
26: -64.4651031, 54.5694351, -64.2599411, 54.5165443, -118.9816284, 118.8293762
27: -47.7955551, 41.6168861, -47.5787582, 41.5609055, -89.3564606, 89.1956482
28: -38.3349686, 40.9877739, -38.1450005, 40.9375687, -79.2725372, 79.1327744
29: -48.5835800, 29.3947983, -48.4095383, 29.3645248, -77.9481049, 77.8043365
30: -50.2965240, 45.1875534, -50.0880737, 45.1232567, -95.4197845, 95.2756271
31: -47.7095833, 41.4268036, -47.4404755, 41.4093704, -89.1189575, 88.8672791
32: -59.8067741, 35.7524147, -59.7368698, 35.7191925, -95.5259552, 95.4892807
33: -85.8582611, 43.8191948, -85.7469254, 43.7845383, -129.6427917, 129.5661163
34: -79.8871765, 28.5046692, -79.7198868, 28.4820576, -108.3692322, 108.2245560
35: -70.2341995, 37.5172806, -70.1219025, 37.4995689, -107.7337646, 107.6391830
36: -71.1376419, 39.3517456, -71.0032654, 39.3152237, -110.4528580, 110.3550110
37: -97.7937317, 34.7225571, -97.5584869, 34.6578827, -132.4516144, 132.2810364
38: -87.1889801, 42.3048363, -87.0056152, 42.2570648, -129.4460449, 129.3104553
39: -96.2866287, 46.4851074, -96.1486282, 46.4361076, -142.7227325, 142.6337280
40: -73.7815628, 32.7629013, -73.7112808, 32.7307281, -106.5122910, 106.4741745
41: -64.5804443, 44.4860153, -64.4980469, 44.4273300, -109.0077744, 108.9840546
42: -48.0176544, 30.0872955, -47.9947968, 30.0189857, -78.0366364, 78.0820923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=387, inp2_unstable=388, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=561, inp2_unstable=559, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 973

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4845979, upper bound: 46.4985890
time: 114.05 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4967778, upper bound: 46.4985890
time: 126.95 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -84.7925034, 39.4068832, -84.6320038, 39.2781792, -124.0706711, 124.0388870
1: -42.5934753, 31.1806831, -42.5463638, 31.0478249, -73.6412964, 73.7270508
2: -34.7139893, 35.5717659, -34.6278000, 35.4197922, -70.1337585, 70.1995697
3: -46.5160675, 39.0863304, -46.4096870, 38.8098221, -85.3258896, 85.4960175
4: -46.9860191, 39.1699829, -46.8582382, 38.9447861, -85.9308014, 86.0282211
5: -43.2564621, 41.4243355, -43.1272736, 41.1539421, -84.4104004, 84.5516052
6: -61.4273224, 41.4360161, -61.3038101, 41.3086243, -102.7359467, 102.7398224
7: -52.2694397, 39.1616592, -52.1825180, 38.9846497, -91.2540894, 91.3441696
8: -66.3553696, 48.4350243, -66.2489319, 48.2249413, -114.5803070, 114.6839600
9: -44.0448685, 41.6216965, -43.9465485, 41.2873611, -85.3322296, 85.5682373
10: -58.5857277, 49.0774422, -58.4788094, 48.8893433, -107.4750671, 107.5562515
11: -49.8968849, 36.1298447, -49.5711899, 36.0276337, -85.9245148, 85.7010345
12: -66.6623077, 50.5735931, -66.4740295, 50.3509598, -117.0132599, 117.0476151
13: -71.6748962, 54.4533882, -71.5026550, 53.9825058, -125.6573944, 125.9560394
14: -104.6374207, 36.7097473, -104.2901993, 36.6485748, -141.2859802, 140.9999390
15: -50.6650658, 35.8470345, -50.5582008, 35.6472321, -86.3123016, 86.4052277
16: -58.9209137, 40.6504898, -58.6897621, 40.4633713, -99.3842850, 99.3402557
17: -101.4971237, 34.3253555, -101.1346130, 34.2038994, -135.7010193, 135.4599609
18: -58.9250450, 52.6415329, -58.4683228, 52.5352631, -111.4602966, 111.1098557
19: -34.7875519, 27.2974472, -34.4858208, 27.2433014, -62.0308456, 61.7832680
20: -39.2378731, 32.8165283, -39.0716095, 32.7663803, -72.0042419, 71.8881378
21: -46.4327698, 34.8073730, -46.1355591, 34.7243576, -81.1571274, 80.9429321
22: -49.8399124, 32.1289978, -49.5566139, 32.0913696, -81.9312820, 81.6856079
23: -36.6818047, 36.9400787, -36.3301315, 36.8463440, -73.5281525, 73.2702026
24: -48.8393517, 40.9936218, -48.3708420, 40.8800278, -89.7193756, 89.3644638
25: -44.5714912, 37.3343430, -44.2749252, 37.2616806, -81.8331680, 81.6092682
26: -64.6339188, 54.6294708, -64.2745819, 54.5319519, -119.1658707, 118.9040451
27: -47.9771461, 41.7224045, -47.5922241, 41.6071510, -89.5842896, 89.3146286
28: -38.4431686, 41.0590591, -38.1522903, 40.9655533, -79.4087219, 79.2113495
29: -48.6908684, 29.4345417, -48.4246712, 29.3740883, -78.0649490, 77.8592072
30: -50.4693756, 45.2895737, -50.1015930, 45.1643448, -95.6337051, 95.3911591
31: -47.8444138, 41.5202560, -47.4472733, 41.4477844, -89.2921982, 88.9675293
32: -59.9789009, 35.9428902, -59.8126183, 35.7280045, -95.7069092, 95.7555084
33: -85.9600220, 43.9490204, -85.7792969, 43.7899513, -129.7499695, 129.7283173
34: -79.9558029, 28.5373344, -79.7375031, 28.4829903, -108.4387970, 108.2748337
35: -70.3002777, 37.5615044, -70.1410980, 37.5020828, -107.8023529, 107.7025986
36: -71.2208252, 39.4480896, -71.0305176, 39.3181725, -110.5389938, 110.4786072
37: -97.8889008, 34.7935371, -97.5844879, 34.6635361, -132.5524292, 132.3780212
38: -87.2659607, 42.4078484, -87.0272064, 42.2651711, -129.5311279, 129.4350586
39: -96.3900070, 46.6401482, -96.1839828, 46.4408646, -142.8308716, 142.8241272
40: -73.8763275, 32.8665276, -73.7389526, 32.7351151, -106.6114426, 106.6054688
41: -64.7059250, 44.6031837, -64.5501328, 44.4348106, -109.1407318, 109.1533203
42: -48.1516228, 30.2831249, -48.0531616, 30.0253601, -78.1769867, 78.3362808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=387, inp2_unstable=388, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=562, inp2_unstable=559, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 973

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4845979, upper bound: 46.5175073
time: 129.83 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4967778, upper bound: 46.5175073
time: 139.89 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -84.7733002, 39.3758049, -84.7348785, 39.3358650, -124.1091309, 124.1106873
1: -42.5723267, 31.1655960, -42.5622749, 31.0941772, -73.6665039, 73.7278671
2: -34.6887283, 35.5329742, -34.6780663, 35.4437904, -70.1325226, 70.2110443
3: -46.5332184, 39.0663300, -46.5222435, 38.8945389, -85.4277573, 85.5885620
4: -46.9664650, 39.1182976, -46.9595108, 39.0009651, -85.9674301, 86.0778046
5: -43.2536621, 41.3744736, -43.2417145, 41.2248764, -84.4785385, 84.6161880
6: -61.3415642, 41.3468361, -61.3190918, 41.3578339, -102.6994019, 102.6659241
7: -52.2111511, 39.1280823, -52.2057343, 39.0057411, -91.2168884, 91.3338165
8: -66.3166275, 48.3921280, -66.3020325, 48.2675896, -114.5842133, 114.6941605
9: -44.0059052, 41.5192566, -44.0018196, 41.3742981, -85.3802032, 85.5210724
10: -58.5688553, 49.0308266, -58.5533829, 48.9825211, -107.5513763, 107.5842056
11: -49.7772942, 36.1478043, -49.6834869, 36.1442032, -85.9214935, 85.8312836
12: -66.5537796, 50.4240646, -66.4807968, 50.4190063, -116.9727859, 116.9048615
13: -71.6765747, 54.2531357, -71.7141571, 54.1651306, -125.8417053, 125.9672928
14: -104.5521774, 36.6247101, -104.4466629, 36.6549797, -141.2071533, 141.0713806
15: -50.6814651, 35.8083344, -50.6753883, 35.7436066, -86.4250565, 86.4837189
16: -58.8027496, 40.6202011, -58.7539673, 40.5178223, -99.3205719, 99.3741684
17: -101.3237915, 34.2086411, -101.2492752, 34.2345314, -135.5583191, 135.4579163
18: -58.7124825, 52.5616226, -58.5773621, 52.5978508, -111.3103104, 111.1389847
19: -34.7021866, 27.2720108, -34.5898438, 27.2827148, -61.9849014, 61.8618546
20: -39.1893463, 32.7975922, -39.1496429, 32.8017807, -71.9911194, 71.9472351
21: -46.3464966, 34.8159981, -46.2665977, 34.8194504, -81.1659241, 81.0825958
22: -49.7520523, 32.0965767, -49.6227455, 32.0988541, -81.8509064, 81.7193222
23: -36.6100197, 36.9670029, -36.4651489, 36.9766617, -73.5866776, 73.4321442
24: -48.6841240, 40.9742241, -48.5105133, 41.0038261, -89.6879425, 89.4847336
25: -44.5109787, 37.3017273, -44.3737373, 37.3221741, -81.8331528, 81.6754608
26: -64.5007477, 54.6271439, -64.3868256, 54.6244965, -119.1252441, 119.0139618
27: -47.8247566, 41.7328796, -47.7111778, 41.7605438, -89.5852966, 89.4440613
28: -38.3563766, 41.0544434, -38.2502213, 41.0619888, -79.4183578, 79.3046570
29: -48.6144371, 29.4257679, -48.4924507, 29.4225101, -78.0369415, 77.9182205
30: -50.3236389, 45.2923203, -50.2298203, 45.3129425, -95.6365814, 95.5221405
31: -47.7479172, 41.4717941, -47.5937157, 41.4877319, -89.2356491, 89.0655060
32: -59.8621445, 35.7712288, -59.8454361, 35.7699585, -95.6320953, 95.6166687
33: -85.8991318, 43.8344498, -85.8528748, 43.8149948, -129.7141266, 129.6873169
34: -79.9164810, 28.5241566, -79.7871399, 28.5130844, -108.4295654, 108.3112946
35: -70.2791290, 37.5258904, -70.2159958, 37.5100975, -107.7892303, 107.7418823
36: -71.2117615, 39.3663940, -71.1439133, 39.3568268, -110.5685883, 110.5103073
37: -97.8434372, 34.7832794, -97.6969910, 34.7674942, -132.6109314, 132.4802704
38: -87.3019562, 42.3218384, -87.2188873, 42.3015366, -129.6034851, 129.5407257
39: -96.3645706, 46.4957809, -96.3079453, 46.4849777, -142.8495483, 142.8037262
40: -73.8178482, 32.7946777, -73.7946320, 32.7935715, -106.6114197, 106.5893097
41: -64.6091309, 44.5354996, -64.5642242, 44.5245743, -109.1337051, 109.0997238
42: -48.0376053, 30.1158066, -48.0458755, 30.0845852, -78.1221924, 78.1616821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=387, inp2_unstable=388, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=561, inp2_unstable=559, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 973

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5054367, upper bound: 46.4985890
time: 162.47 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5175073, upper bound: 46.4985890
time: 118.83 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 283.56 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 283.56
Output dim: 19, lower bound: -46.4845979, upper bound: 46.4758298
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 283.56
Output dim: 19, lower bound: -46.4967778, upper bound: 46.4758298
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 283.56
Output dim: 19, lower bound: -46.4845979, upper bound: 46.4948029
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 283.56
Output dim: 19, lower bound: -46.4967778, upper bound: 46.4948029
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 283.56
Output dim: 19, lower bound: -46.5054367, upper bound: 46.4758298
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 283.56
Output dim: 19, lower bound: -46.5175073, upper bound: 46.4758298
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 283.56
Output dim: 19, lower bound: -46.5054367, upper bound: 46.4948029
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 283.56
Output dim: 19, lower bound: -46.5175073, upper bound: 46.4948029
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 283.56
Output dim: 19, lower bound: -46.4845979, upper bound: 46.4985890
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 283.56
Output dim: 19, lower bound: -46.4967778, upper bound: 46.4985890
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 283.56
Output dim: 19, lower bound: -46.4845979, upper bound: 46.5175073
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 283.56
Output dim: 19, lower bound: -46.4967778, upper bound: 46.5175073
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 283.56
Output dim: 19, lower bound: -46.5054367, upper bound: 46.4985890
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 283.56
Output dim: 19, lower bound: -46.5175073, upper bound: 46.4985890
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 283.56
Output dim: 19, lower bound: -46.4981156, upper bound: 46.5188362

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 164.33 + 3569.81 = 3734.14 seconds
