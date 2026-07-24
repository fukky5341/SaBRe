## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 3600 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.35 + 171.65 = 174.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 19, lower bound: -46.5413551, upper bound: 46.5413551

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1536

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 627

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5400043, upper bound: 46.5277552
time: 127.63 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5277552, upper bound: 46.5400043
time: 108.01 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 235.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 235.76
Output dim: 19, lower bound: -46.5400043, upper bound: 46.5277552
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 235.76
Output dim: 19, lower bound: -46.5277552, upper bound: 46.5400043

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -84.7672882, 39.3736267, -84.7672882, 39.3736267, -124.1409149, 124.1408997
1: -42.5773087, 31.1177921, -42.5773087, 31.1177921, -73.6950912, 73.6950989
2: -34.6925201, 35.4725037, -34.6925201, 35.4725037, -70.1650238, 70.1650162
3: -46.5430717, 38.9130402, -46.5430717, 38.9130402, -85.4561157, 85.4561157
4: -46.9768639, 39.0464897, -46.9768639, 39.0464897, -86.0233536, 86.0233536
5: -43.2595673, 41.2553864, -43.2595673, 41.2553864, -84.5149384, 84.5149536
6: -61.4064789, 41.3731499, -61.4064789, 41.3731499, -102.7796173, 102.7796249
7: -52.2261963, 39.0390739, -52.2261963, 39.0390739, -91.2652740, 91.2652740
8: -66.3197632, 48.3093834, -66.3197632, 48.3093834, -114.6291428, 114.6291428
9: -44.0484085, 41.3931618, -44.0484085, 41.3931618, -85.4415665, 85.4415588
10: -58.5848846, 49.0133591, -58.5848846, 49.0133591, -107.5982361, 107.5982437
11: -49.7151680, 36.1760406, -49.7151680, 36.1760406, -85.8912048, 85.8912048
12: -66.5686646, 50.4459152, -66.5686646, 50.4459152, -117.0145798, 117.0145798
13: -71.8024597, 54.1911697, -71.8024597, 54.1911697, -125.9936295, 125.9936295
14: -104.4877167, 36.7296181, -104.4877167, 36.7296181, -141.2173309, 141.2173309
15: -50.6970444, 35.7664642, -50.6970444, 35.7664642, -86.4635086, 86.4635086
16: -58.7854805, 40.5555725, -58.7854805, 40.5555725, -99.3410416, 99.3410492
17: -101.2854004, 34.3224030, -101.2854004, 34.3224030, -135.6078033, 135.6078033
18: -58.6059418, 52.6934662, -58.6059418, 52.6934662, -111.2993927, 111.2993927
19: -34.6089745, 27.3176270, -34.6089745, 27.3176270, -61.9265976, 61.9266014
20: -39.1725731, 32.8310394, -39.1725731, 32.8310394, -72.0036011, 72.0036011
21: -46.2914467, 34.8583450, -46.2914467, 34.8583450, -81.1497803, 81.1497879
22: -49.6540070, 32.1295013, -49.6540070, 32.1295013, -81.7835083, 81.7835083
23: -36.4852295, 37.0147552, -36.4852295, 37.0147552, -73.4999771, 73.4999847
24: -48.5414314, 41.0755920, -48.5414314, 41.0755920, -89.6170197, 89.6170197
25: -44.3950119, 37.3724670, -44.3950119, 37.3724670, -81.7674713, 81.7674713
26: -64.4201508, 54.6686516, -64.4201508, 54.6686516, -119.0887985, 119.0887985
27: -47.7461967, 41.8227158, -47.7461967, 41.8227158, -89.5689011, 89.5689087
28: -38.2709885, 41.1035767, -38.2709885, 41.1035767, -79.3745651, 79.3745651
29: -48.5310211, 29.4440346, -48.5310211, 29.4440346, -77.9750519, 77.9750519
30: -50.2593193, 45.3741074, -50.2593193, 45.3741074, -95.6334152, 95.6334229
31: -47.6183662, 41.5403900, -47.6183662, 41.5403900, -89.1587524, 89.1587524
32: -59.9542313, 35.7897034, -59.9542313, 35.7897034, -95.7439270, 95.7439270
33: -85.9071808, 43.8326645, -85.9071808, 43.8326645, -129.7398376, 129.7398376
34: -79.8334198, 28.5293617, -79.8334198, 28.5293617, -108.3627777, 108.3627777
35: -70.2571030, 37.5221252, -70.2571030, 37.5221252, -107.7792282, 107.7792282
36: -71.2104950, 39.3675041, -71.2104950, 39.3675041, -110.5780029, 110.5780029
37: -97.7557907, 34.7876053, -97.7557907, 34.7876053, -132.5433960, 132.5433960
38: -87.2721405, 42.3223114, -87.2721405, 42.3223114, -129.5944519, 129.5944519
39: -96.3795929, 46.4996338, -96.3795929, 46.4996338, -142.8792114, 142.8792114
40: -73.8562927, 32.8088455, -73.8562927, 32.8088455, -106.6651382, 106.6651382
41: -64.6400299, 44.5428925, -64.6400299, 44.5428925, -109.1828995, 109.1829071
42: -48.1290588, 30.1027393, -48.1290588, 30.1027393, -78.2317963, 78.2317963

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1536

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 595

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5389336, upper bound: 46.5056472
time: 122.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5179884, upper bound: 46.5266834
time: 481.15 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -84.7672882, 39.3736267, -84.7672882, 39.3736267, -124.1409149, 124.1408997
1: -42.5773087, 31.1177921, -42.5773087, 31.1177921, -73.6950912, 73.6950989
2: -34.6925201, 35.4725037, -34.6925201, 35.4725037, -70.1650238, 70.1650162
3: -46.5430717, 38.9130402, -46.5430717, 38.9130402, -85.4561157, 85.4561157
4: -46.9768639, 39.0464897, -46.9768639, 39.0464897, -86.0233536, 86.0233536
5: -43.2595673, 41.2553864, -43.2595673, 41.2553864, -84.5149384, 84.5149536
6: -61.4064789, 41.3731499, -61.4064789, 41.3731499, -102.7796173, 102.7796249
7: -52.2261963, 39.0390739, -52.2261963, 39.0390739, -91.2652740, 91.2652740
8: -66.3197632, 48.3093834, -66.3197632, 48.3093834, -114.6291428, 114.6291428
9: -44.0484085, 41.3931618, -44.0484085, 41.3931618, -85.4415665, 85.4415588
10: -58.5848846, 49.0133591, -58.5848846, 49.0133591, -107.5982361, 107.5982437
11: -49.7151680, 36.1760406, -49.7151680, 36.1760406, -85.8912048, 85.8912048
12: -66.5686646, 50.4459152, -66.5686646, 50.4459152, -117.0145798, 117.0145798
13: -71.8024597, 54.1911697, -71.8024597, 54.1911697, -125.9936295, 125.9936295
14: -104.4877167, 36.7296181, -104.4877167, 36.7296181, -141.2173309, 141.2173309
15: -50.6970444, 35.7664642, -50.6970444, 35.7664642, -86.4635086, 86.4635086
16: -58.7854805, 40.5555725, -58.7854805, 40.5555725, -99.3410416, 99.3410492
17: -101.2854004, 34.3224030, -101.2854004, 34.3224030, -135.6078033, 135.6078033
18: -58.6059418, 52.6934662, -58.6059418, 52.6934662, -111.2993927, 111.2993927
19: -34.6089745, 27.3176270, -34.6089745, 27.3176270, -61.9265976, 61.9266014
20: -39.1725731, 32.8310394, -39.1725731, 32.8310394, -72.0036011, 72.0036011
21: -46.2914467, 34.8583450, -46.2914467, 34.8583450, -81.1497803, 81.1497879
22: -49.6540070, 32.1295013, -49.6540070, 32.1295013, -81.7835083, 81.7835083
23: -36.4852295, 37.0147552, -36.4852295, 37.0147552, -73.4999771, 73.4999847
24: -48.5414314, 41.0755920, -48.5414314, 41.0755920, -89.6170197, 89.6170197
25: -44.3950119, 37.3724670, -44.3950119, 37.3724670, -81.7674713, 81.7674713
26: -64.4201508, 54.6686516, -64.4201508, 54.6686516, -119.0887985, 119.0887985
27: -47.7461967, 41.8227158, -47.7461967, 41.8227158, -89.5689011, 89.5689087
28: -38.2709885, 41.1035767, -38.2709885, 41.1035767, -79.3745651, 79.3745651
29: -48.5310211, 29.4440346, -48.5310211, 29.4440346, -77.9750519, 77.9750519
30: -50.2593193, 45.3741074, -50.2593193, 45.3741074, -95.6334152, 95.6334229
31: -47.6183662, 41.5403900, -47.6183662, 41.5403900, -89.1587524, 89.1587524
32: -59.9542313, 35.7897034, -59.9542313, 35.7897034, -95.7439270, 95.7439270
33: -85.9071808, 43.8326645, -85.9071808, 43.8326645, -129.7398376, 129.7398376
34: -79.8334198, 28.5293617, -79.8334198, 28.5293617, -108.3627777, 108.3627777
35: -70.2571030, 37.5221252, -70.2571030, 37.5221252, -107.7792282, 107.7792282
36: -71.2104950, 39.3675041, -71.2104950, 39.3675041, -110.5780029, 110.5780029
37: -97.7557907, 34.7876053, -97.7557907, 34.7876053, -132.5433960, 132.5433960
38: -87.2721405, 42.3223114, -87.2721405, 42.3223114, -129.5944519, 129.5944519
39: -96.3795929, 46.4996338, -96.3795929, 46.4996338, -142.8792114, 142.8792114
40: -73.8562927, 32.8088455, -73.8562927, 32.8088455, -106.6651382, 106.6651382
41: -64.6400299, 44.5428925, -64.6400299, 44.5428925, -109.1828995, 109.1829071
42: -48.1290588, 30.1027393, -48.1290588, 30.1027393, -78.2317963, 78.2317963

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1536

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 595

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5266833, upper bound: 46.5179884
time: 132.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5056472, upper bound: 46.5389336
time: 139.19 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 274.01 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 274.01
Output dim: 19, lower bound: -46.5389336, upper bound: 46.5056472
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 274.01
Output dim: 19, lower bound: -46.5179884, upper bound: 46.5266834
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 274.01
Output dim: 19, lower bound: -46.5266833, upper bound: 46.5179884
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 274.01
Output dim: 19, lower bound: -46.5056472, upper bound: 46.5389336

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -84.7672882, 39.3736267, -84.7672882, 39.3736267, -124.1409149, 124.1408997
1: -42.5773087, 31.1177921, -42.5773087, 31.1177921, -73.6950912, 73.6950989
2: -34.6925201, 35.4725037, -34.6925201, 35.4725037, -70.1650238, 70.1650162
3: -46.5430717, 38.9130402, -46.5430717, 38.9130402, -85.4561157, 85.4561157
4: -46.9768639, 39.0464897, -46.9768639, 39.0464897, -86.0233536, 86.0233536
5: -43.2595673, 41.2553864, -43.2595673, 41.2553864, -84.5149384, 84.5149536
6: -61.4064789, 41.3731499, -61.4064789, 41.3731499, -102.7796173, 102.7796249
7: -52.2261963, 39.0390739, -52.2261963, 39.0390739, -91.2652740, 91.2652740
8: -66.3197632, 48.3093834, -66.3197632, 48.3093834, -114.6291428, 114.6291428
9: -44.0484085, 41.3931618, -44.0484085, 41.3931618, -85.4415665, 85.4415588
10: -58.5848846, 49.0133591, -58.5848846, 49.0133591, -107.5982361, 107.5982437
11: -49.7151680, 36.1760406, -49.7151680, 36.1760406, -85.8912048, 85.8912048
12: -66.5686646, 50.4459152, -66.5686646, 50.4459152, -117.0145798, 117.0145798
13: -71.8024597, 54.1911697, -71.8024597, 54.1911697, -125.9936295, 125.9936295
14: -104.4877167, 36.7296181, -104.4877167, 36.7296181, -141.2173309, 141.2173309
15: -50.6970444, 35.7664642, -50.6970444, 35.7664642, -86.4635086, 86.4635086
16: -58.7854805, 40.5555725, -58.7854805, 40.5555725, -99.3410416, 99.3410492
17: -101.2854004, 34.3224030, -101.2854004, 34.3224030, -135.6078033, 135.6078033
18: -58.6059418, 52.6934662, -58.6059418, 52.6934662, -111.2993927, 111.2993927
19: -34.6089745, 27.3176270, -34.6089745, 27.3176270, -61.9265976, 61.9266014
20: -39.1725731, 32.8310394, -39.1725731, 32.8310394, -72.0036011, 72.0036011
21: -46.2914467, 34.8583450, -46.2914467, 34.8583450, -81.1497803, 81.1497879
22: -49.6540070, 32.1295013, -49.6540070, 32.1295013, -81.7835083, 81.7835083
23: -36.4852295, 37.0147552, -36.4852295, 37.0147552, -73.4999771, 73.4999847
24: -48.5414314, 41.0755920, -48.5414314, 41.0755920, -89.6170197, 89.6170197
25: -44.3950119, 37.3724670, -44.3950119, 37.3724670, -81.7674713, 81.7674713
26: -64.4201508, 54.6686516, -64.4201508, 54.6686516, -119.0887985, 119.0887985
27: -47.7461967, 41.8227158, -47.7461967, 41.8227158, -89.5689011, 89.5689087
28: -38.2709885, 41.1035767, -38.2709885, 41.1035767, -79.3745651, 79.3745651
29: -48.5310211, 29.4440346, -48.5310211, 29.4440346, -77.9750519, 77.9750519
30: -50.2593193, 45.3741074, -50.2593193, 45.3741074, -95.6334152, 95.6334229
31: -47.6183662, 41.5403900, -47.6183662, 41.5403900, -89.1587524, 89.1587524
32: -59.9542313, 35.7897034, -59.9542313, 35.7897034, -95.7439270, 95.7439270
33: -85.9071808, 43.8326645, -85.9071808, 43.8326645, -129.7398376, 129.7398376
34: -79.8334198, 28.5293617, -79.8334198, 28.5293617, -108.3627777, 108.3627777
35: -70.2571030, 37.5221252, -70.2571030, 37.5221252, -107.7792282, 107.7792282
36: -71.2104950, 39.3675041, -71.2104950, 39.3675041, -110.5780029, 110.5780029
37: -97.7557907, 34.7876053, -97.7557907, 34.7876053, -132.5433960, 132.5433960
38: -87.2721405, 42.3223114, -87.2721405, 42.3223114, -129.5944519, 129.5944519
39: -96.3795929, 46.4996338, -96.3795929, 46.4996338, -142.8792114, 142.8792114
40: -73.8562927, 32.8088455, -73.8562927, 32.8088455, -106.6651382, 106.6651382
41: -64.6400299, 44.5428925, -64.6400299, 44.5428925, -109.1828995, 109.1829071
42: -48.1290588, 30.1027393, -48.1290588, 30.1027393, -78.2317963, 78.2317963

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1536

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 637

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5382696, upper bound: 46.4819695
time: 127.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5154100, upper bound: 46.5049864
time: 152.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -84.7672882, 39.3736267, -84.7672882, 39.3736267, -124.1409149, 124.1408997
1: -42.5773087, 31.1177921, -42.5773087, 31.1177921, -73.6950912, 73.6950989
2: -34.6925201, 35.4725037, -34.6925201, 35.4725037, -70.1650238, 70.1650162
3: -46.5430717, 38.9130402, -46.5430717, 38.9130402, -85.4561157, 85.4561157
4: -46.9768639, 39.0464897, -46.9768639, 39.0464897, -86.0233536, 86.0233536
5: -43.2595673, 41.2553864, -43.2595673, 41.2553864, -84.5149384, 84.5149536
6: -61.4064789, 41.3731499, -61.4064789, 41.3731499, -102.7796173, 102.7796249
7: -52.2261963, 39.0390739, -52.2261963, 39.0390739, -91.2652740, 91.2652740
8: -66.3197632, 48.3093834, -66.3197632, 48.3093834, -114.6291428, 114.6291428
9: -44.0484085, 41.3931618, -44.0484085, 41.3931618, -85.4415665, 85.4415588
10: -58.5848846, 49.0133591, -58.5848846, 49.0133591, -107.5982361, 107.5982437
11: -49.7151680, 36.1760406, -49.7151680, 36.1760406, -85.8912048, 85.8912048
12: -66.5686646, 50.4459152, -66.5686646, 50.4459152, -117.0145798, 117.0145798
13: -71.8024597, 54.1911697, -71.8024597, 54.1911697, -125.9936295, 125.9936295
14: -104.4877167, 36.7296181, -104.4877167, 36.7296181, -141.2173309, 141.2173309
15: -50.6970444, 35.7664642, -50.6970444, 35.7664642, -86.4635086, 86.4635086
16: -58.7854805, 40.5555725, -58.7854805, 40.5555725, -99.3410416, 99.3410492
17: -101.2854004, 34.3224030, -101.2854004, 34.3224030, -135.6078033, 135.6078033
18: -58.6059418, 52.6934662, -58.6059418, 52.6934662, -111.2993927, 111.2993927
19: -34.6089745, 27.3176270, -34.6089745, 27.3176270, -61.9265976, 61.9266014
20: -39.1725731, 32.8310394, -39.1725731, 32.8310394, -72.0036011, 72.0036011
21: -46.2914467, 34.8583450, -46.2914467, 34.8583450, -81.1497803, 81.1497879
22: -49.6540070, 32.1295013, -49.6540070, 32.1295013, -81.7835083, 81.7835083
23: -36.4852295, 37.0147552, -36.4852295, 37.0147552, -73.4999771, 73.4999847
24: -48.5414314, 41.0755920, -48.5414314, 41.0755920, -89.6170197, 89.6170197
25: -44.3950119, 37.3724670, -44.3950119, 37.3724670, -81.7674713, 81.7674713
26: -64.4201508, 54.6686516, -64.4201508, 54.6686516, -119.0887985, 119.0887985
27: -47.7461967, 41.8227158, -47.7461967, 41.8227158, -89.5689011, 89.5689087
28: -38.2709885, 41.1035767, -38.2709885, 41.1035767, -79.3745651, 79.3745651
29: -48.5310211, 29.4440346, -48.5310211, 29.4440346, -77.9750519, 77.9750519
30: -50.2593193, 45.3741074, -50.2593193, 45.3741074, -95.6334152, 95.6334229
31: -47.6183662, 41.5403900, -47.6183662, 41.5403900, -89.1587524, 89.1587524
32: -59.9542313, 35.7897034, -59.9542313, 35.7897034, -95.7439270, 95.7439270
33: -85.9071808, 43.8326645, -85.9071808, 43.8326645, -129.7398376, 129.7398376
34: -79.8334198, 28.5293617, -79.8334198, 28.5293617, -108.3627777, 108.3627777
35: -70.2571030, 37.5221252, -70.2571030, 37.5221252, -107.7792282, 107.7792282
36: -71.2104950, 39.3675041, -71.2104950, 39.3675041, -110.5780029, 110.5780029
37: -97.7557907, 34.7876053, -97.7557907, 34.7876053, -132.5433960, 132.5433960
38: -87.2721405, 42.3223114, -87.2721405, 42.3223114, -129.5944519, 129.5944519
39: -96.3795929, 46.4996338, -96.3795929, 46.4996338, -142.8792114, 142.8792114
40: -73.8562927, 32.8088455, -73.8562927, 32.8088455, -106.6651382, 106.6651382
41: -64.6400299, 44.5428925, -64.6400299, 44.5428925, -109.1828995, 109.1829071
42: -48.1290588, 30.1027393, -48.1290588, 30.1027393, -78.2317963, 78.2317963

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1536

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 637

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5049864, upper bound: 46.5030921
time: 137.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4944005, upper bound: 46.5260180
time: 114.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -84.7672882, 39.3736267, -84.7672882, 39.3736267, -124.1409149, 124.1408997
1: -42.5773087, 31.1177921, -42.5773087, 31.1177921, -73.6950912, 73.6950989
2: -34.6925201, 35.4725037, -34.6925201, 35.4725037, -70.1650238, 70.1650162
3: -46.5430717, 38.9130402, -46.5430717, 38.9130402, -85.4561157, 85.4561157
4: -46.9768639, 39.0464897, -46.9768639, 39.0464897, -86.0233536, 86.0233536
5: -43.2595673, 41.2553864, -43.2595673, 41.2553864, -84.5149384, 84.5149536
6: -61.4064789, 41.3731499, -61.4064789, 41.3731499, -102.7796173, 102.7796249
7: -52.2261963, 39.0390739, -52.2261963, 39.0390739, -91.2652740, 91.2652740
8: -66.3197632, 48.3093834, -66.3197632, 48.3093834, -114.6291428, 114.6291428
9: -44.0484085, 41.3931618, -44.0484085, 41.3931618, -85.4415665, 85.4415588
10: -58.5848846, 49.0133591, -58.5848846, 49.0133591, -107.5982361, 107.5982437
11: -49.7151680, 36.1760406, -49.7151680, 36.1760406, -85.8912048, 85.8912048
12: -66.5686646, 50.4459152, -66.5686646, 50.4459152, -117.0145798, 117.0145798
13: -71.8024597, 54.1911697, -71.8024597, 54.1911697, -125.9936295, 125.9936295
14: -104.4877167, 36.7296181, -104.4877167, 36.7296181, -141.2173309, 141.2173309
15: -50.6970444, 35.7664642, -50.6970444, 35.7664642, -86.4635086, 86.4635086
16: -58.7854805, 40.5555725, -58.7854805, 40.5555725, -99.3410416, 99.3410492
17: -101.2854004, 34.3224030, -101.2854004, 34.3224030, -135.6078033, 135.6078033
18: -58.6059418, 52.6934662, -58.6059418, 52.6934662, -111.2993927, 111.2993927
19: -34.6089745, 27.3176270, -34.6089745, 27.3176270, -61.9265976, 61.9266014
20: -39.1725731, 32.8310394, -39.1725731, 32.8310394, -72.0036011, 72.0036011
21: -46.2914467, 34.8583450, -46.2914467, 34.8583450, -81.1497803, 81.1497879
22: -49.6540070, 32.1295013, -49.6540070, 32.1295013, -81.7835083, 81.7835083
23: -36.4852295, 37.0147552, -36.4852295, 37.0147552, -73.4999771, 73.4999847
24: -48.5414314, 41.0755920, -48.5414314, 41.0755920, -89.6170197, 89.6170197
25: -44.3950119, 37.3724670, -44.3950119, 37.3724670, -81.7674713, 81.7674713
26: -64.4201508, 54.6686516, -64.4201508, 54.6686516, -119.0887985, 119.0887985
27: -47.7461967, 41.8227158, -47.7461967, 41.8227158, -89.5689011, 89.5689087
28: -38.2709885, 41.1035767, -38.2709885, 41.1035767, -79.3745651, 79.3745651
29: -48.5310211, 29.4440346, -48.5310211, 29.4440346, -77.9750519, 77.9750519
30: -50.2593193, 45.3741074, -50.2593193, 45.3741074, -95.6334152, 95.6334229
31: -47.6183662, 41.5403900, -47.6183662, 41.5403900, -89.1587524, 89.1587524
32: -59.9542313, 35.7897034, -59.9542313, 35.7897034, -95.7439270, 95.7439270
33: -85.9071808, 43.8326645, -85.9071808, 43.8326645, -129.7398376, 129.7398376
34: -79.8334198, 28.5293617, -79.8334198, 28.5293617, -108.3627777, 108.3627777
35: -70.2571030, 37.5221252, -70.2571030, 37.5221252, -107.7792282, 107.7792282
36: -71.2104950, 39.3675041, -71.2104950, 39.3675041, -110.5780029, 110.5780029
37: -97.7557907, 34.7876053, -97.7557907, 34.7876053, -132.5433960, 132.5433960
38: -87.2721405, 42.3223114, -87.2721405, 42.3223114, -129.5944519, 129.5944519
39: -96.3795929, 46.4996338, -96.3795929, 46.4996338, -142.8792114, 142.8792114
40: -73.8562927, 32.8088455, -73.8562927, 32.8088455, -106.6651382, 106.6651382
41: -64.6400299, 44.5428925, -64.6400299, 44.5428925, -109.1828995, 109.1829071
42: -48.1290588, 30.1027393, -48.1290588, 30.1027393, -78.2317963, 78.2317963

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1536

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 637

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5260180, upper bound: 46.4944005
time: 114.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5030921, upper bound: 46.5173217
time: 160.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -84.7672882, 39.3736267, -84.7672882, 39.3736267, -124.1409149, 124.1408997
1: -42.5773087, 31.1177921, -42.5773087, 31.1177921, -73.6950912, 73.6950989
2: -34.6925201, 35.4725037, -34.6925201, 35.4725037, -70.1650238, 70.1650162
3: -46.5430717, 38.9130402, -46.5430717, 38.9130402, -85.4561157, 85.4561157
4: -46.9768639, 39.0464897, -46.9768639, 39.0464897, -86.0233536, 86.0233536
5: -43.2595673, 41.2553864, -43.2595673, 41.2553864, -84.5149384, 84.5149536
6: -61.4064789, 41.3731499, -61.4064789, 41.3731499, -102.7796173, 102.7796249
7: -52.2261963, 39.0390739, -52.2261963, 39.0390739, -91.2652740, 91.2652740
8: -66.3197632, 48.3093834, -66.3197632, 48.3093834, -114.6291428, 114.6291428
9: -44.0484085, 41.3931618, -44.0484085, 41.3931618, -85.4415665, 85.4415588
10: -58.5848846, 49.0133591, -58.5848846, 49.0133591, -107.5982361, 107.5982437
11: -49.7151680, 36.1760406, -49.7151680, 36.1760406, -85.8912048, 85.8912048
12: -66.5686646, 50.4459152, -66.5686646, 50.4459152, -117.0145798, 117.0145798
13: -71.8024597, 54.1911697, -71.8024597, 54.1911697, -125.9936295, 125.9936295
14: -104.4877167, 36.7296181, -104.4877167, 36.7296181, -141.2173309, 141.2173309
15: -50.6970444, 35.7664642, -50.6970444, 35.7664642, -86.4635086, 86.4635086
16: -58.7854805, 40.5555725, -58.7854805, 40.5555725, -99.3410416, 99.3410492
17: -101.2854004, 34.3224030, -101.2854004, 34.3224030, -135.6078033, 135.6078033
18: -58.6059418, 52.6934662, -58.6059418, 52.6934662, -111.2993927, 111.2993927
19: -34.6089745, 27.3176270, -34.6089745, 27.3176270, -61.9265976, 61.9266014
20: -39.1725731, 32.8310394, -39.1725731, 32.8310394, -72.0036011, 72.0036011
21: -46.2914467, 34.8583450, -46.2914467, 34.8583450, -81.1497803, 81.1497879
22: -49.6540070, 32.1295013, -49.6540070, 32.1295013, -81.7835083, 81.7835083
23: -36.4852295, 37.0147552, -36.4852295, 37.0147552, -73.4999771, 73.4999847
24: -48.5414314, 41.0755920, -48.5414314, 41.0755920, -89.6170197, 89.6170197
25: -44.3950119, 37.3724670, -44.3950119, 37.3724670, -81.7674713, 81.7674713
26: -64.4201508, 54.6686516, -64.4201508, 54.6686516, -119.0887985, 119.0887985
27: -47.7461967, 41.8227158, -47.7461967, 41.8227158, -89.5689011, 89.5689087
28: -38.2709885, 41.1035767, -38.2709885, 41.1035767, -79.3745651, 79.3745651
29: -48.5310211, 29.4440346, -48.5310211, 29.4440346, -77.9750519, 77.9750519
30: -50.2593193, 45.3741074, -50.2593193, 45.3741074, -95.6334152, 95.6334229
31: -47.6183662, 41.5403900, -47.6183662, 41.5403900, -89.1587524, 89.1587524
32: -59.9542313, 35.7897034, -59.9542313, 35.7897034, -95.7439270, 95.7439270
33: -85.9071808, 43.8326645, -85.9071808, 43.8326645, -129.7398376, 129.7398376
34: -79.8334198, 28.5293617, -79.8334198, 28.5293617, -108.3627777, 108.3627777
35: -70.2571030, 37.5221252, -70.2571030, 37.5221252, -107.7792282, 107.7792282
36: -71.2104950, 39.3675041, -71.2104950, 39.3675041, -110.5780029, 110.5780029
37: -97.7557907, 34.7876053, -97.7557907, 34.7876053, -132.5433960, 132.5433960
38: -87.2721405, 42.3223114, -87.2721405, 42.3223114, -129.5944519, 129.5944519
39: -96.3795929, 46.4996338, -96.3795929, 46.4996338, -142.8792114, 142.8792114
40: -73.8562927, 32.8088455, -73.8562927, 32.8088455, -106.6651382, 106.6651382
41: -64.6400299, 44.5428925, -64.6400299, 44.5428925, -109.1828995, 109.1829071
42: -48.1290588, 30.1027393, -48.1290588, 30.1027393, -78.2317963, 78.2317963

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1536

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 637

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.5049864, upper bound: 46.5154100
time: 148.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4819694, upper bound: 46.5382696
time: 144.89 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 295.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 295.21
Output dim: 19, lower bound: -46.5382696, upper bound: 46.4819695
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 295.21
Output dim: 19, lower bound: -46.5154100, upper bound: 46.5049864
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 295.21
Output dim: 19, lower bound: -46.5049864, upper bound: 46.5030921
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 295.21
Output dim: 19, lower bound: -46.4944005, upper bound: 46.5260180
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 295.21
Output dim: 19, lower bound: -46.5260180, upper bound: 46.4944005
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 295.21
Output dim: 19, lower bound: -46.5030921, upper bound: 46.5173217
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 295.21
Output dim: 19, lower bound: -46.5049864, upper bound: 46.5154100
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 295.21
Output dim: 19, lower bound: -46.4819694, upper bound: 46.5382696

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -84.7672882, 39.3736267, -84.7672882, 39.3736267, -124.1409149, 124.1408997
1: -42.5773087, 31.1177921, -42.5773087, 31.1177921, -73.6950912, 73.6950989
2: -34.6925201, 35.4725037, -34.6925201, 35.4725037, -70.1650238, 70.1650162
3: -46.5430717, 38.9130402, -46.5430717, 38.9130402, -85.4561157, 85.4561157
4: -46.9768639, 39.0464897, -46.9768639, 39.0464897, -86.0233536, 86.0233536
5: -43.2595673, 41.2553864, -43.2595673, 41.2553864, -84.5149384, 84.5149536
6: -61.4064789, 41.3731499, -61.4064789, 41.3731499, -102.7796173, 102.7796249
7: -52.2261963, 39.0390739, -52.2261963, 39.0390739, -91.2652740, 91.2652740
8: -66.3197632, 48.3093834, -66.3197632, 48.3093834, -114.6291428, 114.6291428
9: -44.0484085, 41.3931618, -44.0484085, 41.3931618, -85.4415665, 85.4415588
10: -58.5848846, 49.0133591, -58.5848846, 49.0133591, -107.5982361, 107.5982437
11: -49.7151680, 36.1760406, -49.7151680, 36.1760406, -85.8912048, 85.8912048
12: -66.5686646, 50.4459152, -66.5686646, 50.4459152, -117.0145798, 117.0145798
13: -71.8024597, 54.1911697, -71.8024597, 54.1911697, -125.9936295, 125.9936295
14: -104.4877167, 36.7296181, -104.4877167, 36.7296181, -141.2173309, 141.2173309
15: -50.6970444, 35.7664642, -50.6970444, 35.7664642, -86.4635086, 86.4635086
16: -58.7854805, 40.5555725, -58.7854805, 40.5555725, -99.3410416, 99.3410492
17: -101.2854004, 34.3224030, -101.2854004, 34.3224030, -135.6078033, 135.6078033
18: -58.6059418, 52.6934662, -58.6059418, 52.6934662, -111.2993927, 111.2993927
19: -34.6089745, 27.3176270, -34.6089745, 27.3176270, -61.9265976, 61.9266014
20: -39.1725731, 32.8310394, -39.1725731, 32.8310394, -72.0036011, 72.0036011
21: -46.2914467, 34.8583450, -46.2914467, 34.8583450, -81.1497803, 81.1497879
22: -49.6540070, 32.1295013, -49.6540070, 32.1295013, -81.7835083, 81.7835083
23: -36.4852295, 37.0147552, -36.4852295, 37.0147552, -73.4999771, 73.4999847
24: -48.5414314, 41.0755920, -48.5414314, 41.0755920, -89.6170197, 89.6170197
25: -44.3950119, 37.3724670, -44.3950119, 37.3724670, -81.7674713, 81.7674713
26: -64.4201508, 54.6686516, -64.4201508, 54.6686516, -119.0887985, 119.0887985
27: -47.7461967, 41.8227158, -47.7461967, 41.8227158, -89.5689011, 89.5689087
28: -38.2709885, 41.1035767, -38.2709885, 41.1035767, -79.3745651, 79.3745651
29: -48.5310211, 29.4440346, -48.5310211, 29.4440346, -77.9750519, 77.9750519
30: -50.2593193, 45.3741074, -50.2593193, 45.3741074, -95.6334152, 95.6334229
31: -47.6183662, 41.5403900, -47.6183662, 41.5403900, -89.1587524, 89.1587524
32: -59.9542313, 35.7897034, -59.9542313, 35.7897034, -95.7439270, 95.7439270
33: -85.9071808, 43.8326645, -85.9071808, 43.8326645, -129.7398376, 129.7398376
34: -79.8334198, 28.5293617, -79.8334198, 28.5293617, -108.3627777, 108.3627777
35: -70.2571030, 37.5221252, -70.2571030, 37.5221252, -107.7792282, 107.7792282
36: -71.2104950, 39.3675041, -71.2104950, 39.3675041, -110.5780029, 110.5780029
37: -97.7557907, 34.7876053, -97.7557907, 34.7876053, -132.5433960, 132.5433960
38: -87.2721405, 42.3223114, -87.2721405, 42.3223114, -129.5944519, 129.5944519
39: -96.3795929, 46.4996338, -96.3795929, 46.4996338, -142.8792114, 142.8792114
40: -73.8562927, 32.8088455, -73.8562927, 32.8088455, -106.6651382, 106.6651382
41: -64.6400299, 44.5428925, -64.6400299, 44.5428925, -109.1828995, 109.1829071
42: -48.1290588, 30.1027393, -48.1290588, 30.1027393, -78.2317963, 78.2317963

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1536

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4701292, upper bound: 46.4114286
time: 118.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4669162, upper bound: 46.4147598
time: 125.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -84.7672882, 39.3736267, -84.7672882, 39.3736267, -124.1409149, 124.1408997
1: -42.5773087, 31.1177921, -42.5773087, 31.1177921, -73.6950912, 73.6950989
2: -34.6925201, 35.4725037, -34.6925201, 35.4725037, -70.1650238, 70.1650162
3: -46.5430717, 38.9130402, -46.5430717, 38.9130402, -85.4561157, 85.4561157
4: -46.9768639, 39.0464897, -46.9768639, 39.0464897, -86.0233536, 86.0233536
5: -43.2595673, 41.2553864, -43.2595673, 41.2553864, -84.5149384, 84.5149536
6: -61.4064789, 41.3731499, -61.4064789, 41.3731499, -102.7796173, 102.7796249
7: -52.2261963, 39.0390739, -52.2261963, 39.0390739, -91.2652740, 91.2652740
8: -66.3197632, 48.3093834, -66.3197632, 48.3093834, -114.6291428, 114.6291428
9: -44.0484085, 41.3931618, -44.0484085, 41.3931618, -85.4415665, 85.4415588
10: -58.5848846, 49.0133591, -58.5848846, 49.0133591, -107.5982361, 107.5982437
11: -49.7151680, 36.1760406, -49.7151680, 36.1760406, -85.8912048, 85.8912048
12: -66.5686646, 50.4459152, -66.5686646, 50.4459152, -117.0145798, 117.0145798
13: -71.8024597, 54.1911697, -71.8024597, 54.1911697, -125.9936295, 125.9936295
14: -104.4877167, 36.7296181, -104.4877167, 36.7296181, -141.2173309, 141.2173309
15: -50.6970444, 35.7664642, -50.6970444, 35.7664642, -86.4635086, 86.4635086
16: -58.7854805, 40.5555725, -58.7854805, 40.5555725, -99.3410416, 99.3410492
17: -101.2854004, 34.3224030, -101.2854004, 34.3224030, -135.6078033, 135.6078033
18: -58.6059418, 52.6934662, -58.6059418, 52.6934662, -111.2993927, 111.2993927
19: -34.6089745, 27.3176270, -34.6089745, 27.3176270, -61.9265976, 61.9266014
20: -39.1725731, 32.8310394, -39.1725731, 32.8310394, -72.0036011, 72.0036011
21: -46.2914467, 34.8583450, -46.2914467, 34.8583450, -81.1497803, 81.1497879
22: -49.6540070, 32.1295013, -49.6540070, 32.1295013, -81.7835083, 81.7835083
23: -36.4852295, 37.0147552, -36.4852295, 37.0147552, -73.4999771, 73.4999847
24: -48.5414314, 41.0755920, -48.5414314, 41.0755920, -89.6170197, 89.6170197
25: -44.3950119, 37.3724670, -44.3950119, 37.3724670, -81.7674713, 81.7674713
26: -64.4201508, 54.6686516, -64.4201508, 54.6686516, -119.0887985, 119.0887985
27: -47.7461967, 41.8227158, -47.7461967, 41.8227158, -89.5689011, 89.5689087
28: -38.2709885, 41.1035767, -38.2709885, 41.1035767, -79.3745651, 79.3745651
29: -48.5310211, 29.4440346, -48.5310211, 29.4440346, -77.9750519, 77.9750519
30: -50.2593193, 45.3741074, -50.2593193, 45.3741074, -95.6334152, 95.6334229
31: -47.6183662, 41.5403900, -47.6183662, 41.5403900, -89.1587524, 89.1587524
32: -59.9542313, 35.7897034, -59.9542313, 35.7897034, -95.7439270, 95.7439270
33: -85.9071808, 43.8326645, -85.9071808, 43.8326645, -129.7398376, 129.7398376
34: -79.8334198, 28.5293617, -79.8334198, 28.5293617, -108.3627777, 108.3627777
35: -70.2571030, 37.5221252, -70.2571030, 37.5221252, -107.7792282, 107.7792282
36: -71.2104950, 39.3675041, -71.2104950, 39.3675041, -110.5780029, 110.5780029
37: -97.7557907, 34.7876053, -97.7557907, 34.7876053, -132.5433960, 132.5433960
38: -87.2721405, 42.3223114, -87.2721405, 42.3223114, -129.5944519, 129.5944519
39: -96.3795929, 46.4996338, -96.3795929, 46.4996338, -142.8792114, 142.8792114
40: -73.8562927, 32.8088455, -73.8562927, 32.8088455, -106.6651382, 106.6651382
41: -64.6400299, 44.5428925, -64.6400299, 44.5428925, -109.1828995, 109.1829071
42: -48.1290588, 30.1027393, -48.1290588, 30.1027393, -78.2317963, 78.2317963

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1536

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4477299, upper bound: 46.4340626
time: 133.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4444743, upper bound: 46.4373364
time: 105.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -84.7672882, 39.3736267, -84.7672882, 39.3736267, -124.1409149, 124.1408997
1: -42.5773087, 31.1177921, -42.5773087, 31.1177921, -73.6950912, 73.6950989
2: -34.6925201, 35.4725037, -34.6925201, 35.4725037, -70.1650238, 70.1650162
3: -46.5430717, 38.9130402, -46.5430717, 38.9130402, -85.4561157, 85.4561157
4: -46.9768639, 39.0464897, -46.9768639, 39.0464897, -86.0233536, 86.0233536
5: -43.2595673, 41.2553864, -43.2595673, 41.2553864, -84.5149384, 84.5149536
6: -61.4064789, 41.3731499, -61.4064789, 41.3731499, -102.7796173, 102.7796249
7: -52.2261963, 39.0390739, -52.2261963, 39.0390739, -91.2652740, 91.2652740
8: -66.3197632, 48.3093834, -66.3197632, 48.3093834, -114.6291428, 114.6291428
9: -44.0484085, 41.3931618, -44.0484085, 41.3931618, -85.4415665, 85.4415588
10: -58.5848846, 49.0133591, -58.5848846, 49.0133591, -107.5982361, 107.5982437
11: -49.7151680, 36.1760406, -49.7151680, 36.1760406, -85.8912048, 85.8912048
12: -66.5686646, 50.4459152, -66.5686646, 50.4459152, -117.0145798, 117.0145798
13: -71.8024597, 54.1911697, -71.8024597, 54.1911697, -125.9936295, 125.9936295
14: -104.4877167, 36.7296181, -104.4877167, 36.7296181, -141.2173309, 141.2173309
15: -50.6970444, 35.7664642, -50.6970444, 35.7664642, -86.4635086, 86.4635086
16: -58.7854805, 40.5555725, -58.7854805, 40.5555725, -99.3410416, 99.3410492
17: -101.2854004, 34.3224030, -101.2854004, 34.3224030, -135.6078033, 135.6078033
18: -58.6059418, 52.6934662, -58.6059418, 52.6934662, -111.2993927, 111.2993927
19: -34.6089745, 27.3176270, -34.6089745, 27.3176270, -61.9265976, 61.9266014
20: -39.1725731, 32.8310394, -39.1725731, 32.8310394, -72.0036011, 72.0036011
21: -46.2914467, 34.8583450, -46.2914467, 34.8583450, -81.1497803, 81.1497879
22: -49.6540070, 32.1295013, -49.6540070, 32.1295013, -81.7835083, 81.7835083
23: -36.4852295, 37.0147552, -36.4852295, 37.0147552, -73.4999771, 73.4999847
24: -48.5414314, 41.0755920, -48.5414314, 41.0755920, -89.6170197, 89.6170197
25: -44.3950119, 37.3724670, -44.3950119, 37.3724670, -81.7674713, 81.7674713
26: -64.4201508, 54.6686516, -64.4201508, 54.6686516, -119.0887985, 119.0887985
27: -47.7461967, 41.8227158, -47.7461967, 41.8227158, -89.5689011, 89.5689087
28: -38.2709885, 41.1035767, -38.2709885, 41.1035767, -79.3745651, 79.3745651
29: -48.5310211, 29.4440346, -48.5310211, 29.4440346, -77.9750519, 77.9750519
30: -50.2593193, 45.3741074, -50.2593193, 45.3741074, -95.6334152, 95.6334229
31: -47.6183662, 41.5403900, -47.6183662, 41.5403900, -89.1587524, 89.1587524
32: -59.9542313, 35.7897034, -59.9542313, 35.7897034, -95.7439270, 95.7439270
33: -85.9071808, 43.8326645, -85.9071808, 43.8326645, -129.7398376, 129.7398376
34: -79.8334198, 28.5293617, -79.8334198, 28.5293617, -108.3627777, 108.3627777
35: -70.2571030, 37.5221252, -70.2571030, 37.5221252, -107.7792282, 107.7792282
36: -71.2104950, 39.3675041, -71.2104950, 39.3675041, -110.5780029, 110.5780029
37: -97.7557907, 34.7876053, -97.7557907, 34.7876053, -132.5433960, 132.5433960
38: -87.2721405, 42.3223114, -87.2721405, 42.3223114, -129.5944519, 129.5944519
39: -96.3795929, 46.4996338, -96.3795929, 46.4996338, -142.8792114, 142.8792114
40: -73.8562927, 32.8088455, -73.8562927, 32.8088455, -106.6651382, 106.6651382
41: -64.6400299, 44.5428925, -64.6400299, 44.5428925, -109.1828995, 109.1829071
42: -48.1290588, 30.1027393, -48.1290588, 30.1027393, -78.2317963, 78.2317963

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1536

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4495183, upper bound: 46.4322562
time: 161.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4462657, upper bound: 46.4355447
time: 125.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -84.7672882, 39.3736267, -84.7672882, 39.3736267, -124.1409149, 124.1408997
1: -42.5773087, 31.1177921, -42.5773087, 31.1177921, -73.6950912, 73.6950989
2: -34.6925201, 35.4725037, -34.6925201, 35.4725037, -70.1650238, 70.1650162
3: -46.5430717, 38.9130402, -46.5430717, 38.9130402, -85.4561157, 85.4561157
4: -46.9768639, 39.0464897, -46.9768639, 39.0464897, -86.0233536, 86.0233536
5: -43.2595673, 41.2553864, -43.2595673, 41.2553864, -84.5149384, 84.5149536
6: -61.4064789, 41.3731499, -61.4064789, 41.3731499, -102.7796173, 102.7796249
7: -52.2261963, 39.0390739, -52.2261963, 39.0390739, -91.2652740, 91.2652740
8: -66.3197632, 48.3093834, -66.3197632, 48.3093834, -114.6291428, 114.6291428
9: -44.0484085, 41.3931618, -44.0484085, 41.3931618, -85.4415665, 85.4415588
10: -58.5848846, 49.0133591, -58.5848846, 49.0133591, -107.5982361, 107.5982437
11: -49.7151680, 36.1760406, -49.7151680, 36.1760406, -85.8912048, 85.8912048
12: -66.5686646, 50.4459152, -66.5686646, 50.4459152, -117.0145798, 117.0145798
13: -71.8024597, 54.1911697, -71.8024597, 54.1911697, -125.9936295, 125.9936295
14: -104.4877167, 36.7296181, -104.4877167, 36.7296181, -141.2173309, 141.2173309
15: -50.6970444, 35.7664642, -50.6970444, 35.7664642, -86.4635086, 86.4635086
16: -58.7854805, 40.5555725, -58.7854805, 40.5555725, -99.3410416, 99.3410492
17: -101.2854004, 34.3224030, -101.2854004, 34.3224030, -135.6078033, 135.6078033
18: -58.6059418, 52.6934662, -58.6059418, 52.6934662, -111.2993927, 111.2993927
19: -34.6089745, 27.3176270, -34.6089745, 27.3176270, -61.9265976, 61.9266014
20: -39.1725731, 32.8310394, -39.1725731, 32.8310394, -72.0036011, 72.0036011
21: -46.2914467, 34.8583450, -46.2914467, 34.8583450, -81.1497803, 81.1497879
22: -49.6540070, 32.1295013, -49.6540070, 32.1295013, -81.7835083, 81.7835083
23: -36.4852295, 37.0147552, -36.4852295, 37.0147552, -73.4999771, 73.4999847
24: -48.5414314, 41.0755920, -48.5414314, 41.0755920, -89.6170197, 89.6170197
25: -44.3950119, 37.3724670, -44.3950119, 37.3724670, -81.7674713, 81.7674713
26: -64.4201508, 54.6686516, -64.4201508, 54.6686516, -119.0887985, 119.0887985
27: -47.7461967, 41.8227158, -47.7461967, 41.8227158, -89.5689011, 89.5689087
28: -38.2709885, 41.1035767, -38.2709885, 41.1035767, -79.3745651, 79.3745651
29: -48.5310211, 29.4440346, -48.5310211, 29.4440346, -77.9750519, 77.9750519
30: -50.2593193, 45.3741074, -50.2593193, 45.3741074, -95.6334152, 95.6334229
31: -47.6183662, 41.5403900, -47.6183662, 41.5403900, -89.1587524, 89.1587524
32: -59.9542313, 35.7897034, -59.9542313, 35.7897034, -95.7439270, 95.7439270
33: -85.9071808, 43.8326645, -85.9071808, 43.8326645, -129.7398376, 129.7398376
34: -79.8334198, 28.5293617, -79.8334198, 28.5293617, -108.3627777, 108.3627777
35: -70.2571030, 37.5221252, -70.2571030, 37.5221252, -107.7792282, 107.7792282
36: -71.2104950, 39.3675041, -71.2104950, 39.3675041, -110.5780029, 110.5780029
37: -97.7557907, 34.7876053, -97.7557907, 34.7876053, -132.5433960, 132.5433960
38: -87.2721405, 42.3223114, -87.2721405, 42.3223114, -129.5944519, 129.5944519
39: -96.3795929, 46.4996338, -96.3795929, 46.4996338, -142.8792114, 142.8792114
40: -73.8562927, 32.8088455, -73.8562927, 32.8088455, -106.6651382, 106.6651382
41: -64.6400299, 44.5428925, -64.6400299, 44.5428925, -109.1828995, 109.1829071
42: -48.1290588, 30.1027393, -48.1290588, 30.1027393, -78.2317963, 78.2317963

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1536

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4270217, upper bound: 46.4547868
time: 189.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4237148, upper bound: 46.4580254
time: 127.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -84.7672882, 39.3736267, -84.7672882, 39.3736267, -124.1409149, 124.1408997
1: -42.5773087, 31.1177921, -42.5773087, 31.1177921, -73.6950912, 73.6950989
2: -34.6925201, 35.4725037, -34.6925201, 35.4725037, -70.1650238, 70.1650162
3: -46.5430717, 38.9130402, -46.5430717, 38.9130402, -85.4561157, 85.4561157
4: -46.9768639, 39.0464897, -46.9768639, 39.0464897, -86.0233536, 86.0233536
5: -43.2595673, 41.2553864, -43.2595673, 41.2553864, -84.5149384, 84.5149536
6: -61.4064789, 41.3731499, -61.4064789, 41.3731499, -102.7796173, 102.7796249
7: -52.2261963, 39.0390739, -52.2261963, 39.0390739, -91.2652740, 91.2652740
8: -66.3197632, 48.3093834, -66.3197632, 48.3093834, -114.6291428, 114.6291428
9: -44.0484085, 41.3931618, -44.0484085, 41.3931618, -85.4415665, 85.4415588
10: -58.5848846, 49.0133591, -58.5848846, 49.0133591, -107.5982361, 107.5982437
11: -49.7151680, 36.1760406, -49.7151680, 36.1760406, -85.8912048, 85.8912048
12: -66.5686646, 50.4459152, -66.5686646, 50.4459152, -117.0145798, 117.0145798
13: -71.8024597, 54.1911697, -71.8024597, 54.1911697, -125.9936295, 125.9936295
14: -104.4877167, 36.7296181, -104.4877167, 36.7296181, -141.2173309, 141.2173309
15: -50.6970444, 35.7664642, -50.6970444, 35.7664642, -86.4635086, 86.4635086
16: -58.7854805, 40.5555725, -58.7854805, 40.5555725, -99.3410416, 99.3410492
17: -101.2854004, 34.3224030, -101.2854004, 34.3224030, -135.6078033, 135.6078033
18: -58.6059418, 52.6934662, -58.6059418, 52.6934662, -111.2993927, 111.2993927
19: -34.6089745, 27.3176270, -34.6089745, 27.3176270, -61.9265976, 61.9266014
20: -39.1725731, 32.8310394, -39.1725731, 32.8310394, -72.0036011, 72.0036011
21: -46.2914467, 34.8583450, -46.2914467, 34.8583450, -81.1497803, 81.1497879
22: -49.6540070, 32.1295013, -49.6540070, 32.1295013, -81.7835083, 81.7835083
23: -36.4852295, 37.0147552, -36.4852295, 37.0147552, -73.4999771, 73.4999847
24: -48.5414314, 41.0755920, -48.5414314, 41.0755920, -89.6170197, 89.6170197
25: -44.3950119, 37.3724670, -44.3950119, 37.3724670, -81.7674713, 81.7674713
26: -64.4201508, 54.6686516, -64.4201508, 54.6686516, -119.0887985, 119.0887985
27: -47.7461967, 41.8227158, -47.7461967, 41.8227158, -89.5689011, 89.5689087
28: -38.2709885, 41.1035767, -38.2709885, 41.1035767, -79.3745651, 79.3745651
29: -48.5310211, 29.4440346, -48.5310211, 29.4440346, -77.9750519, 77.9750519
30: -50.2593193, 45.3741074, -50.2593193, 45.3741074, -95.6334152, 95.6334229
31: -47.6183662, 41.5403900, -47.6183662, 41.5403900, -89.1587524, 89.1587524
32: -59.9542313, 35.7897034, -59.9542313, 35.7897034, -95.7439270, 95.7439270
33: -85.9071808, 43.8326645, -85.9071808, 43.8326645, -129.7398376, 129.7398376
34: -79.8334198, 28.5293617, -79.8334198, 28.5293617, -108.3627777, 108.3627777
35: -70.2571030, 37.5221252, -70.2571030, 37.5221252, -107.7792282, 107.7792282
36: -71.2104950, 39.3675041, -71.2104950, 39.3675041, -110.5780029, 110.5780029
37: -97.7557907, 34.7876053, -97.7557907, 34.7876053, -132.5433960, 132.5433960
38: -87.2721405, 42.3223114, -87.2721405, 42.3223114, -129.5944519, 129.5944519
39: -96.3795929, 46.4996338, -96.3795929, 46.4996338, -142.8792114, 142.8792114
40: -73.8562927, 32.8088455, -73.8562927, 32.8088455, -106.6651382, 106.6651382
41: -64.6400299, 44.5428925, -64.6400299, 44.5428925, -109.1828995, 109.1829071
42: -48.1290588, 30.1027393, -48.1290588, 30.1027393, -78.2317963, 78.2317963

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1536

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4580254, upper bound: 46.4237148
time: 127.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -46.4547868, upper bound: 46.4270217
time: 120.01 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 249.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 249.10
Output dim: 19, lower bound: -46.4701292, upper bound: 46.4114286
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 249.10
Output dim: 19, lower bound: -46.4669162, upper bound: 46.4147598
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 249.10
Output dim: 19, lower bound: -46.4477299, upper bound: 46.4340626
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 249.10
Output dim: 19, lower bound: -46.4444743, upper bound: 46.4373364
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 249.10
Output dim: 19, lower bound: -46.4495183, upper bound: 46.4322562
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 249.10
Output dim: 19, lower bound: -46.4462657, upper bound: 46.4355447
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 249.10
Output dim: 19, lower bound: -46.4270217, upper bound: 46.4547868
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 249.10
Output dim: 19, lower bound: -46.4237148, upper bound: 46.4580254
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 249.10
Output dim: 19, lower bound: -46.4580254, upper bound: 46.4237148
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 249.10
Output dim: 19, lower bound: -46.4547868, upper bound: 46.4270217
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 249.10
Output dim: 19, lower bound: -46.5030921, upper bound: 46.5173217
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 249.10
Output dim: 19, lower bound: -46.5049864, upper bound: 46.5154100
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 249.10
Output dim: 19, lower bound: -46.4819694, upper bound: 46.5382696

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 174.00 + 3566.30 = 3740.31 seconds
