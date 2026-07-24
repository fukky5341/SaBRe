## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 3600 seconds
Split limit: 100
Threshold: 72.4521242511


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-120.6897812, 75.5772247, -120.6897812, 75.5772247, -196.2669983, 196.2669983)
1: (-69.1185150, 62.0534592, -69.1185150, 62.0534592, -131.1719666, 131.1719666)
2: (-59.9247818, 57.4265594, -59.9247818, 57.4265594, -117.3513336, 117.3513336)
3: (-69.0837555, 72.1842041, -69.0837555, 72.1842041, -141.2679443, 141.2679443)
4: (-75.4698944, 68.2243805, -75.4698944, 68.2243805, -143.6942749, 143.6942749)
5: (-67.5072861, 74.3775482, -67.5072861, 74.3775482, -141.8848267, 141.8848267)
6: (-85.3311462, 75.1246033, -85.3311462, 75.1246033, -160.4557495, 160.4557495)
7: (-83.4509277, 75.4055939, -83.4509277, 75.4055939, -158.8565216, 158.8565216)
8: (-81.8659363, 84.5110092, -81.8659363, 84.5110092, -166.3769531, 166.3769531)
9: (-72.3079300, 68.8767319, -72.3079300, 68.8767319, -141.1846619, 141.1846466)
10: (-105.4851379, 97.0667877, -105.4851379, 97.0667877, -202.5519257, 202.5519257)
11: (-98.7560196, 70.3052673, -98.7560196, 70.3052673, -169.0612793, 169.0612793)
12: (-94.6711731, 79.6737671, -94.6711731, 79.6737671, -174.3449402, 174.3449402)
13: (-101.8092804, 95.6050262, -101.8092804, 95.6050262, -197.4143066, 197.4143066)
14: (-153.0820007, 76.5118256, -153.0820007, 76.5118256, -229.5938110, 229.5938263)
15: (-85.4207764, 65.5752563, -85.4207764, 65.5752563, -150.9960327, 150.9960327)
16: (-108.7743988, 80.8979950, -108.7743988, 80.8979950, -189.6723633, 189.6723785)
17: (-160.1158142, 100.7368164, -160.1158142, 100.7368164, -260.8526306, 260.8526306)
18: (-92.6273956, 75.1741791, -92.6273956, 75.1741791, -167.8015594, 167.8015747)
19: (-72.3558044, 42.4454155, -72.3558044, 42.4454155, -114.8012085, 114.8012238)
20: (-68.1926117, 53.3375359, -68.1926117, 53.3375359, -121.5301361, 121.5301361)
21: (-92.3123550, 55.6030388, -92.3123550, 55.6030388, -147.9153748, 147.9153748)
22: (-101.0745773, 60.5765076, -101.0745773, 60.5765076, -161.6510925, 161.6510925)
23: (-72.3953247, 55.8623962, -72.3953247, 55.8623962, -128.2577209, 128.2577209)
24: (-91.0453033, 67.0268860, -91.0453033, 67.0268860, -158.0721741, 158.0721741)
25: (-78.7736359, 63.4368629, -78.7736359, 63.4368629, -142.2104950, 142.2104950)
26: (-107.2412415, 83.0555573, -107.2412415, 83.0555573, -190.2967987, 190.2967987)
27: (-92.9772720, 67.3610229, -92.9772720, 67.3610229, -160.3382874, 160.3382874)
28: (-71.9242935, 59.8130646, -71.9242935, 59.8130646, -131.7373657, 131.7373505)
29: (-107.8225708, 63.9952202, -107.8225708, 63.9952202, -171.8177948, 171.8177948)
30: (-89.9596481, 72.8284531, -89.9596481, 72.8284531, -162.7881012, 162.7881012)
31: (-92.2802734, 61.3140030, -92.2802734, 61.3140030, -153.5942688, 153.5942688)
32: (-90.6332397, 70.1865616, -90.6332397, 70.1865616, -160.8197937, 160.8197937)
33: (-120.4985809, 94.1850891, -120.4985809, 94.1850891, -214.6836548, 214.6836700)
34: (-100.3700409, 73.7348175, -100.3700409, 73.7348175, -174.1048584, 174.1048584)
35: (-104.3925476, 77.7684860, -104.3925476, 77.7684860, -182.1610413, 182.1610260)
36: (-100.9291229, 75.9416580, -100.9291229, 75.9416580, -176.8707886, 176.8707886)
37: (-141.9010010, 84.0858002, -141.9010010, 84.0858002, -225.9868011, 225.9868011)
38: (-122.6707611, 92.7453461, -122.6707611, 92.7453461, -215.4160919, 215.4161072)
39: (-145.5377960, 92.6060257, -145.5377960, 92.6060257, -238.1438141, 238.1438141)
40: (-115.8655548, 81.4251404, -115.8655548, 81.4251404, -197.2906952, 197.2906952)
41: (-89.2050629, 70.4504623, -89.2050629, 70.4504623, -159.6555176, 159.6555176)
42: (-64.9341888, 62.9433632, -64.9341888, 62.9433632, -127.8775482, 127.8775482)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.87 + 210.87 = 213.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -72.5246480, upper bound: 72.5246489

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1617

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4770400, upper bound: 72.5232166
time: 140.80 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4770400, upper bound: 72.5232166
time: 132.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 273.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 273.81
Output dim: 2, lower bound: -72.4770400, upper bound: 72.5232166
IS_A2, status: Status.UNKNOWN, split count: 1, time: 273.81
Output dim: 2, lower bound: -72.4770400, upper bound: 72.5232166

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -120.4648285, 75.5080414, -120.5604095, 75.5373840, -196.0021973, 196.0684509
1: -68.9701691, 62.0014572, -69.0330658, 62.0233612, -130.9935303, 131.0345154
2: -59.6827545, 57.3815155, -59.7855606, 57.4005737, -117.0833282, 117.1670685
3: -68.8027954, 72.1238403, -68.9221802, 72.1493683, -140.9521637, 141.0460205
4: -75.2091217, 68.1693802, -75.3199081, 68.1926041, -143.4017181, 143.4892883
5: -67.2610626, 74.3218460, -67.3657379, 74.3454895, -141.6065521, 141.6875916
6: -85.2345276, 74.9865799, -85.2755432, 75.0447159, -160.2792358, 160.2621155
7: -83.2469254, 75.3571625, -83.3334045, 75.3777466, -158.6246643, 158.6905670
8: -81.6216431, 84.4483337, -81.7253265, 84.4748535, -166.0964966, 166.1736603
9: -72.1242981, 68.7802811, -72.2020340, 68.8210297, -140.9453278, 140.9822998
10: -105.3634491, 96.7401657, -105.4147415, 96.8787537, -202.2422028, 202.1549072
11: -98.6610565, 69.8601837, -98.7011948, 70.0495377, -168.7106018, 168.5613708
12: -94.5941467, 79.2738419, -94.6269150, 79.4438171, -174.0379639, 173.9007416
13: -101.5561371, 95.4751434, -101.6581573, 95.5305176, -197.0866394, 197.1333008
14: -152.9371185, 76.1634827, -152.9986877, 76.3116455, -229.2487640, 229.1621704
15: -85.2215500, 65.4838638, -85.3062057, 65.5225677, -150.7441101, 150.7900696
16: -108.6180954, 80.7004700, -108.6840515, 80.7843628, -189.4024506, 189.3845215
17: -159.9970245, 100.2130814, -160.0475006, 100.4356232, -260.4326172, 260.2605591
18: -92.5177917, 74.8356171, -92.5642853, 74.9777222, -167.4955139, 167.3999023
19: -72.2768250, 42.2307129, -72.3100052, 42.3218842, -114.5987091, 114.5407181
20: -68.1162262, 53.1647644, -68.1485901, 53.2382050, -121.3544312, 121.3133469
21: -92.2252274, 55.2936211, -92.2620773, 55.4252243, -147.6504517, 147.5556946
22: -100.9855957, 60.2981834, -101.0234833, 60.4160919, -161.4016876, 161.3216705
23: -72.3282242, 55.6549225, -72.3566208, 55.7430992, -128.0713196, 128.0115356
24: -90.9622498, 66.8287735, -90.9975739, 66.9130096, -157.8752441, 157.8263397
25: -78.7071838, 63.2438507, -78.7352066, 63.3258247, -142.0329895, 141.9790649
26: -107.1440582, 82.6410675, -107.1852036, 82.8170166, -189.9610748, 189.8262634
27: -92.8624344, 67.1308746, -92.9113312, 67.2286835, -160.0911255, 160.0422058
28: -71.8502960, 59.6439781, -71.8816986, 59.7156868, -131.5659790, 131.5256805
29: -107.7423401, 63.6287651, -107.7764587, 63.7845078, -171.5268250, 171.4052277
30: -89.8779602, 72.5169525, -89.9124374, 72.6493378, -162.5272980, 162.4293823
31: -92.1724548, 61.0911713, -92.2181625, 61.1857834, -153.3582306, 153.3093262
32: -90.5299072, 70.0285339, -90.5735779, 70.0948868, -160.6247864, 160.6021118
33: -120.2309189, 94.1047668, -120.3447723, 94.1388321, -214.3697510, 214.4495392
34: -100.1981812, 73.6386032, -100.2711182, 73.6785126, -173.8766937, 173.9096985
35: -104.2013550, 77.7140350, -104.2813263, 77.7370758, -181.9384308, 181.9953613
36: -100.7988892, 75.8674088, -100.8541565, 75.8987198, -176.6976013, 176.7215576
37: -141.7777710, 83.9378891, -141.8299866, 84.0000687, -225.7778320, 225.7678680
38: -122.4627991, 92.6404877, -122.5509644, 92.6846008, -215.1473999, 215.1914520
39: -145.3027954, 92.5500488, -145.4023743, 92.5737457, -237.8765106, 237.9524231
40: -115.7109528, 81.3584366, -115.7764893, 81.3862000, -197.0971375, 197.1349182
41: -89.1016388, 70.3179550, -89.1455536, 70.3725510, -159.4741821, 159.4635010
42: -64.8565979, 62.7147598, -64.8894424, 62.8110352, -127.6676331, 127.6042023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=367, inp2_unstable=368, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1617

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4759621, upper bound: 72.4940361
time: 168.40 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4759621, upper bound: 72.5219895
time: 197.67 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -120.7087555, 75.6037903, -120.6675262, 75.5675735, -196.2763214, 196.2713165
1: -69.1220322, 62.0654793, -69.1035767, 62.0393372, -131.1613770, 131.1690521
2: -59.9123611, 57.5343933, -59.9030113, 57.4205780, -117.3329391, 117.4374084
3: -69.0710907, 72.3350372, -69.0593567, 72.1762085, -141.2472992, 141.3943787
4: -75.4595566, 68.2966614, -75.4463654, 68.2169647, -143.6765137, 143.7430267
5: -67.4996109, 74.5317993, -67.4863739, 74.3704681, -141.8700867, 142.0181580
6: -85.3588409, 75.0826263, -85.3188934, 75.0682297, -160.4270630, 160.4015198
7: -83.4548569, 75.4262085, -83.4291229, 75.3979187, -158.8527679, 158.8553314
8: -81.8572998, 84.5866241, -81.8435974, 84.5014191, -166.3587189, 166.4302216
9: -72.3111725, 68.9017258, -72.2919006, 68.8595581, -141.1706848, 141.1936340
10: -105.5227814, 97.0825195, -105.4716797, 97.0351715, -202.5579529, 202.5541992
11: -98.9690018, 70.2745361, -98.7402267, 70.2686005, -169.2375793, 169.0147552
12: -94.8674622, 79.6575394, -94.6615906, 79.6390686, -174.5065002, 174.3191223
13: -101.7736206, 95.7133636, -101.7635422, 95.5892105, -197.3628235, 197.4768982
14: -153.2089996, 76.4906769, -153.0664978, 76.4848099, -229.6937714, 229.5571442
15: -85.3939972, 65.5964508, -85.3649902, 65.5618591, -150.9558411, 150.9614258
16: -108.8179626, 80.8314514, -108.7564240, 80.8162994, -189.6342621, 189.5878754
17: -160.3310547, 100.6944580, -160.1040344, 100.6906433, -261.0216675, 260.7984924
18: -92.7879639, 75.1522827, -92.6134567, 75.1433334, -167.9313049, 167.7657471
19: -72.4802399, 42.4304199, -72.3465500, 42.4268265, -114.9070587, 114.7769699
20: -68.2716141, 53.3308487, -68.1824951, 53.3227005, -121.5942993, 121.5133362
21: -92.4890366, 55.5827942, -92.3010864, 55.5789566, -148.0679932, 147.8838806
22: -101.1244965, 60.5604706, -101.0618896, 60.5496559, -161.6741333, 161.6223602
23: -72.5003052, 55.8531914, -72.3880463, 55.8442039, -128.3444977, 128.2412415
24: -91.1388779, 67.0164337, -91.0329208, 67.0115356, -158.1504211, 158.0493469
25: -78.8101425, 63.4218254, -78.7627869, 63.4173279, -142.2274780, 142.1846161
26: -107.4677734, 83.0378265, -107.2269592, 83.0214386, -190.4891968, 190.2647858
27: -93.0825806, 67.3465042, -92.9626694, 67.3422623, -160.4248352, 160.3091736
28: -72.0120850, 59.8115234, -71.9167633, 59.7980576, -131.8101196, 131.7282867
29: -107.9106293, 63.9671021, -107.8105621, 63.9650650, -171.8756714, 171.7776489
30: -90.0762634, 72.8211136, -89.9458313, 72.8023224, -162.8785858, 162.7669373
31: -92.4162445, 61.2958984, -92.2690735, 61.2939529, -153.7102051, 153.5649719
32: -90.6496964, 70.1907349, -90.6141205, 70.1695175, -160.8191986, 160.8048553
33: -120.4947891, 94.2721329, -120.4731369, 94.1749420, -214.6697388, 214.7452698
34: -100.3777618, 73.7594986, -100.3523865, 73.7207184, -174.0984802, 174.1118774
35: -104.3897934, 77.8125153, -104.3689499, 77.7616272, -182.1514282, 182.1814575
36: -100.9391479, 75.9615173, -100.9104309, 75.9336395, -176.8727875, 176.8719330
37: -141.9455566, 84.0678711, -141.8841858, 84.0512619, -225.9968262, 225.9520569
38: -122.6953354, 92.7546387, -122.6479034, 92.7222519, -215.4175873, 215.4025269
39: -145.5389404, 92.6772232, -145.5118256, 92.5990982, -238.1380157, 238.1890564
40: -115.8787766, 81.4086075, -115.8461990, 81.3884506, -197.2672272, 197.2548065
41: -89.2237701, 70.4294891, -89.1935425, 70.4073257, -159.6311035, 159.6230316
42: -64.9678116, 62.9496574, -64.9247131, 62.9206924, -127.8885040, 127.8743591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=367, inp2_unstable=368, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1617

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4759621, upper bound: 72.4940361
time: 222.85 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4759621, upper bound: 72.5219895
time: 185.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 410.28 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 410.28
Output dim: 2, lower bound: -72.4759621, upper bound: 72.4940361
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 410.28
Output dim: 2, lower bound: -72.4759621, upper bound: 72.5219895
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 410.28
Output dim: 2, lower bound: -72.4759621, upper bound: 72.4940361
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 410.28
Output dim: 2, lower bound: -72.4759621, upper bound: 72.5219895

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -120.2921982, 75.4592896, -120.2676239, 75.4549408, -195.7471313, 195.7268982
1: -68.8720398, 61.9660416, -68.8665466, 61.9632645, -130.8352966, 130.8325806
2: -59.5312538, 57.3438492, -59.5292664, 57.3368454, -116.8680801, 116.8731079
3: -68.6314392, 72.0760498, -68.6315918, 72.0686646, -140.7001038, 140.7076416
4: -75.0324860, 68.1231766, -75.0206833, 68.1141434, -143.1466217, 143.1438599
5: -67.1112518, 74.2757339, -67.1118011, 74.2677917, -141.3790283, 141.3875427
6: -85.1641846, 74.8640137, -85.1556244, 74.8375397, -160.0017242, 160.0196381
7: -83.1440735, 75.3035126, -83.1576996, 75.2878952, -158.4319763, 158.4612122
8: -81.4712677, 84.4016113, -81.4705887, 84.3957214, -165.8669891, 165.8721924
9: -71.9611359, 68.7194824, -71.9259949, 68.7178650, -140.6790009, 140.6454773
10: -105.1971664, 96.5803528, -105.1357040, 96.6097946, -201.8069611, 201.7160645
11: -98.5911407, 69.5284882, -98.5822678, 69.4872360, -168.0783691, 168.1107483
12: -94.5277252, 79.0176926, -94.5149536, 79.0097733, -173.5374603, 173.5326538
13: -101.4225006, 95.3818512, -101.4313812, 95.3719635, -196.7944336, 196.8132172
14: -152.8339386, 75.9275970, -152.8238678, 75.9118881, -228.7458191, 228.7514648
15: -85.0513611, 65.4271164, -85.0207214, 65.4251404, -150.4765015, 150.4478302
16: -108.5179596, 80.5463867, -108.5133438, 80.5256042, -189.0435638, 189.0597229
17: -159.8991699, 99.7635956, -159.8820496, 99.6744461, -259.5736084, 259.6456299
18: -92.4368134, 74.6508789, -92.4266815, 74.6630325, -167.0998383, 167.0775604
19: -72.2163162, 42.0510979, -72.2071457, 42.0182800, -114.2345963, 114.2582397
20: -68.0614319, 53.0164337, -68.0556946, 52.9876900, -121.0491028, 121.0721130
21: -92.1563568, 55.0492783, -92.1448822, 55.0118561, -147.1682129, 147.1941528
22: -100.9142227, 60.1039047, -100.9027100, 60.0850029, -160.9992218, 161.0066223
23: -72.2767029, 55.5122414, -72.2692719, 55.5016747, -127.7783813, 127.7815094
24: -90.8963013, 66.7036133, -90.8865585, 66.7008209, -157.5971222, 157.5901794
25: -78.6486511, 63.1075592, -78.6366119, 63.0952225, -141.7438660, 141.7441711
26: -107.0656281, 82.4023743, -107.0528183, 82.4112320, -189.4768677, 189.4552002
27: -92.7834091, 66.9526062, -92.7773666, 66.9272003, -159.7106018, 159.7299805
28: -71.7936401, 59.4730225, -71.7858734, 59.4274559, -131.2210999, 131.2588959
29: -107.6751862, 63.3506470, -107.6627426, 63.3120804, -170.9872742, 171.0133667
30: -89.8160706, 72.2699966, -89.8075714, 72.2311859, -162.0472565, 162.0775757
31: -92.0979614, 60.9298630, -92.0915909, 60.9128723, -153.0108337, 153.0214539
32: -90.4560547, 69.8958130, -90.4482956, 69.8713226, -160.3273621, 160.3441010
33: -120.0234680, 94.0454712, -119.9932480, 94.0379639, -214.0614319, 214.0387268
34: -100.0322571, 73.5885925, -99.9900208, 73.5934448, -173.6257019, 173.5786133
35: -104.0634003, 77.6765747, -104.0473709, 77.6733856, -181.7367859, 181.7239380
36: -100.7077026, 75.7854767, -100.7008514, 75.7614136, -176.4691162, 176.4863281
37: -141.6749573, 83.8416595, -141.6570892, 83.8399200, -225.5148773, 225.4987488
38: -122.2917328, 92.5869293, -122.2608337, 92.5936584, -214.8853607, 214.8477631
39: -145.1177673, 92.5072250, -145.0890198, 92.5010223, -237.6187744, 237.5962524
40: -115.6019058, 81.3086090, -115.5911636, 81.3017044, -196.9036102, 196.8997803
41: -89.0175781, 70.2237701, -89.0030823, 70.2134628, -159.2310486, 159.2268524
42: -64.7853622, 62.5705719, -64.7688217, 62.5670166, -127.3523560, 127.3393936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=367, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1617

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4452308, upper bound: 72.4933623
time: 139.48 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4452308, upper bound: 72.4933614
time: 169.78 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -120.4395447, 75.4940643, -120.5734177, 75.5834656, -196.0230103, 196.0674744
1: -68.9535446, 61.9872437, -69.0393219, 62.0335464, -130.9870911, 131.0265656
2: -59.6603737, 57.3722992, -59.7775841, 57.4857483, -117.1461029, 117.1498871
3: -68.7775955, 72.1107864, -68.9202805, 72.2532043, -141.0307922, 141.0310669
4: -75.1824341, 68.1584778, -75.3098602, 68.2930450, -143.4754791, 143.4683380
5: -67.2407074, 74.3098297, -67.3702927, 74.4733963, -141.7140808, 141.6801147
6: -85.2173462, 74.9268875, -85.3019409, 75.0004272, -160.2177734, 160.2288208
7: -83.2260742, 75.3315811, -83.3424072, 75.3733444, -158.5994263, 158.6739807
8: -81.5954590, 84.4363174, -81.7154846, 84.5225372, -166.1179810, 166.1517944
9: -72.0973663, 68.7648544, -72.1929016, 68.8676453, -140.9650116, 140.9577332
10: -105.3357697, 96.7043762, -105.4300995, 96.9037399, -202.2395020, 202.1344757
11: -98.6370850, 69.8165054, -98.9249268, 70.0154266, -168.6524963, 168.7414246
12: -94.5804672, 79.2367706, -94.7849884, 79.4304504, -174.0109253, 174.0217590
13: -101.4822159, 95.4547577, -101.5900726, 95.5822372, -197.0644379, 197.0448303
14: -152.9168243, 76.1335907, -153.1298828, 76.2918549, -229.2086792, 229.2634735
15: -85.1277466, 65.4660187, -85.2445526, 65.5349579, -150.6627045, 150.7105713
16: -108.5883026, 80.6057205, -108.7012939, 80.7121964, -189.3005066, 189.3070068
17: -159.9806519, 100.1511383, -160.3406677, 100.3769455, -260.3576050, 260.4917603
18: -92.5003815, 74.7913208, -92.6609955, 74.9434586, -167.4438477, 167.4523010
19: -72.2623444, 42.2077484, -72.4496002, 42.3075562, -114.5699005, 114.6573486
20: -68.1020966, 53.1453362, -68.2234192, 53.2285271, -121.3306274, 121.3687592
21: -92.2089157, 55.2647095, -92.4605484, 55.4033051, -147.6122131, 147.7252502
22: -100.9708481, 60.2644501, -101.1021118, 60.3949814, -161.3658295, 161.3665466
23: -72.3170776, 55.6341400, -72.4705276, 55.7293282, -128.0464020, 128.1046448
24: -90.9456482, 66.8106995, -91.0586700, 66.9076309, -157.8532715, 157.8693695
25: -78.6911316, 63.2206612, -78.7759476, 63.3091774, -142.0002899, 141.9966125
26: -107.1242676, 82.6045227, -107.3751831, 82.7925568, -189.9168243, 189.9797058
27: -92.8443298, 67.1063156, -93.0198135, 67.2186127, -160.0629272, 160.1261292
28: -71.8395233, 59.6230164, -72.0042572, 59.7066002, -131.5460968, 131.6272736
29: -107.7290039, 63.5902023, -107.9278793, 63.7477112, -171.4767151, 171.5180817
30: -89.8578415, 72.4851227, -90.0531693, 72.6390381, -162.4968872, 162.5382843
31: -92.1554871, 61.0673904, -92.3448105, 61.1682053, -153.3237000, 153.4122009
32: -90.5003510, 69.9986267, -90.5724716, 70.0911102, -160.5914612, 160.5711060
33: -120.1974106, 94.0906067, -120.3428497, 94.2055511, -214.4029388, 214.4334564
34: -100.1723938, 73.6218414, -100.2694244, 73.7105484, -173.8829346, 173.8912659
35: -104.1688232, 77.7014160, -104.2777557, 77.7660141, -181.9348297, 181.9791718
36: -100.7770309, 75.8548431, -100.8687286, 75.9223480, -176.6993713, 176.7235718
37: -141.7525024, 83.8938446, -141.8586121, 83.9763336, -225.7288361, 225.7524567
38: -122.4340515, 92.6251984, -122.5792389, 92.7228394, -215.1568604, 215.2044373
39: -145.2657166, 92.5413895, -145.3861389, 92.6404190, -237.9061279, 237.9275208
40: -115.6824646, 81.3270264, -115.7729034, 81.3986588, -197.0811157, 197.0999298
41: -89.0813599, 70.2638931, -89.1574554, 70.3398743, -159.4212341, 159.4213562
42: -64.8413773, 62.6886253, -64.9256744, 62.8078003, -127.6491547, 127.6142883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=367, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1617

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4452308, upper bound: 72.5213175
time: 179.64 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4452308, upper bound: 72.5213175
time: 161.80 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -120.5356750, 75.5551529, -120.3746567, 75.4853439, -196.0210266, 195.9298096
1: -69.0235062, 62.0296936, -68.9369507, 61.9791069, -131.0026093, 130.9666443
2: -59.7607422, 57.4964294, -59.6466637, 57.3565063, -117.1172333, 117.1430969
3: -68.8995209, 72.2868347, -68.7687073, 72.0948944, -140.9944153, 141.0555420
4: -75.2827301, 68.2502747, -75.1470490, 68.1382904, -143.4210205, 143.3973236
5: -67.3497009, 74.4856415, -67.2324371, 74.2925262, -141.6422272, 141.7180634
6: -85.2877960, 74.9598160, -85.1988068, 74.8612061, -160.1489868, 160.1586304
7: -83.3509979, 75.3723831, -83.2531891, 75.3078232, -158.6588135, 158.6255646
8: -81.7068481, 84.5397491, -81.5888062, 84.4221344, -166.1289825, 166.1285553
9: -72.1468353, 68.8408203, -72.0145569, 68.7565994, -140.9034119, 140.8553772
10: -105.3540726, 96.9225159, -105.1906433, 96.7662430, -202.1203156, 202.1131592
11: -98.8988876, 69.9427185, -98.6209106, 69.7061691, -168.6050568, 168.5636292
12: -94.8012390, 79.4013214, -94.5498123, 79.2050400, -174.0062866, 173.9511414
13: -101.6321564, 95.6197205, -101.5252686, 95.4315872, -197.0637360, 197.1449585
14: -153.1059570, 76.2542496, -152.8917847, 76.0845184, -229.1904755, 229.1460266
15: -85.2150879, 65.5384445, -85.0748520, 65.4642029, -150.6792908, 150.6132965
16: -108.7156982, 80.6767578, -108.5844040, 80.5582581, -189.2739258, 189.2611389
17: -160.2333984, 100.2440186, -159.9387817, 99.9283905, -260.1618042, 260.1828003
18: -92.7082062, 74.9625778, -92.4760284, 74.8238831, -167.5320892, 167.4385986
19: -72.4187469, 42.2506676, -72.2422943, 42.1231842, -114.5419312, 114.4929657
20: -68.2168045, 53.1822586, -68.0895691, 53.0719566, -121.2887573, 121.2718201
21: -92.4196777, 55.3383560, -92.1835175, 55.1655388, -147.5852203, 147.5218811
22: -101.0532837, 60.3641739, -100.9414520, 60.2176361, -161.2709198, 161.3056335
23: -72.4485397, 55.7104301, -72.3003845, 55.6027260, -128.0512543, 128.0108032
24: -91.0729828, 66.8909607, -90.9219208, 66.7991333, -157.8721008, 157.8128662
25: -78.7509766, 63.2852745, -78.6636810, 63.1866837, -141.9376526, 141.9489441
26: -107.3894958, 82.7976532, -107.0946960, 82.6142120, -190.0037079, 189.8923340
27: -93.0057068, 67.1675415, -92.8292389, 67.0401917, -160.0458984, 159.9967804
28: -71.9554367, 59.6403389, -71.8208771, 59.5096207, -131.4650574, 131.4612122
29: -107.8435287, 63.6882172, -107.6969528, 63.4925842, -171.3361206, 171.3851624
30: -90.0140533, 72.5737152, -89.8406525, 72.3838348, -162.3978882, 162.4143677
31: -92.3423996, 61.1342926, -92.1419067, 61.0209312, -153.3633118, 153.2761993
32: -90.5746002, 70.0576019, -90.4874496, 69.9455109, -160.5200806, 160.5450439
33: -120.2870636, 94.2133026, -120.1213913, 94.0740814, -214.3611450, 214.3346863
34: -100.2116852, 73.7076645, -100.0712051, 73.6336670, -173.8453522, 173.7788696
35: -104.2442780, 77.7746277, -104.1261292, 77.6976089, -181.9418945, 181.9007568
36: -100.8475800, 75.8780670, -100.7569427, 75.7955322, -176.6431122, 176.6350098
37: -141.8420868, 83.9714203, -141.7102966, 83.8909760, -225.7330627, 225.6817169
38: -122.5237656, 92.6994019, -122.3575134, 92.6302185, -215.1539917, 215.0569153
39: -145.3524933, 92.6345520, -145.1969299, 92.5264893, -237.8789673, 237.8314819
40: -115.7688370, 81.3580322, -115.6603470, 81.3036118, -197.0724487, 197.0183716
41: -89.1389008, 70.3351974, -89.0507050, 70.2482224, -159.3871155, 159.3858948
42: -64.8962097, 62.8047180, -64.8039398, 62.6758270, -127.5720367, 127.6086578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=367, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1617

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4452308, upper bound: 72.4933623
time: 207.65 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4452308, upper bound: 72.4933623
time: 149.18 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -120.6832428, 75.5901337, -120.6800232, 75.6132355, -196.2964783, 196.2701569
1: -69.1053085, 62.0514107, -69.1094055, 62.0496597, -131.1549683, 131.1607971
2: -59.8897362, 57.5253830, -59.8943100, 57.5057793, -117.3955078, 117.4196854
3: -69.0456772, 72.3222046, -69.0567703, 72.2801285, -141.3257751, 141.3789673
4: -75.4324875, 68.2861481, -75.4354019, 68.3173370, -143.7498169, 143.7215576
5: -67.4789581, 74.5199280, -67.4901581, 74.4984207, -141.9773865, 142.0100708
6: -85.3419495, 75.0230942, -85.3452454, 75.0240326, -160.3659668, 160.3683472
7: -83.4339752, 75.4004822, -83.4376144, 75.3935394, -158.8275146, 158.8380890
8: -81.8309631, 84.5748062, -81.8329773, 84.5491028, -166.3800507, 166.4077759
9: -72.2845154, 68.8860474, -72.2827606, 68.9055939, -141.1901093, 141.1688080
10: -105.4955902, 97.0462494, -105.4871521, 97.0595932, -202.5551758, 202.5334015
11: -98.9461670, 70.2304001, -98.9642868, 70.2334900, -169.1796570, 169.1946869
12: -94.8538437, 79.6201782, -94.8195648, 79.6247635, -174.4786072, 174.4397430
13: -101.7036133, 95.6931000, -101.6945801, 95.6413956, -197.3449860, 197.3876801
14: -153.1887512, 76.4605408, -153.1976318, 76.4645081, -229.6532593, 229.6581726
15: -85.3100891, 65.5791473, -85.3094940, 65.5746613, -150.8847504, 150.8886414
16: -108.7890930, 80.7363968, -108.7743988, 80.7439728, -189.5330658, 189.5107880
17: -160.3147583, 100.6324997, -160.3972015, 100.6312561, -260.9460144, 261.0296936
18: -92.7707291, 75.1085205, -92.7092438, 75.1088409, -167.8795776, 167.8177643
19: -72.4660339, 42.4073257, -72.4857559, 42.4119797, -114.8779984, 114.8930817
20: -68.2578812, 53.3113289, -68.2574234, 53.3128395, -121.5707245, 121.5687561
21: -92.4732056, 55.5536423, -92.4995499, 55.5564346, -148.0296326, 148.0531921
22: -101.1098633, 60.5270157, -101.1402893, 60.5282898, -161.6381378, 161.6672974
23: -72.4894943, 55.8322830, -72.5019531, 55.8300095, -128.3195038, 128.3342285
24: -91.1224670, 66.9983292, -91.0941391, 67.0060425, -158.1285095, 158.0924683
25: -78.7943420, 63.3986282, -78.8038483, 63.4005089, -142.1948395, 142.2024841
26: -107.4482574, 83.0014496, -107.4169312, 82.9964600, -190.4447021, 190.4183807
27: -93.0649185, 67.3220367, -93.0696106, 67.3321686, -160.3970795, 160.3916473
28: -72.0015106, 59.7904472, -72.0392761, 59.7885170, -131.7900238, 131.8297119
29: -107.8973846, 63.9282227, -107.9619217, 63.9275818, -171.8249664, 171.8901367
30: -90.0569153, 72.7890472, -90.0868835, 72.7914505, -162.8483582, 162.8759155
31: -92.3998260, 61.2720947, -92.3949814, 61.2760849, -153.6759033, 153.6670685
32: -90.6197586, 70.1609421, -90.6124802, 70.1656647, -160.7854309, 160.7734222
33: -120.4610825, 94.2583618, -120.4708099, 94.2410583, -214.7021484, 214.7291718
34: -100.3519669, 73.7428436, -100.3504257, 73.7519989, -174.1039581, 174.0932617
35: -104.3633270, 77.8001404, -104.3675156, 77.7906952, -182.1540222, 182.1676636
36: -100.9171829, 75.9488449, -100.9248428, 75.9571533, -176.8743286, 176.8736877
37: -141.9205933, 84.0270386, -141.9130859, 84.0290146, -225.9496155, 225.9401245
38: -122.6663055, 92.7394028, -122.6754837, 92.7605438, -215.4268494, 215.4148865
39: -145.5021362, 92.6687622, -145.4955750, 92.6656418, -238.1677704, 238.1643066
40: -115.8504333, 81.3783646, -115.8426208, 81.4020844, -197.2525177, 197.2209778
41: -89.2037048, 70.3752136, -89.2054977, 70.3726807, -159.5763855, 159.5807190
42: -64.9529724, 62.9233093, -64.9607239, 62.9162979, -127.8692703, 127.8840332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=367, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1617

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4452308, upper bound: 72.5213175
time: 144.43 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4452308, upper bound: 72.5213175
time: 150.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 297.47 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 297.47
Output dim: 2, lower bound: -72.4452308, upper bound: 72.4933623
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 297.47
Output dim: 2, lower bound: -72.4452308, upper bound: 72.4933614
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 297.47
Output dim: 2, lower bound: -72.4452308, upper bound: 72.5213175
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 297.47
Output dim: 2, lower bound: -72.4452308, upper bound: 72.5213175
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 297.47
Output dim: 2, lower bound: -72.4452308, upper bound: 72.4933623
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 297.47
Output dim: 2, lower bound: -72.4452308, upper bound: 72.4933623
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 297.47
Output dim: 2, lower bound: -72.4452308, upper bound: 72.5213175
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 297.47
Output dim: 2, lower bound: -72.4452308, upper bound: 72.5213175

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -120.0334854, 75.3554840, -120.1164093, 75.3937225, -195.4272156, 195.4718933
1: -68.6836853, 61.8864212, -68.7563171, 61.9161301, -130.5998230, 130.6427307
2: -59.2636414, 57.2788391, -59.3729477, 57.2987900, -116.5624313, 116.6517792
3: -68.2798615, 71.9837189, -68.4268188, 72.0143127, -140.2941742, 140.4105377
4: -74.7281113, 68.0438614, -74.8424835, 68.0677490, -142.7958679, 142.8863525
5: -66.8355255, 74.1893768, -66.9513397, 74.2170258, -141.0525513, 141.1407166
6: -85.0265656, 74.7209549, -85.0750885, 74.7516479, -159.7781982, 159.7960358
7: -82.9161377, 75.2329254, -83.0245209, 75.2466507, -158.1627808, 158.2574463
8: -81.1794281, 84.3065720, -81.2999496, 84.3402328, -165.5196533, 165.6065063
9: -71.7656555, 68.5940170, -71.8106766, 68.6440887, -140.4097443, 140.4046783
10: -105.0411835, 96.0034332, -105.0449295, 96.2729568, -201.3141327, 201.0483398
11: -98.4672089, 68.9559555, -98.5091095, 69.1537018, -167.6209106, 167.4650574
12: -94.4197617, 78.5241318, -94.4519043, 78.7222137, -173.1419678, 172.9760437
13: -101.0590515, 95.2201538, -101.2160797, 95.2777863, -196.3368225, 196.4362183
14: -152.6394653, 75.4411163, -152.7099915, 75.6291428, -228.2686157, 228.1511078
15: -84.7944641, 65.3088226, -84.8657074, 65.3559799, -150.1504364, 150.1745300
16: -108.3084030, 80.2271805, -108.3905716, 80.3370514, -188.6454468, 188.6177521
17: -159.7582703, 99.1039124, -159.7990723, 99.2897949, -259.0480652, 258.9029846
18: -92.2938614, 74.2181396, -92.3426056, 74.4010162, -166.6948853, 166.5607300
19: -72.1105042, 41.7919998, -72.1450500, 41.8671303, -113.9776306, 113.9370499
20: -67.9557114, 52.8161850, -67.9937286, 52.8706818, -120.8263855, 120.8099060
21: -92.0409393, 54.6903648, -92.0772400, 54.8029175, -146.8438568, 146.7676086
22: -100.7896957, 59.8301392, -100.8295135, 59.9247971, -160.7144928, 160.6596375
23: -72.1838608, 55.2396584, -72.2149734, 55.3424873, -127.5263519, 127.4546280
24: -90.7721481, 66.4444046, -90.8133545, 66.5491791, -157.3213196, 157.2577515
25: -78.5505219, 62.8575630, -78.5785446, 62.9492493, -141.4997711, 141.4360962
26: -106.9332581, 81.9194946, -106.9749451, 82.1307602, -189.0640259, 188.8944397
27: -92.6245575, 66.7597733, -92.6841888, 66.8126297, -159.4371948, 159.4439697
28: -71.6856613, 59.3341942, -71.7226181, 59.3450050, -131.0306549, 131.0568085
29: -107.5685730, 62.9950829, -107.6001892, 63.1047096, -170.6732788, 170.5952759
30: -89.7025604, 71.8981323, -89.7406616, 72.0140686, -161.7166290, 161.6387939
31: -91.9465485, 60.6013184, -92.0027618, 60.7214317, -152.6679840, 152.6040802
32: -90.2810822, 69.7710114, -90.3449402, 69.7969360, -160.0780029, 160.1159363
33: -119.6738281, 93.9560318, -119.7894821, 93.9855194, -213.6593475, 213.7454987
34: -99.7727432, 73.4817047, -99.8388367, 73.5306091, -173.3033447, 173.3205414
35: -103.7368317, 77.6046143, -103.8567352, 77.6313629, -181.3681793, 181.4613495
36: -100.4438858, 75.7132187, -100.5471191, 75.7193527, -176.1632385, 176.2603455
37: -141.4908447, 83.6882324, -141.5483856, 83.7509613, -225.2418060, 225.2366028
38: -121.9801331, 92.4755020, -122.0781860, 92.5280762, -214.5082092, 214.5536652
39: -144.8056488, 92.4365768, -144.9056091, 92.4597626, -237.2654114, 237.3421936
40: -115.3549194, 81.2444916, -115.4465179, 81.2640305, -196.6189270, 196.6909943
41: -88.8445129, 70.1014252, -88.9018555, 70.1404724, -158.9849854, 159.0032806
42: -64.6825562, 62.2945938, -64.7087173, 62.4025841, -127.0851440, 127.0033112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=791, inp2_unstable=791, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1617

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -72.4372059, upper bound: 72.4254466
time: 231.02 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4372059, upper bound: 72.4917453
time: 154.36 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -120.3383026, 75.4687958, -120.2386169, 75.4310913, -195.7693939, 195.7074127
1: -68.8852386, 61.9868088, -68.8483582, 61.9449883, -130.8302307, 130.8351746
2: -59.5298347, 57.4287529, -59.5051079, 57.3284378, -116.8582687, 116.9338531
3: -68.6215820, 72.2772217, -68.6021423, 72.0570984, -140.6786652, 140.8793640
4: -75.0419769, 68.1789398, -74.9934540, 68.1031036, -143.1450806, 143.1723938
5: -67.1113129, 74.4347839, -67.0874405, 74.2582016, -141.3694916, 141.5222168
6: -85.1979828, 74.8265457, -85.1377792, 74.7719421, -159.9699097, 159.9643250
7: -83.1603622, 75.3248291, -83.1349258, 75.2732925, -158.4336548, 158.4597473
8: -81.4695892, 84.4748917, -81.4445953, 84.3830109, -165.8525848, 165.9194794
9: -71.9694214, 68.7434769, -71.8953094, 68.6958923, -140.6653137, 140.6387634
10: -105.4367676, 96.5704498, -105.1201019, 96.5569916, -201.9937439, 201.6905518
11: -98.8811264, 69.5039139, -98.5665359, 69.4428864, -168.3240051, 168.0704498
12: -94.7481308, 79.0098495, -94.5011673, 78.9702225, -173.7183533, 173.5110168
13: -101.4408264, 95.6027145, -101.4024582, 95.3529053, -196.7937317, 197.0051575
14: -153.0905762, 75.9051208, -152.8018799, 75.8769379, -228.9675140, 228.7070007
15: -85.0366821, 65.4611206, -84.9577332, 65.4104919, -150.4471588, 150.4188385
16: -108.5893097, 80.5316696, -108.4923859, 80.4688263, -189.0581207, 189.0240479
17: -160.1739349, 99.7402191, -159.8675537, 99.6232681, -259.7971802, 259.6077881
18: -92.6891403, 74.6423187, -92.4083328, 74.6285248, -167.3176575, 167.0506592
19: -72.3851624, 42.0408821, -72.1959991, 41.9961929, -114.3813477, 114.2368622
20: -68.1687088, 53.0215759, -68.0433350, 52.9714890, -121.1401978, 121.0649109
21: -92.3714752, 55.0373459, -92.1305389, 54.9840584, -147.3555298, 147.1678772
22: -100.9462051, 60.1065941, -100.8732529, 60.0604286, -161.0066376, 160.9798279
23: -72.4242401, 55.5129547, -72.2596893, 55.4782867, -127.9025116, 127.7726440
24: -91.0671844, 66.6965179, -90.8677597, 66.6809235, -157.7481079, 157.5642700
25: -78.7519836, 63.1005363, -78.6258850, 63.0730247, -141.8250122, 141.7264252
26: -107.3144836, 82.4122314, -107.0336304, 82.3741150, -189.6885986, 189.4458466
27: -92.8342438, 66.9455643, -92.7548370, 66.9010773, -159.7353210, 159.7004089
28: -71.8364868, 59.4851418, -71.7686462, 59.4099426, -131.2464294, 131.2537842
29: -107.7315598, 63.3378677, -107.6424942, 63.2826080, -171.0141602, 170.9803619
30: -89.9534302, 72.2803345, -89.7925186, 72.2009277, -162.1543579, 162.0728455
31: -92.3473358, 60.9085617, -92.0760880, 60.8857956, -153.2331238, 152.9846344
32: -90.4682083, 69.9172440, -90.4185791, 69.8539276, -160.3221436, 160.3358154
33: -120.0350189, 94.2021027, -119.9623260, 94.0261002, -214.0610962, 214.1644287
34: -100.0339737, 73.6518555, -99.9639130, 73.5820312, -173.6159973, 173.6157684
35: -104.0613861, 77.8230133, -104.0169525, 77.6657944, -181.7271729, 181.8399658
36: -100.6958389, 75.8881378, -100.6718674, 75.7537994, -176.4496155, 176.5599976
37: -141.7235413, 83.8406525, -141.6299133, 83.8060608, -225.5296021, 225.4705658
38: -122.3081284, 92.6367340, -122.2271118, 92.5783844, -214.8864899, 214.8638458
39: -145.1438293, 92.5905304, -145.0542145, 92.4924698, -237.6362915, 237.6447296
40: -115.6329193, 81.3330078, -115.5630035, 81.2778168, -196.9107361, 196.8960114
41: -89.0371399, 70.2319031, -88.9846115, 70.1751099, -159.2122498, 159.2165222
42: -64.8191528, 62.5874519, -64.7569504, 62.5338287, -127.3529816, 127.3444061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=791, inp2_unstable=791, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1617

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -72.4372059, upper bound: 72.4260688
time: 129.57 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4372059, upper bound: 72.4917453
time: 109.88 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -120.1800842, 75.3897171, -120.4207687, 75.5224762, -195.7025604, 195.8104858
1: -68.7646027, 61.9073257, -68.9280243, 61.9859276, -130.7505341, 130.8353424
2: -59.3923759, 57.3071899, -59.6207466, 57.4474411, -116.8398132, 116.9279175
3: -68.4255524, 72.0181427, -68.7149506, 72.1983948, -140.6239319, 140.7330933
4: -74.8774948, 68.0789642, -75.1310272, 68.2462921, -143.1237793, 143.2099915
5: -66.9647064, 74.2233353, -67.2094269, 74.4224777, -141.3871765, 141.4327393
6: -85.0788803, 74.7836609, -85.2193298, 74.9139557, -159.9928284, 160.0029907
7: -82.9971313, 75.2610016, -83.2069016, 75.3315582, -158.3286896, 158.4678955
8: -81.3031540, 84.3411331, -81.5442581, 84.4666748, -165.7698059, 165.8853760
9: -71.9008789, 68.6396332, -72.0765076, 68.7935181, -140.6943970, 140.7161407
10: -105.1779251, 96.1265564, -105.3365402, 96.5657196, -201.7436066, 201.4631042
11: -98.5127029, 69.2432861, -98.8510284, 69.6810608, -168.1937561, 168.0943146
12: -94.4724731, 78.7426147, -94.7217560, 79.1421814, -173.6146545, 173.4643707
13: -101.1174622, 95.2929153, -101.3735504, 95.4863129, -196.6037598, 196.6664581
14: -152.7220001, 75.6464691, -153.0156250, 76.0082397, -228.7302399, 228.6620941
15: -84.8698196, 65.3470840, -85.0878906, 65.4637604, -150.3335876, 150.4349670
16: -108.3778076, 80.2863312, -108.5760117, 80.5231094, -188.9008942, 188.8623199
17: -159.8395386, 99.4902039, -160.2574463, 99.9908218, -259.8303223, 259.7476501
18: -92.3565140, 74.3563004, -92.5761719, 74.6795654, -167.0360565, 166.9324646
19: -72.1560059, 41.9483376, -72.3867569, 42.1559219, -114.3119278, 114.3350983
20: -67.9960632, 52.9447021, -68.1610947, 53.1110077, -121.1070709, 121.1057968
21: -92.0930328, 54.9054794, -92.3920975, 55.1938744, -147.2869110, 147.2975769
22: -100.8460236, 59.9890976, -101.0288086, 60.2313843, -161.0774078, 161.0179138
23: -72.2240295, 55.3611984, -72.4159088, 55.5697060, -127.7937317, 127.7771072
24: -90.8213806, 66.5510025, -90.9851227, 66.7553329, -157.5766907, 157.5361023
25: -78.5928040, 62.9701385, -78.7173920, 63.1623306, -141.7551270, 141.6875305
26: -106.9918137, 82.1207733, -107.2970276, 82.5107269, -189.5025330, 189.4178009
27: -92.6847534, 66.9128723, -92.9271393, 67.1032410, -159.7879944, 159.8399963
28: -71.7312775, 59.4838333, -71.9407883, 59.6235657, -131.3548431, 131.4246216
29: -107.6222610, 63.2337112, -107.8650818, 63.5383377, -171.1605988, 171.0987701
30: -89.7440948, 72.1125336, -89.9857635, 72.4209900, -162.1650848, 162.0982971
31: -92.0030746, 60.7383728, -92.2557831, 60.9759521, -152.9790344, 152.9941559
32: -90.3248291, 69.8732300, -90.4682617, 70.0161133, -160.3409271, 160.3414917
33: -119.8472443, 94.0007629, -120.1382599, 94.1527786, -214.0000153, 214.1390076
34: -99.9124832, 73.5149841, -100.1176147, 73.6472397, -173.5597229, 173.6325989
35: -103.8412247, 77.6291580, -104.0857697, 77.7228470, -181.5640717, 181.7149353
36: -100.5129089, 75.7814026, -100.7141724, 75.8781128, -176.3910217, 176.4955750
37: -141.5676270, 83.7400589, -141.7485504, 83.8866272, -225.4542542, 225.4886169
38: -122.1217957, 92.5134354, -122.3956451, 92.6567230, -214.7785187, 214.9090881
39: -144.9526062, 92.4704742, -145.2013855, 92.5985260, -237.5511169, 237.6718445
40: -115.4343796, 81.2629089, -115.6262054, 81.3606491, -196.7950287, 196.8891144
41: -88.9075089, 70.1414642, -89.0546112, 70.2667542, -159.1742554, 159.1960754
42: -64.7382050, 62.4118118, -64.8648071, 62.6425705, -127.3807526, 127.2766190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=791, inp2_unstable=792, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1617

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4372285, upper bound: 72.4535424
time: 142.96 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4372285, upper bound: 72.5196364
time: 224.02 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -120.4849243, 75.5036087, -120.5439682, 75.5595016, -196.0444031, 196.0475616
1: -68.9662933, 62.0078163, -69.0208282, 62.0151062, -130.9813995, 131.0286407
2: -59.6587181, 57.4568939, -59.7533340, 57.4771996, -117.1358948, 117.2102203
3: -68.7674713, 72.3113937, -68.8907013, 72.2414551, -141.0089111, 141.2020874
4: -75.1915054, 68.2137604, -75.2825089, 68.2817535, -143.4732666, 143.4962769
5: -67.2405777, 74.4686508, -67.3458328, 74.4637451, -141.7042999, 141.8144836
6: -85.2502899, 74.8894958, -85.2833405, 74.9347229, -160.1850128, 160.1728363
7: -83.2415543, 75.3527222, -83.3188629, 75.3583374, -158.5998840, 158.6715698
8: -81.5935211, 84.5093079, -81.6893768, 84.5096283, -166.1031494, 166.1986694
9: -72.1051483, 68.7890167, -72.1619797, 68.8454208, -140.9505615, 140.9509888
10: -105.5737610, 96.6937180, -105.4138565, 96.8507080, -202.4244537, 202.1075745
11: -98.9262924, 69.7913055, -98.9088287, 69.9709320, -168.8972168, 168.7001343
12: -94.8005981, 79.2285004, -94.7710114, 79.3908157, -174.1914062, 173.9995117
13: -101.4997559, 95.6743774, -101.5609589, 95.5625229, -197.0622406, 197.2353363
14: -153.1728821, 76.1107635, -153.1076965, 76.2567215, -229.4295959, 229.2184601
15: -85.1129150, 65.4992065, -85.1812210, 65.5195465, -150.6324615, 150.6804199
16: -108.6583786, 80.5916138, -108.6795044, 80.6552963, -189.3136749, 189.2711182
17: -160.2550354, 100.1268768, -160.3261108, 100.3254929, -260.5805359, 260.4530029
18: -92.7505722, 74.7816391, -92.6421585, 74.9086151, -167.6591644, 167.4237976
19: -72.4305267, 42.1972961, -72.4382095, 42.2853622, -114.7158813, 114.6355057
20: -68.2088470, 53.1502609, -68.2109222, 53.2122383, -121.4210587, 121.3611679
21: -92.4233856, 55.2525749, -92.4458771, 55.3754196, -147.7987976, 147.6984558
22: -101.0025635, 60.2658691, -101.0721741, 60.3693161, -161.3718719, 161.3380432
23: -72.4642029, 55.6345787, -72.4607849, 55.7058334, -128.1700439, 128.0953674
24: -91.1158066, 66.8034363, -91.0396957, 66.8875656, -158.0033722, 157.8431396
25: -78.7941208, 63.2133331, -78.7651138, 63.2867317, -142.0808563, 141.9784393
26: -107.3726883, 82.6137466, -107.3556976, 82.7550659, -190.1277466, 189.9694519
27: -92.8941040, 67.0993347, -92.9971695, 67.1924286, -160.0865021, 160.0964966
28: -71.8822098, 59.6350517, -71.9867249, 59.6889877, -131.5711975, 131.6217804
29: -107.7850800, 63.5767136, -107.9074097, 63.7176132, -171.5026855, 171.4841156
30: -89.9945679, 72.4950867, -90.0379028, 72.6086121, -162.6031799, 162.5329895
31: -92.4028931, 61.0458374, -92.3291245, 61.1409492, -153.5438385, 153.3749695
32: -90.5123520, 70.0194626, -90.5424652, 70.0736389, -160.5859833, 160.5619202
33: -120.2086029, 94.2467499, -120.3117371, 94.1934204, -214.4020233, 214.5584717
34: -100.1738510, 73.6847992, -100.2430878, 73.6989899, -173.8728333, 173.9278870
35: -104.1660309, 77.8470840, -104.2471161, 77.7580414, -181.9240723, 182.0941772
36: -100.7650452, 75.9563065, -100.8394775, 75.9141235, -176.6791534, 176.7957764
37: -141.8003998, 83.8926086, -141.8309784, 83.9423218, -225.7427063, 225.7235870
38: -122.4500809, 92.6745758, -122.5451965, 92.7071838, -215.1572571, 215.2197723
39: -145.2911987, 92.6243134, -145.3510437, 92.6316452, -237.9228516, 237.9753571
40: -115.7125778, 81.3514938, -115.7441101, 81.3746109, -197.0871735, 197.0956116
41: -89.1003113, 70.2721329, -89.1384277, 70.3014984, -159.4017944, 159.4105530
42: -64.8747635, 62.7050667, -64.9133911, 62.7745018, -127.6492615, 127.6184540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=791, inp2_unstable=792, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1617

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4372285, upper bound: 72.4540651
time: 150.52 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4372285, upper bound: 72.4540651
time: 126.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 279.42 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 279.42
Output dim: 2, lower bound: -72.4372059, upper bound: 72.4254466
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 279.42
Output dim: 2, lower bound: -72.4372059, upper bound: 72.4917453
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 279.42
Output dim: 2, lower bound: -72.4372059, upper bound: 72.4260688
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 279.42
Output dim: 2, lower bound: -72.4372059, upper bound: 72.4917453
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 279.42
Output dim: 2, lower bound: -72.4372285, upper bound: 72.4535424
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 279.42
Output dim: 2, lower bound: -72.4372285, upper bound: 72.5196364
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 279.42
Output dim: 2, lower bound: -72.4372285, upper bound: 72.4540651
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 279.42
Output dim: 2, lower bound: -72.4372285, upper bound: 72.4540651
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 279.42
Output dim: 2, lower bound: -72.4452308, upper bound: 72.4933623
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 279.42
Output dim: 2, lower bound: -72.4452308, upper bound: 72.4933623
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 279.42
Output dim: 2, lower bound: -72.4452308, upper bound: 72.5213175
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 279.42
Output dim: 2, lower bound: -72.4452308, upper bound: 72.5213175

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 213.73 + 3642.33 = 3856.06 seconds
