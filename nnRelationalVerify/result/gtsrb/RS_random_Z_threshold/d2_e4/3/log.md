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
execution time: IAR + RelationalAnalysis = 2.79 + 206.82 = 209.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -72.5246480, upper bound: 72.5246489

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1675

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 728

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4876574, upper bound: 72.5240941
time: 131.09 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.5240933, upper bound: 72.4876583
time: 165.22 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 296.32 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 296.32
Output dim: 2, lower bound: -72.4876574, upper bound: 72.5240941
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 296.32
Output dim: 2, lower bound: -72.5240933, upper bound: 72.4876583

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -120.6897812, 75.5772247, -120.6897812, 75.5772247, -196.2669983, 196.2669983
1: -69.1185150, 62.0534592, -69.1185150, 62.0534592, -131.1719666, 131.1719666
2: -59.9247818, 57.4265594, -59.9247818, 57.4265594, -117.3513336, 117.3513336
3: -69.0837555, 72.1842041, -69.0837555, 72.1842041, -141.2679443, 141.2679443
4: -75.4698944, 68.2243805, -75.4698944, 68.2243805, -143.6942749, 143.6942749
5: -67.5072861, 74.3775482, -67.5072861, 74.3775482, -141.8848267, 141.8848267
6: -85.3311462, 75.1246033, -85.3311462, 75.1246033, -160.4557495, 160.4557495
7: -83.4509277, 75.4055939, -83.4509277, 75.4055939, -158.8565216, 158.8565216
8: -81.8659363, 84.5110092, -81.8659363, 84.5110092, -166.3769531, 166.3769531
9: -72.3079300, 68.8767319, -72.3079300, 68.8767319, -141.1846619, 141.1846466
10: -105.4851379, 97.0667877, -105.4851379, 97.0667877, -202.5519257, 202.5519257
11: -98.7560196, 70.3052673, -98.7560196, 70.3052673, -169.0612793, 169.0612793
12: -94.6711731, 79.6737671, -94.6711731, 79.6737671, -174.3449402, 174.3449402
13: -101.8092804, 95.6050262, -101.8092804, 95.6050262, -197.4143066, 197.4143066
14: -153.0820007, 76.5118256, -153.0820007, 76.5118256, -229.5938110, 229.5938263
15: -85.4207764, 65.5752563, -85.4207764, 65.5752563, -150.9960327, 150.9960327
16: -108.7743988, 80.8979950, -108.7743988, 80.8979950, -189.6723633, 189.6723785
17: -160.1158142, 100.7368164, -160.1158142, 100.7368164, -260.8526306, 260.8526306
18: -92.6273956, 75.1741791, -92.6273956, 75.1741791, -167.8015594, 167.8015747
19: -72.3558044, 42.4454155, -72.3558044, 42.4454155, -114.8012085, 114.8012238
20: -68.1926117, 53.3375359, -68.1926117, 53.3375359, -121.5301361, 121.5301361
21: -92.3123550, 55.6030388, -92.3123550, 55.6030388, -147.9153748, 147.9153748
22: -101.0745773, 60.5765076, -101.0745773, 60.5765076, -161.6510925, 161.6510925
23: -72.3953247, 55.8623962, -72.3953247, 55.8623962, -128.2577209, 128.2577209
24: -91.0453033, 67.0268860, -91.0453033, 67.0268860, -158.0721741, 158.0721741
25: -78.7736359, 63.4368629, -78.7736359, 63.4368629, -142.2104950, 142.2104950
26: -107.2412415, 83.0555573, -107.2412415, 83.0555573, -190.2967987, 190.2967987
27: -92.9772720, 67.3610229, -92.9772720, 67.3610229, -160.3382874, 160.3382874
28: -71.9242935, 59.8130646, -71.9242935, 59.8130646, -131.7373657, 131.7373505
29: -107.8225708, 63.9952202, -107.8225708, 63.9952202, -171.8177948, 171.8177948
30: -89.9596481, 72.8284531, -89.9596481, 72.8284531, -162.7881012, 162.7881012
31: -92.2802734, 61.3140030, -92.2802734, 61.3140030, -153.5942688, 153.5942688
32: -90.6332397, 70.1865616, -90.6332397, 70.1865616, -160.8197937, 160.8197937
33: -120.4985809, 94.1850891, -120.4985809, 94.1850891, -214.6836548, 214.6836700
34: -100.3700409, 73.7348175, -100.3700409, 73.7348175, -174.1048584, 174.1048584
35: -104.3925476, 77.7684860, -104.3925476, 77.7684860, -182.1610413, 182.1610260
36: -100.9291229, 75.9416580, -100.9291229, 75.9416580, -176.8707886, 176.8707886
37: -141.9010010, 84.0858002, -141.9010010, 84.0858002, -225.9868011, 225.9868011
38: -122.6707611, 92.7453461, -122.6707611, 92.7453461, -215.4160919, 215.4161072
39: -145.5377960, 92.6060257, -145.5377960, 92.6060257, -238.1438141, 238.1438141
40: -115.8655548, 81.4251404, -115.8655548, 81.4251404, -197.2906952, 197.2906952
41: -89.2050629, 70.4504623, -89.2050629, 70.4504623, -159.6555176, 159.6555176
42: -64.9341888, 62.9433632, -64.9341888, 62.9433632, -127.8775482, 127.8775482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 668

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1616

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4866935, upper bound: 72.5240870
time: 108.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4876574, upper bound: 72.5231889
time: 127.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -120.6897812, 75.5772247, -120.6897812, 75.5772247, -196.2669983, 196.2669983
1: -69.1185150, 62.0534592, -69.1185150, 62.0534592, -131.1719666, 131.1719666
2: -59.9247818, 57.4265594, -59.9247818, 57.4265594, -117.3513336, 117.3513336
3: -69.0837555, 72.1842041, -69.0837555, 72.1842041, -141.2679443, 141.2679443
4: -75.4698944, 68.2243805, -75.4698944, 68.2243805, -143.6942749, 143.6942749
5: -67.5072861, 74.3775482, -67.5072861, 74.3775482, -141.8848267, 141.8848267
6: -85.3311462, 75.1246033, -85.3311462, 75.1246033, -160.4557495, 160.4557495
7: -83.4509277, 75.4055939, -83.4509277, 75.4055939, -158.8565216, 158.8565216
8: -81.8659363, 84.5110092, -81.8659363, 84.5110092, -166.3769531, 166.3769531
9: -72.3079300, 68.8767319, -72.3079300, 68.8767319, -141.1846619, 141.1846466
10: -105.4851379, 97.0667877, -105.4851379, 97.0667877, -202.5519257, 202.5519257
11: -98.7560196, 70.3052673, -98.7560196, 70.3052673, -169.0612793, 169.0612793
12: -94.6711731, 79.6737671, -94.6711731, 79.6737671, -174.3449402, 174.3449402
13: -101.8092804, 95.6050262, -101.8092804, 95.6050262, -197.4143066, 197.4143066
14: -153.0820007, 76.5118256, -153.0820007, 76.5118256, -229.5938110, 229.5938263
15: -85.4207764, 65.5752563, -85.4207764, 65.5752563, -150.9960327, 150.9960327
16: -108.7743988, 80.8979950, -108.7743988, 80.8979950, -189.6723633, 189.6723785
17: -160.1158142, 100.7368164, -160.1158142, 100.7368164, -260.8526306, 260.8526306
18: -92.6273956, 75.1741791, -92.6273956, 75.1741791, -167.8015594, 167.8015747
19: -72.3558044, 42.4454155, -72.3558044, 42.4454155, -114.8012085, 114.8012238
20: -68.1926117, 53.3375359, -68.1926117, 53.3375359, -121.5301361, 121.5301361
21: -92.3123550, 55.6030388, -92.3123550, 55.6030388, -147.9153748, 147.9153748
22: -101.0745773, 60.5765076, -101.0745773, 60.5765076, -161.6510925, 161.6510925
23: -72.3953247, 55.8623962, -72.3953247, 55.8623962, -128.2577209, 128.2577209
24: -91.0453033, 67.0268860, -91.0453033, 67.0268860, -158.0721741, 158.0721741
25: -78.7736359, 63.4368629, -78.7736359, 63.4368629, -142.2104950, 142.2104950
26: -107.2412415, 83.0555573, -107.2412415, 83.0555573, -190.2967987, 190.2967987
27: -92.9772720, 67.3610229, -92.9772720, 67.3610229, -160.3382874, 160.3382874
28: -71.9242935, 59.8130646, -71.9242935, 59.8130646, -131.7373657, 131.7373505
29: -107.8225708, 63.9952202, -107.8225708, 63.9952202, -171.8177948, 171.8177948
30: -89.9596481, 72.8284531, -89.9596481, 72.8284531, -162.7881012, 162.7881012
31: -92.2802734, 61.3140030, -92.2802734, 61.3140030, -153.5942688, 153.5942688
32: -90.6332397, 70.1865616, -90.6332397, 70.1865616, -160.8197937, 160.8197937
33: -120.4985809, 94.1850891, -120.4985809, 94.1850891, -214.6836548, 214.6836700
34: -100.3700409, 73.7348175, -100.3700409, 73.7348175, -174.1048584, 174.1048584
35: -104.3925476, 77.7684860, -104.3925476, 77.7684860, -182.1610413, 182.1610260
36: -100.9291229, 75.9416580, -100.9291229, 75.9416580, -176.8707886, 176.8707886
37: -141.9010010, 84.0858002, -141.9010010, 84.0858002, -225.9868011, 225.9868011
38: -122.6707611, 92.7453461, -122.6707611, 92.7453461, -215.4160919, 215.4161072
39: -145.5377960, 92.6060257, -145.5377960, 92.6060257, -238.1438141, 238.1438141
40: -115.8655548, 81.4251404, -115.8655548, 81.4251404, -197.2906952, 197.2906952
41: -89.2050629, 70.4504623, -89.2050629, 70.4504623, -159.6555176, 159.6555176
42: -64.9341888, 62.9433632, -64.9341888, 62.9433632, -127.8775482, 127.8775482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1684

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 710

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.5096460, upper bound: 72.4870501
time: 294.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.5234068, upper bound: 72.4731692
time: 149.15 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 446.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 446.26
Output dim: 2, lower bound: -72.4866935, upper bound: 72.5240870
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 446.26
Output dim: 2, lower bound: -72.4876574, upper bound: 72.5231889
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 446.26
Output dim: 2, lower bound: -72.5096460, upper bound: 72.4870501
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 446.26
Output dim: 2, lower bound: -72.5234068, upper bound: 72.4731692

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -120.6897812, 75.5772247, -120.6897812, 75.5772247, -196.2669983, 196.2669983
1: -69.1185150, 62.0534592, -69.1185150, 62.0534592, -131.1719666, 131.1719666
2: -59.9247818, 57.4265594, -59.9247818, 57.4265594, -117.3513336, 117.3513336
3: -69.0837555, 72.1842041, -69.0837555, 72.1842041, -141.2679443, 141.2679443
4: -75.4698944, 68.2243805, -75.4698944, 68.2243805, -143.6942749, 143.6942749
5: -67.5072861, 74.3775482, -67.5072861, 74.3775482, -141.8848267, 141.8848267
6: -85.3311462, 75.1246033, -85.3311462, 75.1246033, -160.4557495, 160.4557495
7: -83.4509277, 75.4055939, -83.4509277, 75.4055939, -158.8565216, 158.8565216
8: -81.8659363, 84.5110092, -81.8659363, 84.5110092, -166.3769531, 166.3769531
9: -72.3079300, 68.8767319, -72.3079300, 68.8767319, -141.1846619, 141.1846466
10: -105.4851379, 97.0667877, -105.4851379, 97.0667877, -202.5519257, 202.5519257
11: -98.7560196, 70.3052673, -98.7560196, 70.3052673, -169.0612793, 169.0612793
12: -94.6711731, 79.6737671, -94.6711731, 79.6737671, -174.3449402, 174.3449402
13: -101.8092804, 95.6050262, -101.8092804, 95.6050262, -197.4143066, 197.4143066
14: -153.0820007, 76.5118256, -153.0820007, 76.5118256, -229.5938110, 229.5938263
15: -85.4207764, 65.5752563, -85.4207764, 65.5752563, -150.9960327, 150.9960327
16: -108.7743988, 80.8979950, -108.7743988, 80.8979950, -189.6723633, 189.6723785
17: -160.1158142, 100.7368164, -160.1158142, 100.7368164, -260.8526306, 260.8526306
18: -92.6273956, 75.1741791, -92.6273956, 75.1741791, -167.8015594, 167.8015747
19: -72.3558044, 42.4454155, -72.3558044, 42.4454155, -114.8012085, 114.8012238
20: -68.1926117, 53.3375359, -68.1926117, 53.3375359, -121.5301361, 121.5301361
21: -92.3123550, 55.6030388, -92.3123550, 55.6030388, -147.9153748, 147.9153748
22: -101.0745773, 60.5765076, -101.0745773, 60.5765076, -161.6510925, 161.6510925
23: -72.3953247, 55.8623962, -72.3953247, 55.8623962, -128.2577209, 128.2577209
24: -91.0453033, 67.0268860, -91.0453033, 67.0268860, -158.0721741, 158.0721741
25: -78.7736359, 63.4368629, -78.7736359, 63.4368629, -142.2104950, 142.2104950
26: -107.2412415, 83.0555573, -107.2412415, 83.0555573, -190.2967987, 190.2967987
27: -92.9772720, 67.3610229, -92.9772720, 67.3610229, -160.3382874, 160.3382874
28: -71.9242935, 59.8130646, -71.9242935, 59.8130646, -131.7373657, 131.7373505
29: -107.8225708, 63.9952202, -107.8225708, 63.9952202, -171.8177948, 171.8177948
30: -89.9596481, 72.8284531, -89.9596481, 72.8284531, -162.7881012, 162.7881012
31: -92.2802734, 61.3140030, -92.2802734, 61.3140030, -153.5942688, 153.5942688
32: -90.6332397, 70.1865616, -90.6332397, 70.1865616, -160.8197937, 160.8197937
33: -120.4985809, 94.1850891, -120.4985809, 94.1850891, -214.6836548, 214.6836700
34: -100.3700409, 73.7348175, -100.3700409, 73.7348175, -174.1048584, 174.1048584
35: -104.3925476, 77.7684860, -104.3925476, 77.7684860, -182.1610413, 182.1610260
36: -100.9291229, 75.9416580, -100.9291229, 75.9416580, -176.8707886, 176.8707886
37: -141.9010010, 84.0858002, -141.9010010, 84.0858002, -225.9868011, 225.9868011
38: -122.6707611, 92.7453461, -122.6707611, 92.7453461, -215.4160919, 215.4161072
39: -145.5377960, 92.6060257, -145.5377960, 92.6060257, -238.1438141, 238.1438141
40: -115.8655548, 81.4251404, -115.8655548, 81.4251404, -197.2906952, 197.2906952
41: -89.2050629, 70.4504623, -89.2050629, 70.4504623, -159.6555176, 159.6555176
42: -64.9341888, 62.9433632, -64.9341888, 62.9433632, -127.8775482, 127.8775482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 721

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4863406, upper bound: 72.5067884
time: 158.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4693531, upper bound: 72.5237364
time: 126.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -120.6897812, 75.5772247, -120.6897812, 75.5772247, -196.2669983, 196.2669983
1: -69.1185150, 62.0534592, -69.1185150, 62.0534592, -131.1719666, 131.1719666
2: -59.9247818, 57.4265594, -59.9247818, 57.4265594, -117.3513336, 117.3513336
3: -69.0837555, 72.1842041, -69.0837555, 72.1842041, -141.2679443, 141.2679443
4: -75.4698944, 68.2243805, -75.4698944, 68.2243805, -143.6942749, 143.6942749
5: -67.5072861, 74.3775482, -67.5072861, 74.3775482, -141.8848267, 141.8848267
6: -85.3311462, 75.1246033, -85.3311462, 75.1246033, -160.4557495, 160.4557495
7: -83.4509277, 75.4055939, -83.4509277, 75.4055939, -158.8565216, 158.8565216
8: -81.8659363, 84.5110092, -81.8659363, 84.5110092, -166.3769531, 166.3769531
9: -72.3079300, 68.8767319, -72.3079300, 68.8767319, -141.1846619, 141.1846466
10: -105.4851379, 97.0667877, -105.4851379, 97.0667877, -202.5519257, 202.5519257
11: -98.7560196, 70.3052673, -98.7560196, 70.3052673, -169.0612793, 169.0612793
12: -94.6711731, 79.6737671, -94.6711731, 79.6737671, -174.3449402, 174.3449402
13: -101.8092804, 95.6050262, -101.8092804, 95.6050262, -197.4143066, 197.4143066
14: -153.0820007, 76.5118256, -153.0820007, 76.5118256, -229.5938110, 229.5938263
15: -85.4207764, 65.5752563, -85.4207764, 65.5752563, -150.9960327, 150.9960327
16: -108.7743988, 80.8979950, -108.7743988, 80.8979950, -189.6723633, 189.6723785
17: -160.1158142, 100.7368164, -160.1158142, 100.7368164, -260.8526306, 260.8526306
18: -92.6273956, 75.1741791, -92.6273956, 75.1741791, -167.8015594, 167.8015747
19: -72.3558044, 42.4454155, -72.3558044, 42.4454155, -114.8012085, 114.8012238
20: -68.1926117, 53.3375359, -68.1926117, 53.3375359, -121.5301361, 121.5301361
21: -92.3123550, 55.6030388, -92.3123550, 55.6030388, -147.9153748, 147.9153748
22: -101.0745773, 60.5765076, -101.0745773, 60.5765076, -161.6510925, 161.6510925
23: -72.3953247, 55.8623962, -72.3953247, 55.8623962, -128.2577209, 128.2577209
24: -91.0453033, 67.0268860, -91.0453033, 67.0268860, -158.0721741, 158.0721741
25: -78.7736359, 63.4368629, -78.7736359, 63.4368629, -142.2104950, 142.2104950
26: -107.2412415, 83.0555573, -107.2412415, 83.0555573, -190.2967987, 190.2967987
27: -92.9772720, 67.3610229, -92.9772720, 67.3610229, -160.3382874, 160.3382874
28: -71.9242935, 59.8130646, -71.9242935, 59.8130646, -131.7373657, 131.7373505
29: -107.8225708, 63.9952202, -107.8225708, 63.9952202, -171.8177948, 171.8177948
30: -89.9596481, 72.8284531, -89.9596481, 72.8284531, -162.7881012, 162.7881012
31: -92.2802734, 61.3140030, -92.2802734, 61.3140030, -153.5942688, 153.5942688
32: -90.6332397, 70.1865616, -90.6332397, 70.1865616, -160.8197937, 160.8197937
33: -120.4985809, 94.1850891, -120.4985809, 94.1850891, -214.6836548, 214.6836700
34: -100.3700409, 73.7348175, -100.3700409, 73.7348175, -174.1048584, 174.1048584
35: -104.3925476, 77.7684860, -104.3925476, 77.7684860, -182.1610413, 182.1610260
36: -100.9291229, 75.9416580, -100.9291229, 75.9416580, -176.8707886, 176.8707886
37: -141.9010010, 84.0858002, -141.9010010, 84.0858002, -225.9868011, 225.9868011
38: -122.6707611, 92.7453461, -122.6707611, 92.7453461, -215.4160919, 215.4161072
39: -145.5377960, 92.6060257, -145.5377960, 92.6060257, -238.1438141, 238.1438141
40: -115.8655548, 81.4251404, -115.8655548, 81.4251404, -197.2906952, 197.2906952
41: -89.2050629, 70.4504623, -89.2050629, 70.4504623, -159.6555176, 159.6555176
42: -64.9341888, 62.9433632, -64.9341888, 62.9433632, -127.8775482, 127.8775482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 619

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 721

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4873051, upper bound: 72.5058428
time: 444.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4703435, upper bound: 72.5228371
time: 166.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -120.6897812, 75.5772247, -120.6897812, 75.5772247, -196.2669983, 196.2669983
1: -69.1185150, 62.0534592, -69.1185150, 62.0534592, -131.1719666, 131.1719666
2: -59.9247818, 57.4265594, -59.9247818, 57.4265594, -117.3513336, 117.3513336
3: -69.0837555, 72.1842041, -69.0837555, 72.1842041, -141.2679443, 141.2679443
4: -75.4698944, 68.2243805, -75.4698944, 68.2243805, -143.6942749, 143.6942749
5: -67.5072861, 74.3775482, -67.5072861, 74.3775482, -141.8848267, 141.8848267
6: -85.3311462, 75.1246033, -85.3311462, 75.1246033, -160.4557495, 160.4557495
7: -83.4509277, 75.4055939, -83.4509277, 75.4055939, -158.8565216, 158.8565216
8: -81.8659363, 84.5110092, -81.8659363, 84.5110092, -166.3769531, 166.3769531
9: -72.3079300, 68.8767319, -72.3079300, 68.8767319, -141.1846619, 141.1846466
10: -105.4851379, 97.0667877, -105.4851379, 97.0667877, -202.5519257, 202.5519257
11: -98.7560196, 70.3052673, -98.7560196, 70.3052673, -169.0612793, 169.0612793
12: -94.6711731, 79.6737671, -94.6711731, 79.6737671, -174.3449402, 174.3449402
13: -101.8092804, 95.6050262, -101.8092804, 95.6050262, -197.4143066, 197.4143066
14: -153.0820007, 76.5118256, -153.0820007, 76.5118256, -229.5938110, 229.5938263
15: -85.4207764, 65.5752563, -85.4207764, 65.5752563, -150.9960327, 150.9960327
16: -108.7743988, 80.8979950, -108.7743988, 80.8979950, -189.6723633, 189.6723785
17: -160.1158142, 100.7368164, -160.1158142, 100.7368164, -260.8526306, 260.8526306
18: -92.6273956, 75.1741791, -92.6273956, 75.1741791, -167.8015594, 167.8015747
19: -72.3558044, 42.4454155, -72.3558044, 42.4454155, -114.8012085, 114.8012238
20: -68.1926117, 53.3375359, -68.1926117, 53.3375359, -121.5301361, 121.5301361
21: -92.3123550, 55.6030388, -92.3123550, 55.6030388, -147.9153748, 147.9153748
22: -101.0745773, 60.5765076, -101.0745773, 60.5765076, -161.6510925, 161.6510925
23: -72.3953247, 55.8623962, -72.3953247, 55.8623962, -128.2577209, 128.2577209
24: -91.0453033, 67.0268860, -91.0453033, 67.0268860, -158.0721741, 158.0721741
25: -78.7736359, 63.4368629, -78.7736359, 63.4368629, -142.2104950, 142.2104950
26: -107.2412415, 83.0555573, -107.2412415, 83.0555573, -190.2967987, 190.2967987
27: -92.9772720, 67.3610229, -92.9772720, 67.3610229, -160.3382874, 160.3382874
28: -71.9242935, 59.8130646, -71.9242935, 59.8130646, -131.7373657, 131.7373505
29: -107.8225708, 63.9952202, -107.8225708, 63.9952202, -171.8177948, 171.8177948
30: -89.9596481, 72.8284531, -89.9596481, 72.8284531, -162.7881012, 162.7881012
31: -92.2802734, 61.3140030, -92.2802734, 61.3140030, -153.5942688, 153.5942688
32: -90.6332397, 70.1865616, -90.6332397, 70.1865616, -160.8197937, 160.8197937
33: -120.4985809, 94.1850891, -120.4985809, 94.1850891, -214.6836548, 214.6836700
34: -100.3700409, 73.7348175, -100.3700409, 73.7348175, -174.1048584, 174.1048584
35: -104.3925476, 77.7684860, -104.3925476, 77.7684860, -182.1610413, 182.1610260
36: -100.9291229, 75.9416580, -100.9291229, 75.9416580, -176.8707886, 176.8707886
37: -141.9010010, 84.0858002, -141.9010010, 84.0858002, -225.9868011, 225.9868011
38: -122.6707611, 92.7453461, -122.6707611, 92.7453461, -215.4160919, 215.4161072
39: -145.5377960, 92.6060257, -145.5377960, 92.6060257, -238.1438141, 238.1438141
40: -115.8655548, 81.4251404, -115.8655548, 81.4251404, -197.2906952, 197.2906952
41: -89.2050629, 70.4504623, -89.2050629, 70.4504623, -159.6555176, 159.6555176
42: -64.9341888, 62.9433632, -64.9341888, 62.9433632, -127.8775482, 127.8775482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1603

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 766

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4924838, upper bound: 72.4698653
time: 133.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4924838, upper bound: 72.4867981
time: 195.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -120.6897812, 75.5772247, -120.6897812, 75.5772247, -196.2669983, 196.2669983
1: -69.1185150, 62.0534592, -69.1185150, 62.0534592, -131.1719666, 131.1719666
2: -59.9247818, 57.4265594, -59.9247818, 57.4265594, -117.3513336, 117.3513336
3: -69.0837555, 72.1842041, -69.0837555, 72.1842041, -141.2679443, 141.2679443
4: -75.4698944, 68.2243805, -75.4698944, 68.2243805, -143.6942749, 143.6942749
5: -67.5072861, 74.3775482, -67.5072861, 74.3775482, -141.8848267, 141.8848267
6: -85.3311462, 75.1246033, -85.3311462, 75.1246033, -160.4557495, 160.4557495
7: -83.4509277, 75.4055939, -83.4509277, 75.4055939, -158.8565216, 158.8565216
8: -81.8659363, 84.5110092, -81.8659363, 84.5110092, -166.3769531, 166.3769531
9: -72.3079300, 68.8767319, -72.3079300, 68.8767319, -141.1846619, 141.1846466
10: -105.4851379, 97.0667877, -105.4851379, 97.0667877, -202.5519257, 202.5519257
11: -98.7560196, 70.3052673, -98.7560196, 70.3052673, -169.0612793, 169.0612793
12: -94.6711731, 79.6737671, -94.6711731, 79.6737671, -174.3449402, 174.3449402
13: -101.8092804, 95.6050262, -101.8092804, 95.6050262, -197.4143066, 197.4143066
14: -153.0820007, 76.5118256, -153.0820007, 76.5118256, -229.5938110, 229.5938263
15: -85.4207764, 65.5752563, -85.4207764, 65.5752563, -150.9960327, 150.9960327
16: -108.7743988, 80.8979950, -108.7743988, 80.8979950, -189.6723633, 189.6723785
17: -160.1158142, 100.7368164, -160.1158142, 100.7368164, -260.8526306, 260.8526306
18: -92.6273956, 75.1741791, -92.6273956, 75.1741791, -167.8015594, 167.8015747
19: -72.3558044, 42.4454155, -72.3558044, 42.4454155, -114.8012085, 114.8012238
20: -68.1926117, 53.3375359, -68.1926117, 53.3375359, -121.5301361, 121.5301361
21: -92.3123550, 55.6030388, -92.3123550, 55.6030388, -147.9153748, 147.9153748
22: -101.0745773, 60.5765076, -101.0745773, 60.5765076, -161.6510925, 161.6510925
23: -72.3953247, 55.8623962, -72.3953247, 55.8623962, -128.2577209, 128.2577209
24: -91.0453033, 67.0268860, -91.0453033, 67.0268860, -158.0721741, 158.0721741
25: -78.7736359, 63.4368629, -78.7736359, 63.4368629, -142.2104950, 142.2104950
26: -107.2412415, 83.0555573, -107.2412415, 83.0555573, -190.2967987, 190.2967987
27: -92.9772720, 67.3610229, -92.9772720, 67.3610229, -160.3382874, 160.3382874
28: -71.9242935, 59.8130646, -71.9242935, 59.8130646, -131.7373657, 131.7373505
29: -107.8225708, 63.9952202, -107.8225708, 63.9952202, -171.8177948, 171.8177948
30: -89.9596481, 72.8284531, -89.9596481, 72.8284531, -162.7881012, 162.7881012
31: -92.2802734, 61.3140030, -92.2802734, 61.3140030, -153.5942688, 153.5942688
32: -90.6332397, 70.1865616, -90.6332397, 70.1865616, -160.8197937, 160.8197937
33: -120.4985809, 94.1850891, -120.4985809, 94.1850891, -214.6836548, 214.6836700
34: -100.3700409, 73.7348175, -100.3700409, 73.7348175, -174.1048584, 174.1048584
35: -104.3925476, 77.7684860, -104.3925476, 77.7684860, -182.1610413, 182.1610260
36: -100.9291229, 75.9416580, -100.9291229, 75.9416580, -176.8707886, 176.8707886
37: -141.9010010, 84.0858002, -141.9010010, 84.0858002, -225.9868011, 225.9868011
38: -122.6707611, 92.7453461, -122.6707611, 92.7453461, -215.4160919, 215.4161072
39: -145.5377960, 92.6060257, -145.5377960, 92.6060257, -238.1438141, 238.1438141
40: -115.8655548, 81.4251404, -115.8655548, 81.4251404, -197.2906952, 197.2906952
41: -89.2050629, 70.4504623, -89.2050629, 70.4504623, -159.6555176, 159.6555176
42: -64.9341888, 62.9433632, -64.9341888, 62.9433632, -127.8775482, 127.8775482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1671

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 591

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.5233013, upper bound: 72.4684933
time: 121.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.5187433, upper bound: 72.4730633
time: 142.21 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 265.58 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 265.58
Output dim: 2, lower bound: -72.4863406, upper bound: 72.5067884
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 265.58
Output dim: 2, lower bound: -72.4693531, upper bound: 72.5237364
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 265.58
Output dim: 2, lower bound: -72.4873051, upper bound: 72.5058428
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 265.58
Output dim: 2, lower bound: -72.4703435, upper bound: 72.5228371
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 265.58
Output dim: 2, lower bound: -72.4924838, upper bound: 72.4698653
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 265.58
Output dim: 2, lower bound: -72.4924838, upper bound: 72.4867981
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 265.58
Output dim: 2, lower bound: -72.5233013, upper bound: 72.4684933
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 265.58
Output dim: 2, lower bound: -72.5187433, upper bound: 72.4730633

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -120.6897812, 75.5772247, -120.6897812, 75.5772247, -196.2669983, 196.2669983
1: -69.1185150, 62.0534592, -69.1185150, 62.0534592, -131.1719666, 131.1719666
2: -59.9247818, 57.4265594, -59.9247818, 57.4265594, -117.3513336, 117.3513336
3: -69.0837555, 72.1842041, -69.0837555, 72.1842041, -141.2679443, 141.2679443
4: -75.4698944, 68.2243805, -75.4698944, 68.2243805, -143.6942749, 143.6942749
5: -67.5072861, 74.3775482, -67.5072861, 74.3775482, -141.8848267, 141.8848267
6: -85.3311462, 75.1246033, -85.3311462, 75.1246033, -160.4557495, 160.4557495
7: -83.4509277, 75.4055939, -83.4509277, 75.4055939, -158.8565216, 158.8565216
8: -81.8659363, 84.5110092, -81.8659363, 84.5110092, -166.3769531, 166.3769531
9: -72.3079300, 68.8767319, -72.3079300, 68.8767319, -141.1846619, 141.1846466
10: -105.4851379, 97.0667877, -105.4851379, 97.0667877, -202.5519257, 202.5519257
11: -98.7560196, 70.3052673, -98.7560196, 70.3052673, -169.0612793, 169.0612793
12: -94.6711731, 79.6737671, -94.6711731, 79.6737671, -174.3449402, 174.3449402
13: -101.8092804, 95.6050262, -101.8092804, 95.6050262, -197.4143066, 197.4143066
14: -153.0820007, 76.5118256, -153.0820007, 76.5118256, -229.5938110, 229.5938263
15: -85.4207764, 65.5752563, -85.4207764, 65.5752563, -150.9960327, 150.9960327
16: -108.7743988, 80.8979950, -108.7743988, 80.8979950, -189.6723633, 189.6723785
17: -160.1158142, 100.7368164, -160.1158142, 100.7368164, -260.8526306, 260.8526306
18: -92.6273956, 75.1741791, -92.6273956, 75.1741791, -167.8015594, 167.8015747
19: -72.3558044, 42.4454155, -72.3558044, 42.4454155, -114.8012085, 114.8012238
20: -68.1926117, 53.3375359, -68.1926117, 53.3375359, -121.5301361, 121.5301361
21: -92.3123550, 55.6030388, -92.3123550, 55.6030388, -147.9153748, 147.9153748
22: -101.0745773, 60.5765076, -101.0745773, 60.5765076, -161.6510925, 161.6510925
23: -72.3953247, 55.8623962, -72.3953247, 55.8623962, -128.2577209, 128.2577209
24: -91.0453033, 67.0268860, -91.0453033, 67.0268860, -158.0721741, 158.0721741
25: -78.7736359, 63.4368629, -78.7736359, 63.4368629, -142.2104950, 142.2104950
26: -107.2412415, 83.0555573, -107.2412415, 83.0555573, -190.2967987, 190.2967987
27: -92.9772720, 67.3610229, -92.9772720, 67.3610229, -160.3382874, 160.3382874
28: -71.9242935, 59.8130646, -71.9242935, 59.8130646, -131.7373657, 131.7373505
29: -107.8225708, 63.9952202, -107.8225708, 63.9952202, -171.8177948, 171.8177948
30: -89.9596481, 72.8284531, -89.9596481, 72.8284531, -162.7881012, 162.7881012
31: -92.2802734, 61.3140030, -92.2802734, 61.3140030, -153.5942688, 153.5942688
32: -90.6332397, 70.1865616, -90.6332397, 70.1865616, -160.8197937, 160.8197937
33: -120.4985809, 94.1850891, -120.4985809, 94.1850891, -214.6836548, 214.6836700
34: -100.3700409, 73.7348175, -100.3700409, 73.7348175, -174.1048584, 174.1048584
35: -104.3925476, 77.7684860, -104.3925476, 77.7684860, -182.1610413, 182.1610260
36: -100.9291229, 75.9416580, -100.9291229, 75.9416580, -176.8707886, 176.8707886
37: -141.9010010, 84.0858002, -141.9010010, 84.0858002, -225.9868011, 225.9868011
38: -122.6707611, 92.7453461, -122.6707611, 92.7453461, -215.4160919, 215.4161072
39: -145.5377960, 92.6060257, -145.5377960, 92.6060257, -238.1438141, 238.1438141
40: -115.8655548, 81.4251404, -115.8655548, 81.4251404, -197.2906952, 197.2906952
41: -89.2050629, 70.4504623, -89.2050629, 70.4504623, -159.6555176, 159.6555176
42: -64.9341888, 62.9433632, -64.9341888, 62.9433632, -127.8775482, 127.8775482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 588

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 693

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4693085, upper bound: 72.5060291
time: 139.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4856027, upper bound: 72.4898011
time: 171.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -120.6897812, 75.5772247, -120.6897812, 75.5772247, -196.2669983, 196.2669983
1: -69.1185150, 62.0534592, -69.1185150, 62.0534592, -131.1719666, 131.1719666
2: -59.9247818, 57.4265594, -59.9247818, 57.4265594, -117.3513336, 117.3513336
3: -69.0837555, 72.1842041, -69.0837555, 72.1842041, -141.2679443, 141.2679443
4: -75.4698944, 68.2243805, -75.4698944, 68.2243805, -143.6942749, 143.6942749
5: -67.5072861, 74.3775482, -67.5072861, 74.3775482, -141.8848267, 141.8848267
6: -85.3311462, 75.1246033, -85.3311462, 75.1246033, -160.4557495, 160.4557495
7: -83.4509277, 75.4055939, -83.4509277, 75.4055939, -158.8565216, 158.8565216
8: -81.8659363, 84.5110092, -81.8659363, 84.5110092, -166.3769531, 166.3769531
9: -72.3079300, 68.8767319, -72.3079300, 68.8767319, -141.1846619, 141.1846466
10: -105.4851379, 97.0667877, -105.4851379, 97.0667877, -202.5519257, 202.5519257
11: -98.7560196, 70.3052673, -98.7560196, 70.3052673, -169.0612793, 169.0612793
12: -94.6711731, 79.6737671, -94.6711731, 79.6737671, -174.3449402, 174.3449402
13: -101.8092804, 95.6050262, -101.8092804, 95.6050262, -197.4143066, 197.4143066
14: -153.0820007, 76.5118256, -153.0820007, 76.5118256, -229.5938110, 229.5938263
15: -85.4207764, 65.5752563, -85.4207764, 65.5752563, -150.9960327, 150.9960327
16: -108.7743988, 80.8979950, -108.7743988, 80.8979950, -189.6723633, 189.6723785
17: -160.1158142, 100.7368164, -160.1158142, 100.7368164, -260.8526306, 260.8526306
18: -92.6273956, 75.1741791, -92.6273956, 75.1741791, -167.8015594, 167.8015747
19: -72.3558044, 42.4454155, -72.3558044, 42.4454155, -114.8012085, 114.8012238
20: -68.1926117, 53.3375359, -68.1926117, 53.3375359, -121.5301361, 121.5301361
21: -92.3123550, 55.6030388, -92.3123550, 55.6030388, -147.9153748, 147.9153748
22: -101.0745773, 60.5765076, -101.0745773, 60.5765076, -161.6510925, 161.6510925
23: -72.3953247, 55.8623962, -72.3953247, 55.8623962, -128.2577209, 128.2577209
24: -91.0453033, 67.0268860, -91.0453033, 67.0268860, -158.0721741, 158.0721741
25: -78.7736359, 63.4368629, -78.7736359, 63.4368629, -142.2104950, 142.2104950
26: -107.2412415, 83.0555573, -107.2412415, 83.0555573, -190.2967987, 190.2967987
27: -92.9772720, 67.3610229, -92.9772720, 67.3610229, -160.3382874, 160.3382874
28: -71.9242935, 59.8130646, -71.9242935, 59.8130646, -131.7373657, 131.7373505
29: -107.8225708, 63.9952202, -107.8225708, 63.9952202, -171.8177948, 171.8177948
30: -89.9596481, 72.8284531, -89.9596481, 72.8284531, -162.7881012, 162.7881012
31: -92.2802734, 61.3140030, -92.2802734, 61.3140030, -153.5942688, 153.5942688
32: -90.6332397, 70.1865616, -90.6332397, 70.1865616, -160.8197937, 160.8197937
33: -120.4985809, 94.1850891, -120.4985809, 94.1850891, -214.6836548, 214.6836700
34: -100.3700409, 73.7348175, -100.3700409, 73.7348175, -174.1048584, 174.1048584
35: -104.3925476, 77.7684860, -104.3925476, 77.7684860, -182.1610413, 182.1610260
36: -100.9291229, 75.9416580, -100.9291229, 75.9416580, -176.8707886, 176.8707886
37: -141.9010010, 84.0858002, -141.9010010, 84.0858002, -225.9868011, 225.9868011
38: -122.6707611, 92.7453461, -122.6707611, 92.7453461, -215.4160919, 215.4161072
39: -145.5377960, 92.6060257, -145.5377960, 92.6060257, -238.1438141, 238.1438141
40: -115.8655548, 81.4251404, -115.8655548, 81.4251404, -197.2906952, 197.2906952
41: -89.2050629, 70.4504623, -89.2050629, 70.4504623, -159.6555176, 159.6555176
42: -64.9341888, 62.9433632, -64.9341888, 62.9433632, -127.8775482, 127.8775482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 573

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1566

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4691953, upper bound: 72.5209880
time: 124.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4666627, upper bound: 72.5235816
time: 213.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -120.6897812, 75.5772247, -120.6897812, 75.5772247, -196.2669983, 196.2669983
1: -69.1185150, 62.0534592, -69.1185150, 62.0534592, -131.1719666, 131.1719666
2: -59.9247818, 57.4265594, -59.9247818, 57.4265594, -117.3513336, 117.3513336
3: -69.0837555, 72.1842041, -69.0837555, 72.1842041, -141.2679443, 141.2679443
4: -75.4698944, 68.2243805, -75.4698944, 68.2243805, -143.6942749, 143.6942749
5: -67.5072861, 74.3775482, -67.5072861, 74.3775482, -141.8848267, 141.8848267
6: -85.3311462, 75.1246033, -85.3311462, 75.1246033, -160.4557495, 160.4557495
7: -83.4509277, 75.4055939, -83.4509277, 75.4055939, -158.8565216, 158.8565216
8: -81.8659363, 84.5110092, -81.8659363, 84.5110092, -166.3769531, 166.3769531
9: -72.3079300, 68.8767319, -72.3079300, 68.8767319, -141.1846619, 141.1846466
10: -105.4851379, 97.0667877, -105.4851379, 97.0667877, -202.5519257, 202.5519257
11: -98.7560196, 70.3052673, -98.7560196, 70.3052673, -169.0612793, 169.0612793
12: -94.6711731, 79.6737671, -94.6711731, 79.6737671, -174.3449402, 174.3449402
13: -101.8092804, 95.6050262, -101.8092804, 95.6050262, -197.4143066, 197.4143066
14: -153.0820007, 76.5118256, -153.0820007, 76.5118256, -229.5938110, 229.5938263
15: -85.4207764, 65.5752563, -85.4207764, 65.5752563, -150.9960327, 150.9960327
16: -108.7743988, 80.8979950, -108.7743988, 80.8979950, -189.6723633, 189.6723785
17: -160.1158142, 100.7368164, -160.1158142, 100.7368164, -260.8526306, 260.8526306
18: -92.6273956, 75.1741791, -92.6273956, 75.1741791, -167.8015594, 167.8015747
19: -72.3558044, 42.4454155, -72.3558044, 42.4454155, -114.8012085, 114.8012238
20: -68.1926117, 53.3375359, -68.1926117, 53.3375359, -121.5301361, 121.5301361
21: -92.3123550, 55.6030388, -92.3123550, 55.6030388, -147.9153748, 147.9153748
22: -101.0745773, 60.5765076, -101.0745773, 60.5765076, -161.6510925, 161.6510925
23: -72.3953247, 55.8623962, -72.3953247, 55.8623962, -128.2577209, 128.2577209
24: -91.0453033, 67.0268860, -91.0453033, 67.0268860, -158.0721741, 158.0721741
25: -78.7736359, 63.4368629, -78.7736359, 63.4368629, -142.2104950, 142.2104950
26: -107.2412415, 83.0555573, -107.2412415, 83.0555573, -190.2967987, 190.2967987
27: -92.9772720, 67.3610229, -92.9772720, 67.3610229, -160.3382874, 160.3382874
28: -71.9242935, 59.8130646, -71.9242935, 59.8130646, -131.7373657, 131.7373505
29: -107.8225708, 63.9952202, -107.8225708, 63.9952202, -171.8177948, 171.8177948
30: -89.9596481, 72.8284531, -89.9596481, 72.8284531, -162.7881012, 162.7881012
31: -92.2802734, 61.3140030, -92.2802734, 61.3140030, -153.5942688, 153.5942688
32: -90.6332397, 70.1865616, -90.6332397, 70.1865616, -160.8197937, 160.8197937
33: -120.4985809, 94.1850891, -120.4985809, 94.1850891, -214.6836548, 214.6836700
34: -100.3700409, 73.7348175, -100.3700409, 73.7348175, -174.1048584, 174.1048584
35: -104.3925476, 77.7684860, -104.3925476, 77.7684860, -182.1610413, 182.1610260
36: -100.9291229, 75.9416580, -100.9291229, 75.9416580, -176.8707886, 176.8707886
37: -141.9010010, 84.0858002, -141.9010010, 84.0858002, -225.9868011, 225.9868011
38: -122.6707611, 92.7453461, -122.6707611, 92.7453461, -215.4160919, 215.4161072
39: -145.5377960, 92.6060257, -145.5377960, 92.6060257, -238.1438141, 238.1438141
40: -115.8655548, 81.4251404, -115.8655548, 81.4251404, -197.2906952, 197.2906952
41: -89.2050629, 70.4504623, -89.2050629, 70.4504623, -159.6555176, 159.6555176
42: -64.9341888, 62.9433632, -64.9341888, 62.9433632, -127.8775482, 127.8775482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1740

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 541

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4808463, upper bound: 72.5058262
time: 281.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -72.4872867, upper bound: 72.4994362
time: 133.68 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 416.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 416.92
Output dim: 2, lower bound: -72.4693085, upper bound: 72.5060291
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 416.92
Output dim: 2, lower bound: -72.4856027, upper bound: 72.4898011
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 416.92
Output dim: 2, lower bound: -72.4691953, upper bound: 72.5209880
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 416.92
Output dim: 2, lower bound: -72.4666627, upper bound: 72.5235816
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 416.92
Output dim: 2, lower bound: -72.4808463, upper bound: 72.5058262
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 416.92
Output dim: 2, lower bound: -72.4872867, upper bound: 72.4994362
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 416.92
Output dim: 2, lower bound: -72.4703435, upper bound: 72.5228371
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 416.92
Output dim: 2, lower bound: -72.4924838, upper bound: 72.4698653
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 416.92
Output dim: 2, lower bound: -72.4924838, upper bound: 72.4867981
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 416.92
Output dim: 2, lower bound: -72.5233013, upper bound: 72.4684933
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 416.92
Output dim: 2, lower bound: -72.5187433, upper bound: 72.4730633

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 209.60 + 3547.97 = 3757.57 seconds
