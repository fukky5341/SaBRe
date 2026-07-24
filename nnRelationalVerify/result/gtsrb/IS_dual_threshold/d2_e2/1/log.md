## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 3600 seconds
Split limit: 100
Threshold: 22.4049659067


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301)
1: (-14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610)
2: (-10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408)
3: (-12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606)
4: (-15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124)
5: (-10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690)
6: (-32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7770462, 42.7770462)
7: (-16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3075523, 43.3075523)
8: (-18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311)
9: (-17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898)
10: (-29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482)
11: (-34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441)
12: (-34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480)
13: (-29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433)
14: (-52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4644928, 61.4644852)
15: (-22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842)
16: (-30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341)
17: (-56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064)
18: (-30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531)
19: (-29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924)
20: (-21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896)
21: (-33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033)
22: (-38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131)
23: (-27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346)
24: (-30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835)
25: (-28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707)
26: (-43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777)
27: (-30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377)
28: (-27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994)
29: (-39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092)
30: (-28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813)
31: (-31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658)
32: (-30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079)
33: (-48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7282791, 57.7282867)
34: (-41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1289978, 49.1289978)
35: (-41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5784836, 50.5784836)
36: (-42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865)
37: (-63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7241440, 65.7241516)
38: (-53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258)
39: (-62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682)
40: (-50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0631561, 59.0631485)
41: (-35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932)
42: (-26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0691223, 34.0691223)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.83 + 69.52 = 72.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -22.4273933, upper bound: 22.4273933

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3507611, upper bound: 22.4219063
time: 58.92 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3507611, upper bound: 22.4219063
time: 61.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 120.81 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 120.81
Output dim: 3, lower bound: -22.3507611, upper bound: 22.4219063
IS_B2, status: Status.UNKNOWN, split count: 1, time: 120.81
Output dim: 3, lower bound: -22.3507611, upper bound: 22.4219063

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -29.8690376, 12.5914154, -29.8531570, 12.5847292, -42.4537659, 42.4445724
1: -14.3950481, 20.7124863, -14.3877373, 20.6857338, -35.0807800, 35.1002235
2: -10.3783970, 21.0642834, -10.3719063, 21.0489674, -31.4273643, 31.4361897
3: -12.5727291, 23.6840153, -12.5653696, 23.6375637, -36.2102928, 36.2493858
4: -15.7989674, 20.6705322, -15.7924194, 20.6618290, -36.4607964, 36.4629517
5: -10.8994865, 25.4427204, -10.8924866, 25.4169312, -36.3164177, 36.3352051
6: -32.1161423, 11.5913887, -32.1066246, 11.5835886, -42.7595367, 42.7561722
7: -16.9638977, 26.3728561, -16.9542007, 26.3412933, -43.2562141, 43.2793922
8: -18.4969292, 23.7388763, -18.4904461, 23.7148628, -42.2117920, 42.2293243
9: -17.0998440, 20.6091995, -17.0843353, 20.6026917, -37.7025375, 37.6935349
10: -29.9860344, 29.0636463, -29.9755478, 29.0516205, -59.0376549, 59.0391922
11: -34.8675766, 15.3048830, -34.8558578, 15.2749634, -50.1425400, 50.1607399
12: -34.7262650, 13.5684061, -34.7148361, 13.5576143, -48.2838783, 48.2832413
13: -29.5409431, 22.9498730, -29.5270023, 22.9417725, -52.4827156, 52.4768753
14: -52.3625984, 10.3973598, -52.3476715, 10.3816690, -61.4363022, 61.4355011
15: -22.8534431, 19.0166702, -22.8442383, 19.0057983, -41.8592415, 41.8609085
16: -30.8022156, 25.6222630, -30.7831955, 25.6099663, -56.4121819, 56.4054565
17: -55.9937744, 21.7580833, -55.9785995, 21.6883450, -77.6821213, 77.7366791
18: -30.8240337, 14.3062210, -30.8060188, 14.2965908, -45.1206245, 45.1122398
19: -29.7166290, 3.0112591, -29.6978073, 3.0048671, -32.7214966, 32.7090683
20: -21.8498936, 10.4082842, -21.8410797, 10.4018011, -32.2516937, 32.2493629
21: -33.5510025, 6.8107681, -33.5405502, 6.8013458, -40.3523483, 40.3513184
22: -38.2412949, 10.3547735, -38.2297173, 10.3366871, -48.5779800, 48.5844917
23: -27.6366005, 7.6594930, -27.6282539, 7.6537180, -35.2903175, 35.2877464
24: -30.9811764, 7.9130440, -30.9700794, 7.9059525, -38.8871307, 38.8831253
25: -28.2985725, 11.2272205, -28.2909927, 11.2194424, -39.5180130, 39.5182114
26: -43.3651085, 8.2351007, -43.3537292, 8.2240438, -51.5891533, 51.5888290
27: -30.0278263, 14.0758238, -30.0194397, 14.0651827, -44.0930099, 44.0952644
28: -27.4602909, 9.9769001, -27.4526234, 9.9719725, -37.4322624, 37.4295235
29: -39.8661346, 10.7463799, -39.8540268, 10.7105331, -50.5766678, 50.6004066
30: -28.1182270, 14.7333632, -28.1094322, 14.7219124, -42.8401413, 42.8427963
31: -31.2098198, 8.4931602, -31.1911716, 8.4841270, -39.6939468, 39.6843338
32: -30.9179459, 12.0849733, -30.9090881, 12.0790987, -42.9970436, 42.9940605
33: -48.9568329, 9.3491535, -48.9139481, 9.3419762, -57.6994934, 57.6637878
34: -41.8248177, 7.6099825, -41.8103523, 7.6039495, -49.1132965, 49.1039124
35: -41.0879631, 9.6063147, -41.0614815, 9.6020222, -50.5608673, 50.5380669
36: -42.4358368, 9.9519320, -42.4133453, 9.9471960, -52.3830338, 52.3652763
37: -63.8718224, 2.1072054, -63.7925034, 2.1015863, -65.6818008, 65.6081848
38: -53.2372208, 12.1439877, -53.2115173, 12.1351147, -65.3723373, 65.3555069
39: -62.3181763, 5.9454975, -62.2482834, 5.9383602, -68.2565384, 68.1937790
40: -50.1255341, 9.2994127, -50.0640182, 9.2936468, -59.0278473, 58.9715652
41: -35.2813797, 6.7685194, -35.2561264, 6.7616892, -42.0430679, 42.0246468
42: -26.2444649, 7.8577623, -26.2362823, 7.8432035, -34.0434494, 34.0485764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=212, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3450356, upper bound: 22.3561468
time: 74.76 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3450356, upper bound: 22.4208181
time: 78.51 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -29.8743477, 12.5934839, -29.9608746, 12.6013145, -42.4756622, 42.5543594
1: -14.3976154, 20.7221107, -14.4711485, 20.7276402, -35.1252556, 35.1932602
2: -10.3805208, 21.0695477, -10.4422646, 21.0742626, -31.4547844, 31.5118122
3: -12.5748339, 23.7012424, -12.6865778, 23.7106552, -36.2854881, 36.3878212
4: -15.8011456, 20.6732407, -15.8449297, 20.6970215, -36.4981689, 36.5181694
5: -10.9018784, 25.4522457, -11.0023689, 25.4606552, -36.3625336, 36.4546127
6: -32.1188736, 11.5940809, -32.1326332, 11.6182842, -42.8206558, 42.7827911
7: -16.9672089, 26.3846302, -17.0646858, 26.3878479, -43.3072357, 43.4032593
8: -18.4989281, 23.7473106, -18.5659294, 23.7583733, -42.2573013, 42.3132401
9: -17.1019211, 20.6113434, -17.1201649, 20.6450901, -37.7470093, 37.7315063
10: -29.9890022, 29.0674324, -30.0120010, 29.0893364, -59.0783386, 59.0794334
11: -34.8710594, 15.3143616, -34.9347878, 15.3249474, -50.1960068, 50.2491493
12: -34.7301826, 13.5722589, -34.7445793, 13.6504145, -48.3805962, 48.3168373
13: -29.5456886, 22.9521065, -29.5745468, 22.9757328, -52.5214233, 52.5266533
14: -52.3676071, 10.4013653, -52.4573708, 10.4062548, -61.4648972, 61.5438538
15: -22.8563099, 19.0193157, -22.8927555, 19.0325928, -41.8889008, 41.9120712
16: -30.8075371, 25.6264400, -30.8410530, 25.6392536, -56.4467926, 56.4674911
17: -55.9991989, 21.7840557, -56.1208115, 21.7922096, -77.7914124, 77.9048691
18: -30.8300686, 14.3089714, -30.8545837, 14.3744240, -45.2044907, 45.1635551
19: -29.7200203, 3.0130639, -29.7353859, 3.0366974, -32.7567177, 32.7484512
20: -21.8527431, 10.4104729, -21.8720036, 10.4213047, -32.2740479, 32.2824783
21: -33.5542755, 6.8137951, -33.5875854, 6.8303747, -40.3846512, 40.4013824
22: -38.2453003, 10.3607845, -38.2716827, 10.3723087, -48.6176071, 48.6324692
23: -27.6394501, 7.6611109, -27.6724205, 7.6734271, -35.3128777, 35.3335304
24: -30.9839401, 7.9152193, -30.9966583, 7.9290581, -38.9129982, 38.9118767
25: -28.3010960, 11.2294769, -28.3249283, 11.2485552, -39.5496521, 39.5544052
26: -43.3685684, 8.2387600, -43.3861809, 8.3003807, -51.6689491, 51.6249390
27: -30.0303040, 14.0792303, -30.0693798, 14.0956163, -44.1259193, 44.1486092
28: -27.4628868, 9.9782581, -27.4895725, 9.9973736, -37.4602585, 37.4678307
29: -39.8702927, 10.7596684, -39.9239388, 10.7707510, -50.6410446, 50.6836090
30: -28.1209450, 14.7366123, -28.1758289, 14.7519941, -42.8729401, 42.9124413
31: -31.2142582, 8.4960632, -31.2311268, 8.5259609, -39.7402191, 39.7271881
32: -30.9203949, 12.0870209, -30.9338322, 12.1155930, -43.0359879, 43.0208511
33: -48.9724350, 9.3512430, -48.9822693, 9.4435501, -57.8200073, 57.7341309
34: -41.8294144, 7.6119976, -41.8395844, 7.6451731, -49.1648407, 49.1293335
35: -41.0967178, 9.6075478, -41.1036148, 9.6557312, -50.6263733, 50.5814095
36: -42.4431114, 9.9535885, -42.4486313, 9.9866295, -52.4297409, 52.4022217
37: -63.9012451, 2.1090803, -63.9165039, 2.2559652, -65.8668137, 65.7308884
38: -53.2457390, 12.1470222, -53.2565384, 12.2060366, -65.4517746, 65.4035645
39: -62.3440170, 5.9478168, -62.3552246, 6.0757895, -68.4198074, 68.3030396
40: -50.1477051, 9.3013010, -50.1590652, 9.4149294, -59.1753693, 59.0686188
41: -35.2902374, 6.7705908, -35.2987442, 6.8272648, -42.1175003, 42.0693359
42: -26.2468872, 7.8623466, -26.2897797, 7.8846006, -34.0897408, 34.0998001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=212, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3450356, upper bound: 22.3561468
time: 62.23 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3450356, upper bound: 22.3561468
time: 68.26 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 132.80 seconds
IS_B1_B1, status: Status.VERIFIED, split count: 2, time: 132.80
Output dim: 3, lower bound: -22.3450356, upper bound: 22.3561468
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 132.80
Output dim: 3, lower bound: -22.3450356, upper bound: 22.4208181
IS_B2_B1, status: Status.VERIFIED, split count: 2, time: 132.80
Output dim: 3, lower bound: -22.3450356, upper bound: 22.3561468
IS_B2_B2, status: Status.VERIFIED, split count: 2, time: 132.80
Output dim: 3, lower bound: -22.3450356, upper bound: 22.3561468

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -29.8655834, 12.5897980, -29.8819408, 12.6011295, -42.4667130, 42.4717407
1: -14.3927279, 20.7114716, -14.3907909, 20.7312298, -35.1239586, 35.1022644
2: -10.3762589, 21.0633736, -10.3741245, 21.1114616, -31.4877205, 31.4374981
3: -12.5705433, 23.6829338, -12.5643406, 23.6953106, -36.2658539, 36.2472763
4: -15.7972584, 20.6695251, -15.7943745, 20.6900101, -36.4872665, 36.4638977
5: -10.8973207, 25.4418869, -10.8934002, 25.4486389, -36.3459587, 36.3352890
6: -32.1099701, 11.5901108, -32.1053200, 11.5855560, -42.7547607, 42.7564087
7: -16.9599228, 26.3715229, -16.9499588, 26.4012375, -43.3128204, 43.2714005
8: -18.4936943, 23.7377281, -18.4890003, 23.8188419, -42.3125381, 42.2267303
9: -17.0988121, 20.6065483, -17.0873146, 20.6040287, -37.7028427, 37.6938629
10: -29.9845428, 29.0621605, -29.9878712, 29.0566387, -59.0411835, 59.0500336
11: -34.8606110, 15.3031597, -34.8609161, 15.2715034, -50.1321144, 50.1640778
12: -34.7251511, 13.5660772, -34.7393036, 13.5689144, -48.2940674, 48.3053818
13: -29.5399017, 22.9404602, -29.5215111, 22.9382782, -52.4781799, 52.4619713
14: -52.3594322, 10.3963346, -52.3699608, 10.4591093, -61.5098572, 61.4427299
15: -22.8511238, 19.0086651, -22.8330402, 18.9954529, -41.8465767, 41.8417053
16: -30.7934513, 25.6211700, -30.7868671, 25.6058865, -56.3993378, 56.4080353
17: -55.9919281, 21.7557564, -56.0024033, 21.6940422, -77.6859741, 77.7581635
18: -30.8227997, 14.3042679, -30.8716679, 14.2965527, -45.1193542, 45.1759338
19: -29.7150631, 3.0099845, -29.7422085, 3.0053644, -32.7204285, 32.7521935
20: -21.8483963, 10.4069462, -21.8547802, 10.4085236, -32.2569199, 32.2617264
21: -33.5493164, 6.8099489, -33.5652733, 6.8032236, -40.3525391, 40.3752213
22: -38.2395248, 10.3479633, -38.2855721, 10.3198967, -48.5594215, 48.6335373
23: -27.6353931, 7.6583719, -27.6536484, 7.6578851, -35.2932777, 35.3120193
24: -30.9802151, 7.9114065, -31.0352879, 7.9051046, -38.8853188, 38.9466934
25: -28.2973042, 11.2234755, -28.3582039, 11.2114000, -39.5087051, 39.5816803
26: -43.3634262, 8.2314577, -43.4362259, 8.2269239, -51.5903511, 51.6676826
27: -30.0200043, 14.0752106, -30.0221024, 14.0843859, -44.1043892, 44.0973129
28: -27.4591637, 9.9756813, -27.4881897, 9.9776554, -37.4368210, 37.4638710
29: -39.8642731, 10.7431030, -39.9052658, 10.7051468, -50.5694199, 50.6483688
30: -28.1164799, 14.7320452, -28.1500683, 14.7281551, -42.8446350, 42.8821144
31: -31.2077484, 8.4915466, -31.2446899, 8.4846334, -39.6923828, 39.7362366
32: -30.9119949, 12.0841169, -30.9066429, 12.1091728, -43.0211678, 42.9907608
33: -48.9555817, 9.3463135, -48.9765778, 9.3407650, -57.6898727, 57.7018280
34: -41.8238373, 7.6084509, -41.8652458, 7.6079569, -49.1067581, 49.1575928
35: -41.0866852, 9.6041641, -41.1275406, 9.6004276, -50.5515900, 50.6007538
36: -42.4346771, 9.9503508, -42.4555969, 9.9474716, -52.3821487, 52.4059486
37: -63.8700485, 2.1044693, -63.9061432, 2.0992556, -65.6654663, 65.7177353
38: -53.2359543, 12.1425114, -53.2363892, 12.1446209, -65.3805771, 65.3788986
39: -62.3167343, 5.9409714, -62.2827072, 5.9371853, -68.2539215, 68.2236786
40: -50.1233482, 9.2979012, -50.0900841, 9.3013477, -59.0290833, 58.9954681
41: -35.2793503, 6.7672625, -35.2799530, 6.7719717, -42.0513229, 42.0472145
42: -26.2418232, 7.8564100, -26.2329559, 7.8489695, -34.0373955, 34.0562668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=349, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1684

## Relational analysis of IS_B1_B2_B1

### Relational analysis result of IS_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3146582, upper bound: 22.4170342
time: 151.22 seconds

## Relational analysis of IS_B1_B2_B2

### Relational analysis result of IS_B1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3146582, upper bound: 22.3523614
time: 89.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 242.72 seconds
IS_B1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 242.72
Output dim: 3, lower bound: -22.3146582, upper bound: 22.4170342
IS_B1_B2_B2, status: Status.VERIFIED, split count: 3, time: 242.72
Output dim: 3, lower bound: -22.3146582, upper bound: 22.3523614

## BFS IS instance: IS_B1_B2_B1

### Backsubstitution after applying IS history:
0: -29.8579292, 12.5801220, -29.7990990, 12.5621204, -42.4200516, 42.3792191
1: -14.3895636, 20.7021141, -14.3450623, 20.6962223, -35.0857849, 35.0471764
2: -10.3741035, 21.0585938, -10.3418598, 21.0910549, -31.4651585, 31.4004536
3: -12.5677452, 23.6671848, -12.5098505, 23.6396599, -36.2074051, 36.1770363
4: -15.7957096, 20.6658859, -15.7762966, 20.6708584, -36.4665680, 36.4421844
5: -10.8952656, 25.4272480, -10.8277168, 25.3968868, -36.2921524, 36.2549667
6: -32.0997124, 11.5883389, -32.0583916, 11.5598660, -42.6956444, 42.6997261
7: -16.9566002, 26.3635826, -16.8934212, 26.3724289, -43.2801476, 43.2059479
8: -18.4887371, 23.7324219, -18.4597588, 23.7894955, -42.2782326, 42.1921806
9: -17.0961647, 20.5881691, -17.0150299, 20.5341835, -37.6303482, 37.6031990
10: -29.9820099, 29.0334396, -29.8848457, 28.9494038, -58.9314117, 58.9182854
11: -34.8555336, 15.2953262, -34.7887077, 15.2389145, -50.0944481, 50.0840340
12: -34.7173042, 13.5636225, -34.7034988, 13.5234127, -48.2407150, 48.2671204
13: -29.5352707, 22.9362507, -29.4942665, 22.8808918, -52.4161606, 52.4305191
14: -52.3540840, 10.3939009, -52.3140488, 10.4383116, -61.4763641, 61.3832359
15: -22.8477745, 18.9936905, -22.7935257, 18.9377422, -41.7855148, 41.7872162
16: -30.7869606, 25.6005096, -30.6817150, 25.5320377, -56.3190002, 56.2822266
17: -55.9849434, 21.7453728, -55.9228592, 21.6535835, -77.6385269, 77.6682281
18: -30.8188133, 14.2991238, -30.8326206, 14.2691898, -45.0880051, 45.1317444
19: -29.7118073, 3.0072975, -29.7147903, 2.9875073, -32.6993141, 32.7220879
20: -21.8422985, 10.4041729, -21.8250313, 10.3688984, -32.2111969, 32.2292023
21: -33.5452003, 6.8038797, -33.5116920, 6.7686682, -40.3138695, 40.3155708
22: -38.2327118, 10.3436270, -38.2493095, 10.2956123, -48.5283241, 48.5929375
23: -27.6314240, 7.6554918, -27.6231880, 7.6372232, -35.2686462, 35.2786789
24: -30.9655304, 7.9088001, -30.9818268, 7.8640509, -38.8295822, 38.8906250
25: -28.2910366, 11.2203150, -28.3265896, 11.1733723, -39.4644089, 39.5469055
26: -43.3571167, 8.2278776, -43.4021912, 8.1710939, -51.5282097, 51.6300697
27: -30.0146255, 14.0733728, -29.9871159, 14.0608511, -44.0754776, 44.0604897
28: -27.4545097, 9.9740829, -27.4646301, 9.9442825, -37.3987923, 37.4387131
29: -39.8565979, 10.7322235, -39.8469849, 10.6645603, -50.5211563, 50.5792084
30: -28.1104660, 14.7284985, -28.1092796, 14.6972857, -42.8077507, 42.8377762
31: -31.2024860, 8.4878693, -31.2098827, 8.4658136, -39.6683006, 39.6977539
32: -30.9037132, 12.0825720, -30.8688641, 12.0652390, -42.9689522, 42.9514351
33: -48.9256935, 9.3441172, -48.8664055, 9.2327299, -57.5498505, 57.5876083
34: -41.8061790, 7.6065702, -41.7992020, 7.5446224, -49.0251617, 49.0869827
35: -41.0673065, 9.6027660, -41.0560188, 9.5223322, -50.4540100, 50.5275192
36: -42.4168625, 9.9490919, -42.3915482, 9.8789473, -52.2958107, 52.3406410
37: -63.8330612, 2.1028595, -63.7753792, 1.9879875, -65.5125961, 65.5851288
38: -53.2104645, 12.1393652, -53.1453247, 12.0415564, -65.2520218, 65.2846909
39: -62.2801056, 5.9382153, -62.1583786, 5.8153114, -68.0954132, 68.0965958
40: -50.0887794, 9.2953205, -49.9682007, 9.2100849, -58.9029694, 58.8706055
41: -35.2693405, 6.7656994, -35.2394257, 6.7369003, -42.0062408, 42.0051270
42: -26.2361526, 7.8544340, -26.1915722, 7.8347368, -34.0077629, 34.0082970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=349, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1719

## Relational analysis of IS_B1_B2_B1_B1

### Relational analysis result of IS_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.2454299, upper bound: 22.4047564
time: 62.89 seconds

## Relational analysis of IS_B1_B2_B1_B2

### Relational analysis result of IS_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.2987729, upper bound: 22.3956201
time: 68.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 133.74 seconds
IS_B1_B2_B1_B1, status: Status.VERIFIED, split count: 4, time: 133.74
Output dim: 3, lower bound: -22.2454299, upper bound: 22.4047564
IS_B1_B2_B1_B2, status: Status.VERIFIED, split count: 4, time: 133.74
Output dim: 3, lower bound: -22.2987729, upper bound: 22.3956201

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 72.36 + 785.62 = 857.98 seconds
