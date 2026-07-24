## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 3600 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.20 + 73.73 = 75.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -22.4273933, upper bound: 22.4273933

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3793462, upper bound: 22.4102643
time: 71.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4102643, upper bound: 22.3793462
time: 77.10 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 148.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 148.67
Output dim: 3, lower bound: -22.3793462, upper bound: 22.4102643
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 148.67
Output dim: 3, lower bound: -22.4102643, upper bound: 22.3793462

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7757797, 42.7771950
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3063927, 43.3073616
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4658813, 61.4628677
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7241898, 57.7260284
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1250229, 49.1262741
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5747452, 50.5756683
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7193527, 65.7298279
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0608978, 59.0651398
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0679550, 34.0696373

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3109190, upper bound: 22.4080572
time: 59.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3771069, upper bound: 22.3308225
time: 69.99 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7770462, 42.7757835
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3073616, 43.3075523
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4628601, 61.4644852
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7282791, 57.7241898
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1289978, 49.1250229
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5784836, 50.5747414
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7241440, 65.7193604
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0631561, 59.0608826
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0691223, 34.0679550

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3308225, upper bound: 22.3771069
time: 99.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4080572, upper bound: 22.3109190
time: 64.71 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 165.81 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 165.81
Output dim: 3, lower bound: -22.3109190, upper bound: 22.4080572
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 165.81
Output dim: 3, lower bound: -22.3771069, upper bound: 22.3308225
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 165.81
Output dim: 3, lower bound: -22.3308225, upper bound: 22.3771069
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 165.81
Output dim: 3, lower bound: -22.4080572, upper bound: 22.3109190

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7749825, 42.7756577
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3096466, 43.3105011
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4554214, 61.4494324
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7064438, 57.7093201
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1055069, 49.1068573
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5559006, 50.5580292
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7065735, 65.7213211
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0573730, 59.0623093
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0617447, 34.0618172

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2377593, upper bound: 22.4063400
time: 114.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2377593, upper bound: 22.3299455
time: 64.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7742424, 42.7763939
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3095398, 43.3106079
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4524460, 61.4524078
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7068710, 57.7082825
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1055908, 49.1067505
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5570755, 50.5568275
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7108612, 65.7170410
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0580444, 59.0616150
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0601273, 34.0634232

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2988635, upper bound: 22.3288318
time: 67.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3753678, upper bound: 22.2602337
time: 59.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7762413, 42.7742462
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3106079, 43.3106918
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4524155, 61.4510422
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7105331, 57.7068787
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1094818, 49.1055870
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5596390, 50.5570755
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7113800, 65.7108536
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0596313, 59.0580521
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0629044, 34.0601349

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2602337, upper bound: 22.3753678
time: 84.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2602337, upper bound: 22.2988635
time: 63.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7755089, 42.7749825
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3105011, 43.3107986
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4494247, 61.4540176
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7109756, 57.7064362
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1095657, 49.1055031
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5608292, 50.5559006
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7156677, 65.7065735
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0603180, 59.0573654
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0613022, 34.0617409

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3299456, upper bound: 22.3089433
time: 73.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4063401, upper bound: 22.2377593
time: 233.46 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 309.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 309.16
Output dim: 3, lower bound: -22.2377593, upper bound: 22.4063400
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 309.16
Output dim: 3, lower bound: -22.2377593, upper bound: 22.3299455
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 309.16
Output dim: 3, lower bound: -22.2988635, upper bound: 22.3288318
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 309.16
Output dim: 3, lower bound: -22.3753678, upper bound: 22.2602337
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 309.16
Output dim: 3, lower bound: -22.2602337, upper bound: 22.3753678
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 309.16
Output dim: 3, lower bound: -22.2602337, upper bound: 22.2988635
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 309.16
Output dim: 3, lower bound: -22.3299456, upper bound: 22.3089433
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 309.16
Output dim: 3, lower bound: -22.4063401, upper bound: 22.2377593

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7737656, 42.7743683
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3131332, 43.3135223
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4557648, 61.4497070
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6894302, 57.6930008
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0867004, 49.0888824
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5399551, 50.5423889
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7102432, 65.7265625
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0558929, 59.0617294
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0598602, 34.0597916

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.1988343, upper bound: 22.4037159
time: 66.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.1988343, upper bound: 22.3465481
time: 62.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7736893, 42.7744370
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3126678, 43.3139915
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4556885, 61.4497757
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6901169, 57.6923065
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0879364, 49.0880508
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5406265, 50.5420837
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7118149, 65.7249832
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0567780, 59.0608521
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0597076, 34.0598602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.1988343, upper bound: 22.3257383
time: 62.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.1988343, upper bound: 22.2918009
time: 66.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7730179, 42.7751045
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3130264, 43.3136292
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4527893, 61.4526825
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6898575, 57.6919556
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0867844, 49.0888977
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5411301, 50.5412865
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7145157, 65.7222748
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0565948, 59.0610428
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0581665, 34.0613976

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2640304, upper bound: 22.3263106
time: 75.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2946921, upper bound: 22.2677812
time: 61.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7729568, 42.7751732
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3125610, 43.3140984
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4527130, 61.4527550
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6905594, 57.6912689
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0879211, 49.0879478
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5417099, 50.5408859
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7160873, 65.7207031
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0574799, 59.0601578
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0581055, 34.0615387

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3201907, upper bound: 22.2559375
time: 69.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3727398, upper bound: 22.2180210
time: 60.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7750168, 42.7729568
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3141022, 43.3137131
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4527588, 61.4513130
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6935196, 57.6905518
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0906754, 49.0879211
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5436935, 50.5417099
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7150192, 65.7160950
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0581665, 59.0574722
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0610199, 34.0581055

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2180210, upper bound: 22.3727398
time: 74.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2559375, upper bound: 22.3201907
time: 75.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7749557, 42.7730255
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3136292, 43.3141823
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4526825, 61.4513855
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6942215, 57.6898575
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0919113, 49.0867844
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5443497, 50.5411339
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7166061, 65.7145157
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0590363, 59.0565948
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0608826, 34.0581741

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2180210, upper bound: 22.2946921
time: 69.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3263106, upper bound: 22.2640304
time: 73.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7742844, 42.7736931
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3139954, 43.3138199
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4497833, 61.4542923
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6939621, 57.6901169
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0907669, 49.0879364
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5448685, 50.5406227
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7193222, 65.7118073
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0588531, 59.0567856
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0593414, 34.0597153

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2918009, upper bound: 22.3064229
time: 64.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2918009, upper bound: 22.2517538
time: 83.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7742081, 42.7737617
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3135223, 43.3142891
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4497070, 61.4543610
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6946487, 57.6894226
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0918961, 49.0867004
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5454483, 50.5399590
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7208786, 65.7102356
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0597382, 59.0559082
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0592728, 34.0598564

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3465481, upper bound: 22.2336770
time: 76.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4037159, upper bound: 22.1988343
time: 55.33 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 133.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.1988343, upper bound: 22.4037159
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.1988343, upper bound: 22.3465481
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.1988343, upper bound: 22.3257383
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.1988343, upper bound: 22.2918009
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.2640304, upper bound: 22.3263106
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.2946921, upper bound: 22.2677812
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.3201907, upper bound: 22.2559375
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.3727398, upper bound: 22.2180210
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.2180210, upper bound: 22.3727398
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.2559375, upper bound: 22.3201907
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.2180210, upper bound: 22.2946921
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.3263106, upper bound: 22.2640304
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.2918009, upper bound: 22.3064229
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.2918009, upper bound: 22.2517538
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.3465481, upper bound: 22.2336770
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 133.51
Output dim: 3, lower bound: -22.4037159, upper bound: 22.1988343

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7737770, 42.7763214
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3197479, 43.3204041
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4611664, 61.4541092
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6628876, 57.6676483
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0686340, 49.0716858
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5151520, 50.5181122
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7143707, 65.7351608
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0517349, 59.0604630
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0659142, 34.0690613

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.1778014, upper bound: 22.3302772
time: 60.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.1258963, upper bound: 22.3817224
time: 68.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7756844, 42.7743797
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3200150, 43.3199310
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4601440, 61.4551010
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6640778, 57.6664658
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0690308, 49.0708160
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5153503, 50.5175705
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7183228, 65.7306900
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0546341, 59.0575485
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0689964, 34.0658493

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.1778014, upper bound: 22.2732933
time: 69.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.1258963, upper bound: 22.3250678
time: 62.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7737083, 42.7763519
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3192825, 43.3208733
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4610901, 61.4541550
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6635742, 57.6669540
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0698700, 49.0699768
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5158081, 50.5171051
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7159424, 65.7330780
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0526047, 59.0595856
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0657768, 34.0690002

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2305526, upper bound: 22.2524974
time: 101.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.1258963, upper bound: 22.3042404
time: 65.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7756538, 42.7744484
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3195496, 43.3204041
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4600983, 61.4551773
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6649475, 57.6657791
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0707474, 49.0699844
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5163727, 50.5172729
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7204132, 65.7291107
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0555191, 59.0566711
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0689812, 34.0659180

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.1778014, upper bound: 22.2186555
time: 67.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2332323, upper bound: 22.2705506
time: 57.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7730370, 42.7770615
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3196411, 43.3205109
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4581909, 61.4570847
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6633301, 57.6666031
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0687180, 49.0717010
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5163116, 50.5170288
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7186432, 65.7308807
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0524216, 59.0597687
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0642357, 34.0706711

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2427406, upper bound: 22.2531973
time: 69.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.1909298, upper bound: 22.3048181
time: 61.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7749443, 42.7751160
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3199081, 43.3200417
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4571686, 61.4580803
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6645050, 57.6654205
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0690155, 49.0708313
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5164337, 50.5164719
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7226105, 65.7264099
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0553360, 59.0568619
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0673180, 34.0674591

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2732992, upper bound: 22.1948121
time: 66.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2215839, upper bound: 22.2465504
time: 58.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7729683, 42.7770920
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3191757, 43.3209801
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4581146, 61.4571342
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6640167, 57.6659164
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0698547, 49.0699921
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5168915, 50.5160141
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7202148, 65.7287903
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0532913, 59.0588913
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0641670, 34.0706863

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2985929, upper bound: 22.1829084
time: 56.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2468939, upper bound: 22.2348380
time: 61.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7749138, 42.7751846
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3194427, 43.3205109
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4571228, 61.4581528
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6652679, 57.6647339
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0707245, 49.0698814
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5174561, 50.5160713
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7247009, 65.7248306
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0562057, 59.0559769
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0673790, 34.0676041

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2732992, upper bound: 22.1450472
time: 63.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2993355, upper bound: 22.1970313
time: 51.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7750359, 42.7749100
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3205109, 43.3206024
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4581604, 61.4557152
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6669922, 57.6652756
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0726089, 49.0707283
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5188904, 50.5174522
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7191315, 65.7246933
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0539932, 59.0562057
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0670891, 34.0673752

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.1970313, upper bound: 22.2993355
time: 62.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.1450472, upper bound: 22.3508202
time: 64.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7769432, 42.7729683
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3209839, 43.3201294
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4571228, 61.4567108
15: -22.8573895, 19.0213966, -22.8573895, 19.0213966, -41.8787842, 41.8787842
16: -30.8105545, 25.6274796, -30.8105545, 25.6274796, -56.4380341, 56.4380341
17: -56.0003052, 21.7881050, -56.0003052, 21.7881050, -77.7884064, 77.7884064
18: -30.8318920, 14.3104610, -30.8318920, 14.3104610, -45.1423531, 45.1423531
19: -29.7247734, 3.0140181, -29.7247734, 3.0140181, -32.7387924, 32.7387924
20: -21.8537102, 10.4110794, -21.8537102, 10.4110794, -32.2647896, 32.2647896
21: -33.5555725, 6.8148298, -33.5555725, 6.8148298, -40.3704033, 40.3704033
22: -38.2463188, 10.3630962, -38.2463188, 10.3630962, -48.6094131, 48.6094131
23: -27.6402435, 7.6619906, -27.6402435, 7.6619906, -35.3022346, 35.3022346
24: -30.9860477, 7.9161339, -30.9860477, 7.9161339, -38.9021835, 38.9021835
25: -28.3018532, 11.2306194, -28.3018532, 11.2306194, -39.5324707, 39.5324707
26: -43.3700829, 8.2398968, -43.3700829, 8.2398968, -51.6099777, 51.6099777
27: -30.0314846, 14.0805511, -30.0314846, 14.0805511, -44.1120377, 44.1120377
28: -27.4636269, 9.9790735, -27.4636269, 9.9790735, -37.4426994, 37.4426994
29: -39.8713531, 10.7618580, -39.8713531, 10.7618580, -50.6332092, 50.6332092
30: -28.1220894, 14.7383919, -28.1220894, 14.7383919, -42.8604813, 42.8604813
31: -31.2179165, 8.4970474, -31.2179165, 8.4970474, -39.7149658, 39.7149658
32: -30.9217873, 12.0875225, -30.9217873, 12.0875225, -43.0093079, 43.0093079
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.6681519, 57.6640167
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.0730133, 49.0698547
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5190887, 50.5168953
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7230988, 65.7202148
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0568924, 59.0532913
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0701714, 34.0641670

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.1970313, upper bound: 22.2468939
time: 71.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.1829084, upper bound: 22.2985929
time: 62.98 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 136.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.1778014, upper bound: 22.3302772
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.1258963, upper bound: 22.3817224
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.1778014, upper bound: 22.2732933
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.1258963, upper bound: 22.3250678
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.2305526, upper bound: 22.2524974
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.1258963, upper bound: 22.3042404
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.1778014, upper bound: 22.2186555
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.2332323, upper bound: 22.2705506
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.2427406, upper bound: 22.2531973
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.1909298, upper bound: 22.3048181
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.2732992, upper bound: 22.1948121
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.2215839, upper bound: 22.2465504
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.2985929, upper bound: 22.1829084
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.2468939, upper bound: 22.2348380
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.2732992, upper bound: 22.1450472
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.2993355, upper bound: 22.1970313
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.1970313, upper bound: 22.2993355
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.1450472, upper bound: 22.3508202
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.1970313, upper bound: 22.2468939
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.30
Output dim: 3, lower bound: -22.1829084, upper bound: 22.2985929
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 136.30
Output dim: 3, lower bound: -22.2180210, upper bound: 22.2946921
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 136.30
Output dim: 3, lower bound: -22.3263106, upper bound: 22.2640304
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 136.30
Output dim: 3, lower bound: -22.2918009, upper bound: 22.3064229
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 136.30
Output dim: 3, lower bound: -22.2918009, upper bound: 22.2517538
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 136.30
Output dim: 3, lower bound: -22.3465481, upper bound: 22.2336770
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 136.30
Output dim: 3, lower bound: -22.4037159, upper bound: 22.1988343

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 75.93 + 3649.53 = 3725.46 seconds
