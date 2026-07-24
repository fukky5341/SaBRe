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
execution time: IAR + RelationalAnalysis = 2.89 + 73.33 = 76.22 seconds
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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3793462, upper bound: 22.4102643
time: 71.87 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4102643, upper bound: 22.3793462
time: 76.84 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 148.88 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 148.88
Output dim: 3, lower bound: -22.3793462, upper bound: 22.4102643
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 148.88
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3109190, upper bound: 22.4080572
time: 59.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3771069, upper bound: 22.3308225
time: 70.21 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3308225, upper bound: 22.3771069
time: 98.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4080572, upper bound: 22.3109190
time: 64.04 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 164.75 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 164.75
Output dim: 3, lower bound: -22.3109190, upper bound: 22.4080572
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 164.75
Output dim: 3, lower bound: -22.3771069, upper bound: 22.3308225
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 164.75
Output dim: 3, lower bound: -22.3308225, upper bound: 22.3771069
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 164.75
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2377593, upper bound: 22.4063400
time: 114.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.2377593, upper bound: 22.3299455
time: 63.67 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3299456, upper bound: 22.3089433
time: 72.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4063401, upper bound: 22.2377593
time: 230.46 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 305.80 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 305.80
Output dim: 3, lower bound: -22.2377593, upper bound: 22.4063400
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 305.80
Output dim: 3, lower bound: -22.2377593, upper bound: 22.3299455
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 305.80
Output dim: 3, lower bound: -22.3299456, upper bound: 22.3089433
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 305.80
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.1988343, upper bound: 22.4037159
time: 66.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.1988343, upper bound: 22.3465481
time: 63.21 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3465481, upper bound: 22.2336770
time: 75.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.4037159, upper bound: 22.1988343
time: 54.72 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 132.70 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 132.70
Output dim: 3, lower bound: -22.1988343, upper bound: 22.4037159
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 132.70
Output dim: 3, lower bound: -22.1988343, upper bound: 22.3465481
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 132.70
Output dim: 3, lower bound: -22.3465481, upper bound: 22.2336770
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 132.70
Output dim: 3, lower bound: -22.4037159, upper bound: 22.1988343

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 76.22 + 1195.99 = 1272.21 seconds
