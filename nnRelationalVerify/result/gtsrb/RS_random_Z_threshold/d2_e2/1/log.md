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
execution time: IAR + RelationalAnalysis = 2.82 + 70.64 = 73.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -22.4273933, upper bound: 22.4273933

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 596

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1612

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4192834, upper bound: 22.4040792
time: 58.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4040792, upper bound: 22.4192834
time: 74.92 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 133.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 133.72
Output dim: 3, lower bound: -22.4192834, upper bound: 22.4040792
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 133.72
Output dim: 3, lower bound: -22.4040792, upper bound: 22.4192834

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7768631, 42.7762260
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3071060, 43.3067207
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4645767, 61.4643707
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7265549, 57.7285156
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1225662, 49.1276817
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5764236, 50.5788651
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7248077, 65.7235031
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0631256, 59.0631256
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0689926, 34.0679970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1623

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4057277, upper bound: 22.4024797
time: 73.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4176804, upper bound: 22.3905311
time: 68.73 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7762222, 42.7768669
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3067245, 43.3071060
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4643631, 61.4645882
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7285233, 57.7265472
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1276855, 49.1225662
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5788651, 50.5764313
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7234955, 65.7248154
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0631256, 59.0631256
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0679932, 34.0689926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1686

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1706

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4032056, upper bound: 22.4159313
time: 64.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4006847, upper bound: 22.4184255
time: 58.02 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 124.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 124.46
Output dim: 3, lower bound: -22.4057277, upper bound: 22.4024797
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 124.46
Output dim: 3, lower bound: -22.4176804, upper bound: 22.3905311
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 124.46
Output dim: 3, lower bound: -22.4032056, upper bound: 22.4159313
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 124.46
Output dim: 3, lower bound: -22.4006847, upper bound: 22.4184255

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7764320, 42.7757988
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3054123, 43.3051910
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4531097, 61.4554558
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7296143, 57.7296982
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1251297, 49.1292496
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5796432, 50.5806122
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7200012, 65.7170258
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0678558, 59.0666733
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0696640, 34.0687027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1725

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 713

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3618158, upper bound: 22.4016383
time: 114.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.4048862, upper bound: 22.3583805
time: 76.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7764397, 42.7757874
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3055801, 43.3050308
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4556732, 61.4528847
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7277222, 57.7315826
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1241379, 49.1302414
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5781784, 50.5820770
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7183533, 65.7186890
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0666656, 59.0678482
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0696945, 34.0686722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1534

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 658

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3880020, upper bound: 22.3903294
time: 56.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4174786, upper bound: 22.3608240
time: 94.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7764244, 42.7770157
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3128319, 43.3131943
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4650650, 61.4651680
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7364044, 57.7343674
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1531219, 49.1479187
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5947266, 50.5923157
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7256088, 65.7274628
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0657120, 59.0658722
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0690460, 34.0699844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 582

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 773

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4030447, upper bound: 22.4116982
time: 57.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3989712, upper bound: 22.4157704
time: 68.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7763786, 42.7770691
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3128166, 43.3132095
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4649429, 61.4652748
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7363434, 57.7344360
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1530304, 49.1480103
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5947571, 50.5923004
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7261429, 65.7269287
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0658798, 59.0657120
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0689850, 34.0700417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1720

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3895194, upper bound: 22.3779547
time: 96.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3601025, upper bound: 22.4071520
time: 63.93 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 162.64 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 162.64
Output dim: 3, lower bound: -22.3618158, upper bound: 22.4016383
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 162.64
Output dim: 3, lower bound: -22.4048862, upper bound: 22.3583805
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 162.64
Output dim: 3, lower bound: -22.3880020, upper bound: 22.3903294
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 162.64
Output dim: 3, lower bound: -22.4174786, upper bound: 22.3608240
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 162.64
Output dim: 3, lower bound: -22.4030447, upper bound: 22.4116982
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 162.64
Output dim: 3, lower bound: -22.3989712, upper bound: 22.4157704
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 162.64
Output dim: 3, lower bound: -22.3895194, upper bound: 22.3779547
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 162.64
Output dim: 3, lower bound: -22.3601025, upper bound: 22.4071520

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7738380, 42.7712860
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3067627, 43.3062057
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4449768, 61.4417534
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7393494, 57.7414551
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1377563, 49.1414299
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5861969, 50.5893631
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7264481, 65.7279053
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0661469, 59.0655136
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0763245, 34.0734329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 643

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 627

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3723081, upper bound: 22.3595488
time: 64.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4161549, upper bound: 22.3156438
time: 60.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7739029, 42.7754555
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3130569, 43.3134613
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4672546, 61.4682274
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7375488, 57.7352371
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1535645, 49.1483002
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5961914, 50.5934219
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7277222, 65.7289047
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0654907, 59.0656509
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0630951, 34.0653992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1675

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 602

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3868554, upper bound: 22.4107625
time: 61.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3980394, upper bound: 22.3955119
time: 117.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7748642, 42.7744942
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3130875, 43.3134308
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4681091, 61.4673691
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7372742, 57.7355042
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1535034, 49.1483612
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5958405, 50.5937805
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7270508, 65.7295761
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0654907, 59.0656509
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0644608, 34.0640335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 718

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 563

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3917199, upper bound: 22.4128227
time: 66.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3960016, upper bound: 22.4085232
time: 62.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7744255, 42.7754669
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3126717, 43.3131714
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4583664, 61.4637184
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7351990, 57.7327271
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1525192, 49.1469345
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5946503, 50.5921135
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7241440, 65.7195740
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0647659, 59.0627670
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0667725, 34.0685425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1586

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 654

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3238932, upper bound: 22.4067029
time: 63.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3596524, upper bound: 22.3710411
time: 66.03 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 132.02 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 132.02
Output dim: 3, lower bound: -22.3723081, upper bound: 22.3595488
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 132.02
Output dim: 3, lower bound: -22.4161549, upper bound: 22.3156438
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 132.02
Output dim: 3, lower bound: -22.3868554, upper bound: 22.4107625
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 132.02
Output dim: 3, lower bound: -22.3980394, upper bound: 22.3955119
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 132.02
Output dim: 3, lower bound: -22.3917199, upper bound: 22.4128227
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 132.02
Output dim: 3, lower bound: -22.3960016, upper bound: 22.4085232
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 132.02
Output dim: 3, lower bound: -22.3238932, upper bound: 22.4067029
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 132.02
Output dim: 3, lower bound: -22.3596524, upper bound: 22.3710411

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7737312, 42.7710648
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3065643, 43.3066330
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4441986, 61.4396172
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7358246, 57.7410889
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1359558, 49.1408920
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5832672, 50.5889816
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7239456, 65.7276611
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0644226, 59.0651550
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0768127, 34.0733070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 655

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1568

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4159937, upper bound: 22.3112512
time: 63.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4117665, upper bound: 22.3154818
time: 58.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7633286, 42.7619743
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3145752, 43.3149796
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4705963, 61.4719162
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7402954, 57.7378769
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1572342, 49.1518288
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.6011658, 50.5986328
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7323456, 65.7333755
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0647736, 59.0647812
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0485306, 34.0468369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1305

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3089759, upper bound: 22.4091352
time: 60.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3852281, upper bound: 22.3329553
time: 58.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7745438, 42.7744255
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3133812, 43.3126144
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4666748, 61.4691544
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7360840, 57.7342529
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1493454, 49.1482010
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5941467, 50.5933685
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7301636, 65.7294769
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0656738, 59.0656128
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0643997, 34.0649490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1535

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 644

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3886903, upper bound: 22.4110448
time: 60.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3899414, upper bound: 22.4097974
time: 68.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7748642, 42.7741737
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3122826, 43.3134308
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4681091, 61.4659233
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7372742, 57.7343140
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1535034, 49.1442032
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5958405, 50.5920792
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7269440, 65.7295761
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0654449, 59.0656509
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0644608, 34.0639725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1304

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3870438, upper bound: 22.4076959
time: 61.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3951756, upper bound: 22.3995692
time: 77.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7709694, 42.7733116
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3041725, 43.3064423
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4390411, 61.4402008
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7315674, 57.7280579
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1434402, 49.1354523
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5872803, 50.5825882
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7235107, 65.7188492
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0662537, 59.0648346
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0675774, 34.0696945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1713

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1695

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3234760, upper bound: 22.3885000
time: 62.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3018207, upper bound: 22.4060478
time: 59.01 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 123.98 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 123.98
Output dim: 3, lower bound: -22.4159937, upper bound: 22.3112512
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 123.98
Output dim: 3, lower bound: -22.4117665, upper bound: 22.3154818
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 123.98
Output dim: 3, lower bound: -22.3089759, upper bound: 22.4091352
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 123.98
Output dim: 3, lower bound: -22.3852281, upper bound: 22.3329553
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 123.98
Output dim: 3, lower bound: -22.3886903, upper bound: 22.4110448
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 123.98
Output dim: 3, lower bound: -22.3899414, upper bound: 22.4097974
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 123.98
Output dim: 3, lower bound: -22.3870438, upper bound: 22.4076959
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 123.98
Output dim: 3, lower bound: -22.3951756, upper bound: 22.3995692
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 123.98
Output dim: 3, lower bound: -22.3234760, upper bound: 22.3885000
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 123.98
Output dim: 3, lower bound: -22.3018207, upper bound: 22.4060478

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7734337, 42.7701645
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3060532, 43.3064537
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4462433, 61.4396133
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7353439, 57.7406082
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1353302, 49.1381989
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5826416, 50.5882149
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7239609, 65.7295151
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0644226, 59.0651703
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0772552, 34.0724716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 513

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1622

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4135262, upper bound: 22.3088448
time: 58.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4135636, upper bound: 22.3088141
time: 66.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7728233, 42.7707672
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3063812, 43.3061218
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4441833, 61.4416656
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7353439, 57.7406082
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1332703, 49.1402588
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5825043, 50.5883598
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7258072, 65.7276611
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0644531, 59.0651550
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0759735, 34.0737419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 694

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 601

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4077798, upper bound: 22.3149159
time: 69.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4111958, upper bound: 22.3115033
time: 65.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7620926, 42.7606697
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3180504, 43.3179932
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4709702, 61.4722099
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7232819, 57.7215500
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1386719, 49.1344986
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5852280, 50.5833626
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7351913, 65.7378006
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0633163, 59.0642014
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0466690, 34.0449028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1697

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1691

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3032006, upper bound: 22.4075731
time: 64.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3073228, upper bound: 22.4075667
time: 64.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7575073, 42.7552071
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3145103, 43.3138580
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4684601, 61.4715652
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7337494, 57.7313538
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1396179, 49.1359978
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5944138, 50.5935860
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7392883, 65.7385254
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0634308, 59.0622711
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0585480, 34.0573044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 694

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 610

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3732463, upper bound: 22.4107207
time: 60.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3883654, upper bound: 22.3955953
time: 66.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7553253, 42.7573891
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3146172, 43.3137512
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4690704, 61.4709549
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7331848, 57.7319183
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1371384, 49.1384735
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5943680, 50.5936432
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7392120, 65.7386017
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0623322, 59.0633621
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0567551, 34.0590973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1568

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1781

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3888093, upper bound: 22.4086608
time: 65.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3888051, upper bound: 22.4086648
time: 61.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7747726, 42.7740097
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3113022, 43.3126755
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4643097, 61.4608727
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7375946, 57.7344971
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1535416, 49.1442184
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5958099, 50.5920753
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7274933, 65.7306595
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0659027, 59.0662842
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0643158, 34.0637970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 585

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3867822, upper bound: 22.4072817
time: 64.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3866322, upper bound: 22.4074351
time: 61.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7611847, 42.7661247
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3043594, 43.3066597
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4467697, 61.4480247
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7293396, 57.7251205
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1415482, 49.1328583
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5844116, 50.5788231
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7222290, 65.7173309
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0661850, 59.0649948
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0625076, 34.0685425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 513

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1297

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.2994568, upper bound: 22.4036895
time: 60.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2994568, upper bound: 22.4060360
time: 90.32 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 152.88 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 152.88
Output dim: 3, lower bound: -22.4135262, upper bound: 22.3088448
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 152.88
Output dim: 3, lower bound: -22.4135636, upper bound: 22.3088141
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 152.88
Output dim: 3, lower bound: -22.4077798, upper bound: 22.3149159
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 152.88
Output dim: 3, lower bound: -22.4111958, upper bound: 22.3115033
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 152.88
Output dim: 3, lower bound: -22.3032006, upper bound: 22.4075731
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 152.88
Output dim: 3, lower bound: -22.3073228, upper bound: 22.4075667
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 152.88
Output dim: 3, lower bound: -22.3732463, upper bound: 22.4107207
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 152.88
Output dim: 3, lower bound: -22.3883654, upper bound: 22.3955953
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 152.88
Output dim: 3, lower bound: -22.3888093, upper bound: 22.4086608
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 152.88
Output dim: 3, lower bound: -22.3888051, upper bound: 22.4086648
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 152.88
Output dim: 3, lower bound: -22.3867822, upper bound: 22.4072817
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 152.88
Output dim: 3, lower bound: -22.3866322, upper bound: 22.4074351
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 152.88
Output dim: 3, lower bound: -22.2994568, upper bound: 22.4036895
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 152.88
Output dim: 3, lower bound: -22.2994568, upper bound: 22.4060360

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7654839, 42.7684059
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3047714, 43.3052559
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4451141, 61.4344559
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7292404, 57.7382050
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1290817, 49.1339035
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5780640, 50.5850296
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7154007, 65.7275696
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0588074, 59.0639343
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0685806, 34.0711288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1780

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 761

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.4014921, upper bound: 22.3083860
time: 64.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4130623, upper bound: 22.2967531
time: 68.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7716789, 42.7622108
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3048553, 43.3051720
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4410858, 61.4384842
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7329330, 57.7345047
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1310272, 49.1319580
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.5794678, 50.5836334
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7220078, 65.7209625
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0631866, 59.0595398
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0759125, 34.0638008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 655

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3921187, upper bound: 22.3068970
time: 79.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4116833, upper bound: 22.2873198
time: 62.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.8758945, 12.5943375, -29.8758945, 12.5943375, -42.4702301, 42.4702301
1: -14.3982029, 20.7240562, -14.3982029, 20.7240562, -35.1222610, 35.1222610
2: -10.3812418, 21.0709000, -10.3812418, 21.0709000, -31.4521408, 31.4521408
3: -12.5759840, 23.7039757, -12.5759840, 23.7039757, -36.2799606, 36.2799606
4: -15.8018208, 20.6742897, -15.8018208, 20.6742897, -36.4761124, 36.4761124
5: -10.9025612, 25.4538078, -10.9025612, 25.4538078, -36.3563690, 36.3563690
6: -32.1202812, 11.5947609, -32.1202812, 11.5947609, -42.7416191, 42.7294273
7: -16.9680862, 26.3865700, -16.9680862, 26.3865700, -43.3198013, 43.3151512
8: -18.4997787, 23.7492523, -18.4997787, 23.7492523, -42.2490311, 42.2490311
9: -17.1065979, 20.6119938, -17.1065979, 20.6119938, -37.7185898, 37.7185898
10: -29.9906063, 29.0688438, -29.9906063, 29.0688438, -59.0594482, 59.0594482
11: -34.8727074, 15.3178368, -34.8727074, 15.3178368, -50.1905441, 50.1905441
12: -34.7312050, 13.5731440, -34.7312050, 13.5731440, -48.3043480, 48.3043480
13: -29.5469818, 22.9533596, -29.5469818, 22.9533596, -52.5003433, 52.5003433
14: -52.3690720, 10.4041643, -52.3690720, 10.4041643, -61.4120560, 61.4171524
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
33: -48.9752960, 9.3522329, -48.9752960, 9.3522329, -57.7479477, 57.7553329
34: -41.8310242, 7.6125965, -41.8310242, 7.6125965, -49.1631622, 49.1762352
35: -41.0994415, 9.6082096, -41.0994415, 9.6082096, -50.6095428, 50.6215973
36: -42.4455109, 9.9539757, -42.4455109, 9.9539757, -52.3994865, 52.3994865
37: -63.9058914, 2.1095924, -63.9058914, 2.1095924, -65.7278519, 65.7296753
38: -53.2482986, 12.1478243, -53.2482986, 12.1478243, -65.3961258, 65.3961258
39: -62.3482056, 5.9485617, -62.3482056, 5.9485617, -68.2967682, 68.2967682
40: -50.1519470, 9.3018999, -50.1519470, 9.3018999, -59.0543976, 59.0528183
41: -35.2922745, 6.7715178, -35.2922745, 6.7715178, -42.0637932, 42.0637932
42: -26.2480087, 7.8644753, -26.2480087, 7.8644753, -34.0483322, 34.0351715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=213, inp2_unstable=213, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1721

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 581

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3920148, upper bound: 22.3144842
time: 67.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4073564, upper bound: 22.2992632
time: 62.29 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 132.11 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 132.11
Output dim: 3, lower bound: -22.4014921, upper bound: 22.3083860
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 132.11
Output dim: 3, lower bound: -22.4130623, upper bound: 22.2967531
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 132.11
Output dim: 3, lower bound: -22.3921187, upper bound: 22.3068970
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 132.11
Output dim: 3, lower bound: -22.4116833, upper bound: 22.2873198
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 132.11
Output dim: 3, lower bound: -22.3920148, upper bound: 22.3144842
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 132.11
Output dim: 3, lower bound: -22.4073564, upper bound: 22.2992632
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 132.11
Output dim: 3, lower bound: -22.4111958, upper bound: 22.3115033
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 132.11
Output dim: 3, lower bound: -22.3032006, upper bound: 22.4075731
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 132.11
Output dim: 3, lower bound: -22.3073228, upper bound: 22.4075667
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 132.11
Output dim: 3, lower bound: -22.3732463, upper bound: 22.4107207
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 132.11
Output dim: 3, lower bound: -22.3888093, upper bound: 22.4086608
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 132.11
Output dim: 3, lower bound: -22.3888051, upper bound: 22.4086648
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 132.11
Output dim: 3, lower bound: -22.3867822, upper bound: 22.4072817
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 132.11
Output dim: 3, lower bound: -22.3866322, upper bound: 22.4074351
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 132.11
Output dim: 3, lower bound: -22.2994568, upper bound: 22.4060360

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 73.46 + 3597.73 = 3671.19 seconds
