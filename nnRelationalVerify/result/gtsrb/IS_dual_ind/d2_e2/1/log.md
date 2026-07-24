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
execution time: IAR + RelationalAnalysis = 2.44 + 73.82 = 76.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -22.4273933, upper bound: 22.4273933

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1718

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4219063, upper bound: 22.3507611
time: 64.43 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4219063, upper bound: 22.4219060
time: 70.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 135.23 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 135.23
Output dim: 3, lower bound: -22.4219063, upper bound: 22.3507611
IS_A2, status: Status.UNKNOWN, split count: 1, time: 135.23
Output dim: 3, lower bound: -22.4219063, upper bound: 22.4219060

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -29.8531570, 12.5847292, -29.8690376, 12.5914154, -42.4445724, 42.4537659
1: -14.3877373, 20.6857338, -14.3950481, 20.7124863, -35.1002235, 35.0807800
2: -10.3719063, 21.0489674, -10.3783970, 21.0642834, -31.4361897, 31.4273643
3: -12.5653696, 23.6375637, -12.5727291, 23.6840153, -36.2493858, 36.2102928
4: -15.7924194, 20.6618290, -15.7989674, 20.6705322, -36.4629517, 36.4607964
5: -10.8924866, 25.4169312, -10.8994865, 25.4427204, -36.3352051, 36.3164177
6: -32.1066246, 11.5835886, -32.1161423, 11.5913887, -42.7561760, 42.7595329
7: -16.9542007, 26.3412933, -16.9638977, 26.3728561, -43.2793922, 43.2562180
8: -18.4904461, 23.7148628, -18.4969292, 23.7388763, -42.2293243, 42.2117920
9: -17.0843353, 20.6026917, -17.0998440, 20.6091995, -37.6935349, 37.7025375
10: -29.9755478, 29.0516205, -29.9860344, 29.0636463, -59.0391922, 59.0376549
11: -34.8558578, 15.2749634, -34.8675766, 15.3048830, -50.1607399, 50.1425400
12: -34.7148361, 13.5576143, -34.7262650, 13.5684061, -48.2832413, 48.2838783
13: -29.5270023, 22.9417725, -29.5409431, 22.9498730, -52.4768753, 52.4827156
14: -52.3476715, 10.3816690, -52.3625984, 10.3973598, -61.4355011, 61.4362946
15: -22.8442383, 19.0057983, -22.8534431, 19.0166702, -41.8609085, 41.8592415
16: -30.7831955, 25.6099663, -30.8022156, 25.6222630, -56.4054565, 56.4121819
17: -55.9785995, 21.6883450, -55.9937744, 21.7580833, -77.7366791, 77.6821213
18: -30.8060188, 14.2965908, -30.8240337, 14.3062210, -45.1122398, 45.1206245
19: -29.6978073, 3.0048671, -29.7166290, 3.0112591, -32.7090683, 32.7214966
20: -21.8410797, 10.4018011, -21.8498936, 10.4082842, -32.2493629, 32.2516937
21: -33.5405502, 6.8013458, -33.5510025, 6.8107681, -40.3513184, 40.3523483
22: -38.2297173, 10.3366871, -38.2412949, 10.3547735, -48.5844917, 48.5779800
23: -27.6282539, 7.6537180, -27.6366005, 7.6594930, -35.2877464, 35.2903175
24: -30.9700794, 7.9059525, -30.9811764, 7.9130440, -38.8831253, 38.8871307
25: -28.2909927, 11.2194424, -28.2985725, 11.2272205, -39.5182114, 39.5180130
26: -43.3537292, 8.2240438, -43.3651085, 8.2351007, -51.5888290, 51.5891533
27: -30.0194397, 14.0651827, -30.0278263, 14.0758238, -44.0952644, 44.0930099
28: -27.4526234, 9.9719725, -27.4602909, 9.9769001, -37.4295235, 37.4322624
29: -39.8540268, 10.7105331, -39.8661346, 10.7463799, -50.6004066, 50.5766678
30: -28.1094322, 14.7219124, -28.1182270, 14.7333632, -42.8427963, 42.8401413
31: -31.1911716, 8.4841270, -31.2098198, 8.4931602, -39.6843338, 39.6939468
32: -30.9090881, 12.0790987, -30.9179459, 12.0849733, -42.9940605, 42.9970436
33: -48.9139481, 9.3419762, -48.9568329, 9.3491535, -57.6637878, 57.6995087
34: -41.8103523, 7.6039495, -41.8248177, 7.6099825, -49.1039124, 49.1132965
35: -41.0614815, 9.6020222, -41.0879631, 9.6063147, -50.5380707, 50.5608673
36: -42.4133453, 9.9471960, -42.4358368, 9.9519320, -52.3652763, 52.3830338
37: -63.7925034, 2.1015863, -63.8718224, 2.1072054, -65.6081848, 65.6818008
38: -53.2115173, 12.1351147, -53.2372208, 12.1439877, -65.3555069, 65.3723373
39: -62.2482834, 5.9383602, -62.3181763, 5.9454975, -68.1937790, 68.2565384
40: -50.0640182, 9.2936468, -50.1255341, 9.2994127, -58.9715729, 59.0278397
41: -35.2561264, 6.7616892, -35.2813797, 6.7685194, -42.0246468, 42.0430679
42: -26.2362823, 7.8432035, -26.2444649, 7.8577623, -34.0485764, 34.0434494

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1777

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3759437, upper bound: 22.2240739
time: 64.14 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4194264, upper bound: 22.3482940
time: 55.99 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -29.9608746, 12.6013145, -29.8743477, 12.5934839, -42.5543594, 42.4756622
1: -14.4711485, 20.7276402, -14.3976154, 20.7221107, -35.1932602, 35.1252556
2: -10.4422646, 21.0742626, -10.3805208, 21.0695477, -31.5118122, 31.4547844
3: -12.6865778, 23.7106552, -12.5748339, 23.7012424, -36.3878212, 36.2854881
4: -15.8449297, 20.6970215, -15.8011456, 20.6732407, -36.5181694, 36.4981689
5: -11.0023689, 25.4606552, -10.9018784, 25.4522457, -36.4546127, 36.3625336
6: -32.1326332, 11.6182842, -32.1188736, 11.5940809, -42.7827873, 42.8206596
7: -17.0646858, 26.3878479, -16.9672089, 26.3846302, -43.4032593, 43.3072357
8: -18.5659294, 23.7583733, -18.4989281, 23.7473106, -42.3132401, 42.2573013
9: -17.1201649, 20.6450901, -17.1019211, 20.6113434, -37.7315063, 37.7470093
10: -30.0120010, 29.0893364, -29.9890022, 29.0674324, -59.0794334, 59.0783386
11: -34.9347878, 15.3249474, -34.8710594, 15.3143616, -50.2491493, 50.1960068
12: -34.7445793, 13.6504145, -34.7301826, 13.5722589, -48.3168373, 48.3805962
13: -29.5745468, 22.9757328, -29.5456886, 22.9521065, -52.5266533, 52.5214233
14: -52.4573708, 10.4062548, -52.3676071, 10.4013653, -61.5438538, 61.4648933
15: -22.8927555, 19.0325928, -22.8563099, 19.0193157, -41.9120712, 41.8889008
16: -30.8410530, 25.6392536, -30.8075371, 25.6264400, -56.4674911, 56.4467926
17: -56.1208115, 21.7922096, -55.9991989, 21.7840557, -77.9048691, 77.7914124
18: -30.8545837, 14.3744240, -30.8300686, 14.3089714, -45.1635551, 45.2044907
19: -29.7353859, 3.0366974, -29.7200203, 3.0130639, -32.7484512, 32.7567177
20: -21.8720036, 10.4213047, -21.8527431, 10.4104729, -32.2824783, 32.2740479
21: -33.5875854, 6.8303747, -33.5542755, 6.8137951, -40.4013824, 40.3846512
22: -38.2716827, 10.3723087, -38.2453003, 10.3607845, -48.6324692, 48.6176071
23: -27.6724205, 7.6734271, -27.6394501, 7.6611109, -35.3335304, 35.3128777
24: -30.9966583, 7.9290581, -30.9839401, 7.9152193, -38.9118767, 38.9129982
25: -28.3249283, 11.2485552, -28.3010960, 11.2294769, -39.5544052, 39.5496521
26: -43.3861809, 8.3003807, -43.3685684, 8.2387600, -51.6249390, 51.6689491
27: -30.0693798, 14.0956163, -30.0303040, 14.0792303, -44.1486092, 44.1259193
28: -27.4895725, 9.9973736, -27.4628868, 9.9782581, -37.4678307, 37.4602585
29: -39.9239388, 10.7707510, -39.8702927, 10.7596684, -50.6836090, 50.6410446
30: -28.1758289, 14.7519941, -28.1209450, 14.7366123, -42.9124413, 42.8729401
31: -31.2311268, 8.5259609, -31.2142582, 8.4960632, -39.7271881, 39.7402191
32: -30.9338322, 12.1155930, -30.9203949, 12.0870209, -43.0208511, 43.0359879
33: -48.9822693, 9.4435501, -48.9724350, 9.3512430, -57.7341309, 57.8200073
34: -41.8395844, 7.6451731, -41.8294144, 7.6119976, -49.1293411, 49.1648407
35: -41.1036148, 9.6557312, -41.0967178, 9.6075478, -50.5813980, 50.6263847
36: -42.4486313, 9.9866295, -42.4431114, 9.9535885, -52.4022217, 52.4297409
37: -63.9165039, 2.2559652, -63.9012451, 2.1090803, -65.7308807, 65.8668213
38: -53.2565384, 12.2060366, -53.2457390, 12.1470222, -65.4035645, 65.4517746
39: -62.3552246, 6.0757895, -62.3440170, 5.9478168, -68.3030396, 68.4198074
40: -50.1590652, 9.4149294, -50.1477051, 9.3013010, -59.0686264, 59.1753693
41: -35.2987442, 6.8272648, -35.2902374, 6.7705908, -42.0693359, 42.1175003
42: -26.2897797, 7.8846006, -26.2468872, 7.8623466, -34.0998001, 34.0897369

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1777

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3759437, upper bound: 22.2937689
time: 66.47 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3759437, upper bound: 22.2937689
time: 57.23 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 125.67 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 125.67
Output dim: 3, lower bound: -22.3759437, upper bound: 22.2240739
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 125.67
Output dim: 3, lower bound: -22.4194264, upper bound: 22.3482940
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 125.67
Output dim: 3, lower bound: -22.3759437, upper bound: 22.2937689
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 125.67
Output dim: 3, lower bound: -22.3759437, upper bound: 22.2937689

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -29.7987347, 12.5702286, -29.7048283, 12.5496769, -42.3484116, 42.2750549
1: -14.3532963, 20.6762333, -14.2938147, 20.6825733, -35.0358696, 34.9700470
2: -10.3129597, 21.0394669, -10.2037830, 21.0257568, -31.3387165, 31.2432499
3: -12.4967899, 23.6255131, -12.3681049, 23.6352959, -36.1320877, 35.9936180
4: -15.7305317, 20.6516209, -15.6170311, 20.6437607, -36.3742905, 36.2686539
5: -10.8300304, 25.4075356, -10.7125864, 25.4046135, -36.2346420, 36.1201210
6: -32.0892487, 11.5503483, -32.0873337, 11.4924927, -42.6241226, 42.6842155
7: -16.9075031, 26.3279381, -16.8273144, 26.3317986, -43.1858749, 43.1049500
8: -18.4347992, 23.7020721, -18.3344631, 23.6957531, -42.1305542, 42.0365372
9: -17.0485649, 20.5787411, -16.9942799, 20.5399990, -37.5885620, 37.5730209
10: -29.9523773, 28.9715271, -29.9217262, 28.8296852, -58.7820625, 58.8932533
11: -34.8357773, 15.1753483, -34.7970810, 15.0067844, -49.8425598, 49.9724274
12: -34.6989021, 13.4484568, -34.6588593, 13.2409639, -47.9398651, 48.1073151
13: -29.4749699, 22.9091797, -29.3825569, 22.8743286, -52.3492966, 52.2917366
14: -52.3065147, 10.2811308, -52.2190475, 10.0924950, -61.0880966, 61.1891479
15: -22.8004265, 18.9873810, -22.7208023, 18.9978085, -41.7982330, 41.7081833
16: -30.7551365, 25.5697765, -30.7457161, 25.5047264, -56.2598648, 56.3154907
17: -55.9484177, 21.5680256, -55.8761826, 21.3966331, -77.3450470, 77.4442062
18: -30.7803860, 14.2195244, -30.7539711, 14.0894032, -44.8697891, 44.9734955
19: -29.6809254, 2.9475479, -29.6517220, 2.8426728, -32.5235977, 32.5992699
20: -21.8237228, 10.3490028, -21.7836189, 10.2479992, -32.0717239, 32.1326218
21: -33.5189056, 6.7265654, -33.4678192, 6.5866451, -40.1055527, 40.1943855
22: -38.2079353, 10.2846537, -38.1743927, 10.2035675, -48.4115028, 48.4590454
23: -27.6136570, 7.6053271, -27.5845261, 7.5166883, -35.1303444, 35.1898537
24: -30.9543495, 7.8839827, -30.9317894, 7.8474331, -38.8017807, 38.8157730
25: -28.2756920, 11.1883793, -28.2529678, 11.1377335, -39.4134254, 39.4413452
26: -43.3339233, 8.1301861, -43.2810745, 7.9520130, -51.2859344, 51.4112625
27: -29.9920158, 14.0226135, -29.9635506, 13.9484653, -43.9404831, 43.9861641
28: -27.4364586, 9.9245596, -27.4006386, 9.8367119, -37.2731705, 37.3251991
29: -39.8346863, 10.6374063, -39.8048019, 10.5291348, -50.3638229, 50.4422073
30: -28.0922489, 14.6721191, -28.0692844, 14.5850964, -42.6773453, 42.7414017
31: -31.1685905, 8.4254799, -31.1290054, 8.3194637, -39.4880524, 39.5544853
32: -30.8892136, 12.0377932, -30.8733406, 11.9635220, -42.8527374, 42.9111328
33: -48.8364029, 9.3176327, -48.7215309, 9.2592697, -57.4882736, 57.4203110
34: -41.7705612, 7.5869265, -41.7042542, 7.5625725, -48.9965363, 48.9377823
35: -41.0139160, 9.5861626, -40.9462585, 9.5710087, -50.4284515, 50.3659973
36: -42.3887863, 9.9337912, -42.3636131, 9.9116907, -52.3004761, 52.2974052
37: -63.7708435, 2.0731277, -63.8098984, 2.0244637, -65.4881287, 65.5735016
38: -53.1811752, 12.1059828, -53.1569901, 12.0578766, -65.2390518, 65.2629700
39: -62.1808624, 5.9229488, -62.1173706, 5.8867455, -68.0676117, 68.0403214
40: -50.0318336, 9.2748680, -50.0411606, 9.2429104, -58.8734589, 58.9189835
41: -35.2371826, 6.7329283, -35.2402496, 6.6825953, -41.9197769, 41.9731789
42: -26.2222958, 7.7929115, -26.2104111, 7.7135248, -33.8628082, 33.9438972

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 729

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3103693, upper bound: 22.2183114
time: 62.78 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3748622, upper bound: 22.2229954
time: 74.30 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -29.8531570, 12.5847292, -29.8633461, 12.5894146, -42.4425735, 42.4480743
1: -14.3877373, 20.6857338, -14.3910370, 20.7111549, -35.0988922, 35.0767708
2: -10.3719063, 21.0489674, -10.3719406, 21.0631237, -31.4350300, 31.4209080
3: -12.5653696, 23.6375637, -12.5660915, 23.6822472, -36.2476158, 36.2036552
4: -15.7924194, 20.6618290, -15.7919159, 20.6688499, -36.4612694, 36.4537430
5: -10.8924866, 25.4169312, -10.8932190, 25.4414902, -36.3339767, 36.3101501
6: -32.1066246, 11.5835886, -32.1135597, 11.5758648, -42.7420006, 42.7589149
7: -16.9542007, 26.3412933, -16.9579353, 26.3647404, -43.2799454, 43.2500191
8: -18.4904461, 23.7148628, -18.4907360, 23.7369881, -42.2274323, 42.2055969
9: -17.0843353, 20.6026917, -17.0957355, 20.6047859, -37.6891212, 37.6984253
10: -29.9755478, 29.0516205, -29.9830666, 29.0540695, -59.0296173, 59.0346870
11: -34.8558578, 15.2749634, -34.8638000, 15.2956181, -50.1514740, 50.1387634
12: -34.7148361, 13.5576143, -34.7244720, 13.5582628, -48.2730980, 48.2820854
13: -29.5270023, 22.9417725, -29.5217705, 22.9459915, -52.4729919, 52.4635429
14: -52.3476715, 10.3816690, -52.3583374, 10.3891602, -61.4268036, 61.4363327
15: -22.8442383, 19.0057983, -22.8318748, 19.0132599, -41.8574982, 41.8376732
16: -30.7831955, 25.6099663, -30.7982521, 25.6017342, -56.3849297, 56.4082184
17: -55.9785995, 21.6883450, -55.9909058, 21.7465897, -77.7251892, 77.6792526
18: -30.8060188, 14.2965908, -30.8207283, 14.2988682, -45.1048889, 45.1173172
19: -29.6978073, 3.0048671, -29.7142010, 3.0055470, -32.7033539, 32.7190666
20: -21.8410797, 10.4018011, -21.8473091, 10.4038849, -32.2449646, 32.2491112
21: -33.5405502, 6.8013458, -33.5475693, 6.8042898, -40.3448410, 40.3489151
22: -38.2297173, 10.3366871, -38.2380981, 10.3491755, -48.5788918, 48.5747833
23: -27.6282539, 7.6537180, -27.6347885, 7.6546874, -35.2829399, 35.2885056
24: -30.9700794, 7.9059525, -30.9787731, 7.9111023, -38.8811798, 38.8847275
25: -28.2909927, 11.2194424, -28.2939034, 11.2230902, -39.5140839, 39.5133438
26: -43.3537292, 8.2240438, -43.3624649, 8.2266693, -51.5803986, 51.5865097
27: -30.0194397, 14.0651827, -30.0236855, 14.0725002, -44.0919418, 44.0888672
28: -27.4526234, 9.9719725, -27.4584427, 9.9720869, -37.4247093, 37.4304161
29: -39.8540268, 10.7105331, -39.8629227, 10.7394199, -50.5934448, 50.5734558
30: -28.1094322, 14.7219124, -28.1152115, 14.7285223, -42.8379555, 42.8371239
31: -31.1911716, 8.4841270, -31.2072449, 8.4875250, -39.6786957, 39.6913719
32: -30.9090881, 12.0790987, -30.9148788, 12.0792398, -42.9883270, 42.9939766
33: -48.9139481, 9.3419762, -48.9501648, 9.3461847, -57.6607361, 57.6669540
34: -41.8103523, 7.6039495, -41.8203278, 7.6073904, -49.0965500, 49.0909729
35: -41.0614815, 9.6020222, -41.0786362, 9.6042385, -50.5357819, 50.5297318
36: -42.4133453, 9.9471960, -42.4311333, 9.9504118, -52.3637581, 52.3783302
37: -63.7925034, 2.1015863, -63.8686638, 2.1001139, -65.5989075, 65.6852722
38: -53.2115173, 12.1351147, -53.2330093, 12.1380072, -65.3495255, 65.3681259
39: -62.2482834, 5.9383602, -62.3110390, 5.9435625, -68.1918488, 68.2493973
40: -50.0640182, 9.2936468, -50.1215210, 9.2860003, -58.9577637, 59.0222855
41: -35.2561264, 6.7616892, -35.2788811, 6.7560835, -42.0122108, 42.0405693
42: -26.2362823, 7.8432035, -26.2424965, 7.8490543, -34.0422401, 34.0508423

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2937692, upper bound: 22.3060532
time: 69.10 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2937692, upper bound: 22.3482947
time: 50.36 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -29.9064636, 12.5867996, -29.7101288, 12.5517063, -42.4581680, 42.2969284
1: -14.4367113, 20.7181702, -14.2963667, 20.6922073, -35.1289177, 35.0145378
2: -10.3833065, 21.0647850, -10.2058973, 21.0310326, -31.4143391, 31.2706833
3: -12.6180038, 23.6986351, -12.3702126, 23.6525230, -36.2705269, 36.0688477
4: -15.7830315, 20.6868019, -15.6192112, 20.6464672, -36.4294968, 36.3060150
5: -10.9398746, 25.4512787, -10.7149916, 25.4141388, -36.3540115, 36.1662712
6: -32.1152534, 11.5850277, -32.0900497, 11.4951639, -42.6506805, 42.7453117
7: -17.0179939, 26.3744965, -16.8305950, 26.3435783, -43.3097000, 43.1559372
8: -18.5102997, 23.7455826, -18.3364410, 23.7041969, -42.2144966, 42.0820236
9: -17.0844364, 20.6211281, -16.9963474, 20.5421524, -37.6265869, 37.6174774
10: -29.9888096, 29.0092506, -29.9246941, 28.8334732, -58.8222809, 58.9339447
11: -34.9147186, 15.2253790, -34.8005943, 15.0162067, -49.9309235, 50.0259743
12: -34.7286301, 13.5412350, -34.6627502, 13.2448292, -47.9734573, 48.2039871
13: -29.5224724, 22.9431458, -29.3872738, 22.8765621, -52.3990326, 52.3304214
14: -52.4161491, 10.3056946, -52.2240067, 10.0965338, -61.1963272, 61.2177544
15: -22.8489571, 19.0141983, -22.7236919, 19.0004559, -41.8494110, 41.7378922
16: -30.8130093, 25.5990562, -30.7510509, 25.5088749, -56.3218842, 56.3501053
17: -56.0906105, 21.6719341, -55.8816338, 21.4225922, -77.5131989, 77.5535660
18: -30.8289700, 14.2973423, -30.7600193, 14.0921364, -44.9211044, 45.0573616
19: -29.7185040, 2.9793768, -29.6551094, 2.8444829, -32.5629883, 32.6344872
20: -21.8545761, 10.3685074, -21.7864437, 10.2501793, -32.1047554, 32.1549530
21: -33.5659065, 6.7556057, -33.4710922, 6.5896606, -40.1555672, 40.2266998
22: -38.2499084, 10.3202677, -38.1783371, 10.2095757, -48.4594841, 48.4986038
23: -27.6578007, 7.6250591, -27.5873451, 7.5183063, -35.1761055, 35.2124023
24: -30.9809227, 7.9071321, -30.9345474, 7.8495846, -38.8305054, 38.8416786
25: -28.3095837, 11.2174664, -28.2554855, 11.1399908, -39.4495735, 39.4729538
26: -43.3663483, 8.2064257, -43.2845154, 7.9556618, -51.3220100, 51.4909401
27: -30.0419502, 14.0530643, -29.9660149, 13.9518919, -43.9938431, 44.0190811
28: -27.4733772, 9.9499407, -27.4032059, 9.8380814, -37.3114586, 37.3531456
29: -39.9046402, 10.6976366, -39.8089676, 10.5424414, -50.4470825, 50.5066032
30: -28.1586246, 14.7021751, -28.0720139, 14.5883350, -42.7469597, 42.7741890
31: -31.2085228, 8.4673119, -31.1334057, 8.3223524, -39.5308762, 39.6007156
32: -30.9139309, 12.0742970, -30.8757801, 11.9655609, -42.8794937, 42.9500771
33: -48.9046783, 9.4192095, -48.7371216, 9.2613230, -57.5585709, 57.5408401
34: -41.7997742, 7.6281681, -41.7088623, 7.5646133, -49.0219955, 48.9892883
35: -41.0560532, 9.6398525, -40.9549904, 9.5722504, -50.4718170, 50.4314575
36: -42.4240646, 9.9732170, -42.3708839, 9.9133244, -52.3373871, 52.3441010
37: -63.8948288, 2.2274618, -63.8393364, 2.0263596, -65.6108170, 65.7585449
38: -53.2262344, 12.1769094, -53.1654892, 12.0609322, -65.2871704, 65.3423996
39: -62.2878265, 6.0603390, -62.1432571, 5.8890486, -68.1768723, 68.2035980
40: -50.1268501, 9.3961563, -50.0632896, 9.2447939, -58.9704895, 59.0664520
41: -35.2798233, 6.7984877, -35.2490883, 6.6846952, -41.9645195, 42.0475769
42: -26.2757721, 7.8343401, -26.2128448, 7.7181635, -33.9140549, 33.9901886

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 729

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3103693, upper bound: 22.2878530
time: 68.70 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3748622, upper bound: 22.2926891
time: 70.93 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -29.9608746, 12.6013145, -29.8686619, 12.5914717, -42.5523453, 42.4699783
1: -14.4711485, 20.7276402, -14.3936081, 20.7208118, -35.1919594, 35.1212463
2: -10.4422646, 21.0742626, -10.3740711, 21.0683823, -31.5106468, 31.4483337
3: -12.6865778, 23.7106552, -12.5681925, 23.6994705, -36.3860474, 36.2788467
4: -15.8449297, 20.6970215, -15.7940893, 20.6715622, -36.5164909, 36.4911118
5: -11.0023689, 25.4606552, -10.8956089, 25.4510155, -36.4533844, 36.3562622
6: -32.1326332, 11.6182842, -32.1162796, 11.5785398, -42.7686195, 42.8200264
7: -17.0646858, 26.3878479, -16.9612122, 26.3765125, -43.4038086, 43.3010139
8: -18.5659294, 23.7583733, -18.4926987, 23.7454109, -42.3113403, 42.2510719
9: -17.1201649, 20.6450901, -17.0978394, 20.6069202, -37.7270851, 37.7429276
10: -30.0120010, 29.0893364, -29.9860306, 29.0578537, -59.0698547, 59.0753670
11: -34.9347878, 15.3249474, -34.8673248, 15.3050699, -50.2398567, 50.1922722
12: -34.7445793, 13.6504145, -34.7283783, 13.5621319, -48.3067093, 48.3787918
13: -29.5745468, 22.9757328, -29.5265160, 22.9482288, -52.5227737, 52.5022507
14: -52.4573708, 10.4062548, -52.3633575, 10.3931932, -61.5351715, 61.4649162
15: -22.8927555, 19.0325928, -22.8347340, 19.0159359, -41.9086914, 41.8673248
16: -30.8410530, 25.6392536, -30.8035927, 25.6058846, -56.4469376, 56.4428482
17: -56.1208115, 21.7922096, -55.9963264, 21.7725315, -77.8933411, 77.7885361
18: -30.8545837, 14.3744240, -30.8267612, 14.3016071, -45.1561890, 45.2011871
19: -29.7353859, 3.0366974, -29.7176018, 3.0073538, -32.7427406, 32.7542992
20: -21.8720036, 10.4213047, -21.8501682, 10.4060678, -32.2780724, 32.2714729
21: -33.5875854, 6.8303747, -33.5508347, 6.8072901, -40.3948746, 40.3812103
22: -38.2716827, 10.3723087, -38.2420502, 10.3551884, -48.6268692, 48.6143570
23: -27.6724205, 7.6734271, -27.6376152, 7.6563134, -35.3287354, 35.3110428
24: -30.9966583, 7.9290581, -30.9815273, 7.9132662, -38.9099236, 38.9105835
25: -28.3249283, 11.2485552, -28.2964325, 11.2253571, -39.5502853, 39.5449867
26: -43.3861809, 8.3003807, -43.3659058, 8.2303057, -51.6164856, 51.6662865
27: -30.0693798, 14.0956163, -30.0261726, 14.0759144, -44.1452942, 44.1217880
28: -27.4895725, 9.9973736, -27.4610443, 9.9734497, -37.4630203, 37.4584198
29: -39.9239388, 10.7707510, -39.8670654, 10.7527189, -50.6766586, 50.6378174
30: -28.1758289, 14.7519941, -28.1179218, 14.7317600, -42.9075890, 42.8699150
31: -31.2311268, 8.5259609, -31.2116852, 8.4904137, -39.7215424, 39.7376480
32: -30.9338322, 12.1155930, -30.9173412, 12.0812731, -43.0151062, 43.0329361
33: -48.9822693, 9.4435501, -48.9657707, 9.3482580, -57.7310791, 57.7874680
34: -41.8395844, 7.6451731, -41.8249359, 7.6094494, -49.1219711, 49.1425323
35: -41.1036148, 9.6557312, -41.0874062, 9.6054516, -50.5791321, 50.5952339
36: -42.4486313, 9.9866295, -42.4383850, 9.9520683, -52.4006996, 52.4250145
37: -63.9165039, 2.2559652, -63.8980560, 2.1020155, -65.7215805, 65.8702927
38: -53.2565384, 12.2060366, -53.2415199, 12.1410742, -65.3976135, 65.4475555
39: -62.3552246, 6.0757895, -62.3369141, 5.9458475, -68.3010712, 68.4127045
40: -50.1590652, 9.4149294, -50.1436653, 9.2879276, -59.0548019, 59.1697998
41: -35.2987442, 6.8272648, -35.2877426, 6.7581730, -42.0569153, 42.1150055
42: -26.2897797, 7.8846006, -26.2449284, 7.8536339, -34.0934639, 34.0971298

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2937692, upper bound: 22.3759434
time: 88.80 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2937692, upper bound: 22.4194261
time: 53.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 143.93 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 143.93
Output dim: 3, lower bound: -22.3103693, upper bound: 22.2183114
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 143.93
Output dim: 3, lower bound: -22.3748622, upper bound: 22.2229954
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 143.93
Output dim: 3, lower bound: -22.2937692, upper bound: 22.3060532
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 143.93
Output dim: 3, lower bound: -22.2937692, upper bound: 22.3482947
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 143.93
Output dim: 3, lower bound: -22.3103693, upper bound: 22.2878530
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 143.93
Output dim: 3, lower bound: -22.3748622, upper bound: 22.2926891
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 143.93
Output dim: 3, lower bound: -22.2937692, upper bound: 22.3759434
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 143.93
Output dim: 3, lower bound: -22.2937692, upper bound: 22.4194261

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -29.7087059, 12.5548115, -29.6764870, 12.5446625, -42.2533684, 42.2313004
1: -14.2727089, 20.6626854, -14.2683439, 20.6782379, -34.9509468, 34.9310303
2: -10.2301941, 21.0311546, -10.1777821, 21.0230980, -31.2532921, 31.2089367
3: -12.4201307, 23.6150932, -12.3440256, 23.6319351, -36.0520668, 35.9591179
4: -15.6708431, 20.6420021, -15.5982676, 20.6406860, -36.3115311, 36.2402687
5: -10.7595491, 25.4013290, -10.6904831, 25.4025860, -36.1621361, 36.0918121
6: -32.0741959, 11.5254288, -32.0822792, 11.4846191, -42.5981064, 42.6511917
7: -16.7990952, 26.3137951, -16.7925663, 26.3272514, -43.0737915, 43.0241051
8: -18.3047314, 23.6870079, -18.2935982, 23.6909733, -41.9957047, 41.9806061
9: -17.0239620, 20.5544968, -16.9864883, 20.5319519, -37.5559158, 37.5409851
10: -29.9295158, 28.9368591, -29.9144230, 28.8188343, -58.7483521, 58.8512802
11: -34.8095093, 15.1501169, -34.7882195, 14.9987307, -49.8082390, 49.9383354
12: -34.6880341, 13.3834696, -34.6553040, 13.2205391, -47.9085732, 48.0387726
13: -29.4533043, 22.8819656, -29.3756447, 22.8655815, -52.3188858, 52.2576103
14: -52.2011223, 10.2658701, -52.1858406, 10.0876942, -60.9731293, 61.0946999
15: -22.7492218, 18.9295120, -22.7045670, 18.9783115, -41.7275314, 41.6340790
16: -30.7298965, 25.5491066, -30.7375031, 25.4981098, -56.2280045, 56.2866096
17: -55.9153900, 21.5462475, -55.8656960, 21.3896446, -77.3050385, 77.4119415
18: -30.7621555, 14.1506233, -30.7481689, 14.0677567, -44.8299103, 44.8987923
19: -29.6656170, 2.8983202, -29.6468430, 2.8271937, -32.4928093, 32.5451622
20: -21.8063278, 10.3315048, -21.7779770, 10.2423000, -32.0486298, 32.1094818
21: -33.4971542, 6.7047529, -33.4608612, 6.5797567, -40.0769119, 40.1656151
22: -38.1774330, 10.2298765, -38.1646576, 10.1852827, -48.3627167, 48.3945351
23: -27.5986748, 7.5749722, -27.5797310, 7.5071034, -35.1057777, 35.1547012
24: -30.9388943, 7.8582764, -30.9268589, 7.8391323, -38.7780266, 38.7851334
25: -28.2610970, 11.1230440, -28.2483234, 11.1166916, -39.3777885, 39.3713684
26: -43.3080559, 8.0314436, -43.2727585, 7.9206676, -51.2287216, 51.3042030
27: -29.9645042, 14.0100346, -29.9543648, 13.9444714, -43.9089737, 43.9644012
28: -27.4234238, 9.8813086, -27.3964806, 9.8229284, -37.2463531, 37.2777901
29: -39.8032341, 10.6029987, -39.7948074, 10.5181789, -50.3214111, 50.3978043
30: -28.0722980, 14.6468449, -28.0628300, 14.5771151, -42.6494141, 42.7096748
31: -31.1427326, 8.3656883, -31.1207314, 8.3006039, -39.4433365, 39.4864197
32: -30.8673172, 12.0177336, -30.8652878, 11.9572210, -42.8245392, 42.8830223
33: -48.8201065, 9.2355242, -48.7163696, 9.2332821, -57.3763733, 57.3304138
34: -41.7581558, 7.5345860, -41.7002945, 7.5460939, -48.8664856, 48.8816681
35: -41.0014687, 9.5320177, -40.9423065, 9.5535755, -50.3847351, 50.3106689
36: -42.3744049, 9.8707180, -42.3590469, 9.8914118, -52.2658157, 52.2297668
37: -63.7500038, 1.9670706, -63.8033218, 1.9911661, -65.3500290, 65.4586716
38: -53.1655426, 12.0688086, -53.1519966, 12.0460958, -65.2116394, 65.2208023
39: -62.1659851, 5.8839426, -62.1126862, 5.8732080, -68.0391922, 67.9966278
40: -50.0110703, 9.2259378, -50.0344772, 9.2274761, -58.8007202, 58.8620377
41: -35.2244377, 6.7042732, -35.2360535, 6.6735392, -41.8979759, 41.9403267
42: -26.2108078, 7.7697830, -26.2065907, 7.7062120, -33.8417664, 33.9113121

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1777

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2843038, upper bound: 22.2183114
time: 72.75 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2843038, upper bound: 22.2183114
time: 45.21 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -29.8275433, 12.5866652, -29.7013702, 12.5480671, -42.3756104, 42.2880363
1: -14.3563347, 20.7217236, -14.2914762, 20.6815376, -35.0378723, 35.0131989
2: -10.3151913, 21.1019688, -10.2016621, 21.0248623, -31.3400536, 31.3036308
3: -12.4957848, 23.6832657, -12.3659325, 23.6341877, -36.1299744, 36.0491982
4: -15.7325172, 20.6797962, -15.6153212, 20.6427650, -36.3752823, 36.2951164
5: -10.8309307, 25.4392624, -10.7104282, 25.4037819, -36.2347107, 36.1496887
6: -32.0879402, 11.5522785, -32.0811729, 11.4911957, -42.6243248, 42.6794281
7: -16.9032593, 26.3879089, -16.8233452, 26.3304539, -43.1778412, 43.1615219
8: -18.4333611, 23.8060398, -18.3312340, 23.6945858, -42.1279449, 42.1372757
9: -17.0515175, 20.5800514, -16.9932480, 20.5373478, -37.5888672, 37.5732994
10: -29.9646587, 28.9765415, -29.9202461, 28.8282051, -58.7928619, 58.8967896
11: -34.8408394, 15.1718950, -34.7901230, 15.0049887, -49.8458290, 49.9620171
12: -34.7233696, 13.4597960, -34.6577187, 13.2385998, -47.9619675, 48.1175156
13: -29.4694118, 22.9056816, -29.3815269, 22.8649521, -52.3343658, 52.2872086
14: -52.3287354, 10.3585787, -52.2158813, 10.0915022, -61.0953522, 61.2627525
15: -22.7892265, 18.9770279, -22.7184830, 18.9898357, -41.7790604, 41.6955109
16: -30.7588501, 25.5656166, -30.7369995, 25.5036049, -56.2624550, 56.3026161
17: -55.9721756, 21.5736961, -55.8743706, 21.3942509, -77.3664246, 77.4480667
18: -30.8460007, 14.2195072, -30.7527256, 14.0874310, -44.9334335, 44.9722328
19: -29.7253418, 2.9480410, -29.6501713, 2.8413830, -32.5667267, 32.5982132
20: -21.8374023, 10.3557243, -21.7821350, 10.2466564, -32.0840607, 32.1378593
21: -33.5436363, 6.7284527, -33.4661407, 6.5858097, -40.1294479, 40.1945953
22: -38.2637939, 10.2678661, -38.1726151, 10.1967869, -48.4605789, 48.4404831
23: -27.6390762, 7.6095076, -27.5833092, 7.5155602, -35.1546364, 35.1928177
24: -31.0195465, 7.8831434, -30.9308128, 7.8457718, -38.8653183, 38.8139572
25: -28.3429337, 11.1803064, -28.2517166, 11.1339846, -39.4769173, 39.4320221
26: -43.4164467, 8.1330872, -43.2793732, 7.9483328, -51.3647804, 51.4124603
27: -29.9945889, 14.0418549, -29.9557438, 13.9478579, -43.9424477, 43.9975967
28: -27.4720249, 9.9302540, -27.3995056, 9.8354959, -37.3075218, 37.3297577
29: -39.8859291, 10.6319857, -39.8029594, 10.5258732, -50.4118042, 50.4349442
30: -28.1328716, 14.6783552, -28.0675755, 14.5837612, -42.7166328, 42.7459297
31: -31.2221241, 8.4259834, -31.1269302, 8.3178310, -39.5399551, 39.5529137
32: -30.8866520, 12.0678997, -30.8671837, 11.9626732, -42.8493271, 42.9350815
33: -48.8991013, 9.3164206, -48.7202530, 9.2564220, -57.5262909, 57.4106369
34: -41.8254318, 7.5909338, -41.7032852, 7.5610666, -49.0502548, 48.9312363
35: -41.0800209, 9.5845413, -40.9449768, 9.5688171, -50.4911194, 50.3566742
36: -42.4310799, 9.9340725, -42.3624916, 9.9100752, -52.3411560, 52.2965622
37: -63.8845024, 2.0707388, -63.8081055, 2.0217333, -65.5977783, 65.5571594
38: -53.2060509, 12.1155405, -53.1557121, 12.0564823, -65.2625351, 65.2712555
39: -62.2153397, 5.9217787, -62.1159668, 5.8821621, -68.0975037, 68.0377426
40: -50.0578423, 9.2825413, -50.0389557, 9.2413788, -58.8973694, 58.9201965
41: -35.2610245, 6.7432022, -35.2382202, 6.6813736, -41.9423981, 41.9814224
42: -26.2189846, 7.7986393, -26.2077770, 7.7121801, -33.8704987, 33.9378586

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1777

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2843038, upper bound: 22.2229954
time: 56.77 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2843038, upper bound: 22.2229954
time: 69.01 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -29.6889305, 12.5430145, -29.8633461, 12.5894146, -42.2783432, 42.4063606
1: -14.2865028, 20.6557674, -14.3910370, 20.7111549, -34.9976578, 35.0468063
2: -10.1972971, 21.0104370, -10.3719406, 21.0631237, -31.2604218, 31.3823776
3: -12.3607359, 23.5887947, -12.5660915, 23.6822472, -36.0429840, 36.1548843
4: -15.6104794, 20.6351109, -15.7919159, 20.6688499, -36.2793274, 36.4270248
5: -10.7056046, 25.3787994, -10.8932190, 25.4414902, -36.1470947, 36.2720184
6: -32.0778275, 11.4847021, -32.1135597, 11.5758648, -42.7063408, 42.6428375
7: -16.8176479, 26.3002090, -16.9579353, 26.3647404, -43.1337051, 43.2070541
8: -18.3279762, 23.6717186, -18.4907360, 23.7369881, -42.0649643, 42.1624527
9: -16.9787197, 20.5334969, -17.0957355, 20.6047859, -37.5835037, 37.6292343
10: -29.9112358, 28.8176498, -29.9830666, 29.0540695, -58.9653053, 58.8007164
11: -34.7852936, 14.9767513, -34.8638000, 15.2956181, -50.0809097, 49.8405533
12: -34.6474380, 13.2301474, -34.7244720, 13.5582628, -48.2056999, 47.9546204
13: -29.3686714, 22.8662262, -29.5217705, 22.9459915, -52.3146629, 52.3879967
14: -52.2042847, 10.0767975, -52.3583374, 10.3891602, -61.2829895, 61.1237106
15: -22.7116184, 18.9869347, -22.8318748, 19.0132599, -41.7248764, 41.8188095
16: -30.7267284, 25.4924126, -30.7982521, 25.6017342, -56.3284607, 56.2906647
17: -55.8609962, 21.3267899, -55.9909058, 21.7465897, -77.6075897, 77.3176956
18: -30.7359009, 14.0797386, -30.8207283, 14.2988682, -45.0347672, 44.9004669
19: -29.6329117, 2.8362851, -29.7142010, 3.0055470, -32.6384583, 32.5504875
20: -21.7748833, 10.2415190, -21.8473091, 10.4038849, -32.1787682, 32.0888290
21: -33.4573631, 6.5771923, -33.5475693, 6.8042898, -40.2616539, 40.1247635
22: -38.1628075, 10.1854382, -38.2380981, 10.3491755, -48.5119820, 48.4235382
23: -27.5762253, 7.5108995, -27.6347885, 7.6546874, -35.2309113, 35.1456871
24: -30.9206753, 7.8403187, -30.9787731, 7.9111023, -38.8317795, 38.8190918
25: -28.2454262, 11.1299782, -28.2939034, 11.2230902, -39.4685173, 39.4238815
26: -43.2696877, 7.9409266, -43.3624649, 8.2266693, -51.4963570, 51.3033905
27: -29.9551678, 13.9378242, -30.0236855, 14.0725002, -44.0276680, 43.9615097
28: -27.3930225, 9.8317814, -27.4584427, 9.9720869, -37.3651085, 37.2902222
29: -39.7926712, 10.4932718, -39.8629227, 10.7394199, -50.5320892, 50.3561935
30: -28.0605125, 14.5735912, -28.1152115, 14.7285223, -42.7890358, 42.6888046
31: -31.1104240, 8.3104401, -31.2072449, 8.4875250, -39.5979500, 39.5176849
32: -30.8645229, 11.9576311, -30.9148788, 12.0792398, -42.9437637, 42.8725090
33: -48.6786385, 9.2520819, -48.9501648, 9.3461847, -57.4059143, 57.6009445
34: -41.6898041, 7.5565662, -41.8203278, 7.6073904, -48.9407120, 49.0533257
35: -40.9197655, 9.5667152, -41.0786362, 9.6042385, -50.3563232, 50.5052109
36: -42.3411484, 9.9069452, -42.4311333, 9.9504118, -52.2915611, 52.3380775
37: -63.7305374, 2.0188427, -63.8686638, 2.1001139, -65.5269012, 65.5825272
38: -53.1312180, 12.0490494, -53.2330093, 12.1380072, -65.2692261, 65.2820587
39: -62.0474930, 5.8796501, -62.3110390, 5.9435625, -67.9910583, 68.1906891
40: -49.9796371, 9.2371655, -50.1215210, 9.2860003, -58.8711395, 58.9602509
41: -35.2150116, 6.6757889, -35.2788811, 6.7560835, -41.9710960, 41.9546700
42: -26.2022324, 7.6988707, -26.2424965, 7.8490543, -34.0046959, 33.8695602

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2679523, upper bound: 22.3060526
time: 67.13 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2679523, upper bound: 22.3060526
time: 61.85 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -29.8474712, 12.5827332, -29.8633461, 12.5894146, -42.4368858, 42.4460793
1: -14.3837271, 20.6844063, -14.3910370, 20.7111549, -35.0948830, 35.0754433
2: -10.3654575, 21.0478249, -10.3719406, 21.0631237, -31.4285812, 31.4197655
3: -12.5587091, 23.6357841, -12.5660915, 23.6822472, -36.2409554, 36.2018738
4: -15.7853527, 20.6601410, -15.7919159, 20.6688499, -36.4542007, 36.4520569
5: -10.8862123, 25.4157028, -10.8932190, 25.4414902, -36.3277016, 36.3089218
6: -32.1040192, 11.5680323, -32.1135597, 11.5758648, -42.7414055, 42.7447433
7: -16.9482098, 26.3331604, -16.9579353, 26.3647404, -43.2737198, 43.2505760
8: -18.4842262, 23.7129650, -18.4907360, 23.7369881, -42.2212143, 42.2037010
9: -17.0802193, 20.5982628, -17.0957355, 20.6047859, -37.6850052, 37.6940002
10: -29.9725952, 29.0420303, -29.9830666, 29.0540695, -59.0266647, 59.0250969
11: -34.8520737, 15.2656498, -34.8638000, 15.2956181, -50.1476898, 50.1294479
12: -34.7130203, 13.5474415, -34.7244720, 13.5582628, -48.2712822, 48.2719116
13: -29.5078506, 22.9378853, -29.5217705, 22.9459915, -52.4538422, 52.4596558
14: -52.3434143, 10.3734903, -52.3583374, 10.3891602, -61.4268188, 61.4276352
15: -22.8226738, 19.0024128, -22.8318748, 19.0132599, -41.8359337, 41.8342896
16: -30.7792397, 25.5894318, -30.7982521, 25.6017342, -56.3809738, 56.3876839
17: -55.9757500, 21.6767654, -55.9909058, 21.7465897, -77.7223358, 77.6676712
18: -30.8027344, 14.2892399, -30.8207283, 14.2988682, -45.1016006, 45.1099701
19: -29.6953907, 2.9991498, -29.7142010, 3.0055470, -32.7009392, 32.7133522
20: -21.8384991, 10.3974142, -21.8473091, 10.4038849, -32.2423859, 32.2447243
21: -33.5371246, 6.7948475, -33.5475693, 6.8042898, -40.3414154, 40.3424149
22: -38.2265015, 10.3310499, -38.2380981, 10.3491755, -48.5756760, 48.5691490
23: -27.6264267, 7.6489158, -27.6347885, 7.6546874, -35.2811127, 35.2837029
24: -30.9676781, 7.9039993, -30.9787731, 7.9111023, -38.8787804, 38.8827744
25: -28.2863312, 11.2153244, -28.2939034, 11.2230902, -39.5094223, 39.5092278
26: -43.3510780, 8.2155657, -43.3624649, 8.2266693, -51.5777473, 51.5780296
27: -30.0153179, 14.0618658, -30.0236855, 14.0725002, -44.0878181, 44.0855522
28: -27.4507790, 9.9671583, -27.4584427, 9.9720869, -37.4228668, 37.4256020
29: -39.8507805, 10.7035627, -39.8629227, 10.7394199, -50.5902023, 50.5664864
30: -28.1064053, 14.7170639, -28.1152115, 14.7285223, -42.8349266, 42.8322754
31: -31.1886024, 8.4784746, -31.2072449, 8.4875250, -39.6761284, 39.6857185
32: -30.9060211, 12.0733776, -30.9148788, 12.0792398, -42.9852600, 42.9882584
33: -48.9072685, 9.3389435, -48.9501648, 9.3461847, -57.6281738, 57.6638565
34: -41.8058510, 7.6013613, -41.8203278, 7.6073904, -49.0751190, 49.0845375
35: -41.0521393, 9.5999432, -41.0786362, 9.6042385, -50.5046539, 50.5274734
36: -42.4086456, 9.9456606, -42.4311333, 9.9504118, -52.3590584, 52.3767929
37: -63.7893028, 2.0945759, -63.8686638, 2.1001139, -65.6032257, 65.6768646
38: -53.2072983, 12.1291456, -53.2330093, 12.1380072, -65.3453064, 65.3621521
39: -62.2411499, 5.9364061, -62.3110390, 5.9435625, -68.1847153, 68.2474442
40: -50.0600281, 9.2802839, -50.1215210, 9.2860003, -58.9521866, 59.0084915
41: -35.2536240, 6.7492428, -35.2788811, 6.7560835, -42.0097084, 42.0281219
42: -26.2343216, 7.8345108, -26.2424965, 7.8490543, -34.0496368, 34.0444984

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2679523, upper bound: 22.2898629
time: 49.27 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2679523, upper bound: 22.2240739
time: 114.01 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -29.8163815, 12.5713549, -29.6817856, 12.5467043, -42.3630867, 42.2531395
1: -14.3561087, 20.7046490, -14.2709064, 20.6878719, -35.0439796, 34.9755554
2: -10.3005104, 21.0564651, -10.1799183, 21.0283699, -31.3288803, 31.2363834
3: -12.5413580, 23.6882362, -12.3461208, 23.6491680, -36.1905251, 36.0343552
4: -15.7232971, 20.6771469, -15.6004477, 20.6433964, -36.3666916, 36.2775955
5: -10.8693800, 25.4450531, -10.6928463, 25.4121246, -36.2815056, 36.1379013
6: -32.1002045, 11.5601358, -32.0849876, 11.4872999, -42.6247101, 42.7122650
7: -16.9095383, 26.3603954, -16.7958603, 26.3390541, -43.1976051, 43.0751419
8: -18.3802032, 23.7305355, -18.2955799, 23.6994171, -42.0796204, 42.0261154
9: -17.0598640, 20.5968666, -16.9885445, 20.5340786, -37.5939407, 37.5854111
10: -29.9659595, 28.9745770, -29.9173946, 28.8225784, -58.7885361, 58.8919716
11: -34.8884621, 15.2001848, -34.7917976, 15.0081882, -49.8966522, 49.9919815
12: -34.7177505, 13.4762268, -34.6592102, 13.2243690, -47.9421196, 48.1354370
13: -29.5007534, 22.9158936, -29.3803749, 22.8678493, -52.3686028, 52.2962685
14: -52.3106956, 10.2904863, -52.1908073, 10.0917282, -61.0812607, 61.1233025
15: -22.7977943, 18.9562912, -22.7074165, 18.9809532, -41.7787476, 41.6637077
16: -30.7877560, 25.5783539, -30.7427921, 25.5022774, -56.2900314, 56.3211441
17: -56.0575943, 21.6501808, -55.8711472, 21.4156151, -77.4732056, 77.5213318
18: -30.8107491, 14.2283955, -30.7542381, 14.0705175, -44.8812675, 44.9826355
19: -29.7032013, 2.9301219, -29.6502151, 2.8290086, -32.5322113, 32.5803375
20: -21.8371449, 10.3510056, -21.7807941, 10.2444839, -32.0816269, 32.1317978
21: -33.5441284, 6.7337875, -33.4641457, 6.5827751, -40.1269035, 40.1979332
22: -38.2193947, 10.2655325, -38.1686020, 10.1912966, -48.4106903, 48.4341354
23: -27.6427994, 7.5947022, -27.5825577, 7.5087404, -35.1515388, 35.1772614
24: -30.9654827, 7.8813744, -30.9296112, 7.8413000, -38.8067818, 38.8109856
25: -28.2949944, 11.1522045, -28.2508488, 11.1189384, -39.4139328, 39.4030533
26: -43.3404427, 8.1076651, -43.2761765, 7.9242992, -51.2647400, 51.3838425
27: -30.0144386, 14.0404959, -29.9568405, 13.9478874, -43.9623260, 43.9973373
28: -27.4603577, 9.9067059, -27.3990593, 9.8243046, -37.2846603, 37.3057632
29: -39.8731461, 10.6632719, -39.7989502, 10.5314665, -50.4046135, 50.4622231
30: -28.1386719, 14.6769276, -28.0655537, 14.5803452, -42.7190170, 42.7424812
31: -31.1826534, 8.4075069, -31.1251163, 8.3034859, -39.4861374, 39.5326233
32: -30.8919659, 12.0541935, -30.8677311, 11.9592524, -42.8512192, 42.9219246
33: -48.8883972, 9.3370838, -48.7319794, 9.2353678, -57.4467087, 57.4508896
34: -41.7874069, 7.5758181, -41.7049103, 7.5481234, -48.8928070, 48.9331627
35: -41.0435791, 9.5856686, -40.9510612, 9.5547943, -50.4281616, 50.3761292
36: -42.4096298, 9.9100933, -42.3663101, 9.8930674, -52.3026962, 52.2764053
37: -63.8739700, 2.1214371, -63.8326912, 1.9930468, -65.4727936, 65.6437225
38: -53.2105789, 12.1397209, -53.1604843, 12.0490847, -65.2596664, 65.3002014
39: -62.2729607, 6.0213223, -62.1385498, 5.8754520, -68.1484146, 68.1598740
40: -50.1061325, 9.3471880, -50.0566101, 9.2294064, -58.8977814, 59.0094528
41: -35.2670822, 6.7697840, -35.2449112, 6.6756420, -41.9427261, 42.0146942
42: -26.2642975, 7.8112421, -26.2090225, 7.7108355, -33.8929901, 33.9576035

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1719

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2358813, upper bound: 22.2720738
time: 66.56 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2861421, upper bound: 22.2625357
time: 60.85 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -29.9351997, 12.6032238, -29.7066765, 12.5501032, -42.4853020, 42.3098984
1: -14.4397392, 20.7636738, -14.2940311, 20.6911812, -35.1309204, 35.0577049
2: -10.3855267, 21.1272964, -10.2037611, 21.0301323, -31.4156590, 31.3310585
3: -12.6169891, 23.7564049, -12.3680420, 23.6514359, -36.2684250, 36.1244469
4: -15.7850342, 20.7149582, -15.6175022, 20.6454773, -36.4305115, 36.3324585
5: -10.9407692, 25.4829807, -10.7127991, 25.4133186, -36.3540878, 36.1957779
6: -32.1139297, 11.5870018, -32.0838776, 11.4938765, -42.6509132, 42.7405128
7: -17.0137157, 26.4344749, -16.8266296, 26.3422356, -43.3016586, 43.2125244
8: -18.5088463, 23.8495712, -18.3331814, 23.7030373, -42.2118835, 42.1827545
9: -17.0874481, 20.6224136, -16.9953003, 20.5394878, -37.6269379, 37.6177139
10: -30.0011082, 29.0142746, -29.9231930, 28.8319912, -58.8330994, 58.9374695
11: -34.9197807, 15.2219801, -34.7936172, 15.0144596, -49.9342422, 50.0155983
12: -34.7531013, 13.5525570, -34.6616211, 13.2424669, -47.9955673, 48.2141800
13: -29.5169315, 22.9396667, -29.3862457, 22.8672104, -52.3841400, 52.3259125
14: -52.4383698, 10.3831730, -52.2208405, 10.0955124, -61.2034836, 61.2913322
15: -22.8377686, 19.0038834, -22.7213459, 18.9924774, -41.8302460, 41.7252274
16: -30.8167419, 25.5949802, -30.7423325, 25.5077629, -56.3245049, 56.3373108
17: -56.1144028, 21.6776466, -55.8797913, 21.4202309, -77.5346375, 77.5574341
18: -30.8946018, 14.2972889, -30.7587891, 14.0901699, -44.9847717, 45.0560760
19: -29.7629166, 2.9798799, -29.6535511, 2.8431983, -32.6061134, 32.6334305
20: -21.8682556, 10.3752279, -21.7849522, 10.2488480, -32.1171036, 32.1601791
21: -33.5906258, 6.7575026, -33.4694138, 6.5888271, -40.1794510, 40.2269173
22: -38.3057594, 10.3035307, -38.1765594, 10.2027550, -48.5085144, 48.4800911
23: -27.6832066, 7.6292353, -27.5861282, 7.5171766, -35.2003822, 35.2153625
24: -31.0461407, 7.9062862, -30.9335670, 7.8479605, -38.8941002, 38.8398514
25: -28.3768330, 11.2094126, -28.2542229, 11.1362400, -39.5130730, 39.4636345
26: -43.4488297, 8.2093000, -43.2828026, 7.9519739, -51.4008026, 51.4921036
27: -30.0445595, 14.0722599, -29.9581909, 13.9512749, -43.9958344, 44.0304489
28: -27.5089569, 9.9556332, -27.4020805, 9.8368759, -37.3458328, 37.3577118
29: -39.9558640, 10.6922541, -39.8071136, 10.5391474, -50.4950104, 50.4993668
30: -28.1992760, 14.7084742, -28.0702934, 14.5869961, -42.7862701, 42.7787666
31: -31.2620468, 8.4678221, -31.1313362, 8.3207092, -39.5827560, 39.5991592
32: -30.9113598, 12.1044044, -30.8696098, 11.9647131, -42.8760719, 42.9740143
33: -48.9673576, 9.4180098, -48.7358665, 9.2584944, -57.5966492, 57.5311966
34: -41.8546677, 7.6321611, -41.7078857, 7.5631056, -49.0757065, 48.9827347
35: -41.1221581, 9.6382084, -40.9537239, 9.5700645, -50.5344849, 50.4221764
36: -42.4663239, 9.9734421, -42.3697205, 9.9116974, -52.3780212, 52.3431625
37: -64.0084839, 2.2250605, -63.8375778, 2.0236216, -65.7205353, 65.7422256
38: -53.2511063, 12.1864052, -53.1642151, 12.0594797, -65.3105850, 65.3506165
39: -62.3222885, 6.0591869, -62.1418190, 5.8845091, -68.2067947, 68.2010040
40: -50.1528664, 9.4037800, -50.0611038, 9.2432442, -58.9943695, 59.0676651
41: -35.3036766, 6.8087568, -35.2470818, 6.6834621, -41.9871368, 42.0558395
42: -26.2724533, 7.8401041, -26.2102070, 7.7168026, -33.9217300, 33.9841690

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1719

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3004617, upper bound: 22.2772335
time: 66.67 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3508732, upper bound: 22.2677654
time: 107.43 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -29.7966213, 12.5595284, -29.8686619, 12.5914717, -42.3880920, 42.4281921
1: -14.3699207, 20.6977577, -14.3936081, 20.7208118, -35.0907326, 35.0913658
2: -10.2676315, 21.0357800, -10.3740711, 21.0683823, -31.3360138, 31.4098511
3: -12.4819546, 23.6619415, -12.5681925, 23.6994705, -36.1814270, 36.2301331
4: -15.6630001, 20.6702194, -15.7940893, 20.6715622, -36.3345642, 36.4643097
5: -10.8154154, 25.4225578, -10.8956089, 25.4510155, -36.2664299, 36.3181686
6: -32.1037941, 11.5194006, -32.1162796, 11.5785398, -42.7329292, 42.7038765
7: -16.9280815, 26.3468037, -16.9612122, 26.3765125, -43.2575531, 43.2580605
8: -18.4034538, 23.7152786, -18.4926987, 23.7454109, -42.1488647, 42.2079773
9: -17.0146217, 20.5758495, -17.0978394, 20.6069202, -37.6215439, 37.6736908
10: -29.9476166, 28.8553963, -29.9860306, 29.0578537, -59.0054703, 58.8414268
11: -34.8642273, 15.0268412, -34.8673248, 15.3050699, -50.1692963, 49.8941650
12: -34.6771660, 13.3228874, -34.7283783, 13.5621319, -48.2392960, 48.0512657
13: -29.4161415, 22.9001732, -29.5265160, 22.9482288, -52.3643723, 52.4266891
14: -52.3138657, 10.1013861, -52.3633575, 10.3931932, -61.3910675, 61.1522827
15: -22.7601166, 19.0137043, -22.8347340, 19.0159359, -41.7760544, 41.8484383
16: -30.7845516, 25.5216980, -30.8035927, 25.6058846, -56.3904343, 56.3252907
17: -56.0031433, 21.4307652, -55.9963264, 21.7725315, -77.7756729, 77.4270935
18: -30.7844925, 14.1575336, -30.8267612, 14.3016071, -45.0860977, 44.9842949
19: -29.6705322, 2.8681178, -29.7176018, 3.0073538, -32.6778870, 32.5857201
20: -21.8056526, 10.2610378, -21.8501682, 10.4060678, -32.2117195, 32.1112061
21: -33.5043411, 6.6062412, -33.5508347, 6.8072901, -40.3116302, 40.1570740
22: -38.2047424, 10.2210913, -38.2420502, 10.3551884, -48.5599289, 48.4631424
23: -27.6203251, 7.5306516, -27.6376152, 7.6563134, -35.2766380, 35.1682663
24: -30.9472237, 7.8634224, -30.9815273, 7.9132662, -38.8604889, 38.8449478
25: -28.2792931, 11.1590767, -28.2964325, 11.2253571, -39.5046501, 39.4555092
26: -43.3020630, 8.0170193, -43.3659058, 8.2303057, -51.5323677, 51.3829269
27: -30.0051098, 13.9682703, -30.0261726, 14.0759144, -44.0810242, 43.9944420
28: -27.4299164, 9.8571720, -27.4610443, 9.9734497, -37.4033661, 37.3182144
29: -39.8626633, 10.5535030, -39.8670654, 10.7527189, -50.6153831, 50.4205704
30: -28.1268597, 14.6036758, -28.1179218, 14.7317600, -42.8586197, 42.7215958
31: -31.1503849, 8.3522568, -31.2116852, 8.4904137, -39.6408005, 39.5639420
32: -30.8891602, 11.9941530, -30.9173412, 12.0812731, -42.9704323, 42.9114952
33: -48.7469330, 9.3537016, -48.9657707, 9.3482580, -57.4762573, 57.7214508
34: -41.7190056, 7.5977726, -41.8249359, 7.6094494, -48.9661331, 49.1048584
35: -40.9618912, 9.6204176, -41.0874062, 9.6054516, -50.3996277, 50.5707016
36: -42.3763847, 9.9463177, -42.4383850, 9.9520683, -52.3284531, 52.3847046
37: -63.8545151, 2.1731987, -63.8980560, 2.1020155, -65.6496048, 65.7675095
38: -53.1762810, 12.1199217, -53.2415199, 12.1410742, -65.3173523, 65.3614426
39: -62.1544838, 6.0170546, -62.3369141, 5.9458475, -68.1003342, 68.3539658
40: -50.0746994, 9.3584280, -50.1436653, 9.2879276, -58.9682083, 59.1077194
41: -35.2575836, 6.7412653, -35.2877426, 6.7581730, -42.0157547, 42.0290070
42: -26.2556801, 7.7403545, -26.2449284, 7.8536339, -34.0558968, 33.9158401

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2878533, upper bound: 22.3103690
time: 72.05 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2878533, upper bound: 22.3748619
time: 80.85 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -29.9551868, 12.5993309, -29.8686619, 12.5914717, -42.5466576, 42.4679947
1: -14.4671268, 20.7263298, -14.3936081, 20.7208118, -35.1879387, 35.1199379
2: -10.4358034, 21.0731106, -10.3740711, 21.0683823, -31.5041847, 31.4471817
3: -12.6799345, 23.7088814, -12.5681925, 23.6994705, -36.3794060, 36.2770729
4: -15.8378639, 20.6953430, -15.7940893, 20.6715622, -36.5094261, 36.4894333
5: -10.9961090, 25.4594135, -10.8956089, 25.4510155, -36.4471245, 36.3550224
6: -32.1300316, 11.6027431, -32.1162796, 11.5785398, -42.7680054, 42.8058662
7: -17.0587025, 26.3797054, -16.9612122, 26.3765125, -43.3975525, 43.3015480
8: -18.5597382, 23.7564640, -18.4926987, 23.7454109, -42.3051491, 42.2491608
9: -17.1160755, 20.6406517, -17.0978394, 20.6069202, -37.7229958, 37.7384911
10: -30.0090237, 29.0797443, -29.9860306, 29.0578537, -59.0668793, 59.0657730
11: -34.9310226, 15.3156796, -34.8673248, 15.3050699, -50.2360916, 50.1830063
12: -34.7427826, 13.6402645, -34.7283783, 13.5621319, -48.3049164, 48.3686447
13: -29.5553665, 22.9718800, -29.5265160, 22.9482288, -52.5035934, 52.4983978
14: -52.4531097, 10.3980703, -52.3633575, 10.3931932, -61.5351868, 61.4562378
15: -22.8711910, 19.0292015, -22.8347340, 19.0159359, -41.8871269, 41.8639374
16: -30.8371201, 25.6186981, -30.8035927, 25.6058846, -56.4430046, 56.4222908
17: -56.1179771, 21.7806931, -55.9963264, 21.7725315, -77.8905106, 77.7770233
18: -30.8512897, 14.3670635, -30.8267612, 14.3016071, -45.1528969, 45.1938248
19: -29.7329597, 3.0309811, -29.7176018, 3.0073538, -32.7403145, 32.7485809
20: -21.8694286, 10.4169178, -21.8501682, 10.4060678, -32.2754974, 32.2670860
21: -33.5841293, 6.8239012, -33.5508347, 6.8072901, -40.3914185, 40.3747368
22: -38.2684822, 10.3666992, -38.2420502, 10.3551884, -48.6236725, 48.6087494
23: -27.6706047, 7.6686592, -27.6376152, 7.6563134, -35.3269196, 35.3062744
24: -30.9942627, 7.9271297, -30.9815273, 7.9132662, -38.9075279, 38.9086571
25: -28.3202553, 11.2444420, -28.2964325, 11.2253571, -39.5456123, 39.5408745
26: -43.3835068, 8.2919092, -43.3659058, 8.2303057, -51.6138115, 51.6578140
27: -30.0652390, 14.0923014, -30.0261726, 14.0759144, -44.1411514, 44.1184731
28: -27.4877205, 9.9925518, -27.4610443, 9.9734497, -37.4611702, 37.4535980
29: -39.9207268, 10.7637949, -39.8670654, 10.7527189, -50.6734467, 50.6308594
30: -28.1727905, 14.7471428, -28.1179218, 14.7317600, -42.9045486, 42.8650665
31: -31.2285538, 8.5203066, -31.2116852, 8.4904137, -39.7189674, 39.7319908
32: -30.9307632, 12.1098900, -30.9173412, 12.0812731, -43.0120354, 43.0272293
33: -48.9755783, 9.4405699, -48.9657707, 9.3482580, -57.6985168, 57.7844238
34: -41.8350906, 7.6426172, -41.8249359, 7.6094494, -49.1005859, 49.1360703
35: -41.0942764, 9.6536446, -41.0874062, 9.6054516, -50.5480270, 50.5929947
36: -42.4439011, 9.9850855, -42.4383850, 9.9520683, -52.3959694, 52.4234695
37: -63.9133072, 2.2489157, -63.8980560, 2.1020155, -65.7258759, 65.8619003
38: -53.2523270, 12.2000904, -53.2415199, 12.1410742, -65.3934021, 65.4416122
39: -62.3481026, 6.0738258, -62.3369141, 5.9458475, -68.2939529, 68.4107361
40: -50.1550369, 9.4015760, -50.1436653, 9.2879276, -59.0492477, 59.1559906
41: -35.2962570, 6.8148422, -35.2877426, 6.7581730, -42.0544281, 42.1025848
42: -26.2878036, 7.8758879, -26.2449284, 7.8536339, -34.1008530, 34.0907974

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2878533, upper bound: 22.2953968
time: 63.82 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2878533, upper bound: 22.2282744
time: 172.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 237.78 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.2843038, upper bound: 22.2183114
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.2843038, upper bound: 22.2183114
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.2843038, upper bound: 22.2229954
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.2843038, upper bound: 22.2229954
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.2679523, upper bound: 22.3060526
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.2679523, upper bound: 22.3060526
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.2679523, upper bound: 22.2898629
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.2679523, upper bound: 22.2240739
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.2358813, upper bound: 22.2720738
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.2861421, upper bound: 22.2625357
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.3004617, upper bound: 22.2772335
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.3508732, upper bound: 22.2677654
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.2878533, upper bound: 22.3103690
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.2878533, upper bound: 22.3748619
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.2878533, upper bound: 22.2953968
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 237.78
Output dim: 3, lower bound: -22.2878533, upper bound: 22.2282744

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -29.7087059, 12.5548115, -29.6606121, 12.5380135, -42.2467194, 42.2154236
1: -14.2727089, 20.6626854, -14.2610397, 20.6514435, -34.9241524, 34.9237251
2: -10.2301941, 21.0311546, -10.1713009, 21.0077744, -31.2379684, 31.2024555
3: -12.4201307, 23.6150932, -12.3366385, 23.5854225, -36.0055542, 35.9517326
4: -15.6708431, 20.6420021, -15.5917101, 20.6320305, -36.3028717, 36.2337112
5: -10.7595491, 25.4013290, -10.6834660, 25.3767929, -36.1363411, 36.0847931
6: -32.0741959, 11.5254288, -32.0727730, 11.4768353, -42.5895920, 42.6393394
7: -16.7990952, 26.3137951, -16.7828979, 26.2956753, -43.0408707, 43.0143738
8: -18.3047314, 23.6870079, -18.2871246, 23.6669331, -41.9716644, 41.9741325
9: -17.0239620, 20.5544968, -16.9709244, 20.5254536, -37.5494156, 37.5254211
10: -29.9295158, 28.9368591, -29.9039421, 28.8067970, -58.7363129, 58.8408012
11: -34.8095093, 15.1501169, -34.7764168, 14.9687595, -49.7782669, 49.9265327
12: -34.6880341, 13.3834696, -34.6438637, 13.2096987, -47.8977318, 48.0273323
13: -29.4533043, 22.8819656, -29.3617382, 22.8574982, -52.3108025, 52.2437057
14: -52.2011223, 10.2658701, -52.1710815, 10.0720301, -60.9584885, 61.0793457
15: -22.7492218, 18.9295120, -22.6953831, 18.9674377, -41.7166595, 41.6248932
16: -30.7298965, 25.5491066, -30.7184830, 25.4858131, -56.2157097, 56.2675896
17: -55.9153900, 21.5462475, -55.8504868, 21.3198051, -77.2351990, 77.3967361
18: -30.7621555, 14.1506233, -30.7301311, 14.0581303, -44.8202858, 44.8807526
19: -29.6656170, 2.8983202, -29.6280289, 2.8208227, -32.4864388, 32.5263481
20: -21.8063278, 10.3315048, -21.7692566, 10.2358217, -32.0421486, 32.1007614
21: -33.4971542, 6.7047529, -33.4504204, 6.5703154, -40.0674706, 40.1551743
22: -38.1774330, 10.2298765, -38.1530685, 10.1671515, -48.3445854, 48.3829460
23: -27.5986748, 7.5749722, -27.5714302, 7.5013366, -35.1000099, 35.1464005
24: -30.9388943, 7.8582764, -30.9157429, 7.8320217, -38.7709160, 38.7740173
25: -28.2610970, 11.1230440, -28.2407951, 11.1089268, -39.3700256, 39.3638382
26: -43.3080559, 8.0314436, -43.2613831, 7.9095688, -51.2176247, 51.2928276
27: -29.9645042, 14.0100346, -29.9460239, 13.9338007, -43.8983040, 43.9560585
28: -27.4234238, 9.8813086, -27.3888836, 9.8180294, -37.2414551, 37.2701912
29: -39.8032341, 10.6029987, -39.7826729, 10.4823160, -50.2855492, 50.3856735
30: -28.0722980, 14.6468449, -28.0540390, 14.5656242, -42.6379242, 42.7008820
31: -31.1427326, 8.3656883, -31.1021461, 8.2915802, -39.4343109, 39.4678345
32: -30.8673172, 12.0177336, -30.8564453, 11.9513540, -42.8186722, 42.8741798
33: -48.8201065, 9.2355242, -48.6734848, 9.2261171, -57.3691711, 57.2874832
34: -41.7581558, 7.5345860, -41.6858673, 7.5400801, -48.8602448, 48.8660240
35: -41.0014687, 9.5320177, -40.9158249, 9.5493107, -50.3805542, 50.2836914
36: -42.3744049, 9.8707180, -42.3365707, 9.8866978, -52.2611008, 52.2072906
37: -63.7500038, 1.9670706, -63.7239227, 1.9855690, -65.3442154, 65.3792267
38: -53.1655426, 12.0688086, -53.1262436, 12.0372629, -65.2028046, 65.1950531
39: -62.1659851, 5.8839426, -62.0427437, 5.8661108, -68.0320969, 67.9266891
40: -50.0110703, 9.2259378, -49.9729729, 9.2217731, -58.7948151, 58.7997742
41: -35.2244377, 6.7042732, -35.2107773, 6.6667223, -41.8911591, 41.9150505
42: -26.2108078, 7.7697830, -26.1984253, 7.6915865, -33.8268356, 33.9015541

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2691117, upper bound: 22.1479235
time: 69.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2591718, upper bound: 22.1937816
time: 53.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -29.7087059, 12.5548115, -29.7682762, 12.5545254, -42.2632294, 42.3230896
1: -14.2727089, 20.6626854, -14.3444490, 20.6934280, -34.9661369, 35.0071335
2: -10.2301941, 21.0311546, -10.2416420, 21.0331230, -31.2633171, 31.2727966
3: -12.4201307, 23.6150932, -12.4578571, 23.6585999, -36.0787315, 36.0729523
4: -15.6708431, 20.6420021, -15.6442366, 20.6671410, -36.3379822, 36.2862396
5: -10.7595491, 25.4013290, -10.7932720, 25.4205418, -36.1800919, 36.1946030
6: -32.0741959, 11.5254288, -32.0987396, 11.5115604, -42.6346169, 42.6659737
7: -16.7990952, 26.3137951, -16.8933563, 26.3422813, -43.0889053, 43.1260071
8: -18.3047314, 23.6870079, -18.3626080, 23.7105026, -42.0152359, 42.0496140
9: -17.0239620, 20.5544968, -17.0068359, 20.5677986, -37.5917587, 37.5613327
10: -29.9295158, 28.9368591, -29.9403191, 28.8445244, -58.7740402, 58.8771782
11: -34.8095093, 15.1501169, -34.8553352, 15.0188303, -49.8283386, 50.0054512
12: -34.6880341, 13.3834696, -34.6735992, 13.3024588, -47.9904938, 48.0570679
13: -29.4533043, 22.8819656, -29.4092178, 22.8914566, -52.3447609, 52.2911835
14: -52.2011223, 10.2658701, -52.2806702, 10.0966167, -60.9810257, 61.1841545
15: -22.7492218, 18.9295120, -22.7438965, 18.9942207, -41.7434425, 41.6734085
16: -30.7298965, 25.5491066, -30.7763062, 25.5150948, -56.2449913, 56.3254128
17: -55.9153900, 21.5462475, -55.9926796, 21.4237862, -77.3391724, 77.5389252
18: -30.7621555, 14.1506233, -30.7787209, 14.1358852, -44.8980408, 44.9293442
19: -29.6656170, 2.8983202, -29.6656475, 2.8526487, -32.5182648, 32.5639687
20: -21.8063278, 10.3315048, -21.8000221, 10.2553396, -32.0616684, 32.1315269
21: -33.4971542, 6.7047529, -33.4973946, 6.5993528, -40.0965080, 40.2021484
22: -38.1774330, 10.2298765, -38.1950302, 10.2028217, -48.3802567, 48.4249077
23: -27.5986748, 7.5749722, -27.6155262, 7.5210662, -35.1197395, 35.1904984
24: -30.9388943, 7.8582764, -30.9423141, 7.8551331, -38.7940292, 38.8005905
25: -28.2610970, 11.1230440, -28.2746563, 11.1380444, -39.3991394, 39.3977013
26: -43.3080559, 8.0314436, -43.2937737, 7.9856434, -51.2937012, 51.3252182
27: -29.9645042, 14.0100346, -29.9959412, 13.9642887, -43.9287949, 44.0059738
28: -27.4234238, 9.8813086, -27.4257660, 9.8433952, -37.2668190, 37.3070755
29: -39.8032341, 10.6029987, -39.8526764, 10.5425625, -50.3457947, 50.4556732
30: -28.0722980, 14.6468449, -28.1204414, 14.5957108, -42.6680069, 42.7672882
31: -31.1427326, 8.3656883, -31.1421051, 8.3334141, -39.4761467, 39.5077934
32: -30.8673172, 12.0177336, -30.8810730, 11.9878445, -42.8551636, 42.8988075
33: -48.8201065, 9.2355242, -48.7417946, 9.3277197, -57.4740143, 57.3558197
34: -41.7581558, 7.5345860, -41.7150917, 7.5812874, -48.9017563, 48.8913956
35: -41.0014687, 9.5320177, -40.9579659, 9.6029701, -50.4371185, 50.3263474
36: -42.3744049, 9.8707180, -42.3717880, 9.9260435, -52.3004494, 52.2425079
37: -63.7500038, 1.9670706, -63.8479233, 2.1399117, -65.4998550, 65.5030518
38: -53.1655426, 12.0688086, -53.1712952, 12.1081381, -65.2736816, 65.2401047
39: -62.1659851, 5.8839426, -62.1497574, 6.0035200, -68.1695023, 68.0336990
40: -50.0110703, 9.2259378, -50.0680084, 9.3430395, -58.9189072, 58.8952560
41: -35.2244377, 6.7042732, -35.2533989, 6.7322197, -41.9566574, 41.9576721
42: -26.2108078, 7.7697830, -26.2518692, 7.7330542, -33.8671684, 33.9492035

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2691117, upper bound: 22.1479235
time: 65.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2591718, upper bound: 22.1937816
time: 70.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -29.8275433, 12.5866652, -29.6854668, 12.5414028, -42.3689461, 42.2721329
1: -14.3563347, 20.7217236, -14.2841682, 20.6547470, -35.0110817, 35.0058899
2: -10.3151913, 21.1019688, -10.1951542, 21.0095253, -31.3247166, 31.2971230
3: -12.4957848, 23.6832657, -12.3585424, 23.5876865, -36.0834732, 36.0418091
4: -15.7325172, 20.6797962, -15.6087570, 20.6340942, -36.3666115, 36.2885513
5: -10.8309307, 25.4392624, -10.7034168, 25.3779755, -36.2089081, 36.1426773
6: -32.0879402, 11.5522785, -32.0716629, 11.4834270, -42.6158257, 42.6675644
7: -16.9032593, 26.3879089, -16.8136749, 26.2988777, -43.1449432, 43.1518059
8: -18.4333611, 23.8060398, -18.3247242, 23.6705475, -42.1039085, 42.1307640
9: -17.0515175, 20.5800514, -16.9776821, 20.5308342, -37.5823517, 37.5577316
10: -29.9646587, 28.9765415, -29.9097767, 28.8161659, -58.7808228, 58.8863182
11: -34.8408394, 15.1718950, -34.7783279, 14.9750080, -49.8158493, 49.9502220
12: -34.7233696, 13.4597960, -34.6463013, 13.2277908, -47.9511604, 48.1060982
13: -29.4694118, 22.9056816, -29.3676357, 22.8568726, -52.3262863, 52.2733154
14: -52.3287354, 10.3585787, -52.2011375, 10.0758333, -61.0807190, 61.2474136
15: -22.7892265, 18.9770279, -22.7092972, 18.9789619, -41.7681885, 41.6863251
16: -30.7588501, 25.5656166, -30.7179832, 25.4913273, -56.2501755, 56.2835999
17: -55.9721756, 21.5736961, -55.8591652, 21.3244648, -77.2966385, 77.4328613
18: -30.8460007, 14.2195072, -30.7346802, 14.0778055, -44.9238052, 44.9541855
19: -29.7253418, 2.9480410, -29.6313515, 2.8350196, -32.5603600, 32.5793915
20: -21.8374023, 10.3557243, -21.7734070, 10.2401733, -32.0775757, 32.1291313
21: -33.5436363, 6.7284527, -33.4556961, 6.5763750, -40.1200104, 40.1841507
22: -38.2637939, 10.2678661, -38.1609955, 10.1786289, -48.4424210, 48.4288635
23: -27.6390762, 7.6095076, -27.5749931, 7.5097971, -35.1488724, 35.1845016
24: -31.0195465, 7.8831434, -30.9197063, 7.8386755, -38.8582230, 38.8028488
25: -28.3429337, 11.1803064, -28.2441654, 11.1262436, -39.4691772, 39.4244728
26: -43.4164467, 8.1330872, -43.2679710, 7.9372473, -51.3536949, 51.4010582
27: -29.9945889, 14.0418549, -29.9473534, 13.9371996, -43.9317894, 43.9892082
28: -27.4720249, 9.9302540, -27.3918934, 9.8305721, -37.3025970, 37.3221474
29: -39.8859291, 10.6319857, -39.7908249, 10.4900169, -50.3759460, 50.4228096
30: -28.1328716, 14.6783552, -28.0587769, 14.5722742, -42.7051468, 42.7371330
31: -31.2221241, 8.4259834, -31.1083565, 8.3088083, -39.5309334, 39.5343399
32: -30.8866520, 12.0678997, -30.8583241, 11.9567995, -42.8434525, 42.9262238
33: -48.8991013, 9.3164206, -48.6774025, 9.2492590, -57.5191040, 57.3677216
34: -41.8254318, 7.5909338, -41.6888161, 7.5550556, -49.0440369, 48.9155769
35: -41.0800209, 9.5845413, -40.9184990, 9.5645628, -50.4869690, 50.3297043
36: -42.4310799, 9.9340725, -42.3400116, 9.9053268, -52.3364067, 52.2740860
37: -63.8845024, 2.0707388, -63.7287598, 2.0161648, -65.5919952, 65.4777527
38: -53.2060509, 12.1155405, -53.1299210, 12.0476046, -65.2536545, 65.2454605
39: -62.2153397, 5.9217787, -62.0460129, 5.8751640, -68.0904999, 67.9677887
40: -50.0578423, 9.2825413, -49.9774475, 9.2356730, -58.8914642, 58.8579483
41: -35.2610245, 6.7432022, -35.2129593, 6.6745644, -41.9355888, 41.9561615
42: -26.2189846, 7.7986393, -26.1996078, 7.6975355, -33.8555717, 33.9281082

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3337840, upper bound: 22.1527066
time: 65.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2591718, upper bound: 22.1988680
time: 64.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -29.8275433, 12.5866652, -29.7931519, 12.5579214, -42.3854637, 42.3798180
1: -14.3563347, 20.7217236, -14.3675814, 20.6967201, -35.0530548, 35.0893059
2: -10.3151913, 21.1019688, -10.2654915, 21.0348778, -31.3500690, 31.3674603
3: -12.4957848, 23.6832657, -12.4797812, 23.6608391, -36.1566238, 36.1630478
4: -15.7325172, 20.6797962, -15.6612740, 20.6692047, -36.4017220, 36.3410721
5: -10.8309307, 25.4392624, -10.8132219, 25.4217262, -36.2526550, 36.2524834
6: -32.0879402, 11.5522785, -32.0976219, 11.5181131, -42.6608276, 42.6942177
7: -16.9032593, 26.3879089, -16.9241276, 26.3454323, -43.1929512, 43.2634354
8: -18.4333611, 23.8060398, -18.4002190, 23.7141266, -42.1474876, 42.2062607
9: -17.0515175, 20.5800514, -17.0135727, 20.5731926, -37.6247101, 37.5936241
10: -29.9646587, 28.9765415, -29.9461536, 28.8539162, -58.8185730, 58.9226952
11: -34.8408394, 15.1718950, -34.8572693, 15.0250750, -49.8659134, 50.0291634
12: -34.7233696, 13.4597960, -34.6760216, 13.3205509, -48.0439224, 48.1358185
13: -29.4694118, 22.9056816, -29.4151077, 22.8908138, -52.3602257, 52.3207893
14: -52.3287354, 10.3585787, -52.3107109, 10.1004114, -61.1032562, 61.3522415
15: -22.7892265, 18.9770279, -22.7577839, 19.0057526, -41.7949791, 41.7348099
16: -30.7588501, 25.5656166, -30.7757969, 25.5205650, -56.2794151, 56.3414154
17: -55.9721756, 21.5736961, -56.0012856, 21.4283714, -77.4005432, 77.5749817
18: -30.8460007, 14.2195072, -30.7832527, 14.1555691, -45.0015717, 45.0027618
19: -29.7253418, 2.9480410, -29.6689625, 2.8668447, -32.5921860, 32.6170044
20: -21.8374023, 10.3557243, -21.8041706, 10.2596836, -32.0970840, 32.1598969
21: -33.5436363, 6.7284527, -33.5026474, 6.6054077, -40.1490440, 40.2311020
22: -38.2637939, 10.2678661, -38.2029686, 10.2142754, -48.4780693, 48.4708328
23: -27.6390762, 7.6095076, -27.6191063, 7.5295148, -35.1685905, 35.2286148
24: -31.0195465, 7.8831434, -30.9462910, 7.8618021, -38.8813477, 38.8294334
25: -28.3429337, 11.1803064, -28.2780323, 11.1553230, -39.4982567, 39.4583397
26: -43.4164467, 8.1330872, -43.3003578, 8.0133276, -51.4297752, 51.4334450
27: -29.9945889, 14.0418549, -29.9973106, 13.9676733, -43.9622612, 44.0391655
28: -27.4720249, 9.9302540, -27.4287987, 9.8559780, -37.3280029, 37.3590546
29: -39.8859291, 10.6319857, -39.8608093, 10.5502481, -50.4361763, 50.4927940
30: -28.1328716, 14.6783552, -28.1251602, 14.6023502, -42.7352219, 42.8035164
31: -31.2221241, 8.4259834, -31.1483002, 8.3506269, -39.5727501, 39.5742836
32: -30.8866520, 12.0678997, -30.8830261, 11.9932852, -42.8799362, 42.9509277
33: -48.8991013, 9.3164206, -48.7456970, 9.3508873, -57.6240005, 57.4360352
34: -41.8254318, 7.5909338, -41.7180557, 7.5962648, -49.0855331, 48.9409409
35: -41.0800209, 9.5845413, -40.9606209, 9.6182461, -50.5435867, 50.3723602
36: -42.4310799, 9.9340725, -42.3752251, 9.9447002, -52.3757782, 52.3092957
37: -63.8845024, 2.0707388, -63.8527641, 2.1704483, -65.7476044, 65.6015320
38: -53.2060509, 12.1155405, -53.1749878, 12.1184988, -65.3245468, 65.2905273
39: -62.2153397, 5.9217787, -62.1530380, 6.0125418, -68.2278824, 68.0748138
40: -50.0578423, 9.2825413, -50.0724907, 9.3569078, -59.0155106, 58.9534149
41: -35.2610245, 6.7432022, -35.2555618, 6.7400265, -42.0010529, 41.9987640
42: -26.2189846, 7.7986393, -26.2530556, 7.7390175, -33.8958740, 33.9758224

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3337840, upper bound: 22.1527066
time: 65.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2591718, upper bound: 22.1988680
time: 67.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -29.6889305, 12.5430145, -29.8474712, 12.5827332, -42.2716637, 42.3904877
1: -14.2865028, 20.6557674, -14.3837271, 20.6844063, -34.9709091, 35.0394936
2: -10.1972971, 21.0104370, -10.3654575, 21.0478249, -31.2451210, 31.3758945
3: -12.3607359, 23.5887947, -12.5587091, 23.6357841, -35.9965210, 36.1475029
4: -15.6104794, 20.6351109, -15.7853527, 20.6601410, -36.2706223, 36.4204636
5: -10.7056046, 25.3787994, -10.8862123, 25.4157028, -36.1213074, 36.2650108
6: -32.0778275, 11.4847021, -32.1040192, 11.5680323, -42.6978455, 42.6310005
7: -16.8176479, 26.3002090, -16.9482098, 26.3331604, -43.1008148, 43.1973038
8: -18.3279762, 23.6717186, -18.4842262, 23.7129650, -42.0409393, 42.1559448
9: -16.9787197, 20.5334969, -17.0802193, 20.5982628, -37.5769806, 37.6137161
10: -29.9112358, 28.8176498, -29.9725952, 29.0420303, -58.9532661, 58.7902451
11: -34.7852936, 14.9767513, -34.8520737, 15.2656498, -50.0509415, 49.8288269
12: -34.6474380, 13.2301474, -34.7130203, 13.5474415, -48.1948776, 47.9431686
13: -29.3686714, 22.8662262, -29.5078506, 22.9378853, -52.3065567, 52.3740768
14: -52.2042847, 10.0767975, -52.3434143, 10.3734903, -61.2683945, 61.1082687
15: -22.7116184, 18.9869347, -22.8226738, 19.0024128, -41.7140312, 41.8096085
16: -30.7267284, 25.4924126, -30.7792397, 25.5894318, -56.3161621, 56.2716522
17: -55.8609962, 21.3267899, -55.9757500, 21.6767654, -77.5377655, 77.3025360
18: -30.7359009, 14.0797386, -30.8027344, 14.2892399, -45.0251389, 44.8824730
19: -29.6329117, 2.8362851, -29.6953907, 2.9991498, -32.6320610, 32.5316772
20: -21.7748833, 10.2415190, -21.8384991, 10.3974142, -32.1722984, 32.0800171
21: -33.4573631, 6.5771923, -33.5371246, 6.7948475, -40.2522125, 40.1143188
22: -38.1628075, 10.1854382, -38.2265015, 10.3310499, -48.4938583, 48.4119415
23: -27.5762253, 7.5108995, -27.6264267, 7.6489158, -35.2251396, 35.1373253
24: -30.9206753, 7.8403187, -30.9676781, 7.9039993, -38.8246765, 38.8079987
25: -28.2454262, 11.1299782, -28.2863312, 11.2153244, -39.4607506, 39.4163094
26: -43.2696877, 7.9409266, -43.3510780, 8.2155657, -51.4852524, 51.2920036
27: -29.9551678, 13.9378242, -30.0153179, 14.0618658, -44.0170326, 43.9531403
28: -27.3930225, 9.8317814, -27.4507790, 9.9671583, -37.3601799, 37.2825623
29: -39.7926712, 10.4932718, -39.8507805, 10.7035627, -50.4962349, 50.3440514
30: -28.0605125, 14.5735912, -28.1064053, 14.7170639, -42.7775764, 42.6799965
31: -31.1104240, 8.3104401, -31.1886024, 8.4784746, -39.5888977, 39.4990425
32: -30.8645229, 11.9576311, -30.9060211, 12.0733776, -42.9379005, 42.8636513
33: -48.6786385, 9.2520819, -48.9072685, 9.3389435, -57.3986435, 57.5580292
34: -41.6898041, 7.5565662, -41.8058510, 7.6013613, -48.9344254, 49.0376511
35: -40.9197655, 9.5667152, -41.0521393, 9.5999432, -50.3521576, 50.4782295
36: -42.3411484, 9.9069452, -42.4086456, 9.9456606, -52.2868080, 52.3155899
37: -63.7305374, 2.0188427, -63.7893028, 2.0945759, -65.5211105, 65.5031433
38: -53.1312180, 12.0490494, -53.2072983, 12.1291456, -65.2603607, 65.2563477
39: -62.0474930, 5.8796501, -62.2411499, 5.9364061, -67.9839020, 68.1208038
40: -49.9796371, 9.2371655, -50.0600281, 9.2802839, -58.8652267, 58.8980408
41: -35.2150116, 6.6757889, -35.2536240, 6.7492428, -41.9642563, 41.9294128
42: -26.2022324, 7.6988707, -26.2343216, 7.8345108, -33.9898109, 33.8598175

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 729

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2024980, upper bound: 22.3003074
time: 71.50 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2024980, upper bound: 22.3049778
time: 60.29 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -29.6889305, 12.5430145, -29.9551868, 12.5993309, -42.2882614, 42.4981995
1: -14.2865028, 20.6557674, -14.4671268, 20.7263298, -35.0128326, 35.1228943
2: -10.1972971, 21.0104370, -10.4358034, 21.0731106, -31.2704086, 31.4462395
3: -12.3607359, 23.5887947, -12.6799345, 23.7088814, -36.0696182, 36.2687302
4: -15.6104794, 20.6351109, -15.8378639, 20.6953430, -36.3058243, 36.4729767
5: -10.7056046, 25.3787994, -10.9961090, 25.4594135, -36.1650162, 36.3749084
6: -32.0778275, 11.4847021, -32.1300316, 11.6027431, -42.7428894, 42.6576729
7: -16.8176479, 26.3002090, -17.0587025, 26.3797054, -43.1487770, 43.3089371
8: -18.3279762, 23.6717186, -18.5597382, 23.7564640, -42.0844421, 42.2314568
9: -16.9787197, 20.5334969, -17.1160755, 20.6406517, -37.6193695, 37.6495743
10: -29.9112358, 28.8176498, -30.0090237, 29.0797443, -58.9909821, 58.8266754
11: -34.7852936, 14.9767513, -34.9310226, 15.3156796, -50.1009750, 49.9077759
12: -34.6474380, 13.2301474, -34.7427826, 13.6402645, -48.2877045, 47.9729309
13: -29.3686714, 22.8662262, -29.5553665, 22.9718800, -52.3405533, 52.4215927
14: -52.2042847, 10.0767975, -52.4531097, 10.3980703, -61.2909012, 61.2134323
15: -22.7116184, 18.9869347, -22.8711910, 19.0292015, -41.7408218, 41.8581238
16: -30.7267284, 25.4924126, -30.8371201, 25.6186981, -56.3454285, 56.3295326
17: -55.8609962, 21.3267899, -56.1179771, 21.7806931, -77.6416931, 77.4447632
18: -30.7359009, 14.0797386, -30.8512897, 14.3670635, -45.1029663, 44.9310303
19: -29.6329117, 2.8362851, -29.7329597, 3.0309811, -32.6638947, 32.5692444
20: -21.7748833, 10.2415190, -21.8694286, 10.4169178, -32.1918030, 32.1109467
21: -33.4573631, 6.5771923, -33.5841293, 6.8239012, -40.2812653, 40.1613235
22: -38.1628075, 10.1854382, -38.2684822, 10.3666992, -48.5295067, 48.4539185
23: -27.5762253, 7.5108995, -27.6706047, 7.6686592, -35.2448845, 35.1815033
24: -30.9206753, 7.8403187, -30.9942627, 7.9271297, -38.8478050, 38.8345795
25: -28.2454262, 11.1299782, -28.3202553, 11.2444420, -39.4898682, 39.4502335
26: -43.2696877, 7.9409266, -43.3835068, 8.2919092, -51.5615959, 51.3244324
27: -29.9551678, 13.9378242, -30.0652390, 14.0923014, -44.0474701, 44.0030632
28: -27.3930225, 9.8317814, -27.4877205, 9.9925518, -37.3855743, 37.3195038
29: -39.7926712, 10.4932718, -39.9207268, 10.7637949, -50.5564651, 50.4139977
30: -28.0605125, 14.5735912, -28.1727905, 14.7471428, -42.8076553, 42.7463837
31: -31.1104240, 8.3104401, -31.2285538, 8.5203066, -39.6307297, 39.5389938
32: -30.8645229, 11.9576311, -30.9307632, 12.1098900, -42.9744110, 42.8883934
33: -48.6786385, 9.2520819, -48.9755783, 9.4405699, -57.5035934, 57.6263199
34: -41.6898041, 7.5565662, -41.8350906, 7.6426172, -48.9759674, 49.0630341
35: -40.9197655, 9.5667152, -41.0942764, 9.6536446, -50.4087753, 50.5208969
36: -42.3411484, 9.9069452, -42.4439011, 9.9850855, -52.3262329, 52.3508453
37: -63.7305374, 2.0188427, -63.9133072, 2.2489157, -65.6767807, 65.6268539
38: -53.1312180, 12.0490494, -53.2523270, 12.2000904, -65.3313065, 65.3013763
39: -62.0474930, 5.8796501, -62.3481026, 6.0738258, -68.1213226, 68.2277527
40: -49.9796371, 9.2371655, -50.1550369, 9.4015760, -58.9893188, 58.9934921
41: -35.2150116, 6.6757889, -35.2962570, 6.8148422, -42.0298538, 41.9720459
42: -26.2022324, 7.6988707, -26.2878036, 7.8758879, -34.0300331, 33.9075546

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1777

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 729

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2024980, upper bound: 22.3003068
time: 70.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2024980, upper bound: 22.3049772
time: 75.48 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -29.8474712, 12.5827332, -29.8474712, 12.5827332, -42.4302063, 42.4302063
1: -14.3837271, 20.6844063, -14.3837271, 20.6844063, -35.0681343, 35.0681343
2: -10.3654575, 21.0478249, -10.3654575, 21.0478249, -31.4132824, 31.4132824
3: -12.5587091, 23.6357841, -12.5587091, 23.6357841, -36.1944923, 36.1944923
4: -15.7853527, 20.6601410, -15.7853527, 20.6601410, -36.4454956, 36.4454956
5: -10.8862123, 25.4157028, -10.8862123, 25.4157028, -36.3019142, 36.3019142
6: -32.1040192, 11.5680323, -32.1040192, 11.5680323, -42.7329102, 42.7329102
7: -16.9482098, 26.3331604, -16.9482098, 26.3331604, -43.2408295, 43.2408371
8: -18.4842262, 23.7129650, -18.4842262, 23.7129650, -42.1971893, 42.1971893
9: -17.0802193, 20.5982628, -17.0802193, 20.5982628, -37.6784821, 37.6784821
10: -29.9725952, 29.0420303, -29.9725952, 29.0420303, -59.0146255, 59.0146255
11: -34.8520737, 15.2656498, -34.8520737, 15.2656498, -50.1177216, 50.1177216
12: -34.7130203, 13.5474415, -34.7130203, 13.5474415, -48.2604599, 48.2604599
13: -29.5078506, 22.9378853, -29.5078506, 22.9378853, -52.4457359, 52.4457359
14: -52.3434143, 10.3734903, -52.3434143, 10.3734903, -61.4122009, 61.4121933
15: -22.8226738, 19.0024128, -22.8226738, 19.0024128, -41.8250885, 41.8250885
16: -30.7792397, 25.5894318, -30.7792397, 25.5894318, -56.3686714, 56.3686714
17: -55.9757500, 21.6767654, -55.9757500, 21.6767654, -77.6525116, 77.6525116
18: -30.8027344, 14.2892399, -30.8027344, 14.2892399, -45.0919724, 45.0919724
19: -29.6953907, 2.9991498, -29.6953907, 2.9991498, -32.6945419, 32.6945419
20: -21.8384991, 10.3974142, -21.8384991, 10.3974142, -32.2359123, 32.2359123
21: -33.5371246, 6.7948475, -33.5371246, 6.7948475, -40.3319702, 40.3319702
22: -38.2265015, 10.3310499, -38.2265015, 10.3310499, -48.5575523, 48.5575523
23: -27.6264267, 7.6489158, -27.6264267, 7.6489158, -35.2753410, 35.2753410
24: -30.9676781, 7.9039993, -30.9676781, 7.9039993, -38.8716774, 38.8716774
25: -28.2863312, 11.2153244, -28.2863312, 11.2153244, -39.5016556, 39.5016556
26: -43.3510780, 8.2155657, -43.3510780, 8.2155657, -51.5666428, 51.5666428
27: -30.0153179, 14.0618658, -30.0153179, 14.0618658, -44.0771828, 44.0771828
28: -27.4507790, 9.9671583, -27.4507790, 9.9671583, -37.4179382, 37.4179382
29: -39.8507805, 10.7035627, -39.8507805, 10.7035627, -50.5543442, 50.5543442
30: -28.1064053, 14.7170639, -28.1064053, 14.7170639, -42.8234711, 42.8234711
31: -31.1886024, 8.4784746, -31.1886024, 8.4784746, -39.6670761, 39.6670761
32: -30.9060211, 12.0733776, -30.9060211, 12.0733776, -42.9794006, 42.9794006
33: -48.9072685, 9.3389435, -48.9072685, 9.3389435, -57.6209412, 57.6209488
34: -41.8058510, 7.6013613, -41.8058510, 7.6013613, -49.0688477, 49.0688477
35: -41.0521393, 9.5999432, -41.0521393, 9.5999432, -50.5004883, 50.5004883
36: -42.4086456, 9.9456606, -42.4086456, 9.9456606, -52.3543053, 52.3543053
37: -63.7893028, 2.0945759, -63.7893028, 2.0945759, -65.5974655, 65.5974808
38: -53.2072983, 12.1291456, -53.2072983, 12.1291456, -65.3364410, 65.3364410
39: -62.2411499, 5.9364061, -62.2411499, 5.9364061, -68.1775589, 68.1775589
40: -50.0600281, 9.2802839, -50.0600281, 9.2802839, -58.9462738, 58.9462814
41: -35.2536240, 6.7492428, -35.2536240, 6.7492428, -42.0028687, 42.0028687
42: -26.2343216, 7.8345108, -26.2343216, 7.8345108, -34.0347519, 34.0347519

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 729

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3266211, upper bound: 22.2841268
time: 61.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3912577, upper bound: 22.2887722
time: 96.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -29.8474712, 12.5827332, -29.9551868, 12.5993309, -42.4468002, 42.5379181
1: -14.3837271, 20.6844063, -14.4671268, 20.7263298, -35.1100578, 35.1515350
2: -10.3654575, 21.0478249, -10.4358034, 21.0731106, -31.4385681, 31.4836273
3: -12.5587091, 23.6357841, -12.6799345, 23.7088814, -36.2675896, 36.3157196
4: -15.7853527, 20.6601410, -15.8378639, 20.6953430, -36.4806976, 36.4980049
5: -10.8862123, 25.4157028, -10.9961090, 25.4594135, -36.3456268, 36.4118118
6: -32.1040192, 11.5680323, -32.1300316, 11.6027431, -42.7779579, 42.7595863
7: -16.9482098, 26.3331604, -17.0587025, 26.3797054, -43.2887917, 43.3524704
8: -18.4842262, 23.7129650, -18.5597382, 23.7564640, -42.2406921, 42.2727051
9: -17.0802193, 20.5982628, -17.1160755, 20.6406517, -37.7208710, 37.7143402
10: -29.9725952, 29.0420303, -30.0090237, 29.0797443, -59.0523376, 59.0510559
11: -34.8520737, 15.2656498, -34.9310226, 15.3156796, -50.1677551, 50.1966705
12: -34.7130203, 13.5474415, -34.7427826, 13.6402645, -48.3532867, 48.2902222
13: -29.5078506, 22.9378853, -29.5553665, 22.9718800, -52.4797287, 52.4932518
14: -52.3434143, 10.3734903, -52.4531097, 10.3980703, -61.4347153, 61.5173683
15: -22.8226738, 19.0024128, -22.8711910, 19.0292015, -41.8518753, 41.8736038
16: -30.7792397, 25.5894318, -30.8371201, 25.6186981, -56.3979378, 56.4265518
17: -55.9757500, 21.6767654, -56.1179771, 21.7806931, -77.7564392, 77.7947388
18: -30.8027344, 14.2892399, -30.8512897, 14.3670635, -45.1697998, 45.1405296
19: -29.6953907, 2.9991498, -29.7329597, 3.0309811, -32.7263718, 32.7321091
20: -21.8384991, 10.3974142, -21.8694286, 10.4169178, -32.2554169, 32.2668419
21: -33.5371246, 6.7948475, -33.5841293, 6.8239012, -40.3610268, 40.3789749
22: -38.2265015, 10.3310499, -38.2684822, 10.3666992, -48.5932007, 48.5995331
23: -27.6264267, 7.6489158, -27.6706047, 7.6686592, -35.2950859, 35.3195190
24: -30.9676781, 7.9039993, -30.9942627, 7.9271297, -38.8948059, 38.8982620
25: -28.2863312, 11.2153244, -28.3202553, 11.2444420, -39.5307732, 39.5355797
26: -43.3510780, 8.2155657, -43.3835068, 8.2919092, -51.6429863, 51.5990715
27: -30.0153179, 14.0618658, -30.0652390, 14.0923014, -44.1076202, 44.1271057
28: -27.4507790, 9.9671583, -27.4877205, 9.9925518, -37.4433289, 37.4548798
29: -39.8507805, 10.7035627, -39.9207268, 10.7637949, -50.6145744, 50.6242905
30: -28.1064053, 14.7170639, -28.1727905, 14.7471428, -42.8535461, 42.8898544
31: -31.1886024, 8.4784746, -31.2285538, 8.5203066, -39.7089081, 39.7070274
32: -30.9060211, 12.0733776, -30.9307632, 12.1098900, -43.0159111, 43.0041428
33: -48.9072685, 9.3389435, -48.9755783, 9.4405699, -57.7258911, 57.6892319
34: -41.8058510, 7.6013613, -41.8350906, 7.6426172, -49.1103973, 49.0942383
35: -41.0521393, 9.5999432, -41.0942764, 9.6536446, -50.5570908, 50.5431595
36: -42.4086456, 9.9456606, -42.4439011, 9.9850855, -52.3937302, 52.3895607
37: -63.7893028, 2.0945759, -63.9133072, 2.2489157, -65.7531357, 65.7211761
38: -53.2072983, 12.1291456, -53.2523270, 12.2000904, -65.4073868, 65.3814697
39: -62.2411499, 5.9364061, -62.3481026, 6.0738258, -68.3149719, 68.2845078
40: -50.0600281, 9.2802839, -50.1550369, 9.4015760, -59.0703735, 59.0417328
41: -35.2536240, 6.7492428, -35.2962570, 6.8148422, -42.0684662, 42.0455017
42: -26.2343216, 7.8345108, -26.2878036, 7.8758879, -34.0749779, 34.0824928

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 729

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3266211, upper bound: 22.2841260
time: 66.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.3912577, upper bound: 22.2887722
time: 71.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -29.8163815, 12.5713549, -29.6700115, 12.5404758, -42.3568573, 42.2413673
1: -14.3561087, 20.7046490, -14.2656155, 20.6713753, -35.0274849, 34.9702644
2: -10.3005104, 21.0564651, -10.1743269, 21.0137253, -31.3142357, 31.2307930
3: -12.5413580, 23.6882362, -12.3399591, 23.6173973, -36.1587563, 36.0281944
4: -15.7232971, 20.6771469, -15.5927219, 20.6375351, -36.3608322, 36.2698669
5: -10.8693800, 25.4450531, -10.6873875, 25.3985615, -36.2679405, 36.1324387
6: -32.1002045, 11.5601358, -32.0728149, 11.4761486, -42.6122437, 42.6979294
7: -16.9095383, 26.3603954, -16.7882652, 26.3186035, -43.1747742, 43.0671387
8: -18.3802032, 23.7305355, -18.2873402, 23.6783867, -42.0585899, 42.0178757
9: -17.0598640, 20.5968666, -16.9735374, 20.5268173, -37.5866814, 37.5704041
10: -29.9659595, 28.9745770, -29.9004269, 28.8109818, -58.7769394, 58.8750038
11: -34.8884621, 15.2001848, -34.7802353, 14.9930811, -49.8815422, 49.9804192
12: -34.7177505, 13.4762268, -34.6524887, 13.2053776, -47.9231262, 48.1287155
13: -29.5007534, 22.9158936, -29.3702850, 22.8344650, -52.3352203, 52.2861786
14: -52.3106956, 10.2904863, -52.1771507, 10.0463085, -61.0362091, 61.1094475
15: -22.7977943, 18.9562912, -22.6949463, 18.9730434, -41.7708359, 41.6512375
16: -30.7877560, 25.5783539, -30.7181740, 25.4929047, -56.2806625, 56.2965279
17: -56.0575943, 21.6501808, -55.8574409, 21.3326721, -77.3902664, 77.5076218
18: -30.8107491, 14.2283955, -30.7268562, 14.0628567, -44.8736038, 44.9552536
19: -29.7032013, 2.9301219, -29.6330185, 2.8249202, -32.5281219, 32.5631409
20: -21.8371449, 10.3510056, -21.7728539, 10.2383804, -32.0755234, 32.1238594
21: -33.5441284, 6.7337875, -33.4537201, 6.5776920, -40.1218185, 40.1875076
22: -38.2193947, 10.2655325, -38.1570549, 10.1791325, -48.3985291, 48.4225883
23: -27.6427994, 7.5947022, -27.5760002, 7.5037246, -35.1465225, 35.1707039
24: -30.9654827, 7.8813744, -30.9148178, 7.8379116, -38.8033943, 38.7961922
25: -28.2949944, 11.1522045, -28.2400970, 11.1115532, -39.4065475, 39.3923035
26: -43.3404427, 8.1076651, -43.2635231, 7.9159060, -51.2563477, 51.3711891
27: -30.0144386, 14.0404959, -29.9438114, 13.9415464, -43.9559860, 43.9843063
28: -27.4603577, 9.9067059, -27.3925514, 9.8189507, -37.2793083, 37.2992554
29: -39.8731461, 10.6632719, -39.7894516, 10.5132151, -50.3863602, 50.4527245
30: -28.1386719, 14.6769276, -28.0554104, 14.5685940, -42.7072678, 42.7323380
31: -31.1826534, 8.4075069, -31.1059952, 8.2984104, -39.4810638, 39.5135040
32: -30.8919659, 12.0541935, -30.8605881, 11.9479399, -42.8399048, 42.9147797
33: -48.8883972, 9.3370838, -48.7038612, 9.2282400, -57.4396896, 57.4222412
34: -41.7874069, 7.5758181, -41.6949654, 7.5417643, -48.8868637, 48.9192810
35: -41.0435791, 9.5856686, -40.9388580, 9.5501089, -50.4222717, 50.3613358
36: -42.4096298, 9.9100933, -42.3554192, 9.8883057, -52.2979355, 52.2655106
37: -63.8739700, 2.1214371, -63.7721443, 1.9881916, -65.4675598, 65.5818481
38: -53.2105789, 12.1397209, -53.1451645, 12.0419712, -65.2525482, 65.2848816
39: -62.2729607, 6.0213223, -62.1009064, 5.8707762, -68.1437378, 68.1222305
40: -50.1061325, 9.3471880, -50.0029144, 9.2232399, -58.8914566, 58.9543076
41: -35.2670822, 6.7697840, -35.2141647, 6.6678286, -41.9349098, 41.9839478
42: -26.2642975, 7.8112421, -26.2019272, 7.6998911, -33.8802643, 33.9470253

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1777

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2358813, upper bound: 22.2128123
time: 76.82 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2358813, upper bound: 22.2625357
time: 64.07 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -29.8127708, 12.5697479, -29.8168869, 12.5766191, -42.3893890, 42.3866348
1: -14.3546324, 20.7015648, -14.3875942, 20.6974907, -35.0521240, 35.0891571
2: -10.2989283, 21.0538273, -10.2975616, 21.0338631, -31.3327904, 31.3513889
3: -12.5393991, 23.6833286, -12.5382118, 23.6613503, -36.2007484, 36.2215424
4: -15.7216187, 20.6750240, -15.6898756, 20.6695938, -36.3912125, 36.3648987
5: -10.8681316, 25.4420662, -10.8491049, 25.4234886, -36.2916183, 36.2911720
6: -32.0919914, 11.5578737, -32.0966797, 11.5562153, -42.7010193, 42.7199097
7: -16.9080009, 26.3571835, -16.9614086, 26.3439827, -43.2001762, 43.2381210
8: -18.3783951, 23.7266464, -18.4510345, 23.7140408, -42.0924377, 42.1776810
9: -17.0491085, 20.5949612, -17.0070343, 20.6226501, -37.6717606, 37.6019974
10: -29.9512024, 28.9710484, -29.9253578, 28.9250126, -58.8762131, 58.8964081
11: -34.8851700, 15.1960201, -34.8573608, 15.0274220, -49.9125900, 50.0533829
12: -34.7156410, 13.4729338, -34.6911469, 13.3078623, -48.0235023, 48.1640816
13: -29.4977016, 22.9098015, -29.5091457, 22.8911495, -52.3888512, 52.4189453
14: -52.3060188, 10.2839432, -52.4257736, 10.1033611, -61.0814819, 61.3273277
15: -22.7952595, 18.9536629, -22.7743702, 19.0451775, -41.8404388, 41.7280350
16: -30.7694073, 25.5760040, -30.7498589, 25.5893402, -56.3587494, 56.3258629
17: -56.0536423, 21.6395321, -56.1161499, 21.4250946, -77.4787369, 77.7556839
18: -30.8051491, 14.2257996, -30.7856331, 14.2187786, -45.0239258, 45.0114326
19: -29.6931686, 2.9285812, -29.6643906, 2.8958182, -32.5889854, 32.5929718
20: -21.8351135, 10.3492985, -21.8200188, 10.2638378, -32.0989532, 32.1693192
21: -33.5410652, 6.7321730, -33.5181274, 6.6193895, -40.1604538, 40.2503014
22: -38.2168770, 10.2566786, -38.2116547, 10.2045641, -48.4214401, 48.4683342
23: -27.6410599, 7.5932226, -27.6246681, 7.5439692, -35.1850281, 35.2178917
24: -30.9619904, 7.8801117, -30.9468155, 7.8751898, -38.8371811, 38.8269272
25: -28.2924767, 11.1482611, -28.2880230, 11.1583920, -39.4508667, 39.4362831
26: -43.3362885, 8.1053200, -43.3002586, 8.0363359, -51.3726234, 51.4055786
27: -30.0106850, 14.0385332, -29.9862862, 13.9810982, -43.9917831, 44.0248184
28: -27.4588070, 9.9048777, -27.4332008, 9.8711281, -37.3299332, 37.3380775
29: -39.8708763, 10.6589670, -39.8551636, 10.5475225, -50.4183998, 50.5141296
30: -28.1359119, 14.6727924, -28.1276302, 14.6122379, -42.7481499, 42.8004227
31: -31.1752357, 8.4058285, -31.1560822, 8.3792057, -39.5544434, 39.5619125
32: -30.8897572, 12.0516014, -30.9000816, 11.9902534, -42.8800125, 42.9516830
33: -48.8831635, 9.3356018, -48.7474556, 9.3402958, -57.5492401, 57.4663849
34: -41.7845230, 7.5741978, -41.7243500, 7.6077175, -48.9534225, 48.9442673
35: -41.0402946, 9.5841074, -40.9660950, 9.5929232, -50.4623871, 50.3893433
36: -42.4064026, 9.9090157, -42.3831367, 9.9310989, -52.3375015, 52.2921524
37: -63.8648720, 2.1206093, -63.8555107, 2.2451582, -65.7157288, 65.6686020
38: -53.2068787, 12.1376057, -53.1954269, 12.0967455, -65.3036270, 65.3330307
39: -62.2666206, 6.0203056, -62.1690826, 6.0052605, -68.2718811, 68.1893921
40: -50.0980721, 9.3455267, -50.0813293, 9.4609594, -59.1235428, 59.0324936
41: -35.2627182, 6.7674189, -35.2640190, 6.8214083, -42.0841255, 42.0314369
42: -26.2606430, 7.8086290, -26.2366009, 7.7575598, -33.9460831, 33.9731827

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1777

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1684

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2823997, upper bound: 22.2321669
time: 68.20 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.2823997, upper bound: 22.2588173
time: 59.14 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 129.26 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.2691117, upper bound: 22.1479235
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.2591718, upper bound: 22.1937816
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.2691117, upper bound: 22.1479235
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.2591718, upper bound: 22.1937816
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.3337840, upper bound: 22.1527066
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.2591718, upper bound: 22.1988680
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.3337840, upper bound: 22.1527066
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.2591718, upper bound: 22.1988680
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.2024980, upper bound: 22.3003074
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.2024980, upper bound: 22.3049778
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.2024980, upper bound: 22.3003068
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.2024980, upper bound: 22.3049772
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.3266211, upper bound: 22.2841268
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.3912577, upper bound: 22.2887722
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.3266211, upper bound: 22.2841260
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.3912577, upper bound: 22.2887722
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.2358813, upper bound: 22.2128123
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.2358813, upper bound: 22.2625357
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.2823997, upper bound: 22.2321669
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 129.26
Output dim: 3, lower bound: -22.2823997, upper bound: 22.2588173
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 129.26
Output dim: 3, lower bound: -22.3004617, upper bound: 22.2772335
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 129.26
Output dim: 3, lower bound: -22.3508732, upper bound: 22.2677654
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 129.26
Output dim: 3, lower bound: -22.2878533, upper bound: 22.3103690
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 129.26
Output dim: 3, lower bound: -22.2878533, upper bound: 22.3748619
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 129.26
Output dim: 3, lower bound: -22.2878533, upper bound: 22.2953968
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 129.26
Output dim: 3, lower bound: -22.2878533, upper bound: 22.2282744

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 76.26 + 3556.02 = 3632.28 seconds
