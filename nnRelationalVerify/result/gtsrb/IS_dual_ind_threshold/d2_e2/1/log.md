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
execution time: IAR + RelationalAnalysis = 2.82 + 71.74 = 74.56 seconds
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
time: 61.31 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4219063, upper bound: 22.4219060
time: 67.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 129.23 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 129.23
Output dim: 3, lower bound: -22.4219063, upper bound: 22.3507611
IS_A2, status: Status.UNKNOWN, split count: 1, time: 129.23
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=213, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3759437, upper bound: 22.2240739
time: 61.88 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -22.4194264, upper bound: 22.3482940
time: 54.17 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=213, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3759437, upper bound: 22.2937689
time: 63.77 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.3759437, upper bound: 22.2937689
time: 54.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 120.85 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 120.85
Output dim: 3, lower bound: -22.3759437, upper bound: 22.2240739
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 120.85
Output dim: 3, lower bound: -22.4194264, upper bound: 22.3482940
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 120.85
Output dim: 3, lower bound: -22.3759437, upper bound: 22.2937689
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 120.85
Output dim: 3, lower bound: -22.3759437, upper bound: 22.2937689

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=350, inp2_unstable=350, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=22, inp2_unstable=22, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

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
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.2937692, upper bound: 22.3060532
time: 66.59 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -22.2937692, upper bound: 22.3482947
time: 48.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 117.33 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 117.33
Output dim: 3, lower bound: -22.2937692, upper bound: 22.3060532
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 117.33
Output dim: 3, lower bound: -22.2937692, upper bound: 22.3482947

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 74.56 + 485.75 = 560.31 seconds
