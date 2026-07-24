## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 5)
Time budget: 1800 seconds
Split limit: 100
Threshold: 4.5670275009


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6798630, 15.6798630)
1: (0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2691345, 9.2691345)
2: (2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0294800, 9.0294800)
3: (1.5895219, 14.1851063, 1.5895219, 14.1851063, -9.0066338, 9.0066338)
4: (-4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.6086578, 12.6086655)
5: (2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4130630, 8.4130630)
6: (-25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3687363, 13.3687363)
7: (2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3604660, 9.3604660)
8: (-4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6617126, 15.6617050)
9: (0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2787743, 9.2787743)
10: (-4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0734787, 12.0734749)
11: (-4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5991440, 8.5991440)
12: (-26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9329147, 10.9329147)
13: (-14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3786621, 13.3786659)
14: (-24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3273315, 16.3273315)
15: (-7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.3071594, 11.3071594)
16: (-7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4354362, 9.4354362)
17: (-26.7557220, -11.1057987, -26.7557220, -11.1057987, -11.0222092, 11.0222092)
18: (-17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.6050987, 10.6050949)
19: (-10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8847923, 6.8847942)
20: (-5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5516205, 7.5516224)
21: (-8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7141228, 9.7141228)
22: (-10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8515701, 7.8515720)
23: (-4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8088226, 8.8088226)
24: (-8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5757370, 10.5757370)
25: (-8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3544083, 8.3544102)
26: (-16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6793213, 11.6793251)
27: (-7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2137756, 12.2137794)
28: (-6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1729126, 10.1729126)
29: (-7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6785126, 8.6785126)
30: (-3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3045807, 12.3045769)
31: (-14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8111343, 10.8111343)
32: (-20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0389709, 12.0389671)
33: (-38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6259651, 10.6259651)
34: (-35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5543938, 11.5543938)
35: (-33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3333244, 11.3333244)
36: (-31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6401138, 12.6401138)
37: (-50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9371605, 9.9371586)
38: (-38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4227600, 11.4227600)
39: (-42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6862717, 10.6862717)
40: (-38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9440098, 7.9440098)
41: (-24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1554718, 13.1554680)
42: (-15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1264839, 8.1264858)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.54 + 35.14 = 37.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -4.5715991, upper bound: 4.5715991

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 875

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 571

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5563961, upper bound: 4.5713117
time: 40.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5713117, upper bound: 4.5563961
time: 34.24 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 74.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 74.59
Output dim: 3, lower bound: -4.5563961, upper bound: 4.5713117
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 74.59
Output dim: 3, lower bound: -4.5713117, upper bound: 4.5563961

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6633606, 15.6674500
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2653236, 9.2660446
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0235214, 9.0250015
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9938965, 8.9960289
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5891876, 12.5940094
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4026756, 8.4045868
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3598251, 13.3579330
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3633804, 9.3631554
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6662216, 15.6658783
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2733841, 9.2747078
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0662727, 12.0639381
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5780792, 8.5711441
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9261818, 10.9239998
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3513680, 13.3565521
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3261032, 16.3240128
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2837105, 11.2895126
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4362984, 9.4365082
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -11.0166779, 11.0172615
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.6007652, 10.6004944
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8806572, 6.8799953
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5363560, 7.5303764
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7041168, 9.7009888
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8560181, 7.8575535
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7914162, 8.7855873
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5683289, 10.5655899
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3481865, 8.3458710
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6673393, 11.6624298
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2016144, 12.1973114
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1634903, 10.1599045
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6802444, 8.6805496
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2919617, 12.2860260
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8115501, 10.8115883
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0336990, 12.0327988
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6059761, 10.6112595
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5578728, 11.5582962
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3068695, 11.3126869
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6395721, 12.6397400
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9361916, 9.9368839
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4223213, 11.4221649
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6848793, 10.6878357
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9442062, 7.9441662
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1505432, 13.1499443
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1085739, 8.1026649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1593

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5547421, upper bound: 4.5706001
time: 19.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5556856, upper bound: 4.5696555
time: 33.44 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6674500, 15.6633606
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2660446, 9.2653236
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0249977, 9.0235214
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9960289, 8.9938965
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5940094, 12.5891876
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4045868, 8.4026756
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3579330, 13.3598289
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3631554, 9.3633804
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6658783, 15.6662140
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2747078, 9.2733841
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0639381, 12.0662727
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5711441, 8.5780792
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9239998, 10.9261818
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3565483, 13.3513680
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3240128, 16.3261032
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2895088, 11.2837067
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4365082, 9.4363003
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -11.0172615, 11.0166779
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.6004944, 10.6007652
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8799934, 6.8806553
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5303745, 7.5363541
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7009850, 9.7041168
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8575554, 7.8560181
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7855873, 8.7914162
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5655899, 10.5683289
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3458710, 8.3481865
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6624298, 11.6673393
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1973114, 12.2016144
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1599045, 10.1634903
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6805496, 8.6802444
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2860260, 12.2919617
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8115883, 10.8115501
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0327988, 12.0336990
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6112595, 10.6059761
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5582962, 11.5578766
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3126907, 11.3068695
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6397400, 12.6395721
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9368858, 9.9361916
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4221649, 11.4223213
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6878357, 10.6848793
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9441643, 7.9442062
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1499481, 13.1505432
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1026649, 8.1085758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 767

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1781

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5712441, upper bound: 4.5563278
time: 17.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5712434, upper bound: 4.5563285
time: 38.95 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 58.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 58.12
Output dim: 3, lower bound: -4.5547421, upper bound: 4.5706001
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 58.12
Output dim: 3, lower bound: -4.5556856, upper bound: 4.5696555
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 58.12
Output dim: 3, lower bound: -4.5712441, upper bound: 4.5563278
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 58.12
Output dim: 3, lower bound: -4.5712434, upper bound: 4.5563285

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6659241, 15.6658401
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2653618, 9.2659645
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0238266, 9.0247879
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9926643, 8.9974976
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5908813, 12.5925293
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4023209, 8.4043922
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3546143, 13.3604202
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3612633, 9.3636246
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6657944, 15.6653137
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2730484, 9.2749481
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0644646, 12.0640755
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5762100, 8.5750351
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9235001, 10.9242249
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3438683, 13.3585129
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3249817, 16.3222961
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2855453, 11.2825279
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4340286, 9.4394073
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -11.0164032, 11.0169296
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.6038284, 10.5894089
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8804436, 6.8799038
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5362206, 7.5302563
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7032967, 9.7041245
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8568649, 7.8537216
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7915268, 8.7849960
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5700989, 10.5635757
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3479881, 8.3452511
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6690331, 11.6548347
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2031250, 12.1929283
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1634254, 10.1586838
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6803398, 8.6795540
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2908249, 12.2889709
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8114738, 10.8113937
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0293503, 12.0336685
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5989647, 10.6132240
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5563545, 11.5612984
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3029671, 11.3170395
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6375580, 12.6415176
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9340591, 9.9368362
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4210815, 11.4223213
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6749229, 10.6907825
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9435520, 7.9443474
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1466827, 13.1513062
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1039848, 8.1044598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 750

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5546519, upper bound: 4.5636368
time: 10.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5477788, upper bound: 4.5705099
time: 28.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6617432, 15.6674500
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2652473, 9.2660446
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0233116, 9.0250015
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9938965, 8.9947968
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5876999, 12.5940094
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4026756, 8.4042320
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3598251, 13.3527222
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3633804, 9.3610344
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6662216, 15.6654510
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2733841, 9.2743721
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0662727, 12.0621300
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5780792, 8.5692787
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9261818, 10.9213181
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3513680, 13.3490562
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3243866, 16.3240128
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2767181, 11.2895126
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4362984, 9.4342384
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -11.0163460, 11.0172615
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5896797, 10.6004944
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8806572, 6.8797817
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5362358, 7.5303764
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7041168, 9.7001686
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8521881, 7.8575535
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7908211, 8.7855873
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5663147, 10.5655899
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3475647, 8.3458710
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6597481, 11.6624298
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1972351, 12.1973114
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1622696, 10.1599045
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6792488, 8.6805496
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2919617, 12.2848892
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8113556, 10.8115883
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0336990, 12.0284538
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6059761, 10.6042480
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5578728, 11.5567741
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3068695, 11.3087845
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6395721, 12.6377258
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9361916, 9.9347534
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4223213, 11.4209270
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6848793, 10.6778793
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9442062, 7.9435139
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1505432, 13.1460876
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1085739, 8.0980740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 688

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 670

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5400554, upper bound: 4.5696510
time: 32.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5556811, upper bound: 4.5540297
time: 36.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6476669, 15.6480141
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2596550, 9.2605515
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0272293, 9.0254097
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9969139, 8.9951019
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5932236, 12.5881691
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4019241, 8.4006920
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3590775, 13.3609390
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3643074, 9.3637009
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6590500, 15.6572876
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2727776, 9.2733765
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0532799, 12.0582657
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5729942, 8.5772705
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8985939, 10.9063301
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3466072, 13.3444862
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2909622, 16.3007660
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2931862, 11.2883492
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4186630, 9.4218750
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9783173, 10.9867592
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5866051, 10.5902290
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8682117, 6.8650570
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5135117, 7.5148277
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6877403, 9.6866837
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8516960, 7.8482666
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7753830, 8.7766533
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5667572, 10.5653152
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3399200, 8.3398571
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6516762, 11.6533165
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1825180, 12.1821823
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1497459, 10.1500893
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6786613, 8.6770859
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2787170, 12.2809219
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8125648, 10.8104019
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0350418, 12.0356369
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6058884, 10.5972080
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5578651, 11.5574303
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3121719, 11.3061180
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6347275, 12.6329460
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9366455, 9.9349499
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4175301, 11.4171944
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6758842, 10.6688232
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9468422, 7.9474545
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1460114, 13.1437302
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0960312, 8.1000862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1594

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5695998, upper bound: 4.5552833
time: 44.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5695998, upper bound: 4.5562241
time: 24.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6521072, 15.6435738
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2612686, 9.2589378
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0268860, 9.0257530
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9972343, 8.9947815
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5929947, 12.5883980
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4025993, 8.4000168
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3590469, 13.3609695
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3634758, 9.3645325
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6569443, 15.6593857
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2747002, 9.2714500
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0559349, 12.0556107
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5703392, 8.5799294
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9041443, 10.9007759
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3496742, 13.3414268
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2986679, 16.2930527
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2941551, 11.2873764
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4220848, 9.4184532
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9873428, 10.9777374
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5899582, 10.5868797
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8643970, 6.8688755
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5088463, 7.5194893
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6835556, 9.6908722
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8498001, 7.8501625
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7708282, 8.7812080
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5625763, 10.5694885
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3375435, 8.3422337
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6484070, 11.6565895
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1778793, 12.1868210
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1465034, 10.1533318
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6773872, 8.6783562
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2749863, 12.2846489
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8104439, 10.8125267
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0347366, 12.0359383
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6024933, 10.6006031
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5578499, 11.5574455
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3119392, 11.3063507
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6331100, 12.6345558
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9356461, 9.9359531
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4170418, 11.4176826
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6717796, 10.6729298
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9474144, 7.9468842
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1431274, 13.1466141
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0941772, 8.1019440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 749

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5470654, upper bound: 4.5470661
time: 38.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5619810, upper bound: 4.5561702
time: 22.14 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 61.92 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 61.92
Output dim: 3, lower bound: -4.5546519, upper bound: 4.5636368
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 61.92
Output dim: 3, lower bound: -4.5477788, upper bound: 4.5705099
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 61.92
Output dim: 3, lower bound: -4.5400554, upper bound: 4.5696510
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 61.92
Output dim: 3, lower bound: -4.5556811, upper bound: 4.5540297
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 61.92
Output dim: 3, lower bound: -4.5695998, upper bound: 4.5552833
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 61.92
Output dim: 3, lower bound: -4.5695998, upper bound: 4.5562241
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 61.92
Output dim: 3, lower bound: -4.5470654, upper bound: 4.5470661
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 61.92
Output dim: 3, lower bound: -4.5619810, upper bound: 4.5561702

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6608124, 15.6619911
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2618408, 9.2633133
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0191422, 9.0212631
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9919777, 8.9969902
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5868835, 12.5890350
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4011955, 8.4035492
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3409882, 13.3437309
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3634415, 9.3666992
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6582947, 15.6589355
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2691727, 9.2720337
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0571365, 12.0585403
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5692596, 8.5657310
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9212761, 10.9212952
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3425827, 13.3573761
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3152084, 16.3141212
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2776489, 11.2739792
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4260445, 9.4334068
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -11.0034981, 11.0070267
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.6039925, 10.5894928
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8784313, 6.8771858
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5265083, 7.5173340
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6985703, 9.6975670
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8572083, 7.8539543
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7919235, 8.7851562
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5714531, 10.5643806
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3480644, 8.3453064
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6656876, 11.6503868
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2012405, 12.1899681
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1630554, 10.1581955
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6799698, 8.6789970
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2908325, 12.2880287
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8092308, 10.8079491
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0197830, 12.0214767
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5921707, 10.6041908
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5550690, 11.5595932
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3013420, 11.3148766
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6296806, 12.6310387
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9321976, 9.9343605
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4065170, 11.4029522
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6670799, 10.6803513
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9366379, 7.9351482
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1418991, 13.1449432
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0956230, 8.0945396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 674

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 638

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5234777, upper bound: 4.5690275
time: 31.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5462995, upper bound: 4.5462077
time: 26.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6535110, 15.6613464
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2527924, 9.2567024
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0136490, 9.0176926
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9811745, 8.9852600
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5675049, 12.5788422
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3921967, 8.3963737
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3632126, 13.3565903
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3516388, 9.3518524
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6545258, 15.6566696
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2727966, 9.2738457
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0677185, 12.0626259
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5746536, 8.5642586
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9105988, 10.9005470
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3299789, 13.3201065
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3176727, 16.3138962
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2686081, 11.2834320
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4286118, 9.4284515
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9965324, 10.9908714
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5824585, 10.5992699
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8785858, 6.8763161
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5357189, 7.5297947
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7040329, 9.6999931
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8481789, 7.8520336
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7829323, 8.7753792
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5671844, 10.5663528
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3383217, 8.3346748
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6604118, 11.6629028
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1891174, 12.1930923
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1601562, 10.1568680
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6741447, 8.6723213
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2837296, 12.2737350
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8131676, 10.8129272
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0317879, 12.0262566
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5978203, 10.5933800
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5569305, 11.5559273
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3002892, 11.3000298
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6319084, 12.6275101
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9195747, 9.9126015
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4241791, 11.4227295
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6715355, 10.6600971
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9481201, 7.9480915
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1495132, 13.1450615
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1008720, 8.0909824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1524

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 638

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5157440, upper bound: 4.5681708
time: 24.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5385815, upper bound: 4.5453500
time: 28.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6485367, 15.6458817
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2615395, 9.2614899
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0309486, 9.0280266
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9992981, 8.9977531
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5903931, 12.5833054
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4006538, 8.3993034
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3391953, 13.3459969
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3647156, 9.3645515
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6581955, 15.6563416
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2697411, 9.2698936
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0518684, 12.0571327
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5701752, 8.5776291
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8889847, 10.8991127
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3380508, 13.3401337
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2880707, 16.2977638
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2766953, 11.2666321
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4207687, 9.4250450
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9761047, 10.9845543
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5759697, 10.5730438
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8674316, 6.8665543
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5156116, 7.5182133
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6892090, 9.6923141
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8485374, 7.8439445
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7754745, 8.7769241
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5683479, 10.5663414
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3393822, 8.3398285
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6436882, 11.6426048
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1785889, 12.1766701
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1501579, 10.1504593
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6785469, 8.6769562
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2821198, 12.2870140
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8087349, 10.8075294
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0216904, 12.0256119
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5819969, 10.5787277
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5641937, 11.5666695
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3145638, 11.3130913
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6387138, 12.6402321
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9217873, 9.9226036
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4064140, 11.4101295
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6411858, 10.6417313
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9468517, 7.9479294
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1315155, 13.1329346
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0832787, 8.0902367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 670

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5555184, upper bound: 4.5552789
time: 47.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5539739, upper bound: 4.5396554
time: 25.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6455460, 15.6488800
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2605934, 9.2624245
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0298462, 9.0291328
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9994965, 8.9974861
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5883560, 12.5853424
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4005356, 8.3993988
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3441315, 13.3410568
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3651581, 9.3641090
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6581039, 15.6564331
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2692909, 9.2703323
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0520973, 12.0568581
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5733528, 8.5744514
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8913727, 10.8967247
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3418274, 13.3359261
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2879639, 16.2978630
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2714691, 11.2718544
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4217834, 9.4239845
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9761124, 10.9844971
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5694199, 10.5794373
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8696518, 6.8642731
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5168972, 7.5169277
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6932755, 9.6881523
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8473740, 7.8449955
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7756081, 8.7767487
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5677834, 10.5667610
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3398895, 8.3393192
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6409645, 11.6451302
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1770096, 12.1782761
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1501083, 10.1505013
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6785355, 8.6769676
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2848053, 12.2843285
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8096733, 10.8065720
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0250168, 12.0222893
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5868225, 10.5733166
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5670166, 11.5637550
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3186035, 11.3085098
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6419868, 12.6369362
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9241867, 9.9200916
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4104614, 11.4060822
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6484299, 10.6341228
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9472942, 7.9474640
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1351547, 13.1292267
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0861816, 8.0873337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1664

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 638

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5452987, upper bound: 4.5547430
time: 19.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5681198, upper bound: 4.5319220
time: 38.02 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 59.61 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 59.61
Output dim: 3, lower bound: -4.5234777, upper bound: 4.5690275
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 59.61
Output dim: 3, lower bound: -4.5462995, upper bound: 4.5462077
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 59.61
Output dim: 3, lower bound: -4.5157440, upper bound: 4.5681708
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 59.61
Output dim: 3, lower bound: -4.5385815, upper bound: 4.5453500
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 59.61
Output dim: 3, lower bound: -4.5555184, upper bound: 4.5552789
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 59.61
Output dim: 3, lower bound: -4.5539739, upper bound: 4.5396554
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 59.61
Output dim: 3, lower bound: -4.5452987, upper bound: 4.5547430
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 59.61
Output dim: 3, lower bound: -4.5681198, upper bound: 4.5319220

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6588898, 15.6621513
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2590981, 9.2635536
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0164986, 9.0214958
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9876785, 8.9973564
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5836868, 12.5893059
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3976822, 8.4038467
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3409348, 13.3447456
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3610954, 9.3669052
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6560516, 15.6591415
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2681694, 9.2720184
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0599976, 12.0582085
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5696602, 8.5630608
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9214478, 10.9189682
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3414154, 13.3562775
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3172379, 16.3140297
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2754669, 11.2733917
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4259644, 9.4335499
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -11.0043068, 11.0034637
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.6038780, 10.5890541
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8787556, 6.8741684
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5268097, 7.5156078
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6988792, 9.6967125
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8575706, 7.8525314
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7922134, 8.7817955
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5716896, 10.5616608
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3484535, 8.3414612
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6658478, 11.6489410
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2012177, 12.1900787
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1633224, 10.1550789
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6803436, 8.6774712
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2913208, 12.2849159
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8095284, 10.8051643
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0197754, 12.0215721
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5921631, 10.6040173
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5542793, 11.5586548
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3013000, 11.3143082
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6296272, 12.6306267
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9328194, 9.9319687
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4057007, 11.4015083
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6672745, 10.6801224
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9354248, 7.9343147
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1418915, 13.1450958
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0937386, 8.0934868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1399

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 838

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5224946, upper bound: 4.5680419
time: 22.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5224946, upper bound: 4.5680419
time: 22.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6515884, 15.6615067
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2500496, 9.2569427
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0110054, 9.0179214
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9768791, 8.9856262
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5643158, 12.5791130
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3886871, 8.3966751
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3631668, 13.3576088
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3492889, 9.3520584
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6522598, 15.6568680
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2717896, 9.2738342
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0705719, 12.0622864
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5750542, 8.5615883
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9107704, 10.8982201
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3288040, 13.3190002
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3197098, 16.3138046
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2664185, 11.2828407
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4285278, 9.4285927
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9973335, 10.9873009
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5823479, 10.5988426
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8789139, 6.8733006
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5360241, 7.5280743
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7043343, 9.6991348
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8485451, 7.8506165
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7832146, 8.7720261
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5674210, 10.5636253
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3387108, 8.3308258
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6605644, 11.6614532
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1891022, 12.1932106
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1604271, 10.1537514
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6745224, 8.6707878
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2842255, 12.2706337
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8134766, 10.8101501
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0317955, 12.0263367
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5978088, 10.5932045
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5561447, 11.5549927
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3002434, 11.2994537
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6318550, 12.6270943
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9201889, 9.9102020
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4233627, 11.4212856
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6717415, 10.6598778
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9469051, 7.9472561
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1495056, 13.1452141
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0989914, 8.0899391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1680

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5157249, upper bound: 4.5668951
time: 34.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5144709, upper bound: 4.5681524
time: 32.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6455460, 15.6469650
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2605934, 9.2596779
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0298462, 9.0264854
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9994965, 8.9931946
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5883560, 12.5821495
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4005356, 8.3958931
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3441315, 13.3410034
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3651581, 9.3617592
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6581039, 15.6541672
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2692909, 9.2693214
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0517578, 12.0568581
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5706863, 8.5744514
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8890419, 10.8967247
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3418274, 13.3347511
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2878723, 16.2978630
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2714691, 11.2696724
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4217834, 9.4239044
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9725418, 10.9844971
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5689888, 10.5794373
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8666363, 6.8642731
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5151787, 7.5169277
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6924171, 9.6881523
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8459511, 7.8449955
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7722511, 8.7767487
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5650597, 10.5667610
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3360443, 8.3393192
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6395187, 11.6451302
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1770096, 12.1782684
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1469994, 10.1505013
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6770096, 8.6769676
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2816925, 12.2843285
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8068848, 10.8065720
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0250168, 12.0222855
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5866470, 10.5733166
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5660820, 11.5637550
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3186035, 11.3084641
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6419868, 12.6368866
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9217873, 9.9200916
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4104614, 11.4052639
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6482086, 10.6341228
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9464645, 7.9474640
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1351547, 13.1292229
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0851364, 8.0873337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 765

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 670

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5525036, upper bound: 4.5319175
time: 48.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5681153, upper bound: 4.5162854
time: 42.46 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 92.77 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 92.77
Output dim: 3, lower bound: -4.5224946, upper bound: 4.5680419
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 92.77
Output dim: 3, lower bound: -4.5224946, upper bound: 4.5680419
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 92.77
Output dim: 3, lower bound: -4.5157249, upper bound: 4.5668951
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 92.77
Output dim: 3, lower bound: -4.5144709, upper bound: 4.5681524
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 92.77
Output dim: 3, lower bound: -4.5525036, upper bound: 4.5319175
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 92.77
Output dim: 3, lower bound: -4.5681153, upper bound: 4.5162854

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6585312, 15.6619644
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2584457, 9.2637444
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0155640, 9.0217514
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9875069, 8.9971199
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5829773, 12.5895271
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3972778, 8.4037971
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3410492, 13.3427734
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3605194, 9.3666458
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6541748, 15.6590195
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2693214, 9.2719688
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0616455, 12.0572929
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5698395, 8.5628548
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9215775, 10.9165421
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3424034, 13.3551826
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3171005, 16.3139267
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2753792, 11.2733688
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4264183, 9.4334564
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -11.0029449, 11.0035210
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.6027603, 10.5891838
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8784542, 6.8744602
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5266609, 7.5154629
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6985588, 9.6967888
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8573074, 7.8528557
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7914200, 8.7816048
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5704079, 10.5616379
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3469048, 8.3412018
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6652107, 11.6488113
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2005997, 12.1907806
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1626968, 10.1552734
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6802788, 8.6774178
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2913666, 12.2849007
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8090744, 10.8059464
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0199165, 12.0189743
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5937462, 10.6039581
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5558510, 11.5573845
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3021469, 11.3137856
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6293831, 12.6287727
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9341316, 9.9312897
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4054146, 11.4002419
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6676483, 10.6783333
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9353065, 7.9318752
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1419411, 13.1432343
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0938072, 8.0922604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1399

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1782

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5223908, upper bound: 4.5621542
time: 31.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5166078, upper bound: 4.5679380
time: 15.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6588898, 15.6618042
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2590981, 9.2629013
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0164986, 9.0205612
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9876785, 8.9971848
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5836868, 12.5886040
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3976822, 8.4034424
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3389664, 13.3447456
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3610954, 9.3663292
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6560516, 15.6572571
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2681198, 9.2720184
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0590820, 12.0582085
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5694542, 8.5630608
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9190254, 10.9189682
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3403206, 13.3562775
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3172379, 16.3138885
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2754402, 11.2733917
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4258728, 9.4335499
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -11.0043068, 11.0021019
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.6038780, 10.5879364
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8787556, 6.8738670
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5268097, 7.5154572
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6988792, 9.6963921
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8575706, 7.8522682
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7922134, 8.7810020
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5716896, 10.5603752
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3484535, 8.3399143
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6658478, 11.6483040
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2012177, 12.1894646
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1633224, 10.1544533
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6803436, 8.6774063
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2913055, 12.2849159
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8095284, 10.8047104
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0171776, 12.0215721
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5921021, 10.6040173
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5530090, 11.5586548
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3007774, 11.3143082
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6277733, 12.6306267
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9321404, 9.9319687
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4044342, 11.4015083
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6654854, 10.6801224
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9329834, 7.9343147
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1400337, 13.1450958
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0925102, 8.0934868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1632

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5212578, upper bound: 4.5672084
time: 44.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5216684, upper bound: 4.5667984
time: 51.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6505432, 15.6606789
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2487030, 9.2566299
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0077248, 9.0155640
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9767227, 8.9871788
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5605774, 12.5776405
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3853722, 8.3957062
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3532715, 13.3520775
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3471451, 9.3521996
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6521378, 15.6567383
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2665443, 9.2673531
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0704422, 12.0603638
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5745125, 8.5610161
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9121399, 10.8993111
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3257675, 13.3158417
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3082275, 16.2981377
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2636108, 11.2782097
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4270706, 9.4272346
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9922333, 10.9804649
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5809593, 10.5979424
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8782158, 6.8725262
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5347862, 7.5265274
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7050133, 9.6998711
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8464699, 7.8478889
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7755623, 8.7618103
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5644531, 10.5597305
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3304138, 8.3197594
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6494560, 11.6466446
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1892395, 12.1933670
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1549721, 10.1464691
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6735115, 8.6696091
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2850647, 12.2714157
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8133888, 10.8100662
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0290222, 12.0250130
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5879059, 10.5858898
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5578957, 11.5584259
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.2955894, 11.2973862
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6307449, 12.6274834
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9204636, 9.9104080
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4171982, 11.4176159
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6641541, 10.6539707
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9456806, 7.9461632
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1462517, 13.1441193
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0993385, 8.0888805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 825

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1784

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5139461, upper bound: 4.5551286
time: 20.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5014402, upper bound: 4.5676259
time: 36.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6394348, 15.6387177
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2512627, 9.2472343
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0225372, 9.0168266
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9899559, 8.9804726
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5731888, 12.5619774
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3926811, 8.3854141
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3479881, 13.3443794
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3559647, 9.3500099
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6493149, 15.6424789
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2687607, 9.2687302
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0522652, 12.0583153
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5656700, 8.5710335
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8682709, 10.8811417
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3128738, 13.3133621
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2777634, 16.2911644
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2653809, 11.2615509
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4160080, 9.4162197
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9461517, 10.9646721
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5677567, 10.5722046
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8631744, 6.8621941
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5145950, 7.5164127
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6922417, 9.6880684
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8404503, 7.8409920
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7620468, 8.7688560
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5658188, 10.5676308
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3248520, 8.3300819
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6399956, 11.6457977
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1728058, 12.1701546
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1439438, 10.1483841
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6687775, 8.6718597
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2705536, 12.2761040
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8082237, 10.8083801
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0228004, 12.0203857
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5757828, 10.5651703
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5652428, 11.5628052
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3098450, 11.3018913
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6317825, 12.6292305
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.8996391, 9.9034729
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4122581, 11.4071159
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6304283, 10.6207924
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9510384, 7.9513741
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1341362, 13.1281967
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0780296, 8.0796242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1772

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5535408, upper bound: 4.5157238
time: 31.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5675544, upper bound: 4.5016902
time: 28.80 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 62.34 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 62.34
Output dim: 3, lower bound: -4.5223908, upper bound: 4.5621542
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 62.34
Output dim: 3, lower bound: -4.5166078, upper bound: 4.5679380
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 62.34
Output dim: 3, lower bound: -4.5212578, upper bound: 4.5672084
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 62.34
Output dim: 3, lower bound: -4.5216684, upper bound: 4.5667984
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 62.34
Output dim: 3, lower bound: -4.5139461, upper bound: 4.5551286
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 62.34
Output dim: 3, lower bound: -4.5014402, upper bound: 4.5676259
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 62.34
Output dim: 3, lower bound: -4.5535408, upper bound: 4.5157238
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 62.34
Output dim: 3, lower bound: -4.5675544, upper bound: 4.5016902

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6494522, 15.6498795
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2584114, 9.2632370
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0128250, 9.0204086
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9865379, 8.9962311
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5764847, 12.5845451
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3975945, 8.4041252
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3403587, 13.3419456
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3540344, 9.3616829
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6429520, 15.6504822
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2685585, 9.2712975
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0605774, 12.0555954
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5733566, 8.5667305
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9050446, 10.8943787
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3461304, 13.3562469
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2963486, 16.2862473
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2764816, 11.2743263
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4226418, 9.4296989
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9763870, 10.9683495
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5924454, 10.5754967
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8721924, 6.8709488
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5156174, 7.5071316
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6939240, 9.6947060
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8559265, 7.8517780
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7963638, 8.7893257
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5780640, 10.5717392
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3471870, 8.3416061
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6581764, 11.6427116
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1932755, 12.1870689
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1573601, 10.1518021
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6808434, 8.6780472
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2947769, 12.2903519
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8124008, 10.8099823
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0211182, 12.0202408
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5991287, 10.6114216
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5536537, 11.5546570
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3008804, 11.3123550
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6297607, 12.6293449
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9353676, 9.9327507
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4101639, 11.4034557
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6580048, 10.6718674
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9354687, 7.9320126
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1490021, 13.1530304
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0902939, 8.0895729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1596

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1766

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5165788, upper bound: 4.5575839
time: 33.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5093190, upper bound: 4.5679114
time: 50.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6587830, 15.6617279
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2587166, 9.2631721
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0167656, 9.0205498
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9875870, 8.9971428
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5845795, 12.5885658
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3980217, 8.4033852
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3392029, 13.3442307
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3609543, 9.3662682
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6560516, 15.6572723
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2675858, 9.2719154
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0586548, 12.0578957
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5694427, 8.5631256
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9190102, 10.9190445
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3393364, 13.3562737
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3166733, 16.3142815
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2761536, 11.2733345
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4253769, 9.4332123
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -11.0037117, 11.0025520
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.6049232, 10.5866814
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8789787, 6.8734989
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5269814, 7.5151196
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6988525, 9.6962051
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8575783, 7.8519745
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7922134, 8.7810097
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5715218, 10.5602837
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3481560, 8.3401756
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6668854, 11.6472168
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2012100, 12.1889954
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1633186, 10.1544838
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6803360, 8.6773071
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2911072, 12.2854080
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8097954, 10.8042831
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0171204, 12.0214424
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5923309, 10.6039257
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5530014, 11.5586433
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3008156, 11.3141403
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6277962, 12.6302719
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9328499, 9.9315567
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4049721, 11.4007874
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6654778, 10.6801128
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9334412, 7.9335423
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1403046, 13.1446075
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0925102, 8.0939255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 767

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5182979, upper bound: 4.5639457
time: 21.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5182979, upper bound: 4.5671602
time: 19.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6497498, 15.6596947
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2441139, 9.2548981
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0030098, 9.0137596
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9727478, 8.9856987
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5573730, 12.5764427
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3827438, 8.3947067
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3531036, 13.3526611
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3414421, 9.3500710
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6439896, 15.6536789
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2618828, 9.2656136
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0714798, 12.0600624
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5726700, 8.5574074
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9099121, 10.8935890
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3203239, 13.3137741
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3067780, 16.2942619
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2643127, 11.2780609
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4265327, 9.4312363
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9904556, 10.9756927
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5757408, 10.5839424
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8769035, 6.8699341
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5355263, 7.5255604
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7038879, 9.6969376
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8447723, 7.8431759
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7736320, 8.7583809
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5631104, 10.5561218
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3290520, 8.3173885
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6457748, 11.6369705
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1891251, 12.1941566
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1542587, 10.1449165
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6725731, 8.6670837
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2850189, 12.2714500
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8112869, 10.8051262
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0289001, 12.0249138
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5878754, 10.5860023
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5562286, 11.5539474
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.2948380, 11.2958145
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6304932, 12.6271362
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9184761, 9.9053516
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4154968, 11.4106903
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6597176, 10.6522217
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9442978, 7.9441090
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1461182, 13.1449089
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0982170, 8.0885620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1664

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5011074, upper bound: 4.5665480
time: 30.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5007039, upper bound: 4.5673256
time: 50.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6244583, 15.6189461
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2462921, 9.2410011
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0140877, 9.0057793
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9791641, 8.9665451
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5702286, 12.5581894
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3802109, 8.3689003
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3535385, 13.3492851
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3459778, 9.3369637
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6423187, 15.6335678
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2604294, 9.2585564
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0437469, 12.0485878
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5641518, 8.5696297
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8651237, 10.8787460
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3081436, 13.3078728
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2740021, 16.2868271
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2679596, 11.2646866
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4119072, 9.4070950
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9367905, 10.9522133
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5651627, 10.5701027
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8523102, 6.8533039
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.4951401, 7.5010681
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6768646, 9.6756134
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8162155, 7.8211918
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7499008, 8.7597275
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5487976, 10.5544205
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3044167, 8.3138885
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6126556, 11.6240845
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1621323, 12.1619110
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1254692, 10.1343269
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6555634, 8.6614571
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2537537, 12.2637138
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8044472, 10.8051872
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0247688, 12.0221100
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5659370, 10.5564060
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5523529, 11.5531921
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.2963524, 11.2904243
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6174355, 12.6175346
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.8943214, 9.8988609
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.3883781, 11.3868256
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6268272, 10.6177864
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9615898, 7.9575939
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1354218, 13.1293640
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0825615, 8.0839729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1664

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1645

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5347708, upper bound: 4.4840603
time: 32.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5191520, upper bound: 4.5012298
time: 35.99 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 70.09 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 70.09
Output dim: 3, lower bound: -4.5165788, upper bound: 4.5575839
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 70.09
Output dim: 3, lower bound: -4.5093190, upper bound: 4.5679114
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 70.09
Output dim: 3, lower bound: -4.5182979, upper bound: 4.5639457
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 70.09
Output dim: 3, lower bound: -4.5182979, upper bound: 4.5671602
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 70.09
Output dim: 3, lower bound: -4.5011074, upper bound: 4.5665480
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 70.09
Output dim: 3, lower bound: -4.5007039, upper bound: 4.5673256
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 70.09
Output dim: 3, lower bound: -4.5347708, upper bound: 4.4840603
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 70.09
Output dim: 3, lower bound: -4.5191520, upper bound: 4.5012298

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6401596, 15.6373177
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2593613, 9.2639618
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0117340, 9.0212288
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9844971, 8.9945564
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5691986, 12.5792618
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3968849, 8.4034576
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3388939, 13.3400726
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3472328, 9.3564415
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6294174, 15.6403198
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2689438, 9.2716637
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0604782, 12.0542068
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5770607, 8.5704651
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8855591, 10.8682137
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3555222, 13.3629074
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2780914, 16.2601089
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2777290, 11.2756119
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4182014, 9.4242268
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9600105, 10.9424629
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5810699, 10.5602074
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8680992, 6.8688011
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5042763, 7.4983463
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6915436, 9.6945992
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8550606, 7.8514328
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8077011, 8.8039780
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5898056, 10.5862274
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3483047, 8.3429222
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6514664, 11.6371155
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1866989, 12.1846695
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1524391, 10.1489792
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6819878, 8.6793556
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3006287, 12.2985115
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8163147, 10.8139687
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0214577, 12.0205612
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6084824, 10.6221962
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5513992, 11.5517082
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.2989845, 11.3101845
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6301651, 12.6299171
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9347839, 9.9320984
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4177437, 11.4091892
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6644554, 10.6804371
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9379082, 7.9336147
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1603470, 13.1666069
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0879555, 8.0876274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 932

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 940

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5092302, upper bound: 4.5646099
time: 31.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5060174, upper bound: 4.5678226
time: 31.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6547394, 15.6586761
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2578506, 9.2625198
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0128670, 9.0176048
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9878807, 8.9980087
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5808029, 12.5855331
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3980713, 8.4034653
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3268433, 13.3280296
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3645821, 9.3705368
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6491394, 15.6521072
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2678566, 9.2723274
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0532913, 12.0538864
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5650368, 8.5560417
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9157944, 10.9147644
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3316269, 13.3475189
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3086472, 16.3077011
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2669678, 11.2651443
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4210739, 9.4299641
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9918251, 10.9935989
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.6059990, 10.5872803
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8791580, 6.8734970
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5198326, 7.5056229
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6970444, 9.6930313
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8588791, 7.8525581
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7921257, 8.7808418
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5712090, 10.5596542
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3485756, 8.3405151
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6686745, 11.6483955
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2011032, 12.1887398
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1647873, 10.1556778
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6805191, 8.6773224
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2897415, 12.2829170
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8096161, 10.8039207
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0071907, 12.0082550
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5857277, 10.5951653
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5517960, 11.5570450
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.2989578, 11.3116531
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6207199, 12.6208878
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9341202, 9.9323311
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.3931732, 11.3851185
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6570396, 10.6688766
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9287643, 7.9273338
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1367455, 13.1398849
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0849838, 8.0842934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 842

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5180518, upper bound: 4.5669500
time: 61.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5180491, upper bound: 4.5669483
time: 21.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6497498, 15.6594009
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2441139, 9.2548943
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0027580, 9.0137596
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9724808, 8.9856987
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5567322, 12.5764427
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3822517, 8.3947067
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3517990, 13.3526611
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3410988, 9.3500710
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6438980, 15.6536789
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2618828, 9.2647400
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0714798, 12.0595016
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5726700, 8.5573349
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9099121, 10.8932762
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3203239, 13.3133316
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3067780, 16.2921181
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2643127, 11.2777824
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4265327, 9.4311523
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9904556, 10.9746628
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5754776, 10.5839424
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8768349, 6.8699341
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5353680, 7.5255604
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7038879, 9.6969376
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8447723, 7.8430443
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7736320, 8.7572517
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5631104, 10.5557976
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3290520, 8.3158493
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6457748, 11.6365089
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1890335, 12.1941566
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1542587, 10.1443596
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6725731, 8.6670341
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.2850113, 12.2714500
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8110580, 10.8051262
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0284538, 12.0249138
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5872803, 10.5860023
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5561829, 11.5539474
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.2946815, 11.2958145
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6303558, 12.6271362
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9184761, 9.9052792
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4150162, 11.4106903
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6596107, 10.6522217
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9430962, 7.9441090
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1455307, 13.1449089
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0982170, 8.0883179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1638

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.4860943, upper bound: 4.5667659
time: 21.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5001424, upper bound: 4.5527574
time: 41.81 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 65.58 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 65.58
Output dim: 3, lower bound: -4.5092302, upper bound: 4.5646099
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 65.58
Output dim: 3, lower bound: -4.5060174, upper bound: 4.5678226
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 65.58
Output dim: 3, lower bound: -4.5180518, upper bound: 4.5669500
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 65.58
Output dim: 3, lower bound: -4.5180491, upper bound: 4.5669483
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 65.58
Output dim: 3, lower bound: -4.4860943, upper bound: 4.5667659
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 65.58
Output dim: 3, lower bound: -4.5001424, upper bound: 4.5527574

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6416245, 15.6382599
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2603149, 9.2641373
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0117073, 9.0211983
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9830551, 8.9934158
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5686111, 12.5785103
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3966255, 8.4032097
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3338356, 13.3373909
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3470154, 9.3561935
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6294022, 15.6401749
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2703056, 9.2719421
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0576515, 12.0498009
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5770798, 8.5696526
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8809166, 10.8664894
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3530693, 13.3629265
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2791901, 16.2618332
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2774353, 11.2747612
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4177818, 9.4224434
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9468193, 10.9323044
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5786591, 10.5574150
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8675232, 6.8672028
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5035477, 7.4972496
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6899643, 9.6921005
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8551254, 7.8514862
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8079948, 8.8034935
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5866699, 10.5816727
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3434296, 8.3362865
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6512527, 11.6369209
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.1889267, 12.1853905
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1542435, 10.1497803
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6824951, 8.6797104
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3005447, 12.2983360
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8136024, 10.8093987
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0137787, 12.0150452
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6103935, 10.6274567
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5415688, 11.5443192
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.2913971, 11.3047676
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6215744, 12.6234665
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9334869, 9.9343452
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4134407, 11.4064007
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6642418, 10.6824646
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9371243, 7.9353085
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1540413, 13.1621666
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0847931, 8.0854645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1576

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5002203, upper bound: 4.5668612
time: 21.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5050338, upper bound: 4.5619898
time: 37.72 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 60.55 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 60.55
Output dim: 3, lower bound: -4.5002203, upper bound: 4.5668612
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 60.55
Output dim: 3, lower bound: -4.5050338, upper bound: 4.5619898

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 37.68 + 1712.12 = 1749.80 seconds
