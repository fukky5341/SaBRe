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
execution time: IAR + RelationalAnalysis = 2.30 + 34.94 = 37.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -4.5715991, upper bound: 4.5715991

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1785

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5710360, upper bound: 4.5591325
time: 42.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5591325, upper bound: 4.5710360
time: 33.97 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 76.84 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 76.84
Output dim: 3, lower bound: -4.5710360, upper bound: 4.5591325
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 76.84
Output dim: 3, lower bound: -4.5591325, upper bound: 4.5710360

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6774979, 15.6776390
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2666664, 9.2638931
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0269508, 9.0243187
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -9.0047798, 9.0026360
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.6067123, 12.6044693
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4119911, 8.4106903
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3696213, 13.3683701
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3569527, 9.3526001
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6571350, 15.6520157
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2767563, 9.2743034
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0730209, 12.0746155
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5971947, 8.5979691
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9247437, 10.9289246
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3768234, 13.3756828
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3207397, 16.3248138
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.3066483, 11.3091011
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4392509, 9.4343204
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -11.0137024, 11.0183640
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5897331, 10.5979576
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8841133, 6.8842430
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5513382, 7.5518169
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7133331, 9.7136345
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8459282, 7.8491058
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8061104, 8.8067360
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5733948, 10.5746689
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3523636, 8.3529015
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6697807, 11.6757088
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2153549, 12.2133789
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1727180, 10.1727943
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6749420, 8.6768951
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3046570, 12.3042641
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8079224, 10.8092804
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0389519, 12.0388832
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6265678, 10.6256523
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5497894, 11.5522156
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3311844, 11.3317070
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6388779, 12.6390648
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9312553, 9.9339638
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4142799, 11.4191284
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6835289, 10.6802368
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9427929, 7.9427567
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1566544, 13.1549950
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1265488, 8.1253071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5704130, upper bound: 4.5434144
time: 30.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5553179, upper bound: 4.5585093
time: 44.44 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6776352, 15.6775017
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2638931, 9.2666664
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0243187, 9.0269508
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -9.0026360, 9.0047798
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.6044693, 12.6067123
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4106903, 8.4119911
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3683701, 13.3696213
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3526001, 9.3569489
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6520157, 15.6571350
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2743034, 9.2767563
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0746155, 12.0730209
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5979691, 8.5971947
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9289246, 10.9247437
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3756790, 13.3768272
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3248138, 16.3207397
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.3091049, 11.3066483
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4343185, 9.4392490
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -11.0183640, 11.0137024
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5979576, 10.5897331
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8842430, 6.8841095
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5518188, 7.5513401
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7136345, 9.7133331
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8491058, 7.8459282
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8067322, 8.8061104
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5746689, 10.5733986
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3529015, 8.3523636
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6757050, 11.6697807
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2133789, 12.2153587
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1727943, 10.1727180
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6768951, 8.6749420
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3042603, 12.3046570
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8092804, 10.8079224
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0388832, 12.0389557
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6256523, 10.6265678
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5522156, 11.5497932
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3317070, 11.3311844
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6390686, 12.6388741
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9339638, 9.9312553
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4191284, 11.4142799
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6802368, 10.6835289
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9427547, 7.9427929
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1549911, 13.1566505
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1253090, 8.1265488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5585093, upper bound: 4.5553179
time: 37.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5434144, upper bound: 4.5704130
time: 38.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 78.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 78.29
Output dim: 3, lower bound: -4.5704130, upper bound: 4.5434144
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 78.29
Output dim: 3, lower bound: -4.5553179, upper bound: 4.5585093
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 78.29
Output dim: 3, lower bound: -4.5585093, upper bound: 4.5553179
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 78.29
Output dim: 3, lower bound: -4.5434144, upper bound: 4.5704130

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6640625, 15.6675034
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2612152, 9.2570229
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0133781, 9.0069389
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9985886, 8.9942894
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5976257, 12.5924454
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4085464, 8.4063683
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3712997, 13.3699570
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3448219, 9.3361206
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6366348, 15.6247787
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2687187, 9.2636414
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0812836, 12.0849037
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5938530, 8.5952454
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8914223, 10.9037209
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3749084, 13.3736000
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2880020, 16.3004913
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.3101006, 11.3131905
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4509087, 9.4433537
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9678459, 10.9832954
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5451012, 10.5638008
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8896713, 6.8890533
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5480194, 7.5482445
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7170029, 9.7169952
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8376160, 7.8427391
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8113098, 8.8110123
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5752182, 10.5763283
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3475761, 8.3486252
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6709061, 11.6798019
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2358704, 12.2296829
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1765442, 10.1759834
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6710052, 8.6739731
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3147812, 12.3125153
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.7999535, 10.8028450
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0396805, 12.0395966
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6361237, 10.6330681
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5376625, 11.5428581
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3277245, 11.3290558
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6405258, 12.6405716
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9244194, 9.9278049
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4257927, 11.4364738
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6729851, 10.6644249
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9369087, 7.9380531
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1725082, 13.1672821
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1232338, 8.1206322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1784

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5698848, upper bound: 4.5311371
time: 45.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5576709, upper bound: 4.5431415
time: 27.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6675034, 15.6640625
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2570229, 9.2612152
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0069389, 9.0133781
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9942894, 8.9985886
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5924454, 12.5976257
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4063683, 8.4085464
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3699570, 13.3713036
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3361206, 9.3448219
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6247787, 15.6366348
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2636414, 9.2687187
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0848999, 12.0812798
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5952454, 8.5938530
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9037209, 10.8914223
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3736038, 13.3749123
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3004913, 16.2880020
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.3131905, 11.3101006
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4433556, 9.4509087
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9832954, 10.9678459
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5638008, 10.5451012
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8890533, 6.8896732
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5482445, 7.5480213
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7169952, 9.7170029
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8427391, 7.8376179
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8110123, 8.8113098
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5763245, 10.5752182
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3486252, 8.3475761
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6798019, 11.6709061
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2296829, 12.2358665
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1759834, 10.1765442
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6739731, 8.6710052
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3125153, 12.3147812
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8028450, 10.7999535
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0396042, 12.0396843
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6330681, 10.6361237
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5428581, 11.5376625
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3290558, 11.3277245
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6405716, 12.6405220
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9278030, 9.9244194
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4364738, 11.4257927
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6644249, 10.6729851
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9380531, 7.9369106
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1672821, 13.1725121
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1206322, 8.1232338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1784

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5431415, upper bound: 4.5576709
time: 26.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5311371, upper bound: 4.5698848
time: 27.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 56.77 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 56.77
Output dim: 3, lower bound: -4.5698848, upper bound: 4.5311371
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 56.77
Output dim: 3, lower bound: -4.5576709, upper bound: 4.5431415
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 56.77
Output dim: 3, lower bound: -4.5431415, upper bound: 4.5576709
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 56.77
Output dim: 3, lower bound: -4.5311371, upper bound: 4.5698848

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6630859, 15.6667366
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2593994, 9.2524376
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0115128, 9.0022240
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9969597, 8.9903107
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5962677, 12.5892487
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4075432, 8.4037399
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3718910, 13.3697891
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3426323, 9.3304176
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6333771, 15.6166306
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2669792, 9.2589798
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0809784, 12.0858650
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5902443, 8.5933990
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8857002, 10.9014931
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3728333, 13.3681488
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2841339, 16.2990570
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.3099480, 11.3140068
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4549065, 9.4428082
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9629021, 10.9813423
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5311050, 10.5585480
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8870811, 6.8877449
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5470467, 7.5489769
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7140694, 9.7158508
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8328171, 7.8409500
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8078957, 8.8091087
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5716019, 10.5749817
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3452110, 8.3473511
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6610184, 11.6759109
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2365952, 12.2295685
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1749954, 10.1752739
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6684837, 8.6730347
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3147659, 12.3124619
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.7950134, 10.8006401
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0395889, 12.0394936
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6362419, 10.6330414
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5331917, 11.5411606
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3261452, 11.3282585
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6401825, 12.6403503
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9193611, 9.9258080
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4198723, 11.4357796
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6710815, 10.6598358
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9348564, 7.9366493
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1733017, 13.1671486
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1229172, 8.1195107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1768

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5692701, upper bound: 4.5117935
time: 25.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5462594, upper bound: 4.5302156
time: 33.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6667328, 15.6630821
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2524414, 9.2593994
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0022240, 9.0115128
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9903107, 8.9969597
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5892487, 12.5962677
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4037399, 8.4075432
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3697929, 13.3718910
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3304176, 9.3426323
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6166306, 15.6333771
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2589798, 9.2669792
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0858612, 12.0809746
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5933990, 8.5902443
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.9014931, 10.8857040
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3681488, 13.3728333
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2990570, 16.2841339
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.3140068, 11.3099480
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4428101, 9.4549065
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9813423, 10.9629021
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5585480, 10.5311050
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8877449, 6.8870792
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5489769, 7.5470486
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7158508, 9.7140694
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8409500, 7.8328190
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8091087, 8.8078957
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5749817, 10.5716057
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3473511, 8.3452091
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6759071, 11.6610184
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2295685, 12.2365952
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1752739, 10.1749954
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6730347, 8.6684837
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3124619, 12.3147697
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.8006401, 10.7950134
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0394974, 12.0395851
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6330414, 10.6362419
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5411606, 11.5331917
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3282585, 11.3261452
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6403503, 12.6401787
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9258080, 9.9193611
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4357796, 11.4198742
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6598358, 10.6710815
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9366493, 7.9348564
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1671524, 13.1733017
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1195107, 8.1229153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1768

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5302156, upper bound: 4.5462594
time: 28.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5117935, upper bound: 4.5692701
time: 31.92 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 62.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 62.56
Output dim: 3, lower bound: -4.5692701, upper bound: 4.5117935
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 62.56
Output dim: 3, lower bound: -4.5462594, upper bound: 4.5302156
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 62.56
Output dim: 3, lower bound: -4.5302156, upper bound: 4.5462594
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 62.56
Output dim: 3, lower bound: -4.5117935, upper bound: 4.5692701

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6619568, 15.6659508
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2484856, 9.2379456
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -9.0007095, 8.9879112
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9878693, 8.9782372
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5888519, 12.5793991
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.4014473, 8.3956451
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3761902, 13.3731232
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3304024, 9.3141785
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6151733, 15.5924759
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2581482, 9.2477264
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0884781, 12.0955544
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5783920, 8.5844765
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8688622, 10.8888092
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3614349, 13.3531151
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2829895, 16.3004379
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.3143616, 11.3196144
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4747200, 9.4582825
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9531479, 10.9742928
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.4921532, 10.5289803
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8782234, 6.8813877
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5503120, 7.5549011
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7046242, 9.7088814
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8197632, 7.8309841
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7990761, 8.8025436
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5617943, 10.5677261
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3386078, 8.3423157
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6303749, 11.6519852
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2421799, 12.2339058
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1703033, 10.1719246
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6611290, 8.6674957
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3165894, 12.3140564
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.7793045, 10.7886505
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0375977, 12.0378036
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6355362, 10.6324043
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5218201, 11.5325966
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3220100, 11.3250961
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6392708, 12.6395416
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9032249, 9.9137726
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4195251, 11.4410439
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6649780, 10.6513500
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9247952, 7.9290924
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1763153, 13.1700821
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1232452, 8.1196976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5690227, upper bound: 4.5047102
time: 27.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5621853, upper bound: 4.5115535
time: 39.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6659546, 15.6619606
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2379456, 9.2484856
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -8.9879112, 9.0007095
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9782372, 8.9878693
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5793991, 12.5888519
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3956490, 8.4014511
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3731232, 13.3761902
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3141785, 9.3304024
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.5924759, 15.6151733
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2477303, 9.2581482
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0955582, 12.0884819
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5844765, 8.5783920
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8888092, 10.8688622
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3531189, 13.3614388
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.3004379, 16.2829895
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.3196106, 11.3143616
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4582825, 9.4747200
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9742928, 10.9531479
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5289803, 10.4921532
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8813896, 6.8782253
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5549011, 7.5503101
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7088814, 9.7046242
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8309860, 7.8197632
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8025436, 8.7990761
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5677299, 10.5617981
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3423157, 8.3386078
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6519852, 11.6303749
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2339020, 12.2421799
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1719284, 10.1703033
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6674957, 8.6611290
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3140564, 12.3165894
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.7886505, 10.7793045
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0378036, 12.0375977
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6324043, 10.6355343
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5325966, 11.5218201
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3250999, 11.3220100
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6395378, 12.6392746
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9137726, 9.9032249
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4410439, 11.4195251
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6513481, 10.6649780
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9290905, 7.9247952
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1700821, 13.1763153
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1196976, 8.1232452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5115535, upper bound: 4.5621853
time: 34.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5047102, upper bound: 4.5690227
time: 28.77 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 64.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 64.76
Output dim: 3, lower bound: -4.5690227, upper bound: 4.5047102
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 64.76
Output dim: 3, lower bound: -4.5621853, upper bound: 4.5115535
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 64.76
Output dim: 3, lower bound: -4.5115535, upper bound: 4.5621853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 64.76
Output dim: 3, lower bound: -4.5047102, upper bound: 4.5690227

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6533508, 15.6545525
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2452927, 9.2337151
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -8.9953880, 8.9808960
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9843216, 8.9734612
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5896454, 12.5799141
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3971863, 8.3900032
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3554077, 13.3566513
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3313713, 9.3147049
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6059418, 15.5816193
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2528992, 9.2408104
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0798759, 12.0848274
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5682640, 8.5771217
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8635902, 10.8841782
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3586731, 13.3493500
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2728577, 16.2870712
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.3104324, 11.3153229
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4640694, 9.4447155
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9354630, 10.9508629
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.4920197, 10.5296707
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8742752, 6.8784046
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5333557, 7.5421047
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6957932, 9.7022057
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8153763, 7.8275928
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7994194, 8.8040009
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5648155, 10.5721321
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3386955, 8.3426895
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6207275, 11.6447029
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2308884, 12.2254333
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1698570, 10.1715927
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6604996, 8.6670074
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3118057, 12.3109474
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.7752037, 10.7858658
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0193214, 12.0237923
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6184196, 10.6194839
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5081406, 11.5222702
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3100700, 11.3160210
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6146698, 12.6207047
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.8937073, 9.9065857
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.3822327, 11.4125137
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6403389, 10.6327858
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9118252, 7.9193020
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1635590, 13.1604576
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1184425, 8.1160049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 767

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5689745, upper bound: 4.5014409
time: 94.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5657641, upper bound: 4.5046621
time: 28.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6545563, 15.6533546
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2337151, 9.2452927
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -8.9808960, 8.9953880
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9734612, 8.9843216
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5799179, 12.5896492
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3900032, 8.3971863
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3566513, 13.3554077
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3147049, 9.3313713
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.5816193, 15.6059418
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2408104, 9.2528992
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0848274, 12.0798721
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5771217, 8.5682640
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8841782, 10.8635902
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3493500, 13.3586731
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2870712, 16.2728577
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.3153229, 11.3104362
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4447174, 9.4640694
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9508667, 10.9354630
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5296707, 10.4920197
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8784027, 6.8742752
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5421066, 7.5333576
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7022095, 9.6957970
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8275909, 7.8153763
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8040009, 8.7994194
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5721321, 10.5648193
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3426895, 8.3386936
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6447029, 11.6207275
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2254333, 12.2308884
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1715927, 10.1698570
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6670074, 8.6604996
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3109436, 12.3118057
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.7858696, 10.7752037
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0237923, 12.0193176
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6194839, 10.6184196
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5222702, 11.5081406
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3160210, 11.3100700
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6207123, 12.6146774
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9065857, 9.8937054
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4125137, 11.3822327
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6327858, 10.6403389
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9193020, 7.9118271
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1604538, 13.1635628
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1160049, 8.1184425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 767

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5046621, upper bound: 4.5657641
time: 34.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5014409, upper bound: 4.5689745
time: 33.43 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 70.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 70.36
Output dim: 3, lower bound: -4.5689745, upper bound: 4.5014409
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 70.36
Output dim: 3, lower bound: -4.5657641, upper bound: 4.5046621
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 70.36
Output dim: 3, lower bound: -4.5046621, upper bound: 4.5657641
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 70.36
Output dim: 3, lower bound: -4.5014409, upper bound: 4.5689745

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6502991, 15.6504936
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2446480, 9.2328568
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -8.9924545, 8.9769974
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9851952, 8.9737663
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5866089, 12.5761337
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3972702, 8.3900604
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3392029, 13.3442192
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3356361, 9.3183250
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.6007309, 15.5747223
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2532997, 9.2410736
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0758629, 12.0794563
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5607605, 8.5721016
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8593140, 10.8809547
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3499222, 13.3416100
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2661896, 16.2790489
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.3021812, 11.3061333
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4608192, 9.4404030
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9265175, 10.9389877
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.4926109, 10.5307426
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8742752, 6.8785839
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5238647, 7.5349617
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6923866, 9.7001610
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8159523, 7.8288879
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7992477, 8.8039207
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5642052, 10.5718193
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3390312, 8.3431129
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6219101, 11.6464920
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2306366, 12.2253342
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1710434, 10.1730537
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6605110, 8.6671829
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3093262, 12.3094406
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.7748337, 10.7856827
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0061493, 12.0138779
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6096725, 10.6128998
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5065384, 11.5210609
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3075829, 11.3141632
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6052856, 12.6136360
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.8944740, 9.9078522
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.3665657, 11.4007168
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6291122, 10.6243534
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9056149, 7.9146252
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1588516, 13.1569061
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1088104, 8.1082668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 766

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5458427, upper bound: 4.4983344
time: 33.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5277824, upper bound: 4.5011246
time: 28.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6504974, 15.6502953
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2328568, 9.2446480
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -8.9770012, 8.9924545
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9737663, 8.9851952
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5761337, 12.5866089
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3900604, 8.3972702
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3442230, 13.3392029
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3183250, 9.3356361
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.5747223, 15.6007309
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2410736, 9.2532997
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0794563, 12.0758629
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5721016, 8.5607605
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8809586, 10.8593102
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3416061, 13.3499222
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2790527, 16.2661934
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.3061333, 11.3021812
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4404030, 9.4608212
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9389877, 10.9265175
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5307426, 10.4926109
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8785858, 6.8742771
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5349655, 7.5238686
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.7001610, 9.6923866
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8288879, 7.8159523
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8039207, 8.7992477
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5718193, 10.5642014
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3431129, 8.3390312
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6464920, 11.6219101
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2253342, 12.2306366
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1730499, 10.1710434
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6671829, 8.6605110
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3094406, 12.3093262
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.7856827, 10.7748337
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0138779, 12.0061493
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6128998, 10.6096706
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5210609, 11.5065384
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3141670, 11.3075829
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6136322, 12.6052895
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9078522, 9.8944740
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.4007149, 11.3665638
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6243515, 10.6291122
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9146252, 7.9056168
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1569061, 13.1588478
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.1082649, 8.1088104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 766

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5011246, upper bound: 4.5635527
time: 37.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.4983344, upper bound: 4.5688528
time: 23.18 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 62.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 62.24
Output dim: 3, lower bound: -4.5458427, upper bound: 4.4983344
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 62.24
Output dim: 3, lower bound: -4.5277824, upper bound: 4.5011246
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 62.24
Output dim: 3, lower bound: -4.5011246, upper bound: 4.5635527
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 62.24
Output dim: 3, lower bound: -4.4983344, upper bound: 4.5688528

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6443405, 15.6456757
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2293358, 9.2420006
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -8.9710999, 8.9880257
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9739380, 8.9854660
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5705719, 12.5818443
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3890114, 8.3964806
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3263474, 13.3166275
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3220329, 9.3405876
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.5651016, 15.5929489
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2380981, 9.2510643
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0722885, 12.0704765
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5644112, 8.5495148
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8764534, 10.8533287
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3358994, 13.3439407
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2681427, 16.2575836
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2945099, 11.2914886
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4328003, 9.4551067
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9233284, 10.9144592
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5316696, 10.4928207
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8776798, 6.8729839
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5239487, 7.5092182
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6958618, 9.6860428
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8310738, 7.8173695
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8049240, 8.7997284
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5720482, 10.5642815
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3441963, 8.3398571
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6446953, 11.6195183
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2243500, 12.2287598
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1740646, 10.1717606
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6674767, 8.6606293
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3087006, 12.3070984
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.7846870, 10.7729149
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -12.0003967, 11.9886131
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6041183, 10.5979881
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5181770, 11.5026970
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3117065, 11.3043137
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.6035080, 12.5918121
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9056473, 9.8915367
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.3825073, 11.3423386
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6134071, 10.6145477
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9051933, 7.8930645
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1501083, 13.1497993
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0973167, 8.0950451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 750

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.4928303, upper bound: 4.5616171
time: 23.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.4928303, upper bound: 4.5687653
time: 17.54 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 42.81 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 42.81
Output dim: 3, lower bound: -4.4928303, upper bound: 4.5616171
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 42.81
Output dim: 3, lower bound: -4.4928303, upper bound: 4.5687653

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6392212, 15.6418266
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2258263, 9.2393608
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -8.9664040, 8.9844894
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9732513, 8.9849586
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5665817, 12.5783577
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3878784, 8.3956261
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3134003, 13.2999115
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3242188, 9.3436623
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.5576096, 15.5867462
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2342300, 9.2481575
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0649643, 12.0649681
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5586700, 8.5410995
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8742409, 10.8503914
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3346710, 13.3428040
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2583771, 16.2496643
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2868500, 11.2835426
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4248123, 9.4491005
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.9104195, 10.9046936
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5318222, 10.4928932
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8756752, 6.8702717
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5142326, 7.4962940
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6914864, 9.6797676
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8314133, 7.8175983
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8054504, 8.8000374
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5734024, 10.5650902
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3442764, 8.3399258
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6413498, 11.6150703
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2224655, 12.2257957
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1737022, 10.1712761
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6671104, 8.6600761
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3092422, 12.3064384
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.7824402, 10.7694664
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -11.9910507, 11.9764175
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5973282, 10.5889587
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5168915, 11.5009918
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3100739, 11.3021469
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.5956268, 12.5813293
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.9037781, 9.8890572
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.3679428, 11.3229675
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6055679, 10.6041203
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.8982735, 7.8838634
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1453362, 13.1434555
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0894470, 8.0851212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1663

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.4744300, upper bound: 4.5503721
time: 30.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.4744300, upper bound: 4.5682323
time: 28.48 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 60.84 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 60.84
Output dim: 3, lower bound: -4.4744300, upper bound: 4.5503721
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 60.84
Output dim: 3, lower bound: -4.4744300, upper bound: 4.5682323

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6303101, 15.6350861
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2138252, 9.2302895
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -8.9576492, 8.9778709
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9588966, 8.9740677
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5605927, 12.5737228
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3778267, 8.3880157
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3177681, 13.3067245
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3131256, 9.3352776
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.5464554, 15.5783234
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2227859, 9.2395058
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0649872, 12.0649834
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5531998, 8.5341339
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8708611, 10.8462524
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3305206, 13.3385544
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2587662, 16.2483711
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2871780, 11.2838860
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4260521, 9.4543095
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.8979645, 10.8881836
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5222015, 10.4812241
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8646336, 6.8556709
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.5061703, 7.4856319
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6852226, 9.6714821
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8194504, 7.8018456
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.7981148, 8.7903862
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5613708, 10.5491714
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3302193, 8.3213272
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6290131, 11.5987511
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2211838, 12.2245560
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1648064, 10.1595078
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6586494, 8.6488876
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3035355, 12.2988930
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.7698364, 10.7527924
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -11.9935150, 11.9802246
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.5963135, 10.5878162
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5167046, 11.5008240
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3084145, 11.3002853
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.5950165, 12.5806808
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.8988495, 9.8826809
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.3698769, 11.3240585
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6056519, 10.6040993
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9005814, 7.8871803
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1469345, 13.1471977
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0921898, 8.0891476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1766

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.4744021, upper bound: 4.5587952
time: 28.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.4644670, upper bound: 4.5682044
time: 24.78 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 54.89 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 54.89
Output dim: 3, lower bound: -4.4744021, upper bound: 4.5587952
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 54.89
Output dim: 3, lower bound: -4.4644670, upper bound: 4.5682044

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6210403, 15.6225471
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2145042, 9.2307396
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -8.9554787, 8.9775391
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9568558, 8.9723969
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5532150, 12.5681763
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3771324, 8.3873596
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3163109, 13.3048592
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.3058662, 9.3294296
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.5329514, 15.5681915
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2231712, 9.2398720
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0645599, 12.0633392
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5568428, 8.5378075
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8508072, 10.8195229
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3375359, 13.3428345
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2382736, 16.2200012
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2884216, 11.2851601
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4215431, 9.4488602
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.8780136, 10.8587227
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.5095367, 10.4646492
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8601494, 6.8531361
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.4948273, 7.4768429
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6817932, 9.6703300
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8186531, 7.8015518
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8079872, 8.8035736
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5724258, 10.5629807
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3313408, 8.3226433
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6222992, 11.5931511
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2135696, 12.2211151
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1598816, 10.1566811
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6597824, 8.6501579
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3087769, 12.3064041
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.7736969, 10.7566795
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -11.9938583, 11.9805374
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6053276, 10.5982513
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5143471, 11.4977951
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3065033, 11.2981110
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.5954208, 12.5812416
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.8982658, 9.8820286
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.3759155, 11.3283806
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6091652, 10.6097279
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9029922, 7.8887634
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1577721, 13.1602707
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0896454, 8.0869980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1782

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.4637221, upper bound: 4.5626333
time: 25.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.4611948, upper bound: 4.5681022
time: 36.95 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 64.55 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 64.55
Output dim: 3, lower bound: -4.4637221, upper bound: 4.5626333
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 64.55
Output dim: 3, lower bound: -4.4611948, upper bound: 4.5681022

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3809347, 4.3146091, -13.3809347, 4.3146091, -15.6117172, 15.6103210
1: 0.4285917, 12.3426075, 0.4285917, 12.3426075, -9.2148209, 9.2305870
2: 2.0693097, 13.4902630, 2.0693097, 13.4902630, -8.9536362, 8.9771652
3: 1.5895219, 14.1851063, 1.5895219, 14.1851063, -8.9558830, 8.9714890
4: -4.1968122, 10.4691925, -4.1968122, 10.4691925, -12.5467148, 12.5631866
5: 2.1020551, 13.7777576, 2.1020551, 13.7777576, -8.3774376, 8.3876762
6: -25.1710892, -8.7839451, -25.1710892, -8.7839451, -13.3156128, 13.3040237
7: 2.5634320, 15.3062096, 2.5634320, 15.3062096, -9.2994995, 9.3245659
8: -4.4499159, 14.2579231, -4.4499159, 14.2579231, -15.5217056, 15.5596313
9: 0.5905259, 13.6066589, 0.5905259, 13.6066589, -9.2223930, 9.2391853
10: -4.4052677, 11.3081303, -4.4052677, 11.3081303, -12.0635605, 12.0616760
11: -4.4716330, 6.8974557, -4.4716330, 6.8974557, -8.5604744, 8.5417976
12: -26.2457123, -11.1710958, -26.2457123, -11.1710958, -10.8334465, 10.7965279
13: -14.1602573, 4.6837254, -14.1602573, 4.6837254, -13.3437881, 13.3464241
14: -24.1747360, -5.2175064, -24.1747360, -5.2175064, -16.2178345, 16.1926346
15: -7.6069403, 4.7103748, -7.6069403, 4.7103748, -11.2895966, 11.2861137
16: -7.6732941, 5.0040016, -7.6732941, 5.0040016, -9.4179039, 9.4451466
17: -26.7557220, -11.1057987, -26.7557220, -11.1057987, -10.8525124, 10.8246078
18: -17.7020721, -2.0515327, -17.7020721, -2.0515327, -10.4990768, 10.4508133
19: -10.4936953, -0.0656571, -10.4936953, -0.0656571, -6.8537159, 6.8494568
20: -5.8909655, 4.7191973, -5.8909655, 4.7191973, -7.4837837, 7.4684601
21: -8.5942287, 3.8232310, -8.5942287, 3.8232310, -9.6779442, 9.6690292
22: -10.8043451, 0.8349555, -10.8043451, 0.8349555, -7.8173141, 7.8005333
23: -4.6564751, 6.9074693, -4.6564751, 6.9074693, -8.8143425, 8.8127022
24: -8.1061096, 5.2016850, -8.1061096, 5.2016850, -10.5807571, 10.5737572
25: -8.3765574, 4.8564501, -8.3765574, 4.8564501, -8.3316460, 8.3230743
26: -16.5926285, 0.1533532, -16.5926285, 0.1533532, -11.6152725, 11.5870590
27: -7.8430915, 6.2024174, -7.8430915, 6.2024174, -12.2066498, 12.2178040
28: -6.5493727, 6.3284578, -6.5493727, 6.3284578, -10.1544571, 10.1531258
29: -7.7773094, 2.8724625, -7.7773094, 2.8724625, -8.6603508, 8.6508217
30: -3.8678496, 10.3132706, -3.8678496, 10.3132706, -12.3127441, 12.3123283
31: -14.8697681, -0.1749096, -14.8697681, -0.1749096, -10.7770844, 10.7608223
32: -20.8186321, -5.8451862, -20.8186321, -5.8451862, -11.9950714, 11.9818077
33: -38.5928497, -20.0071220, -38.5928497, -20.0071220, -10.6109924, 10.6059952
34: -35.7664642, -20.2659626, -35.7664642, -20.2659626, -11.5120506, 11.4949646
35: -33.0715828, -16.7966881, -33.0715828, -16.7966881, -11.3052521, 11.2966843
36: -31.1271877, -13.5126791, -31.1271877, -13.5126791, -12.5957870, 12.5818291
37: -50.2989235, -32.2753296, -50.2989235, -32.2753296, -9.8995132, 9.8834991
38: -38.8015518, -20.2444553, -38.8015518, -20.2444553, -11.3821831, 11.3329830
39: -42.5673218, -23.5579166, -42.5673218, -23.5579166, -10.6043777, 10.6081181
40: -38.0392609, -24.5524178, -38.0392609, -24.5524178, -7.9031754, 7.8889122
41: -24.8254318, -8.8904037, -24.8254318, -8.8904037, -13.1653404, 13.1705742
42: -15.1472034, -4.8143821, -15.1472034, -4.8143821, -8.0864182, 8.0845966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=69, inp2_unstable=69, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 638

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.4400185, upper bound: 4.5666103
time: 32.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.4598031, upper bound: 4.5437294
time: 28.98 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 63.55 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 63.55
Output dim: 3, lower bound: -4.4400185, upper bound: 4.5666103
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 63.55
Output dim: 3, lower bound: -4.4598031, upper bound: 4.5437294

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 37.24 + 1228.68 = 1265.92 seconds
