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
execution time: IAR + RelationalAnalysis = 2.28 + 34.59 = 36.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -4.5715991, upper bound: 4.5715991

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 971

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1785

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5710360, upper bound: 4.5591325
time: 29.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5710360, upper bound: 4.5710359
time: 26.98 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 56.38 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 56.38
Output dim: 3, lower bound: -4.5710360, upper bound: 4.5591325
IS_A2, status: Status.UNKNOWN, split count: 1, time: 56.38
Output dim: 3, lower bound: -4.5710360, upper bound: 4.5710359

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.3693085, 4.3124585, -13.3773298, 4.3140078, -15.6661530, 15.6733894
1: 0.4295800, 12.3322554, 0.4288960, 12.3395929, -9.2648315, 9.2579956
2: 2.0698588, 13.4767818, 2.0694709, 13.4863605, -9.0245361, 9.0148048
3: 1.5900605, 14.1692924, 1.5896690, 14.1806049, -9.0015221, 8.9906006
4: -4.1958284, 10.4439440, -4.1965284, 10.4619999, -12.6007843, 12.5832100
5: 2.1027603, 13.7661142, 2.1022658, 13.7744617, -8.4092598, 8.4013062
6: -25.1595688, -8.7851257, -25.1677971, -8.7842665, -13.3514481, 13.3628654
7: 2.5641656, 15.2877827, 2.5636296, 15.3009644, -9.3545952, 9.3419228
8: -4.4486580, 14.2189236, -4.4495635, 14.2468071, -15.6496735, 15.6225586
9: 0.5915604, 13.5922413, 0.5908129, 13.6025553, -9.2739983, 9.2643433
10: -4.4038978, 11.3027744, -4.4048634, 11.3065853, -12.0693054, 12.0633125
11: -4.4533052, 6.8968534, -4.4663200, 6.8972797, -8.5803299, 8.5932999
12: -26.2261581, -11.1724453, -26.2401352, -11.1714649, -10.9132729, 10.9264526
13: -14.1594944, 4.6624451, -14.1600447, 4.6776166, -13.3715439, 13.3561859
14: -24.1594505, -5.2198820, -24.1703892, -5.2182026, -16.3134613, 16.3196602
15: -7.6047659, 4.6991096, -7.6063466, 4.7071013, -11.3010864, 11.2899857
16: -7.6700640, 5.0029273, -7.6723356, 5.0036941, -9.4212990, 9.4298553
17: -26.7396088, -11.1071844, -26.7511234, -11.1062155, -11.0052795, 11.0153465
18: -17.6663055, -2.0523257, -17.6919079, -2.0517392, -10.5691910, 10.5941620
19: -10.4884357, -0.0658555, -10.4921951, -0.0657170, -6.8789005, 6.8828907
20: -5.8896446, 4.7183599, -5.8905830, 4.7189479, -7.5496140, 7.5484676
21: -8.5843887, 3.8227477, -8.5913649, 3.8231010, -9.7025986, 9.7101097
22: -10.7924824, 0.8348796, -10.8009529, 0.8349545, -7.8399887, 7.8480225
23: -4.6413321, 6.9069557, -4.6520939, 6.9073195, -8.7923431, 8.8035202
24: -8.0861721, 5.2013330, -8.1004477, 5.2015619, -10.5554962, 10.5693550
25: -8.3688345, 4.8557372, -8.3743515, 4.8562450, -8.3467903, 8.3515587
26: -16.5779190, 0.1526071, -16.5884399, 0.1531694, -11.6660957, 11.6745033
27: -7.8300290, 6.2015162, -7.8393650, 6.2021532, -12.1952057, 12.2074890
28: -6.5421953, 6.3275461, -6.5472851, 6.3281770, -10.1651649, 10.1698761
29: -7.7607527, 2.8723259, -7.7725229, 2.8724263, -8.6616859, 8.6734657
30: -3.8512270, 10.3122377, -3.8631344, 10.3129873, -12.2835617, 12.2974625
31: -14.8514233, -0.1757097, -14.8645306, -0.1751533, -10.7920227, 10.8049240
32: -20.8109303, -5.8463984, -20.8164482, -5.8455276, -12.0295334, 12.0354080
33: -38.5918770, -20.0083618, -38.5925827, -20.0074883, -10.6217155, 10.6230354
34: -35.7419205, -20.2663555, -35.7594681, -20.2660542, -11.5297737, 11.5469055
35: -33.0543823, -16.7972775, -33.0666580, -16.7968330, -11.3153114, 11.3275719
36: -31.1147251, -13.5132694, -31.1236153, -13.5128288, -12.6275787, 12.6359596
37: -50.2821350, -32.2758331, -50.2941360, -32.2754784, -9.9191856, 9.9315281
38: -38.7898979, -20.2448845, -38.7982635, -20.2445908, -11.4086761, 11.4142914
39: -42.5670929, -23.5675335, -42.5672569, -23.5606880, -10.6828690, 10.6757240
40: -38.0344467, -24.5536823, -38.0378418, -24.5527611, -7.9374771, 7.9411411
41: -24.8160038, -8.8913412, -24.8227081, -8.8906937, -13.1415329, 13.1504517
42: -15.1466789, -4.8172631, -15.1470623, -4.8152189, -8.1245232, 8.1238899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=68, inp2_unstable=69, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1784

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5582941, upper bound: 4.5586024
time: 19.63 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5582941, upper bound: 4.5586024
time: 24.25 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -13.3915176, 4.3243332, -13.3803444, 4.3143439, -15.6868286, 15.6910629
1: 0.4080882, 12.3430738, 0.4286838, 12.3421345, -9.2887611, 9.2676773
2: 2.0466073, 13.4909859, 2.0693498, 13.4896946, -9.0524216, 9.0283241
3: 1.5638514, 14.1861477, 1.5895574, 14.1844826, -9.0322838, 9.0056534
4: -4.2322903, 10.4700012, -4.1967826, 10.4682093, -12.6434937, 12.6074448
5: 2.0859303, 13.7774677, 2.1021149, 13.7772903, -8.4295273, 8.4118919
6: -25.1735439, -8.7643986, -25.1701889, -8.7839823, -13.3644562, 13.3957825
7: 2.5368173, 15.3048859, 2.5634484, 15.3055096, -9.3870392, 9.3564873
8: -4.5073295, 14.2571754, -4.4498644, 14.2564011, -15.7179642, 15.6574402
9: 0.5692906, 13.6070347, 0.5905626, 13.6060820, -9.3009720, 9.2779388
10: -4.4133382, 11.3117399, -4.4051881, 11.3078365, -12.0884857, 12.0707626
11: -4.4750967, 6.9215198, -4.4708767, 6.8974266, -8.6000595, 8.6244545
12: -26.2446289, -11.1383791, -26.2449951, -11.1711597, -10.9288635, 10.9652863
13: -14.1886034, 4.6908908, -14.1602325, 4.6828213, -13.4080124, 13.3816261
14: -24.1852436, -5.1986456, -24.1741371, -5.2177019, -16.3358688, 16.3376312
15: -7.6232743, 4.7122936, -7.6068702, 4.7098680, -11.3334503, 11.3012657
16: -7.6803322, 5.0142431, -7.6725564, 5.0038733, -9.4259300, 9.4640312
17: -26.7592449, -11.0797958, -26.7551346, -11.1059332, -11.0225258, 11.0485115
18: -17.7020283, -2.0008311, -17.7006874, -2.0515637, -10.5983582, 10.6545258
19: -10.4950562, -0.0584338, -10.4930725, -0.0656614, -6.8852806, 6.8923817
20: -5.8951335, 4.7224336, -5.8907948, 4.7188082, -7.5606270, 7.5525379
21: -8.5979004, 3.8345511, -8.5934334, 3.8232179, -9.7167397, 9.7251740
22: -10.8059120, 0.8485966, -10.8038635, 0.8349705, -7.8507004, 7.8654366
23: -4.6601505, 6.9277868, -4.6558485, 6.9074583, -8.8092270, 8.8303833
24: -8.1112156, 5.2310510, -8.1053152, 5.2016745, -10.5775909, 10.6069641
25: -8.3794746, 4.8687453, -8.3762331, 4.8564382, -8.3553696, 8.3672161
26: -16.5956326, 0.1772244, -16.5920067, 0.1533229, -11.6801186, 11.7015953
27: -7.8499846, 6.2185988, -7.8425388, 6.2023721, -12.2134857, 12.2367516
28: -6.5527778, 6.3427653, -6.5487485, 6.3283901, -10.1750641, 10.1869545
29: -7.7808218, 2.8924708, -7.7766385, 2.8724632, -8.6793633, 8.6986656
30: -3.8736196, 10.3348770, -3.8671596, 10.3132448, -12.3034668, 12.3316269
31: -14.8711138, -0.1498084, -14.8689137, -0.1749296, -10.8102112, 10.8364716
32: -20.8204231, -5.8309498, -20.8180084, -5.8452263, -12.0396423, 12.0553856
33: -38.5956154, -20.0015087, -38.5926056, -20.0071983, -10.6262398, 10.6343803
34: -35.7705536, -20.2342052, -35.7655029, -20.2659397, -11.5545921, 11.5864296
35: -33.0758324, -16.7725010, -33.0709305, -16.7967262, -11.3341446, 11.3570786
36: -31.1282902, -13.4958181, -31.1266670, -13.5127048, -12.6394882, 12.6572838
37: -50.2997360, -32.2499771, -50.2982101, -32.2753601, -9.9344711, 9.9633827
38: -38.8022919, -20.2273178, -38.8010406, -20.2444859, -11.4207191, 11.4277153
39: -42.5811996, -23.5560932, -42.5672913, -23.5582809, -10.7000122, 10.6853580
40: -38.0390015, -24.5401096, -38.0388870, -24.5524788, -7.9430199, 7.9559765
41: -24.8292465, -8.8744345, -24.8247795, -8.8904676, -13.1539192, 13.1769524
42: -15.1528063, -4.8103924, -15.1471634, -4.8145847, -8.1275139, 8.1312714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=68, inp2_unstable=69, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1784

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5582941, upper bound: 4.5705077
time: 28.87 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5705076, upper bound: 4.5705077
time: 31.06 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 61.91 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 61.91
Output dim: 3, lower bound: -4.5582941, upper bound: 4.5586024
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 61.91
Output dim: 3, lower bound: -4.5582941, upper bound: 4.5586024
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 61.91
Output dim: 3, lower bound: -4.5582941, upper bound: 4.5705077
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 61.91
Output dim: 3, lower bound: -4.5705076, upper bound: 4.5705077

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.3894768, 4.3240099, -13.3714714, 4.3130083, -15.6828766, 15.6816254
1: 0.4082558, 12.3407688, 0.4294417, 12.3323307, -9.2786026, 9.2644615
2: 2.0467105, 13.4881792, 2.0697823, 13.4776363, -9.0394402, 9.0247879
3: 1.5639286, 14.1827698, 1.5898883, 14.1698084, -9.0175247, 9.0018120
4: -4.2321391, 10.4652977, -4.1960692, 10.4477367, -12.6229095, 12.6022339
5: 2.0860636, 13.7749271, 2.1026697, 13.7662764, -8.4184341, 8.4089890
6: -25.1712074, -8.7646122, -25.1601486, -8.7849712, -13.3608055, 13.3827171
7: 2.5369289, 15.3015347, 2.5639935, 15.2909660, -9.3724213, 9.3527184
8: -4.5071082, 14.2496548, -4.4488730, 14.2239304, -15.6853333, 15.6492691
9: 0.5694947, 13.6042862, 0.5914488, 13.5942907, -9.2890968, 9.2747231
10: -4.4130602, 11.3106613, -4.4040513, 11.3031263, -12.0799141, 12.0677490
11: -4.4711099, 6.9214344, -4.4534807, 6.8969345, -8.5955734, 8.6069107
12: -26.2414417, -11.1386328, -26.2311325, -11.1722984, -10.9249496, 10.9513550
13: -14.1884441, 4.6861863, -14.1596003, 4.6624861, -13.3870430, 13.3762703
14: -24.1829948, -5.1990309, -24.1644840, -5.2193861, -16.3311005, 16.3280334
15: -7.6228743, 4.7095618, -7.6051188, 4.6982646, -11.3195267, 11.2973366
16: -7.6791406, 5.0141487, -7.6676111, 5.0032673, -9.4220810, 9.4511490
17: -26.7569084, -11.0800314, -26.7450066, -11.1069164, -11.0184517, 11.0377922
18: -17.6950474, -2.0009971, -17.6703243, -2.0522075, -10.5907097, 10.6240425
19: -10.4933290, -0.0584683, -10.4856234, -0.0658455, -6.8831348, 6.8840904
20: -5.8947687, 4.7223258, -5.8892035, 4.7184081, -7.5585175, 7.5504951
21: -8.5951385, 3.8344879, -8.5815802, 3.8228002, -9.7131577, 9.7120590
22: -10.8036518, 0.8485734, -10.7940664, 0.8348916, -7.8483238, 7.8557701
23: -4.6569200, 6.9276748, -4.6418376, 6.9070311, -8.8052979, 8.8152237
24: -8.1070051, 5.2310200, -8.0871382, 5.2014790, -10.5729141, 10.5884895
25: -8.3776331, 4.8686838, -8.3683500, 4.8561130, -8.3531990, 8.3593102
26: -16.5926189, 0.1770421, -16.5788956, 0.1527549, -11.6762543, 11.6887665
27: -7.8469687, 6.2184243, -7.8295760, 6.2016082, -12.2091217, 12.2207642
28: -6.5507717, 6.3426123, -6.5401683, 6.3276634, -10.1723022, 10.1779060
29: -7.7778416, 2.8924456, -7.7636509, 2.8723457, -8.6761208, 8.6854401
30: -3.8700955, 10.3346853, -3.8518023, 10.3124285, -12.2987366, 12.3142967
31: -14.8668365, -0.1499119, -14.8500023, -0.1755104, -10.8052521, 10.8174438
32: -20.8190804, -5.8311977, -20.8121872, -5.8462315, -12.0374756, 12.0486603
33: -38.5949364, -20.0017281, -38.5897408, -20.0079079, -10.6242332, 10.6298733
34: -35.7662239, -20.2342548, -35.7465820, -20.2663059, -11.5498886, 11.5674629
35: -33.0726624, -16.7725773, -33.0571480, -16.7971287, -11.3303528, 11.3427734
36: -31.1261120, -13.4959297, -31.1172485, -13.5131416, -12.6368675, 12.6477661
37: -50.2961998, -32.2500763, -50.2826157, -32.2757034, -9.9304543, 9.9477005
38: -38.8000183, -20.2274017, -38.7913208, -20.2448235, -11.4129448, 11.4163589
39: -42.5811691, -23.5574036, -42.5671234, -23.5639210, -10.6934814, 10.6835289
40: -38.0378113, -24.5403557, -38.0337067, -24.5535240, -7.9405689, 7.9488525
41: -24.8272991, -8.8745947, -24.8165092, -8.8912239, -13.1507607, 13.1660194
42: -15.1527414, -4.8109593, -15.1469221, -4.8171158, -8.1247902, 8.1298904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=68, inp2_unstable=68, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 971

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 717

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5579663, upper bound: 4.5496720
time: 26.02 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5579663, upper bound: 4.5701800
time: 19.61 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.3910351, 4.3240733, -13.3898392, 4.3202815, -15.6928253, 15.6985741
1: 0.4081361, 12.3425636, 0.4071975, 12.3422375, -9.2875290, 9.2882881
2: 2.0466275, 13.4903851, 2.0457077, 13.4900150, -9.0511894, 9.0508347
3: 1.5638807, 14.1854496, 1.5629735, 14.1847506, -9.0311203, 9.0310936
4: -4.2322435, 10.4690228, -4.2301316, 10.4684706, -12.6424713, 12.6400490
5: 2.0859694, 13.7769299, 2.0835323, 13.7764244, -8.4279480, 8.4300766
6: -25.1727886, -8.7644367, -25.1721916, -8.7646351, -13.3882599, 13.3937073
7: 2.5368443, 15.3042107, 2.5388288, 15.3039818, -9.3841019, 9.3805809
8: -4.5073013, 14.2556267, -4.5026579, 14.2550869, -15.7145004, 15.7089081
9: 0.5693519, 13.6064415, 0.5692184, 13.6057014, -9.2993774, 9.2987022
10: -4.4132643, 11.3114319, -4.4127235, 11.3101625, -12.0853348, 12.0860825
11: -4.4742746, 6.9215169, -4.4731693, 6.9257932, -8.6276741, 8.6239357
12: -26.2439728, -11.1384850, -26.2438354, -11.1456928, -10.9536972, 10.9627495
13: -14.1885805, 4.6898661, -14.1933022, 4.6876807, -13.4093857, 13.4138107
14: -24.1847878, -5.1988029, -24.1828861, -5.2068567, -16.3432312, 16.3453903
15: -7.6231985, 4.7116971, -7.6257439, 4.7109289, -11.3312683, 11.3244400
16: -7.6799879, 5.0141268, -7.6795201, 5.0125208, -9.4505234, 9.4553375
17: -26.7587814, -11.0799179, -26.7581291, -11.0898991, -11.0381851, 11.0496712
18: -17.7005920, -2.0008731, -17.7001114, -2.0007162, -10.6475182, 10.6494141
19: -10.4945030, -0.0584319, -10.4940243, -0.0522473, -6.8983192, 6.8917389
20: -5.8948479, 4.7224107, -5.8929543, 4.7243986, -7.5605316, 7.5584774
21: -8.5972443, 3.8345506, -8.5960560, 3.8406277, -9.7334976, 9.7264404
22: -10.8054180, 0.8485885, -10.8049917, 0.8488944, -7.8643112, 7.8649788
23: -4.6594763, 6.9277635, -4.6578894, 6.9306240, -8.8317375, 8.8298416
24: -8.1103373, 5.2310505, -8.1089544, 5.2328815, -10.6080093, 10.6076622
25: -8.3790808, 4.8687167, -8.3785229, 4.8694639, -8.3686371, 8.3677273
26: -16.5949516, 0.1771563, -16.5939236, 0.1778501, -11.7050705, 11.7010078
27: -7.8493137, 6.2185674, -7.8474293, 6.2218003, -12.2365799, 12.2377739
28: -6.5523171, 6.3427439, -6.5511599, 6.3461294, -10.1924706, 10.1877556
29: -7.7801914, 2.8924716, -7.7793341, 2.8921852, -8.6984291, 8.6994095
30: -3.8728385, 10.3348541, -3.8715992, 10.3372030, -12.3278198, 12.3323135
31: -14.8701458, -0.1498232, -14.8691216, -0.1452909, -10.8390884, 10.8346863
32: -20.8199310, -5.8310137, -20.8196468, -5.8316870, -12.0538406, 12.0566406
33: -38.5954475, -20.0015221, -38.5953827, -20.0001163, -10.6334610, 10.6356659
34: -35.7696381, -20.2341995, -35.7690125, -20.2353592, -11.5842819, 11.5867233
35: -33.0752029, -16.7725296, -33.0746689, -16.7740154, -11.3560753, 11.3583298
36: -31.1278324, -13.4958267, -31.1275272, -13.4971590, -12.6554909, 12.6571999
37: -50.2988243, -32.2500038, -50.2981873, -32.2489738, -9.9599380, 9.9616013
38: -38.8017578, -20.2273483, -38.8012390, -20.2281799, -11.4228516, 11.4276447
39: -42.5811768, -23.5563698, -42.5775185, -23.5566006, -10.7001495, 10.6959057
40: -38.0383263, -24.5401669, -38.0380936, -24.5384464, -7.9566803, 7.9543781
41: -24.8287239, -8.8744440, -24.8276234, -8.8733587, -13.1739578, 13.1764297
42: -15.1527576, -4.8105049, -15.1509380, -4.8109937, -8.1310883, 8.1327267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=68, inp2_unstable=68, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 971

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 717

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5701801, upper bound: 4.5496720
time: 24.36 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5579663, upper bound: 4.5701800
time: 36.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 63.28 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 63.28
Output dim: 3, lower bound: -4.5579663, upper bound: 4.5496720
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 63.28
Output dim: 3, lower bound: -4.5579663, upper bound: 4.5701800
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 63.28
Output dim: 3, lower bound: -4.5701801, upper bound: 4.5496720
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 63.28
Output dim: 3, lower bound: -4.5579663, upper bound: 4.5701800

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.3892307, 4.3234310, -13.3714046, 4.3128529, -15.6824570, 15.6750259
1: 0.4083347, 12.3402023, 0.4294672, 12.3321762, -9.2783813, 9.2546158
2: 2.0467534, 13.4875183, 2.0697975, 13.4774590, -9.0391960, 9.0108643
3: 1.5639420, 14.1821423, 1.5899031, 14.1696510, -9.0173378, 8.9880447
4: -4.2320623, 10.4645844, -4.1960392, 10.4475508, -12.6226425, 12.5970573
5: 2.0860989, 13.7744389, 2.1026824, 13.7661419, -8.4182816, 8.4020119
6: -25.1704865, -8.7647123, -25.1599751, -8.7850046, -13.3590164, 13.3888779
7: 2.5369835, 15.3007889, 2.5640154, 15.2907858, -9.3721962, 9.3384361
8: -4.5070114, 14.2485046, -4.4488673, 14.2236156, -15.6849747, 15.6318817
9: 0.5695529, 13.6035738, 0.5914416, 13.5940962, -9.2888641, 9.2588997
10: -4.4129858, 11.3100967, -4.4040194, 11.3029804, -12.0796967, 12.0630531
11: -4.4704885, 6.9213929, -4.4533033, 6.8969188, -8.5876846, 8.6067505
12: -26.2411003, -11.1391382, -26.2310333, -11.1724472, -10.9245834, 10.9516907
13: -14.1884193, 4.6853986, -14.1595821, 4.6622553, -13.3867798, 13.3641853
14: -24.1827621, -5.1994133, -24.1644211, -5.2195015, -16.3334274, 16.3270302
15: -7.6227674, 4.7087126, -7.6051044, 4.6980577, -11.3248138, 11.2960968
16: -7.6790876, 5.0135603, -7.6676044, 5.0030918, -9.4172859, 9.4542656
17: -26.7567673, -11.0801649, -26.7449856, -11.1069489, -11.0189171, 11.0374603
18: -17.6937180, -2.0010200, -17.6699715, -2.0522132, -10.5650406, 10.6237793
19: -10.4925718, -0.0584977, -10.4854259, -0.0658484, -6.8665848, 6.8838959
20: -5.8941393, 4.7222910, -5.8890057, 4.7183933, -7.5396862, 7.5503616
21: -8.5942879, 3.8344584, -8.5813217, 3.8228018, -9.6955338, 9.7118263
22: -10.8027878, 0.8485749, -10.7938118, 0.8348823, -7.8241043, 7.8555374
23: -4.6562452, 6.9276557, -4.6416621, 6.9070053, -8.7949104, 8.8150177
24: -8.1060543, 5.2309856, -8.0868931, 5.2014790, -10.5521164, 10.5883675
25: -8.3766365, 4.8686404, -8.3680801, 4.8561339, -8.3229256, 8.3590107
26: -16.5916615, 0.1769959, -16.5786591, 0.1527585, -11.6550407, 11.6886063
27: -7.8460922, 6.2184114, -7.8293457, 6.2016048, -12.1979752, 12.2205009
28: -6.5499992, 6.3425455, -6.5399752, 6.3276477, -10.1600037, 10.1776390
29: -7.7770405, 2.8924391, -7.7634568, 2.8723292, -8.6617889, 8.6852226
30: -3.8694320, 10.3346519, -3.8516271, 10.3124113, -12.2932968, 12.3140793
31: -14.8655701, -0.1499667, -14.8496628, -0.1754985, -10.7761040, 10.8170738
32: -20.8188515, -5.8318062, -20.8121223, -5.8463974, -12.0357819, 12.0568657
33: -38.5944328, -20.0017357, -38.5896034, -20.0079498, -10.6236191, 10.6312752
34: -35.7646637, -20.2343216, -35.7461624, -20.2663212, -11.5491982, 11.5675888
35: -33.0718918, -16.7726135, -33.0569000, -16.7971420, -11.3266945, 11.3425636
36: -31.1247902, -13.4959393, -31.1168919, -13.5131607, -12.6268120, 12.6475296
37: -50.2959213, -32.2501144, -50.2825394, -32.2757111, -9.9295692, 9.9508686
38: -38.7979889, -20.2274513, -38.7907829, -20.2448196, -11.3935509, 11.4153214
39: -42.5802155, -23.5574055, -42.5668869, -23.5639324, -10.6919479, 10.6826534
40: -38.0377884, -24.5428581, -38.0336990, -24.5543938, -7.9389381, 7.9583073
41: -24.8270988, -8.8754950, -24.8164558, -8.8914518, -13.1479950, 13.1796265
42: -15.1527300, -4.8142629, -15.1469355, -4.8179674, -8.1235466, 8.1372528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=68, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1682

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5396313, upper bound: 4.5691501
time: 32.81 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5574229, upper bound: 4.5696368
time: 55.36 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.3750839, 4.3048115, -13.3881989, 4.3135066, -15.6700287, 15.6778564
1: 0.4209139, 12.3225422, 0.4075484, 12.3351784, -9.2675858, 9.2680511
2: 2.0614438, 13.4661865, 2.0460939, 13.4816437, -9.0278320, 9.0262909
3: 1.5779881, 14.1643543, 1.5631418, 14.1772327, -9.0094376, 9.0098801
4: -4.2171898, 10.4440088, -4.2297387, 10.4597330, -12.6185303, 12.6147346
5: 2.0968804, 13.7592134, 2.0838611, 13.7701445, -8.4108810, 8.4123344
6: -25.1642609, -8.7705860, -25.1693573, -8.7654200, -13.3784866, 13.3783226
7: 2.5525620, 15.2787685, 2.5391488, 15.2949238, -9.3594933, 9.3549156
8: -4.4819813, 14.2154322, -4.5020113, 14.2407970, -15.6752243, 15.6683350
9: 0.5865285, 13.5813217, 0.5696478, 13.5966759, -9.2735710, 9.2732315
10: -4.4005847, 11.2916851, -4.4118586, 11.3034039, -12.0652161, 12.0653648
11: -4.4548430, 6.9107676, -4.4665961, 6.9255462, -8.6081314, 8.6060028
12: -26.2414932, -11.1441765, -26.2429543, -11.1466131, -10.9504509, 10.9561157
13: -14.1723118, 4.6615744, -14.1929083, 4.6781030, -13.3835831, 13.3854218
14: -24.1706982, -5.2140551, -24.1806870, -5.2120810, -16.3216782, 16.3281174
15: -7.6153841, 4.6983223, -7.6247940, 4.7064691, -11.3132515, 11.3101540
16: -7.6718550, 5.0029621, -7.6793842, 5.0085058, -9.4444618, 9.4460449
17: -26.7535992, -11.0824594, -26.7569122, -11.0908079, -11.0310555, 11.0445175
18: -17.6690407, -2.0201764, -17.6891899, -2.0010839, -10.6157417, 10.6180534
19: -10.4682388, -0.0736325, -10.4848061, -0.0523424, -6.8721886, 6.8673115
20: -5.8781152, 4.7120147, -5.8871007, 4.7241764, -7.5426273, 7.5412159
21: -8.5678988, 3.8187366, -8.5858746, 3.8403006, -9.7042007, 9.7004852
22: -10.7754459, 0.8317275, -10.7944345, 0.8488028, -7.8344383, 7.8371964
23: -4.6359272, 6.9139452, -4.6496525, 6.9303637, -8.8080635, 8.8078537
24: -8.0796919, 5.2123270, -8.0978050, 5.2327499, -10.5762749, 10.5777397
25: -8.3434324, 4.8470235, -8.3658161, 4.8692818, -8.3326492, 8.3327255
26: -16.5731850, 0.1653949, -16.5862560, 0.1776855, -11.6821938, 11.6800308
27: -7.8189082, 6.2020516, -7.8371234, 6.2214031, -12.2059631, 12.2108765
28: -6.5246930, 6.3262272, -6.5416784, 6.3457737, -10.1646767, 10.1618652
29: -7.7532554, 2.8781948, -7.7700872, 2.8921378, -8.6715355, 8.6757851
30: -3.8510931, 10.3225641, -3.8639829, 10.3367434, -12.3063736, 12.3128433
31: -14.8277473, -0.1747990, -14.8539352, -0.1455665, -10.7962837, 10.7945023
32: -20.8179245, -5.8363733, -20.8192520, -5.8327332, -12.0494766, 12.0451889
33: -38.5901794, -20.0063324, -38.5937386, -20.0007095, -10.6274834, 10.6258984
34: -35.7523689, -20.2414398, -35.7637062, -20.2355671, -11.5665550, 11.5709991
35: -33.0555191, -16.7829666, -33.0681610, -16.7742500, -11.3363457, 11.3397980
36: -31.1061268, -13.5077295, -31.1199837, -13.4975939, -12.6327591, 12.6345901
37: -50.2965393, -32.2529678, -50.2975349, -32.2495613, -9.9567451, 9.9538269
38: -38.7712936, -20.2415695, -38.7905731, -20.2283611, -11.3921013, 11.4100647
39: -42.5796471, -23.5578995, -42.5770721, -23.5569382, -10.6982422, 10.6947441
40: -38.0371475, -24.5533333, -38.0379868, -24.5429478, -7.9612675, 7.9452324
41: -24.8262157, -8.8788252, -24.8271370, -8.8741245, -13.1696358, 13.1612892
42: -15.1549854, -4.8159380, -15.1509085, -4.8123817, -8.1344757, 8.1280251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=68, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1682

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5396313, upper bound: 4.5486408
time: 32.08 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5696367, upper bound: 4.5491288
time: 33.09 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3907814, 4.3235021, -13.3897648, 4.3201380, -15.6924133, 15.6919937
1: 0.4082153, 12.3419962, 0.4072161, 12.3420954, -9.2873001, 9.2784386
2: 2.0466852, 13.4896927, 2.0457234, 13.4898338, -9.0509377, 9.0369225
3: 1.5638773, 14.1848145, 1.5629802, 14.1845970, -9.0309296, 9.0173264
4: -4.2321916, 10.4683180, -4.2301407, 10.4682951, -12.6422272, 12.6348495
5: 2.0859888, 13.7764406, 2.0835295, 13.7762833, -8.4277954, 8.4231033
6: -25.1720924, -8.7645369, -25.1719818, -8.7646456, -13.3864822, 13.3998299
7: 2.5368834, 15.3034868, 2.5388727, 15.3037930, -9.3838768, 9.3663139
8: -4.5072060, 14.2544746, -4.5026264, 14.2547827, -15.7141647, 15.6915359
9: 0.5693874, 13.6057224, 0.5692346, 13.6055355, -9.2991486, 9.2828865
10: -4.4131594, 11.3108969, -4.4126763, 11.3100166, -12.0851479, 12.0813599
11: -4.4736519, 6.9214835, -4.4729452, 6.9257927, -8.6197853, 8.6237450
12: -26.2436523, -11.1389732, -26.2437572, -11.1458054, -10.9533501, 10.9631119
13: -14.1885185, 4.6890326, -14.1932936, 4.6874828, -13.4091110, 13.4017067
14: -24.1845284, -5.1992092, -24.1827755, -5.2069530, -16.3455658, 16.3443642
15: -7.6230836, 4.7108240, -7.6257229, 4.7107019, -11.3365707, 11.3231964
16: -7.6799412, 5.0135756, -7.6795049, 5.0123377, -9.4457512, 9.4584503
17: -26.7586002, -11.0800667, -26.7580605, -11.0899343, -11.0386620, 11.0493546
18: -17.6992931, -2.0008883, -17.6997776, -2.0007348, -10.6218491, 10.6491470
19: -10.4937382, -0.0584607, -10.4938326, -0.0522437, -6.8817673, 6.8915405
20: -5.8942037, 4.7223754, -5.8927827, 4.7243843, -7.5417042, 7.5583439
21: -8.5963764, 3.8345137, -8.5958433, 3.8406148, -9.7158508, 9.7261810
22: -10.8045712, 0.8485909, -10.8047781, 0.8488703, -7.8400841, 7.8647423
23: -4.6588001, 6.9277301, -4.6577082, 6.9306135, -8.8213615, 8.8296280
24: -8.1093922, 5.2310314, -8.1086788, 5.2328668, -10.5871964, 10.6075287
25: -8.3780308, 4.8687105, -8.3782635, 4.8694696, -8.3383827, 8.3674335
26: -16.5940094, 0.1771166, -16.5937080, 0.1778605, -11.6838570, 11.7008438
27: -7.8484297, 6.2185316, -7.8471918, 6.2217970, -12.2254333, 12.2374954
28: -6.5515738, 6.3426704, -6.5509582, 6.3461366, -10.1801682, 10.1875076
29: -7.7794209, 2.8924656, -7.7791328, 2.8921831, -8.6840897, 8.6991920
30: -3.8721962, 10.3348036, -3.8714530, 10.3371935, -12.3223877, 12.3321075
31: -14.8688469, -0.1498542, -14.8687849, -0.1452930, -10.8099785, 10.8343086
32: -20.8197021, -5.8316345, -20.8195839, -5.8318529, -12.0521240, 12.0648689
33: -38.5949287, -20.0016060, -38.5952454, -20.0001392, -10.6328506, 10.6370583
34: -35.7681198, -20.2342167, -35.7686081, -20.2353935, -11.5836182, 11.5868568
35: -33.0744438, -16.7725677, -33.0743904, -16.7740269, -11.3524132, 11.3581085
36: -31.1264954, -13.4958973, -31.1271763, -13.4971867, -12.6454582, 12.6569748
37: -50.2985535, -32.2500801, -50.2981110, -32.2489853, -9.9590607, 9.9647579
38: -38.7997780, -20.2273712, -38.8006630, -20.2281837, -11.4034233, 11.4266205
39: -42.5802078, -23.5563927, -42.5772858, -23.5566139, -10.6986275, 10.6950016
40: -38.0382996, -24.5426598, -38.0381012, -24.5392799, -7.9550610, 7.9638290
41: -24.8285275, -8.8753672, -24.8275528, -8.8736153, -13.1711807, 13.1900406
42: -15.1527538, -4.8137856, -15.1509361, -4.8118515, -8.1298332, 8.1400890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=68, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1682

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5396313, upper bound: 4.5691501
time: 30.57 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5396313, upper bound: 4.5696368
time: 25.22 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 57.78 seconds
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 57.78
Output dim: 3, lower bound: -4.5396313, upper bound: 4.5691501
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 57.78
Output dim: 3, lower bound: -4.5574229, upper bound: 4.5696368
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 57.78
Output dim: 3, lower bound: -4.5396313, upper bound: 4.5486408
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 57.78
Output dim: 3, lower bound: -4.5696367, upper bound: 4.5491288
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 57.78
Output dim: 3, lower bound: -4.5396313, upper bound: 4.5691501
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 57.78
Output dim: 3, lower bound: -4.5396313, upper bound: 4.5696368

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.3882561, 4.3198724, -13.3619347, 4.3028083, -15.6729736, 15.6645126
1: 0.4088495, 12.3381767, 0.4356339, 12.3263340, -9.2727585, 9.2478981
2: 2.0469847, 13.4840965, 2.0751252, 13.4674854, -9.0292358, 9.0026474
3: 1.5640769, 14.1730900, 1.6000860, 14.1436701, -8.9908218, 8.9683418
4: -4.2317362, 10.4542227, -4.1810217, 10.4179239, -12.5925598, 12.5715179
5: 2.0863090, 13.7654419, 2.1131063, 13.7406187, -8.3923531, 8.3820992
6: -25.1685638, -8.7653599, -25.1529789, -8.7880592, -13.3476257, 13.3780975
7: 2.5372577, 15.2962627, 2.5719051, 15.2776031, -9.3584900, 9.3257751
8: -4.5063982, 14.2446461, -4.4425430, 14.2121658, -15.6723862, 15.6205063
9: 0.5698762, 13.5944195, 0.6068709, 13.5679474, -9.2623024, 9.2345314
10: -4.4125924, 11.2980280, -4.3823214, 11.2686501, -12.0451813, 12.0299644
11: -4.4655666, 6.9212446, -4.4385757, 6.8905301, -8.5767326, 8.5920029
12: -26.2335682, -11.1394329, -26.2096500, -11.1820440, -10.9068108, 10.9294167
13: -14.1842642, 4.6848993, -14.1477547, 4.6513815, -13.3837280, 13.3566780
14: -24.1756287, -5.1994553, -24.1409016, -5.2224483, -16.3317108, 16.3087273
15: -7.6223116, 4.6999454, -7.5930505, 4.6719494, -11.2977142, 11.2749062
16: -7.6789293, 5.0080233, -7.6550159, 4.9875145, -9.4004631, 9.4343147
17: -26.7420502, -11.0802431, -26.7028008, -11.1279926, -10.9849091, 10.9985657
18: -17.6935158, -2.0064073, -17.6613388, -2.0680952, -10.5506897, 10.6144714
19: -10.4876852, -0.0586686, -10.4710102, -0.0729280, -6.8552113, 6.8695030
20: -5.8914070, 4.7220817, -5.8809724, 4.7157025, -7.5337543, 7.5416508
21: -8.5916309, 3.8340938, -8.5728130, 3.8211844, -9.6897316, 9.7022820
22: -10.7989073, 0.8484454, -10.7822199, 0.8285761, -7.8151093, 7.8444176
23: -4.6488161, 6.9274836, -4.6196547, 6.8968320, -8.7773247, 8.7930298
24: -8.0983191, 5.2309427, -8.0637226, 5.1914124, -10.5341606, 10.5652390
25: -8.3650761, 4.8685598, -8.3351841, 4.8419347, -8.2978134, 8.3267632
26: -16.5894146, 0.1766857, -16.5713139, 0.1477547, -11.6462517, 11.6801109
27: -7.8426256, 6.2181520, -7.8177533, 6.1980033, -12.1869736, 12.2078552
28: -6.5413189, 6.3422713, -6.5146618, 6.3154192, -10.1391335, 10.1528549
29: -7.7700324, 2.8923733, -7.7425117, 2.8603985, -8.6429596, 8.6644440
30: -3.8617544, 10.3344784, -3.8290083, 10.3053322, -12.2798767, 12.2922401
31: -14.8620768, -0.1505778, -14.8384838, -0.1793146, -10.7661896, 10.8035812
32: -20.8182640, -5.8327188, -20.8100967, -5.8500185, -12.0251465, 12.0533981
33: -38.5900612, -20.0020256, -38.5775070, -20.0149536, -10.6133194, 10.6191521
34: -35.7642174, -20.2349205, -35.7439728, -20.2688236, -11.5409088, 11.5641632
35: -33.0671005, -16.7729034, -33.0427322, -16.8065739, -11.3134193, 11.3287544
36: -31.1180630, -13.4962454, -31.0972443, -13.5263147, -12.6066132, 12.6277809
37: -50.2870293, -32.2502747, -50.2573471, -32.2888336, -9.9074707, 9.9253941
38: -38.7924881, -20.2276802, -38.7744751, -20.2529068, -11.3783569, 11.3979530
39: -42.5747375, -23.5576878, -42.5509796, -23.5725117, -10.6772690, 10.6662598
40: -38.0377121, -24.5458679, -38.0300217, -24.5642376, -7.9278088, 7.9506626
41: -24.8254948, -8.8760681, -24.8102875, -8.8950834, -13.1360474, 13.1716995
42: -15.1473808, -4.8144965, -15.1316023, -4.8259687, -8.1099167, 8.1214066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 971

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1663

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5390986, upper bound: 4.5507204
time: 28.34 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5390986, upper bound: 4.5686170
time: 38.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.3891773, 4.3233285, -13.3712263, 4.3123884, -15.6798401, 15.6745758
1: 0.4083736, 12.3401222, 0.4295592, 12.3318033, -9.2769508, 9.2543144
2: 2.0467780, 13.4873638, 2.0698442, 13.4768467, -9.0353088, 9.0106087
3: 1.5639467, 14.1818705, 1.5899215, 14.1684933, -9.0073929, 8.9877052
4: -4.2320433, 10.4642658, -4.1959915, 10.4462271, -12.6077271, 12.5966530
5: 2.0860984, 13.7741575, 2.1027250, 13.7649269, -8.4086266, 8.4016685
6: -25.1702042, -8.7647877, -25.1588173, -8.7852268, -13.3613701, 13.3879967
7: 2.5369937, 15.3006744, 2.5640697, 15.2902079, -9.3675041, 9.3382263
8: -4.5069790, 14.2482777, -4.4486961, 14.2227669, -15.6841049, 15.6316452
9: 0.5695579, 13.6032963, 0.5915024, 13.5929413, -9.2728119, 9.2585869
10: -4.4129448, 11.3097410, -4.4039369, 11.3014460, -12.0619202, 12.0626297
11: -4.4702158, 6.9213848, -4.4522057, 6.8968916, -8.5873566, 8.5975037
12: -26.2408428, -11.1391687, -26.2300339, -11.1724901, -10.9242401, 10.9448280
13: -14.1882133, 4.6853814, -14.1589375, 4.6622057, -13.3850250, 13.3723183
14: -24.1824417, -5.1994324, -24.1631355, -5.2195110, -16.3322449, 16.3290710
15: -7.6227527, 4.7084508, -7.6050100, 4.6969547, -11.3133545, 11.2957573
16: -7.6791034, 5.0133691, -7.6675825, 5.0023460, -9.4065132, 9.4540405
17: -26.7562428, -11.0801697, -26.7428913, -11.1069574, -11.0184059, 11.0074005
18: -17.6937103, -2.0012035, -17.6699505, -2.0529256, -10.5574036, 10.6234550
19: -10.4924307, -0.0585070, -10.4847994, -0.0658998, -6.8663635, 6.8747921
20: -5.8939600, 4.7222705, -5.8882813, 4.7183623, -7.5394516, 7.5486164
21: -8.5940657, 3.8344305, -8.5803509, 3.8226864, -9.6952591, 9.7090797
22: -10.8026733, 0.8485599, -10.7933207, 0.8348620, -7.8239212, 7.8528862
23: -4.6560030, 6.9276323, -4.6406803, 6.9069967, -8.7946434, 8.8065643
24: -8.1058197, 5.2309718, -8.0858822, 5.2014422, -10.5518494, 10.5794487
25: -8.3762569, 4.8686566, -8.3665638, 4.8561101, -8.3225479, 8.3397388
26: -16.5915527, 0.1769779, -16.5781422, 0.1526819, -11.6549339, 11.6875343
27: -7.8459349, 6.2183886, -7.8286562, 6.2015405, -12.2007675, 12.2192802
28: -6.5497332, 6.3425298, -6.5388041, 6.3276176, -10.1596870, 10.1663704
29: -7.7768297, 2.8924177, -7.7625647, 2.8723159, -8.6615448, 8.6730843
30: -3.8690777, 10.3346310, -3.8501751, 10.3124132, -12.2927399, 12.3065186
31: -14.8652115, -0.1500158, -14.8482418, -0.1757023, -10.7757797, 10.8095055
32: -20.8188286, -5.8320694, -20.8120632, -5.8474951, -12.0451660, 12.0552635
33: -38.5942841, -20.0017967, -38.5889854, -20.0079765, -10.6232567, 10.6249619
34: -35.7646866, -20.2344532, -35.7461243, -20.2668476, -11.5550537, 11.5660057
35: -33.0717506, -16.7726135, -33.0562897, -16.7972126, -11.3264160, 11.3350906
36: -31.1245842, -13.4959831, -31.1160583, -13.5132341, -12.6265297, 12.6408958
37: -50.2956314, -32.2501183, -50.2813911, -32.2757416, -9.9292564, 9.9358311
38: -38.7978210, -20.2274323, -38.7900124, -20.2448597, -11.3932915, 11.4088936
39: -42.5800400, -23.5574532, -42.5661774, -23.5639668, -10.6917267, 10.6731339
40: -38.0377846, -24.5430603, -38.0336838, -24.5552464, -7.9336128, 7.9581623
41: -24.8270149, -8.8755159, -24.8161621, -8.8916378, -13.1532555, 13.1784897
42: -15.1525679, -4.8142576, -15.1462116, -4.8180251, -8.1233215, 8.1286697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 971

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1663

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5568901, upper bound: 4.5512084
time: 36.50 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5568901, upper bound: 4.5691037
time: 30.30 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.3750324, 4.3047009, -13.3880100, 4.3130751, -15.6674118, 15.6773834
1: 0.4209442, 12.3224468, 0.4076507, 12.3348055, -9.2661781, 9.2677498
2: 2.0614374, 13.4660482, 2.0461178, 13.4810286, -9.0239792, 9.0260391
3: 1.5780091, 14.1640663, 1.5631762, 14.1760540, -8.9995003, 9.0095367
4: -4.2171888, 10.4436913, -4.2296658, 10.4583902, -12.6035919, 12.6142731
5: 2.0968862, 13.7589245, 2.0838594, 13.7689285, -8.4012222, 8.4119797
6: -25.1639519, -8.7706528, -25.1681957, -8.7656441, -13.3808517, 13.3774414
7: 2.5525701, 15.2786398, 2.5391808, 15.2943754, -9.3548012, 9.3546944
8: -4.4819212, 14.2151756, -4.5019207, 14.2399120, -15.6743698, 15.6680832
9: 0.5865326, 13.5810432, 0.5697103, 13.5955410, -9.2575302, 9.2728996
10: -4.4005532, 11.2913074, -4.4117532, 11.3018293, -12.0474319, 12.0649452
11: -4.4545789, 6.9107666, -4.4655104, 6.9255185, -8.6077995, 8.5967636
12: -26.2412281, -11.1441975, -26.2419701, -11.1466694, -10.9501305, 10.9492531
13: -14.1721439, 4.6615734, -14.1922531, 4.6779985, -13.3818169, 13.3935547
14: -24.1703606, -5.2140579, -24.1794147, -5.2120686, -16.3205109, 16.3301811
15: -7.6153479, 4.6980529, -7.6247215, 4.7053480, -11.3017960, 11.3098335
16: -7.6718655, 5.0027862, -7.6793642, 5.0077758, -9.4336739, 9.4458122
17: -26.7530899, -11.0824718, -26.7548428, -11.0908251, -11.0305595, 11.0144463
18: -17.6690331, -2.0203290, -17.6891403, -2.0017309, -10.6081085, 10.6177521
19: -10.4680901, -0.0736408, -10.4841919, -0.0523860, -6.8719521, 6.8582230
20: -5.8779359, 4.7120180, -5.8864002, 4.7241387, -7.5423851, 7.5394707
21: -8.5676765, 3.8187091, -8.5849104, 3.8402231, -9.7039452, 9.6977692
22: -10.7753077, 0.8317280, -10.7939625, 0.8487973, -7.8342476, 7.8345432
23: -4.6356940, 6.9139385, -4.6486597, 6.9303389, -8.8077965, 8.7993660
24: -8.0794430, 5.2123380, -8.0968037, 5.2327423, -10.5760231, 10.5688362
25: -8.3430777, 4.8470402, -8.3642807, 4.8692656, -8.3322525, 8.3134613
26: -16.5730591, 0.1653631, -16.5857410, 0.1776323, -11.6820831, 11.6789665
27: -7.8187656, 6.2020450, -7.8364253, 6.2213531, -12.2087631, 12.2096634
28: -6.5244122, 6.3262119, -6.5404983, 6.3457351, -10.1643486, 10.1505966
29: -7.7530603, 2.8781812, -7.7691994, 2.8921094, -8.6712952, 8.6636467
30: -3.8507485, 10.3225651, -3.8625240, 10.3367043, -12.3058090, 12.3052826
31: -14.8273592, -0.1748543, -14.8525124, -0.1457868, -10.7959328, 10.7869377
32: -20.8179092, -5.8366232, -20.8191681, -5.8338242, -12.0588837, 12.0435905
33: -38.5900345, -20.0063438, -38.5931358, -20.0007229, -10.6271248, 10.6195774
34: -35.7523575, -20.2415638, -35.7636414, -20.2360725, -11.5723877, 11.5694275
35: -33.0553665, -16.7829685, -33.0675774, -16.7742977, -11.3360825, 11.3323593
36: -31.1059380, -13.5077410, -31.1191540, -13.4976406, -12.6325226, 12.6279488
37: -50.2962570, -32.2529831, -50.2963791, -32.2495918, -9.9564209, 9.9387875
38: -38.7711296, -20.2415657, -38.7898865, -20.2283974, -11.3918533, 11.4036312
39: -42.5794983, -23.5579472, -42.5763779, -23.5569878, -10.6980286, 10.6852341
40: -38.0371399, -24.5535431, -38.0379715, -24.5437737, -7.9559498, 7.9450836
41: -24.8261318, -8.8788509, -24.8268070, -8.8742857, -13.1749153, 13.1601219
42: -15.1548157, -4.8159323, -15.1501818, -4.8124304, -8.1342468, 8.1194420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 971

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1663

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5390986, upper bound: 4.5307003
time: 30.09 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5691037, upper bound: 4.5485959
time: 23.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.3898239, 4.3199434, -13.3802357, 4.3101034, -15.6829300, 15.6814041
1: 0.4087131, 12.3399630, 0.4133983, 12.3362246, -9.2816887, 9.2717171
2: 2.0469084, 13.4863167, 2.0510697, 13.4798412, -9.0409698, 9.0286674
3: 1.5639908, 14.1757469, 1.5731359, 14.1586227, -9.0044060, 8.9976158
4: -4.2318888, 10.4579639, -4.2150888, 10.4386940, -12.6121368, 12.6093025
5: 2.0861828, 13.7674446, 2.0939484, 13.7507763, -8.4018784, 8.4031944
6: -25.1702003, -8.7651825, -25.1650085, -8.7676897, -13.3750916, 13.3890610
7: 2.5371501, 15.2989302, 2.5467548, 15.2906675, -9.3701401, 9.3536339
8: -4.5065794, 14.2505875, -4.4962764, 14.2433271, -15.7016144, 15.6801224
9: 0.5697103, 13.5965452, 0.5846672, 13.5793362, -9.2725906, 9.2585068
10: -4.4127903, 11.2988367, -4.3909674, 11.2756910, -12.0506668, 12.0482750
11: -4.4687438, 6.9213433, -4.4582720, 6.9194221, -8.6088295, 8.6090126
12: -26.2360878, -11.1392765, -26.2223587, -11.1554546, -10.9355812, 10.9408226
13: -14.1843786, 4.6885471, -14.1814384, 4.6765180, -13.4060555, 13.3941956
14: -24.1773834, -5.1992340, -24.1592464, -5.2099276, -16.3438568, 16.3260460
15: -7.6226358, 4.7020645, -7.6136866, 4.6845951, -11.3094826, 11.3020134
16: -7.6797676, 5.0080242, -7.6668830, 4.9967513, -9.4289322, 9.4384689
17: -26.7439079, -11.0801086, -26.7159138, -11.1109562, -11.0046616, 11.0104256
18: -17.6991138, -2.0063143, -17.6911469, -2.0165997, -10.6074905, 10.6398201
19: -10.4888735, -0.0586321, -10.4794426, -0.0593138, -6.8703823, 6.8771496
20: -5.8914928, 4.7221594, -5.8847299, 4.7216606, -7.5357571, 7.5496387
21: -8.5937252, 3.8341808, -8.5873032, 3.8389931, -9.7100601, 9.7166519
22: -10.8006735, 0.8484602, -10.7931538, 0.8425539, -7.8311043, 7.8536148
23: -4.6513500, 6.9275560, -4.6356955, 6.9204388, -8.8037567, 8.8076363
24: -8.1015949, 5.2309570, -8.0855465, 5.2228370, -10.5692291, 10.5843849
25: -8.3665133, 4.8685999, -8.3453579, 4.8552938, -8.3132439, 8.3351860
26: -16.5917702, 0.1767908, -16.5863609, 0.1728964, -11.6750603, 11.6923714
27: -7.8449464, 6.2182961, -7.8355970, 6.2182055, -12.2144699, 12.2248840
28: -6.5428715, 6.3423882, -6.5256739, 6.3338537, -10.1593132, 10.1627083
29: -7.7723889, 2.8923850, -7.7581863, 2.8802569, -8.6652451, 8.6784210
30: -3.8644950, 10.3346357, -3.8488121, 10.3300886, -12.3090134, 12.3103104
31: -14.8653917, -0.1504576, -14.8575687, -0.1491199, -10.8000221, 10.8208237
32: -20.8191490, -5.8325539, -20.8175354, -5.8354702, -12.0415268, 12.0614128
33: -38.5905724, -20.0019398, -38.5831375, -20.0071564, -10.6225510, 10.6249504
34: -35.7676620, -20.2348900, -35.7663956, -20.2378902, -11.5753250, 11.5834198
35: -33.0696335, -16.7728519, -33.0602570, -16.7834816, -11.3391418, 11.3442764
36: -31.1197605, -13.4961777, -31.1075687, -13.5103292, -12.6252670, 12.6372337
37: -50.2896500, -32.2502365, -50.2729111, -32.2621689, -9.9369545, 9.9392681
38: -38.7942238, -20.2276402, -38.7843704, -20.2363129, -11.3882446, 11.4092388
39: -42.5747452, -23.5566292, -42.5613747, -23.5652637, -10.6839600, 10.6786423
40: -38.0382233, -24.5456848, -38.0343895, -24.5490932, -7.9439335, 7.9561749
41: -24.8269081, -8.8759241, -24.8214016, -8.8772211, -13.1592484, 13.1821404
42: -15.1473885, -4.8140430, -15.1356115, -4.8198152, -8.1162186, 8.1242561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 971

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1663

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5513134, upper bound: 4.5507204
time: 16.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5390986, upper bound: 4.5686170
time: 35.04 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.3907356, 4.3233995, -13.3895664, 4.3197041, -15.6898041, 15.6915131
1: 0.4082448, 12.3419085, 0.4073040, 12.3417149, -9.2858925, 9.2781296
2: 2.0466931, 13.4895382, 2.0457711, 13.4892178, -9.0470619, 9.0366478
3: 1.5638671, 14.1845360, 1.5630093, 14.1834249, -9.0209846, 9.0169830
4: -4.2321663, 10.4680061, -4.2300749, 10.4669600, -12.6272812, 12.6344643
5: 2.0860009, 13.7761593, 2.0835593, 13.7750740, -8.4181366, 8.4227753
6: -25.1717834, -8.7646112, -25.1708450, -8.7648697, -13.3888397, 13.3989334
7: 2.5369062, 15.3033419, 2.5389068, 15.3032150, -9.3791847, 9.3660774
8: -4.5071602, 14.2542343, -4.5024877, 14.2538815, -15.7132797, 15.6912842
9: 0.5693989, 13.6054440, 0.5692902, 13.6043863, -9.2831078, 9.2825508
10: -4.4131346, 11.3105259, -4.4126072, 11.3084564, -12.0673447, 12.0809441
11: -4.4734173, 6.9214902, -4.4718800, 6.9257722, -8.6194305, 8.6145172
12: -26.2434006, -11.1389790, -26.2427654, -11.1458645, -10.9530029, 10.9562340
13: -14.1883440, 4.6890249, -14.1926155, 4.6873717, -13.4073486, 13.4098434
14: -24.1841927, -5.1992121, -24.1815071, -5.2069693, -16.3443832, 16.3463974
15: -7.6230545, 4.7105627, -7.6256247, 4.7095952, -11.3251190, 11.3228683
16: -7.6799278, 5.0133772, -7.6794968, 5.0115767, -9.4349709, 9.4582214
17: -26.7580910, -11.0800714, -26.7559929, -11.0899582, -11.0381584, 11.0192642
18: -17.6992874, -2.0010715, -17.6997433, -2.0014119, -10.6142120, 10.6488266
19: -10.4936047, -0.0584695, -10.4932175, -0.0523102, -6.8815365, 6.8824387
20: -5.8940468, 4.7223635, -5.8920808, 4.7243676, -7.5414658, 7.5566082
21: -8.5961542, 3.8344946, -8.5948524, 3.8405287, -9.7155914, 9.7234497
22: -10.8044643, 0.8485761, -10.8042746, 0.8488646, -7.8399086, 7.8620872
23: -4.6585665, 6.9277210, -4.6567121, 6.9305887, -8.8211021, 8.8211594
24: -8.1091585, 5.2310257, -8.1076813, 5.2328672, -10.5869370, 10.5986290
25: -8.3776665, 4.8687038, -8.3767338, 4.8694458, -8.3379822, 8.3481693
26: -16.5939217, 0.1770841, -16.5931892, 0.1777996, -11.6837578, 11.6997948
27: -7.8482494, 6.2185173, -7.8464985, 6.2217469, -12.2282715, 12.2362862
28: -6.5512786, 6.3426542, -6.5497994, 6.3460665, -10.1798553, 10.1762314
29: -7.7792020, 2.8924596, -7.7782435, 2.8921700, -8.6838455, 8.6870422
30: -3.8718300, 10.3347969, -3.8699861, 10.3371334, -12.3218460, 12.3245544
31: -14.8684740, -0.1499104, -14.8673515, -0.1454911, -10.8096313, 10.8267288
32: -20.8196754, -5.8318901, -20.8195095, -5.8329296, -12.0615158, 12.0632515
33: -38.5947800, -20.0016327, -38.5946426, -20.0001564, -10.6324730, 10.6307564
34: -35.7681007, -20.2343655, -35.7685280, -20.2359161, -11.5894585, 11.5852737
35: -33.0742874, -16.7725735, -33.0738068, -16.7740936, -11.3521385, 11.3506546
36: -31.1263142, -13.4959002, -31.1263428, -13.4972363, -12.6451797, 12.6503487
37: -50.2982864, -32.2501068, -50.2969284, -32.2490616, -9.9587326, 9.9497089
38: -38.7995720, -20.2273712, -38.7999115, -20.2282391, -11.4031601, 11.4201794
39: -42.5800781, -23.5564404, -42.5765915, -23.5566940, -10.6984062, 10.6855087
40: -38.0382919, -24.5428734, -38.0380821, -24.5401306, -7.9497414, 7.9636803
41: -24.8284512, -8.8754177, -24.8272610, -8.8737583, -13.1764488, 13.1888924
42: -15.1525908, -4.8137879, -15.1502256, -4.8118725, -8.1296158, 8.1314983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 971

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1663

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5691037, upper bound: 4.5512084
time: 25.45 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5691037, upper bound: 4.5691037
time: 25.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 53.20 seconds
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 53.20
Output dim: 3, lower bound: -4.5390986, upper bound: 4.5507204
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 53.20
Output dim: 3, lower bound: -4.5390986, upper bound: 4.5686170
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 53.20
Output dim: 3, lower bound: -4.5568901, upper bound: 4.5512084
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 53.20
Output dim: 3, lower bound: -4.5568901, upper bound: 4.5691037
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 53.20
Output dim: 3, lower bound: -4.5390986, upper bound: 4.5307003
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 53.20
Output dim: 3, lower bound: -4.5691037, upper bound: 4.5485959
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 53.20
Output dim: 3, lower bound: -4.5513134, upper bound: 4.5507204
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 53.20
Output dim: 3, lower bound: -4.5390986, upper bound: 4.5686170
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 53.20
Output dim: 3, lower bound: -4.5691037, upper bound: 4.5512084
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 53.20
Output dim: 3, lower bound: -4.5691037, upper bound: 4.5691037

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.3881197, 4.3191752, -13.3619041, 4.3027263, -15.6727142, 15.6572266
1: 0.4088628, 12.3374624, 0.4356337, 12.3262348, -9.2726364, 9.2383842
2: 2.0470209, 13.4835434, 2.0751271, 13.4674101, -9.0290947, 8.9956245
3: 1.5640926, 14.1721897, 1.6000850, 14.1435661, -8.9906845, 8.9569206
4: -4.2317286, 10.4534359, -4.1810131, 10.4178410, -12.5924225, 12.5662155
5: 2.0863185, 13.7647219, 2.1131153, 13.7405310, -8.3922424, 8.3740196
6: -25.1684723, -8.7656651, -25.1529732, -8.7880859, -13.3466415, 13.3848038
7: 2.5372651, 15.2955704, 2.5719182, 15.2775402, -9.3583832, 9.3169594
8: -4.5063677, 14.2436399, -4.4425049, 14.2120571, -15.6722336, 15.6113434
9: 0.5699122, 13.5937881, 0.6068883, 13.5678749, -9.2621994, 9.2255249
10: -4.4124646, 11.2977257, -4.3822975, 11.2686176, -12.0450249, 12.0296593
11: -4.4648876, 6.9212427, -4.4384966, 6.8905334, -8.5707474, 8.5918922
12: -26.2331200, -11.1394787, -26.2095985, -11.1820393, -10.9030800, 10.9293251
13: -14.1842003, 4.6841927, -14.1477547, 4.6512942, -13.3835983, 13.3517838
14: -24.1750793, -5.1995687, -24.1408291, -5.2224569, -16.3317108, 16.3079262
15: -7.6222658, 4.6994042, -7.5930505, 4.6718893, -11.2979469, 11.2743263
16: -7.6789064, 5.0077038, -7.6550055, 4.9874802, -9.3990364, 9.4394608
17: -26.7413673, -11.0802298, -26.7027187, -11.1279802, -10.9721642, 10.9984627
18: -17.6927509, -2.0064688, -17.6612473, -2.0680866, -10.5405769, 10.6143227
19: -10.4869785, -0.0586667, -10.4709072, -0.0729308, -6.8438187, 6.8694096
20: -5.8908534, 4.7220774, -5.8809090, 4.7156992, -7.5253868, 7.5415268
21: -8.5910454, 3.8340669, -8.5727444, 3.8211710, -9.6830750, 9.7021713
22: -10.7983093, 0.8484247, -10.7821503, 0.8285642, -7.8029289, 7.8443222
23: -4.6480150, 6.9274621, -4.6195440, 6.8968115, -8.7694168, 8.7929077
24: -8.0973473, 5.2309256, -8.0636120, 5.1914182, -10.5215225, 10.5650826
25: -8.3641396, 4.8685355, -8.3350639, 4.8419390, -8.2832603, 8.3266258
26: -16.5886536, 0.1766742, -16.5712433, 0.1477790, -11.6335411, 11.6799507
27: -7.8421125, 6.2181482, -7.8176804, 6.1980047, -12.1852417, 12.2077789
28: -6.5405660, 6.3422432, -6.5145683, 6.3154101, -10.1297722, 10.1527328
29: -7.7695045, 2.8923516, -7.7424464, 2.8603878, -8.6342163, 8.6643524
30: -3.8609078, 10.3344440, -3.8288982, 10.3053341, -12.2735291, 12.2920990
31: -14.8610725, -0.1505713, -14.8383350, -0.1793294, -10.7529449, 10.8034363
32: -20.8181953, -5.8329725, -20.8100891, -5.8500595, -12.0246391, 12.0571480
33: -38.5895538, -20.0021591, -38.5774269, -20.0149727, -10.6118202, 10.6189728
34: -35.7638588, -20.2349453, -35.7439194, -20.2688160, -11.5403442, 11.5641136
35: -33.0667496, -16.7729607, -33.0426636, -16.8066044, -11.3114738, 11.3286400
36: -31.1173496, -13.4963255, -31.0971451, -13.5263281, -12.6056328, 12.6276817
37: -50.2864609, -32.2503777, -50.2572289, -32.2888565, -9.9022217, 9.9252663
38: -38.7914276, -20.2277336, -38.7743301, -20.2529373, -11.3801498, 11.3973770
39: -42.5740128, -23.5577145, -42.5508881, -23.5725040, -10.6771660, 10.6660213
40: -38.0376625, -24.5460396, -38.0300102, -24.5642471, -7.9274731, 7.9538822
41: -24.8254395, -8.8762379, -24.8102760, -8.8950863, -13.1350708, 13.1753616
42: -15.1473045, -4.8147955, -15.1316071, -4.8260036, -8.1095352, 8.1253815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5540931
time: 29.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5680603
time: 22.54 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3890400, 4.3226156, -13.3712139, 4.3123136, -15.6795731, 15.6673088
1: 0.4084029, 12.3393850, 0.4295812, 12.3317127, -9.2768288, 9.2448120
2: 2.0468142, 13.4867811, 2.0698311, 13.4767790, -9.0351791, 9.0036011
3: 1.5639925, 14.1809635, 1.5899358, 14.1683741, -9.0072670, 8.9762802
4: -4.2320137, 10.4634724, -4.1959987, 10.4461126, -12.6075897, 12.5913582
5: 2.0861161, 13.7734289, 2.1027269, 13.7648373, -8.4085274, 8.3935661
6: -25.1701317, -8.7650528, -25.1588154, -8.7852592, -13.3603973, 13.3946800
7: 2.5370073, 15.2999821, 2.5640826, 15.2901154, -9.3673973, 9.3294029
8: -4.5069408, 14.2473011, -4.4486966, 14.2226295, -15.6839523, 15.6225052
9: 0.5696084, 13.6026707, 0.5914893, 13.5928688, -9.2726936, 9.2495766
10: -4.4128170, 11.3094215, -4.4038982, 11.3014069, -12.0617638, 12.0623131
11: -4.4695606, 6.9213681, -4.4521322, 6.8968954, -8.5813751, 8.5974007
12: -26.2404099, -11.1391935, -26.2299881, -11.1724834, -10.9205055, 10.9447556
13: -14.1881952, 4.6846852, -14.1589460, 4.6621389, -13.3848724, 13.3674431
14: -24.1819019, -5.1995382, -24.1630707, -5.2195253, -16.3322220, 16.3282738
15: -7.6226850, 4.7079096, -7.6050167, 4.6968865, -11.3135643, 11.2951775
16: -7.6790495, 5.0130873, -7.6675701, 5.0023193, -9.4050827, 9.4591751
17: -26.7555733, -11.0801935, -26.7428112, -11.1069775, -11.0056267, 11.0072937
18: -17.6929436, -2.0012493, -17.6698589, -2.0529404, -10.5472717, 10.6233215
19: -10.4917240, -0.0585124, -10.4847050, -0.0659025, -6.8549786, 6.8746948
20: -5.8934221, 4.7222505, -5.8882189, 4.7183585, -7.5310745, 7.5484982
21: -8.5934439, 3.8343883, -8.5802584, 3.8226814, -9.6885948, 9.7089653
22: -10.8020773, 0.8485324, -10.7932673, 0.8348727, -7.8117485, 7.8527908
23: -4.6552305, 6.9276333, -4.6405859, 6.9069920, -8.7867584, 8.8064346
24: -8.1048775, 5.2309856, -8.0857792, 5.2014403, -10.5392151, 10.5793114
25: -8.3753357, 4.8686237, -8.3664513, 4.8561068, -8.3080025, 8.3395977
26: -16.5908108, 0.1769595, -16.5780411, 0.1526692, -11.6422348, 11.6874046
27: -7.8454452, 6.2183461, -7.8285904, 6.2015424, -12.1990128, 12.2191696
28: -6.5489874, 6.3425016, -6.5387187, 6.3276086, -10.1503258, 10.1662483
29: -7.7762861, 2.8924344, -7.7624998, 2.8723135, -8.6527977, 8.6730003
30: -3.8682377, 10.3346043, -3.8500583, 10.3123999, -12.2863922, 12.3063660
31: -14.8641682, -0.1500463, -14.8481150, -0.1757164, -10.7625465, 10.8093452
32: -20.8187542, -5.8323002, -20.8120461, -5.8475380, -12.0446472, 12.0590096
33: -38.5937309, -20.0018806, -38.5889206, -20.0080128, -10.6217499, 10.6247787
34: -35.7643127, -20.2344379, -35.7460785, -20.2668419, -11.5544930, 11.5659409
35: -33.0714188, -16.7726955, -33.0562439, -16.7972202, -11.3244743, 11.3350067
36: -31.1238976, -13.4960527, -31.1159573, -13.5132332, -12.6255341, 12.6407776
37: -50.2950554, -32.2501869, -50.2813072, -32.2757492, -9.9240112, 9.9356995
38: -38.7967606, -20.2275066, -38.7898788, -20.2448654, -11.3950806, 11.4083214
39: -42.5793381, -23.5575085, -42.5661011, -23.5639801, -10.6916275, 10.6729012
40: -38.0377350, -24.5432053, -38.0336800, -24.5552559, -7.9332809, 7.9613781
41: -24.8269501, -8.8756933, -24.8161564, -8.8916473, -13.1522789, 13.1821175
42: -15.1524839, -4.8145638, -15.1462002, -4.8180599, -8.1229362, 8.1326256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5563304, upper bound: 4.5545600
time: 45.91 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5563304, upper bound: 4.5685449
time: 32.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.3749018, 4.3040047, -13.3879929, 4.3129926, -15.6671600, 15.6700974
1: 0.4209652, 12.3217258, 0.4076591, 12.3347216, -9.2660484, 9.2582245
2: 2.0615106, 13.4654655, 2.0461369, 13.4809523, -9.0238342, 9.0190315
3: 1.5780311, 14.1631975, 1.5631771, 14.1759481, -8.9993629, 8.9981079
4: -4.2171583, 10.4429054, -4.2296724, 10.4582968, -12.6034546, 12.6090088
5: 2.0969026, 13.7582035, 2.0838699, 13.7688618, -8.4011269, 8.4038811
6: -25.1638908, -8.7709293, -25.1681919, -8.7656784, -13.3798790, 13.3841324
7: 2.5525994, 15.2779465, 2.5391870, 15.2942820, -9.3546906, 9.3458900
8: -4.4818978, 14.2142143, -4.5018954, 14.2397881, -15.6742096, 15.6589432
9: 0.5865979, 13.5804195, 0.5697143, 13.5954762, -9.2574043, 9.2638855
10: -4.4004183, 11.2909908, -4.4117584, 11.3017969, -12.0472641, 12.0646324
11: -4.4539161, 6.9107275, -4.4654207, 6.9255056, -8.6018486, 8.5966530
12: -26.2407799, -11.1442509, -26.2419128, -11.1466818, -10.9463959, 10.9491653
13: -14.1720886, 4.6608543, -14.1922398, 4.6779180, -13.3816910, 13.3886909
14: -24.1698112, -5.2141800, -24.1793385, -5.2120972, -16.3205261, 16.3293686
15: -7.6152868, 4.6975126, -7.6247125, 4.7052708, -11.3020020, 11.3092384
16: -7.6718106, 5.0024738, -7.6793509, 5.0077333, -9.4322281, 9.4509315
17: -26.7524261, -11.0824871, -26.7547607, -11.0908270, -11.0177956, 11.0143204
18: -17.6682625, -2.0203900, -17.6890392, -2.0017538, -10.5979691, 10.6175842
19: -10.4673901, -0.0736918, -10.4841194, -0.0524049, -6.8605671, 6.8581238
20: -5.8773913, 4.7119670, -5.8863220, 4.7241335, -7.5340118, 7.5393448
21: -8.5670471, 3.8186984, -8.5848246, 3.8402240, -9.6972923, 9.6976318
22: -10.7747421, 0.8317084, -10.7939014, 0.8488009, -7.8220825, 7.8344460
23: -4.6348772, 6.9138942, -4.6485605, 6.9303493, -8.7999077, 8.7992554
24: -8.0785007, 5.2123051, -8.0966825, 5.2327552, -10.5634041, 10.5686874
25: -8.3421421, 4.8469844, -8.3641605, 4.8692527, -8.3176994, 8.3133221
26: -16.5722904, 0.1653212, -16.5856514, 0.1776329, -11.6693726, 11.6788292
27: -7.8182549, 6.2019949, -7.8363733, 6.2213683, -12.2070160, 12.2095718
28: -6.5236530, 6.3261757, -6.5404177, 6.3457336, -10.1550026, 10.1504669
29: -7.7525311, 2.8781824, -7.7691326, 2.8921065, -8.6625481, 8.6635551
30: -3.8499506, 10.3225155, -3.8624094, 10.3367128, -12.2994461, 12.3051338
31: -14.8263531, -0.1748781, -14.8523922, -0.1457789, -10.7827187, 10.7867928
32: -20.8178406, -5.8368702, -20.8191566, -5.8338547, -12.0583839, 12.0473289
33: -38.5895081, -20.0064278, -38.5930672, -20.0007439, -10.6256371, 10.6193981
34: -35.7519760, -20.2415810, -35.7635956, -20.2360916, -11.5718155, 11.5693474
35: -33.0550346, -16.7830334, -33.0675049, -16.7742958, -11.3341331, 11.3322449
36: -31.1052551, -13.5078068, -31.1190491, -13.4976711, -12.6315155, 12.6278267
37: -50.2956734, -32.2530365, -50.2962875, -32.2495728, -9.9511948, 9.9386749
38: -38.7700920, -20.2416172, -38.7897377, -20.2284107, -11.3936234, 11.4030533
39: -42.5787811, -23.5579586, -42.5762939, -23.5569973, -10.6979332, 10.6849976
40: -38.0371056, -24.5536842, -38.0379715, -24.5437889, -7.9556026, 7.9483013
41: -24.8260727, -8.8789988, -24.8268032, -8.8743114, -13.1739426, 13.1637917
42: -15.1547470, -4.8162384, -15.1501637, -4.8124595, -8.1338577, 8.1234131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5340313
time: 29.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5480363
time: 19.89 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.3896713, 4.3192539, -13.3802290, 4.3100052, -15.6826630, 15.6741295
1: 0.4087431, 12.3392487, 0.4133964, 12.3361406, -9.2815552, 9.2622032
2: 2.0469522, 13.4857340, 2.0510719, 13.4797735, -9.0408401, 9.0216827
3: 1.5640216, 14.1748476, 1.5731435, 14.1585035, -9.0042725, 8.9861984
4: -4.2318430, 10.4571781, -4.2150965, 10.4385786, -12.6120377, 12.6040001
5: 2.0862076, 13.7667255, 2.0939498, 13.7506809, -8.4017639, 8.3951187
6: -25.1701050, -8.7654552, -25.1649990, -8.7677393, -13.3741264, 13.3957596
7: 2.5371938, 15.2982454, 2.5467441, 15.2905598, -9.3700333, 9.3448219
8: -4.5065413, 14.2496243, -4.4962845, 14.2432041, -15.7014389, 15.6709900
9: 0.5697539, 13.5959206, 0.5846920, 13.5792713, -9.2724838, 9.2494926
10: -4.4126472, 11.2984896, -4.3909521, 11.2756519, -12.0504875, 12.0479736
11: -4.4680667, 6.9213028, -4.4581838, 6.9194326, -8.6028481, 8.6089096
12: -26.2356720, -11.1393003, -26.2223072, -11.1554480, -10.9318314, 10.9407387
13: -14.1843243, 4.6878586, -14.1814327, 4.6764364, -13.4059372, 13.3893089
14: -24.1768379, -5.1993465, -24.1591759, -5.2099352, -16.3438339, 16.3252373
15: -7.6225686, 4.7015162, -7.6136727, 4.6845198, -11.3097076, 11.3014297
16: -7.6797061, 5.0077133, -7.6668873, 4.9967251, -9.4275055, 9.4435921
17: -26.7432613, -11.0801334, -26.7158184, -11.1109447, -10.9918900, 11.0103149
18: -17.6983414, -2.0063672, -17.6910400, -2.0165987, -10.5973625, 10.6396790
19: -10.4881592, -0.0586534, -10.4793482, -0.0593159, -6.8589821, 6.8770618
20: -5.8909483, 4.7221317, -5.8846750, 4.7216640, -7.5273743, 7.5495052
21: -8.5931282, 3.8341300, -8.5872202, 3.8389959, -9.7033920, 9.7165222
22: -10.8001003, 0.8484552, -10.7930698, 0.8425651, -7.8189163, 7.8535500
23: -4.6505785, 6.9275284, -4.6355848, 6.9204254, -8.7958641, 8.8075294
24: -8.1006470, 5.2309551, -8.0854273, 5.2228203, -10.5566101, 10.5842438
25: -8.3655796, 4.8685818, -8.3452301, 4.8552971, -8.2986794, 8.3350487
26: -16.5910015, 0.1767650, -16.5862808, 0.1728766, -11.6623573, 11.6922150
27: -7.8444209, 6.2182608, -7.8355246, 6.2182140, -12.2127228, 12.2247849
28: -6.5421348, 6.3423600, -6.5255632, 6.3338461, -10.1499405, 10.1625824
29: -7.7718692, 2.8923786, -7.7581210, 2.8802614, -8.6564941, 8.6783562
30: -3.8636663, 10.3345871, -3.8486936, 10.3300819, -12.3026810, 12.3101768
31: -14.8643780, -0.1504893, -14.8574419, -0.1491256, -10.7868080, 10.8206711
32: -20.8190422, -5.8327732, -20.8175182, -5.8355026, -12.0410004, 12.0651474
33: -38.5900345, -20.0019951, -38.5830879, -20.0071774, -10.6210556, 10.6247787
34: -35.7673264, -20.2348785, -35.7663422, -20.2378864, -11.5747681, 11.5833588
35: -33.0692825, -16.7728672, -33.0602036, -16.7834740, -11.3372040, 11.3441772
36: -31.1190586, -13.4962492, -31.1074848, -13.5103273, -12.6242523, 12.6371269
37: -50.2890511, -32.2502899, -50.2728119, -32.2621765, -9.9317131, 9.9391556
38: -38.7931595, -20.2276821, -38.7842293, -20.2363205, -11.3900185, 11.4086628
39: -42.5740204, -23.5566998, -42.5612831, -23.5652714, -10.6838493, 10.6783943
40: -38.0381889, -24.5458622, -38.0343857, -24.5491257, -7.9435997, 7.9593925
41: -24.8268433, -8.8760595, -24.8213997, -8.8772383, -13.1582718, 13.1857605
42: -15.1473207, -4.8143334, -15.1356077, -4.8198729, -8.1158257, 8.1282310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5507566, upper bound: 4.5540931
time: 34.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5680603
time: 49.16 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.3785706, 4.3017569, -13.3882828, 4.3121891, -15.6693649, 15.6672974
1: 0.4171662, 12.3197212, 0.4076009, 12.3339939, -9.2688446, 9.2554054
2: 2.0532742, 13.4720154, 2.0461898, 13.4831543, -9.0340500, 9.0184975
3: 1.5728359, 14.1578970, 1.5631258, 14.1740999, -9.0026970, 8.9902573
4: -4.2229595, 10.4435301, -4.2297873, 10.4584866, -12.6096039, 12.6096420
5: 2.0938950, 13.7541609, 2.0838029, 13.7674122, -8.4026260, 8.4005547
6: -25.1697750, -8.7695532, -25.1703033, -8.7662735, -13.3839531, 13.3899879
7: 2.5457382, 15.2824173, 2.5391483, 15.2959118, -9.3630333, 9.3449478
8: -4.4938192, 14.2249813, -4.5019870, 14.2436962, -15.6899261, 15.6615982
9: 0.5787947, 13.5862904, 0.5696235, 13.5976801, -9.2672691, 9.2630806
10: -4.4072304, 11.3090563, -4.4111161, 11.3080263, -12.0593872, 12.0756950
11: -4.4527607, 6.9156303, -4.4647913, 6.9255719, -8.5987206, 8.6016121
12: -26.2306824, -11.1443911, -26.2383423, -11.1462879, -10.9398193, 10.9464073
13: -14.1859770, 4.6675539, -14.1922626, 4.6799946, -13.3959846, 13.3877602
14: -24.1657448, -5.1993618, -24.1758232, -5.2070293, -16.3256073, 16.3396568
15: -7.6162262, 4.6944132, -7.6249619, 4.7040491, -11.3127823, 11.3062210
16: -7.6733894, 5.0043144, -7.6791878, 5.0084491, -9.4298820, 9.4511719
17: -26.7367687, -11.0873108, -26.7489014, -11.0899925, -11.0168648, 11.0044975
18: -17.6751633, -2.0096936, -17.6914482, -2.0019178, -10.5899239, 10.6317940
19: -10.4720039, -0.0651202, -10.4856968, -0.0523765, -6.8599091, 6.8682938
20: -5.8769293, 4.7171679, -5.8861866, 4.7241077, -7.5234375, 7.5450745
21: -8.5773888, 3.8301141, -8.5885067, 3.8402364, -9.6966858, 9.7122993
22: -10.7863541, 0.8423057, -10.7980442, 0.8487816, -7.8217239, 7.8489799
23: -4.6342564, 6.9202542, -4.6482964, 6.9303851, -8.7966080, 8.8052711
24: -8.0800829, 5.2217913, -8.0976152, 5.2327595, -10.5574074, 10.5790787
25: -8.3487682, 4.8588676, -8.3667297, 4.8692465, -8.3088112, 8.3282127
26: -16.5701885, 0.1691884, -16.5850334, 0.1776415, -11.6596603, 11.6835098
27: -7.8327751, 6.2158527, -7.8412828, 6.2214456, -12.2124939, 12.2284012
28: -6.5279641, 6.3343925, -6.5417213, 6.3457718, -10.1562920, 10.1598320
29: -7.7631445, 2.8877673, -7.7726860, 2.8921335, -8.6673126, 8.6762695
30: -3.8459015, 10.3275623, -3.8611360, 10.3367672, -12.2957230, 12.3083725
31: -14.8380833, -0.1586840, -14.8567448, -0.1457262, -10.7790184, 10.8073692
32: -20.8182564, -5.8362741, -20.8190651, -5.8339968, -12.0583420, 12.0567436
33: -38.5894928, -20.0045929, -38.5928116, -20.0010147, -10.6258698, 10.6256962
34: -35.7574730, -20.2384567, -35.7649994, -20.2360744, -11.5781097, 11.5771866
35: -33.0673370, -16.7768574, -33.0715256, -16.7745609, -11.3431435, 11.3427773
36: -31.1219540, -13.4996490, -31.1248055, -13.4978542, -12.6386414, 12.6442528
37: -50.2859955, -32.2559280, -50.2926979, -32.2497253, -9.9448738, 9.9411793
38: -38.7932587, -20.2302170, -38.7977753, -20.2288475, -11.3955078, 11.4149151
39: -42.5781021, -23.5577984, -42.5758934, -23.5571556, -10.6957741, 10.6836967
40: -38.0366096, -24.5461540, -38.0377388, -24.5411472, -7.9465675, 7.9579697
41: -24.8269997, -8.8815899, -24.8269634, -8.8754969, -13.1733475, 13.1812820
42: -15.1507740, -4.8171406, -15.1496544, -4.8128309, -8.1266975, 8.1262398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5366441
time: 34.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5685449, upper bound: 4.5506487
time: 29.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3905869, 4.3226862, -13.3895512, 4.3196220, -15.6895599, 15.6842422
1: 0.4082754, 12.3411789, 0.4073417, 12.3416309, -9.2857475, 9.2686234
2: 2.0467536, 13.4889870, 2.0457590, 13.4891348, -9.0469284, 9.0296478
3: 1.5638988, 14.1836519, 1.5630076, 14.1833124, -9.0208549, 9.0055466
4: -4.2321205, 10.4672136, -4.2300663, 10.4668455, -12.6271591, 12.6291466
5: 2.0860422, 13.7754316, 2.0835838, 13.7749777, -8.4180374, 8.4146652
6: -25.1717224, -8.7648840, -25.1708069, -8.7649183, -13.3878860, 13.4056473
7: 2.5368989, 15.3026495, 2.5389123, 15.3031292, -9.3790703, 9.3572617
8: -4.5071058, 14.2532568, -4.5024796, 14.2537422, -15.7131424, 15.6821518
9: 0.5694585, 13.6048040, 0.5693085, 13.6043053, -9.2829895, 9.2735405
10: -4.4130220, 11.3102236, -4.4125996, 11.3084030, -12.0671997, 12.0806236
11: -4.4727163, 6.9214644, -4.4717999, 6.9257755, -8.6134682, 8.6144333
12: -26.2429695, -11.1390438, -26.2427254, -11.1458740, -10.9492683, 10.9561539
13: -14.1883259, 4.6883364, -14.1926403, 4.6872988, -13.4072189, 13.4049721
14: -24.1836395, -5.1993027, -24.1814404, -5.2069902, -16.3443756, 16.3456001
15: -7.6230011, 4.7100139, -7.6256399, 4.7095275, -11.3253174, 11.3222809
16: -7.6798959, 5.0130959, -7.6794777, 5.0115499, -9.4335251, 9.4633255
17: -26.7574234, -11.0800734, -26.7559071, -11.0899649, -11.0253754, 11.0191460
18: -17.6985283, -2.0011106, -17.6996536, -2.0014262, -10.6040878, 10.6486816
19: -10.4928970, -0.0584834, -10.4931450, -0.0523140, -6.8701649, 6.8823414
20: -5.8935170, 4.7223048, -5.8920116, 4.7243485, -7.5330734, 7.5564823
21: -8.5955791, 3.8344665, -8.5947762, 3.8405390, -9.7089272, 9.7233505
22: -10.8038607, 0.8485703, -10.8042059, 0.8488605, -7.8277397, 7.8620071
23: -4.6577721, 6.9277077, -4.6566234, 6.9305692, -8.8132095, 8.8210335
24: -8.1081934, 5.2310166, -8.1075792, 5.2328496, -10.5742874, 10.5984802
25: -8.3767309, 4.8686790, -8.3766136, 4.8694339, -8.3234177, 8.3480206
26: -16.5931511, 0.1770501, -16.5930729, 0.1778080, -11.6710587, 11.6996651
27: -7.8477612, 6.2185049, -7.8464255, 6.2217484, -12.2265167, 12.2362213
28: -6.5505252, 6.3426037, -6.5497141, 6.3460579, -10.1704636, 10.1761055
29: -7.7786598, 2.8924453, -7.7781782, 2.8921745, -8.6750984, 8.6869698
30: -3.8710353, 10.3347559, -3.8698714, 10.3371334, -12.3155060, 12.3244286
31: -14.8674774, -0.1499109, -14.8672247, -0.1454909, -10.7964134, 10.8265877
32: -20.8195896, -5.8321180, -20.8194923, -5.8329859, -12.0610199, 12.0669899
33: -38.5942535, -20.0017357, -38.5945740, -20.0001640, -10.6309891, 10.6305771
34: -35.7677536, -20.2343807, -35.7684669, -20.2359161, -11.5888977, 11.5852165
35: -33.0739441, -16.7726192, -33.0737457, -16.7740955, -11.3502007, 11.3505363
36: -31.1256180, -13.4959583, -31.1262302, -13.4972429, -12.6441727, 12.6502151
37: -50.2976990, -32.2501488, -50.2968445, -32.2490692, -9.9535027, 9.9495964
38: -38.7985077, -20.2274628, -38.7997818, -20.2282429, -11.4049530, 11.4196129
39: -42.5793686, -23.5564804, -42.5764999, -23.5566864, -10.6982994, 10.6852646
40: -38.0382462, -24.5430260, -38.0380783, -24.5401726, -7.9493999, 7.9668884
41: -24.8283482, -8.8755999, -24.8272438, -8.8737803, -13.1755066, 13.1925278
42: -15.1525145, -4.8140855, -15.1502075, -4.8119068, -8.1292305, 8.1354713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5545600
time: 20.00 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5685449
time: 20.05 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 42.07 seconds
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 42.07
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5540931
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 42.07
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5680603
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 42.07
Output dim: 3, lower bound: -4.5563304, upper bound: 4.5545600
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 42.07
Output dim: 3, lower bound: -4.5563304, upper bound: 4.5685449
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 42.07
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5340313
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 42.07
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5480363
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 42.07
Output dim: 3, lower bound: -4.5507566, upper bound: 4.5540931
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 42.07
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5680603
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 42.07
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5366441
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 42.07
Output dim: 3, lower bound: -4.5685449, upper bound: 4.5506487
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 42.07
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5545600
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 42.07
Output dim: 3, lower bound: -4.5385403, upper bound: 4.5685449

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.3880692, 4.3191800, -13.3611383, 4.3026681, -15.6726532, 15.6411591
1: 0.4089255, 12.3374405, 0.4364412, 12.3262014, -9.2725449, 9.2327385
2: 2.0470915, 13.4835491, 2.0760067, 13.4673729, -9.0290108, 8.9865341
3: 1.5641232, 14.1721830, 1.6008711, 14.1435547, -8.9906082, 8.9456520
4: -4.2316289, 10.4534245, -4.1800861, 10.4177485, -12.5922928, 12.5622711
5: 2.0863805, 13.7647438, 2.1136811, 13.7404642, -8.3921356, 8.3610840
6: -25.1684570, -8.7657223, -25.1529102, -8.7892342, -13.3457146, 13.3902321
7: 2.5373182, 15.2955580, 2.5725350, 15.2774982, -9.3582764, 9.3063698
8: -4.5062485, 14.2436523, -4.4409933, 14.2120266, -15.6721115, 15.6033020
9: 0.5700088, 13.5937757, 0.6082923, 13.5678349, -9.2621269, 9.2166901
10: -4.4123030, 11.2977009, -4.3801351, 11.2685261, -12.0448914, 12.0203514
11: -4.4648714, 6.9211988, -4.4383912, 6.8899221, -8.5688324, 8.5917435
12: -26.2331238, -11.1395454, -26.2095680, -11.1829042, -10.9000282, 10.9292183
13: -14.1841412, 4.6841884, -14.1472616, 4.6512041, -13.3834686, 13.3465652
14: -24.1749992, -5.1995792, -24.1395702, -5.2224674, -16.3316193, 16.3032990
15: -7.6221943, 4.6993980, -7.5920911, 4.6718335, -11.3009872, 11.2735710
16: -7.6787758, 5.0076981, -7.6535273, 4.9874563, -9.3987732, 9.4352722
17: -26.7413063, -11.0802460, -26.7018700, -11.1280041, -10.9720230, 10.9883766
18: -17.6927547, -2.0065203, -17.6611996, -2.0690584, -10.5380096, 10.6142464
19: -10.4869690, -0.0587134, -10.4708290, -0.0736763, -6.8344955, 6.8693180
20: -5.8908443, 4.7220116, -5.8808694, 4.7147932, -7.5094929, 7.5414391
21: -8.5910234, 3.8340023, -8.5726576, 3.8202605, -9.6700211, 9.7020340
22: -10.7983179, 0.8482940, -10.7820835, 0.8275840, -7.7827072, 7.8442497
23: -4.6480227, 6.9274092, -4.6194143, 6.8958969, -8.7597504, 8.7927322
24: -8.0973349, 5.2308450, -8.0635080, 5.1903853, -10.5077972, 10.5648994
25: -8.3641472, 4.8684521, -8.3350067, 4.8409100, -8.2666550, 8.3265228
26: -16.5886650, 0.1765995, -16.5711670, 0.1468420, -11.6113853, 11.6798668
27: -7.8421431, 6.2180481, -7.8175693, 6.1968904, -12.1762009, 12.2076416
28: -6.5405617, 6.3421264, -6.5144672, 6.3145213, -10.1149597, 10.1525955
29: -7.7694902, 2.8922870, -7.7423635, 2.8594177, -8.6232681, 8.6642342
30: -3.8608952, 10.3343573, -3.8287790, 10.3041687, -12.2603760, 12.2919617
31: -14.8610706, -0.1506660, -14.8382473, -0.1804142, -10.7492065, 10.8032990
32: -20.8181896, -5.8330202, -20.8100319, -5.8508711, -12.0240784, 12.0590286
33: -38.5895653, -20.0022831, -38.5773430, -20.0163612, -10.6025276, 10.6188755
34: -35.7638626, -20.2350063, -35.7437820, -20.2697315, -11.5302391, 11.5639648
35: -33.0667191, -16.7730389, -33.0425720, -16.8079090, -11.2993813, 11.3285027
36: -31.1173649, -13.4964046, -31.0971222, -13.5274839, -12.5932159, 12.6275864
37: -50.2864647, -32.2504730, -50.2571945, -32.2901459, -9.8971443, 9.9251862
38: -38.7914276, -20.2278042, -38.7742767, -20.2539577, -11.3595352, 11.3972969
39: -42.5739899, -23.5578194, -42.5508537, -23.5735321, -10.6737251, 10.6659069
40: -38.0376511, -24.5460567, -38.0296173, -24.5643902, -7.9250641, 7.9644203
41: -24.8254185, -8.8762789, -24.8101921, -8.8958187, -13.1345291, 13.1765099
42: -15.1472988, -4.8148165, -15.1315517, -4.8262234, -8.1090393, 8.1298676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 971

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5245462, upper bound: 4.5680603
time: 31.12 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5245462, upper bound: 4.5680603
time: 41.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.3889656, 4.3226199, -13.3704414, 4.3122935, -15.6795120, 15.6512375
1: 0.4084570, 12.3393860, 0.4303634, 12.3316774, -9.2767563, 9.2391548
2: 2.0468774, 13.4867706, 2.0707278, 13.4767485, -9.0350914, 8.9944916
3: 1.5640402, 14.1809864, 1.5907533, 14.1683245, -9.0071754, 8.9650002
4: -4.2319283, 10.4634676, -4.1950655, 10.4460516, -12.6074524, 12.5874100
5: 2.0861781, 13.7734280, 2.1033106, 13.7647934, -8.4084320, 8.3806267
6: -25.1701183, -8.7651434, -25.1587753, -8.7864199, -13.3594780, 13.4001274
7: 2.5370610, 15.2999630, 2.5646806, 15.2900887, -9.3673058, 9.3188057
8: -4.5068359, 14.2473135, -4.4471788, 14.2225981, -15.6838226, 15.6144409
9: 0.5696907, 13.6026745, 0.5929110, 13.5928516, -9.2726173, 9.2407341
10: -4.4126849, 11.3094130, -4.4017644, 11.3013067, -12.0616074, 12.0530205
11: -4.4695358, 6.9213147, -4.4520202, 6.8962750, -8.5794449, 8.5972633
12: -26.2404213, -11.1392593, -26.2299805, -11.1733608, -10.9174500, 10.9446526
13: -14.1881504, 4.6846409, -14.1584358, 4.6620417, -13.3847809, 13.3622475
14: -24.1818123, -5.1995668, -24.1618519, -5.2195749, -16.3321533, 16.3236465
15: -7.6226287, 4.7079105, -7.6040468, 4.6968160, -11.3165970, 11.2944183
16: -7.6789341, 5.0130897, -7.6661091, 5.0022802, -9.4048157, 9.4549866
17: -26.7555084, -11.0802088, -26.7419395, -11.1069841, -11.0054817, 10.9972076
18: -17.6929398, -2.0013089, -17.6698036, -2.0539126, -10.5447235, 10.6232338
19: -10.4917068, -0.0585814, -10.4846420, -0.0666709, -6.8456593, 6.8745975
20: -5.8933978, 4.7221999, -5.8881869, 4.7174830, -7.5151901, 7.5484200
21: -8.5934362, 3.8343234, -8.5801783, 3.8218000, -9.6755486, 9.7088242
22: -10.8020630, 0.8484170, -10.7932158, 0.8338771, -7.7915230, 7.8527336
23: -4.6552162, 6.9275584, -4.6404800, 6.9060450, -8.7770882, 8.8062592
24: -8.1048574, 5.2308979, -8.0856609, 5.2004371, -10.5255013, 10.5791435
25: -8.3753128, 4.8685479, -8.3663855, 4.8550792, -8.2913971, 8.3395004
26: -16.5908070, 0.1768855, -16.5779419, 0.1517366, -11.6200600, 11.6873169
27: -7.8454599, 6.2182679, -7.8284855, 6.2004337, -12.1900024, 12.2190285
28: -6.5489678, 6.3423929, -6.5386105, 6.3267031, -10.1355209, 10.1660995
29: -7.7762794, 2.8923390, -7.7623935, 2.8713276, -8.6418381, 8.6728401
30: -3.8682640, 10.3345013, -3.8499541, 10.3112488, -12.2732162, 12.3062286
31: -14.8641758, -0.1501265, -14.8480339, -0.1767886, -10.7588196, 10.8092232
32: -20.8187408, -5.8323727, -20.8120003, -5.8483143, -12.0440903, 12.0608749
33: -38.5937271, -20.0019779, -38.5888519, -20.0094471, -10.6124649, 10.6246643
34: -35.7643166, -20.2344990, -35.7459641, -20.2677612, -11.5443611, 11.5658112
35: -33.0714035, -16.7727909, -33.0560989, -16.7985497, -11.3123856, 11.3348732
36: -31.1238956, -13.4961405, -31.1159019, -13.5144167, -12.6131172, 12.6406708
37: -50.2950439, -32.2503204, -50.2812538, -32.2770538, -9.9189224, 9.9356422
38: -38.7967606, -20.2275734, -38.7898712, -20.2458973, -11.3744583, 11.4082489
39: -42.5793381, -23.5575867, -42.5660477, -23.5649891, -10.6881676, 10.6727734
40: -38.0377121, -24.5432243, -38.0332794, -24.5553818, -7.9308796, 7.9719162
41: -24.8269310, -8.8757734, -24.8160973, -8.8923683, -13.1517334, 13.1833000
42: -15.1524935, -4.8145618, -15.1461468, -4.8182740, -8.1224442, 8.1371040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 971

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5423198, upper bound: 4.5685449
time: 35.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5423198, upper bound: 4.5685449
time: 39.04 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.3895988, 4.3192425, -13.3794308, 4.3099942, -15.6825867, 15.6580658
1: 0.4087973, 12.3392487, 0.4142027, 12.3361149, -9.2814789, 9.2565575
2: 2.0470071, 13.4857388, 2.0519550, 13.4797449, -9.0407524, 9.0125847
3: 1.5640829, 14.1748486, 1.5739489, 14.1584835, -9.0041924, 8.9749184
4: -4.2317581, 10.4571733, -4.2141566, 10.4384785, -12.6118469, 12.6000786
5: 2.0862763, 13.7667313, 2.0945339, 13.7506294, -8.4016609, 8.3821869
6: -25.1700935, -8.7655315, -25.1649437, -8.7688942, -13.3732071, 13.4011955
7: 2.5372407, 15.2982407, 2.5473614, 15.2905331, -9.3699455, 9.3342323
8: -4.5064211, 14.2496376, -4.4947691, 14.2431412, -15.7012939, 15.6629410
9: 0.5698502, 13.5959301, 0.5860729, 13.5792150, -9.2723999, 9.2406540
10: -4.4124908, 11.2984953, -4.3888254, 11.2755604, -12.0503578, 12.0386620
11: -4.4680715, 6.9212656, -4.4580808, 6.9188061, -8.6009293, 8.6087914
12: -26.2356682, -11.1393356, -26.2222843, -11.1562986, -10.9288025, 10.9406395
13: -14.1842937, 4.6878595, -14.1809225, 4.6763773, -13.4058342, 13.3841133
14: -24.1767616, -5.1993513, -24.1579590, -5.2099390, -16.3437424, 16.3206406
15: -7.6224966, 4.7015100, -7.6127257, 4.6844716, -11.3127403, 11.3006821
16: -7.6796064, 5.0077229, -7.6654229, 4.9966774, -9.4272308, 9.4393692
17: -26.7431736, -11.0801249, -26.7149601, -11.1109762, -10.9917412, 11.0002174
18: -17.6983376, -2.0064344, -17.6909943, -2.0175467, -10.5948143, 10.6396065
19: -10.4881573, -0.0586996, -10.4792624, -0.0600700, -6.8496628, 6.8769684
20: -5.8909307, 4.7220654, -5.8846364, 4.7207808, -7.5114880, 7.5494194
21: -8.5931187, 3.8340502, -8.5871420, 3.8380456, -9.6903572, 9.7163925
22: -10.8000927, 0.8483253, -10.7930260, 0.8415766, -7.7986984, 7.8534737
23: -4.6505704, 6.9274516, -4.6354723, 6.9194803, -8.7861862, 8.8073616
24: -8.1006489, 5.2308903, -8.0853014, 5.2217879, -10.5428848, 10.5840569
25: -8.3655634, 4.8684983, -8.3451653, 4.8542867, -8.2820816, 8.3349438
26: -16.5910053, 0.1766919, -16.5861626, 0.1719332, -11.6402092, 11.6921272
27: -7.8444185, 6.2181830, -7.8354225, 6.2170682, -12.2037048, 12.2246399
28: -6.5421367, 6.3422585, -6.5254650, 6.3329849, -10.1351128, 10.1624680
29: -7.7718620, 2.8922830, -7.7580376, 2.8792832, -8.6455498, 8.6781921
30: -3.8636515, 10.3344936, -3.8486152, 10.3289394, -12.2894974, 12.3100128
31: -14.8643923, -0.1505692, -14.8573675, -0.1501818, -10.7830658, 10.8205414
32: -20.8190536, -5.8328180, -20.8174400, -5.8363037, -12.0404434, 12.0670242
33: -38.5900459, -20.0021057, -38.5829926, -20.0085773, -10.6117630, 10.6246758
34: -35.7673187, -20.2349434, -35.7662048, -20.2388000, -11.5646591, 11.5832176
35: -33.0693016, -16.7729816, -33.0600662, -16.7848110, -11.3251038, 11.3440590
36: -31.1190720, -13.4963322, -31.1074047, -13.5115013, -12.6118622, 12.6370163
37: -50.2890625, -32.2504044, -50.2727661, -32.2634506, -9.9266205, 9.9390945
38: -38.7931290, -20.2277641, -38.7841682, -20.2373371, -11.3693924, 11.4085732
39: -42.5740204, -23.5567799, -42.5612335, -23.5662632, -10.6803970, 10.6782818
40: -38.0381622, -24.5458584, -38.0339966, -24.5492630, -7.9411907, 7.9699249
41: -24.8268394, -8.8761263, -24.8212776, -8.8779821, -13.1577225, 13.1869469
42: -15.1473141, -4.8143549, -15.1355305, -4.8201032, -8.1153297, 8.1327209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 971

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5367916, upper bound: 4.5680603
time: 22.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5245462, upper bound: 4.5680603
time: 33.03 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.3784943, 4.3017612, -13.3874626, 4.3121781, -15.6692963, 15.6512451
1: 0.4172218, 12.3197145, 0.4083903, 12.3339596, -9.2687721, 9.2497635
2: 2.0533309, 13.4720211, 2.0470793, 13.4831038, -9.0339432, 9.0093918
3: 1.5729203, 14.1578970, 1.5639217, 14.1740742, -9.0026131, 8.9789963
4: -4.2228961, 10.4435482, -4.2288485, 10.4584103, -12.6094589, 12.6057129
5: 2.0939801, 13.7541742, 2.0843847, 13.7673683, -8.4025345, 8.3876228
6: -25.1697769, -8.7696276, -25.1702538, -8.7674122, -13.3830032, 13.3954430
7: 2.5457997, 15.2824135, 2.5397353, 15.2958775, -9.3629379, 9.3343353
8: -4.4937062, 14.2249641, -4.5004969, 14.2436790, -15.6897964, 15.6535568
9: 0.5788846, 13.5862856, 0.5710020, 13.5976400, -9.2671928, 9.2542458
10: -4.4070826, 11.3090401, -4.4090128, 11.3079739, -12.0592842, 12.0663795
11: -4.4527364, 6.9155788, -4.4646959, 6.9249535, -8.5967941, 8.6014824
12: -26.2306709, -11.1444607, -26.2383232, -11.1471643, -10.9367867, 10.9463005
13: -14.1859312, 4.6675467, -14.1917686, 4.6798978, -13.3958511, 13.3825684
14: -24.1656590, -5.1993437, -24.1745911, -5.2070723, -16.3254852, 16.3350105
15: -7.6161690, 4.6944084, -7.6240158, 4.7039871, -11.3158112, 11.3054504
16: -7.6732755, 5.0043154, -7.6777110, 5.0084171, -9.4296227, 9.4469490
17: -26.7367172, -11.0873127, -26.7480507, -11.0899954, -11.0167313, 10.9944153
18: -17.6751518, -2.0097680, -17.6914120, -2.0028877, -10.5873756, 10.6317406
19: -10.4720087, -0.0651736, -10.4856348, -0.0531290, -6.8505936, 6.8681908
20: -5.8769126, 4.7170935, -5.8861408, 4.7232237, -7.5075607, 7.5449867
21: -8.5773964, 3.8300493, -8.5884285, 3.8393402, -9.6836510, 9.7121506
22: -10.7863550, 0.8421831, -10.7979937, 0.8478010, -7.8014984, 7.8489189
23: -4.6342487, 6.9201860, -4.6481671, 6.9294610, -8.7869415, 8.8051071
24: -8.0800638, 5.2217178, -8.0974894, 5.2317371, -10.5436630, 10.5789070
25: -8.3487768, 4.8587832, -8.3666964, 4.8682270, -8.2922096, 8.3281078
26: -16.5701790, 0.1691064, -16.5849457, 0.1766535, -11.6375160, 11.6834183
27: -7.8327827, 6.2157745, -7.8411884, 6.2203226, -12.2034836, 12.2282715
28: -6.5279379, 6.3342834, -6.5416083, 6.3449030, -10.1414871, 10.1597137
29: -7.7631445, 2.8876870, -7.7725811, 2.8911471, -8.6563683, 8.6761055
30: -3.8458676, 10.3274746, -3.8610303, 10.3356323, -12.2825699, 12.3082619
31: -14.8380966, -0.1587718, -14.8566856, -0.1468120, -10.7752876, 10.8072510
32: -20.8182373, -5.8363075, -20.8190269, -5.8347731, -12.0577888, 12.0586243
33: -38.5894775, -20.0047035, -38.5927429, -20.0024147, -10.6165924, 10.6255798
34: -35.7574463, -20.2385292, -35.7648773, -20.2369366, -11.5679932, 11.5770607
35: -33.0673218, -16.7769527, -33.0714149, -16.7758751, -11.3310509, 11.3426514
36: -31.1219273, -13.4997311, -31.1247654, -13.4990635, -12.6262398, 12.6441307
37: -50.2860146, -32.2560577, -50.2926445, -32.2509842, -9.9397774, 9.9411182
38: -38.7932510, -20.2302933, -38.7977333, -20.2298660, -11.3748665, 11.4148216
39: -42.5780716, -23.5578766, -42.5758133, -23.5581932, -10.6923294, 10.6835766
40: -38.0366058, -24.5461597, -38.0373459, -24.5412884, -7.9441624, 7.9685078
41: -24.8270187, -8.8816319, -24.8268967, -8.8762236, -13.1728096, 13.1824646
42: -15.1507692, -4.8171616, -15.1495619, -4.8130450, -8.1262207, 8.1307278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 971

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5245462, upper bound: 4.5506488
time: 52.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5245462, upper bound: 4.5506488
time: 24.06 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.3905306, 4.3226905, -13.3887711, 4.3195958, -15.6894684, 15.6681862
1: 0.4083211, 12.3411789, 0.4081419, 12.3415737, -9.2856750, 9.2629776
2: 2.0468240, 13.4889832, 2.0466356, 13.4890909, -9.0468483, 9.0205498
3: 1.5639670, 14.1836605, 1.5637918, 14.1832933, -9.0207672, 8.9942894
4: -4.2320585, 10.4672089, -4.2291236, 10.4667463, -12.6270142, 12.6252174
5: 2.0860963, 13.7754211, 2.0841615, 13.7749414, -8.4179344, 8.4017410
6: -25.1717319, -8.7649746, -25.1707687, -8.7660761, -13.3869553, 13.4111023
7: 2.5369530, 15.3026371, 2.5395246, 15.3031178, -9.3789597, 9.3466530
8: -4.5069933, 14.2532682, -4.5009670, 14.2537556, -15.7130127, 15.6741028
9: 0.5695546, 13.6047993, 0.5706973, 13.6042500, -9.2829056, 9.2647057
10: -4.4128819, 11.3101988, -4.4104624, 11.3083506, -12.0670815, 12.0713196
11: -4.4727201, 6.9214211, -4.4716949, 6.9251547, -8.6115417, 8.6142921
12: -26.2429485, -11.1391010, -26.2426662, -11.1467581, -10.9462128, 10.9560471
13: -14.1882954, 4.6883287, -14.1921530, 4.6872005, -13.4071083, 13.3997536
14: -24.1835403, -5.1993141, -24.1802158, -5.2070179, -16.3442764, 16.3409691
15: -7.6229300, 4.7100153, -7.6246910, 4.7094488, -11.3283768, 11.3215332
16: -7.6797962, 5.0130897, -7.6780229, 5.0115261, -9.4332809, 9.4590912
17: -26.7573566, -11.0800810, -26.7550449, -11.0899725, -11.0252304, 11.0090599
18: -17.6985073, -2.0011835, -17.6995831, -2.0024204, -10.6015320, 10.6485977
19: -10.4928951, -0.0585299, -10.4930754, -0.0530527, -6.8608303, 6.8822460
20: -5.8935151, 4.7222462, -5.8919568, 4.7234569, -7.5171967, 7.5563869
21: -8.5955467, 3.8344181, -8.5946903, 3.8396266, -9.6958885, 9.7231903
22: -10.8038521, 0.8484557, -10.8041553, 0.8478951, -7.8075180, 7.8619270
23: -4.6577559, 6.9276466, -4.6565018, 6.9296293, -8.8035240, 8.8208694
24: -8.1081572, 5.2309361, -8.1074514, 5.2318335, -10.5605659, 10.5982895
25: -8.3767300, 4.8685951, -8.3765640, 4.8684535, -8.3068123, 8.3479233
26: -16.5931396, 0.1769845, -16.5930042, 0.1768427, -11.6488953, 11.6995735
27: -7.8477354, 6.2184353, -7.8463383, 6.2206230, -12.2174988, 12.2360840
28: -6.5504961, 6.3425169, -6.5495758, 6.3451571, -10.1556702, 10.1759720
29: -7.7786522, 2.8923724, -7.7780795, 2.8911886, -8.6641579, 8.6868248
30: -3.8710289, 10.3346672, -3.8697760, 10.3359900, -12.3023071, 12.3242874
31: -14.8674822, -0.1499984, -14.8671503, -0.1465712, -10.7926598, 10.8264656
32: -20.8195839, -5.8321848, -20.8194160, -5.8337364, -12.0604553, 12.0688629
33: -38.5942535, -20.0018311, -38.5944748, -20.0015621, -10.6216965, 10.6304684
34: -35.7677460, -20.2344513, -35.7683792, -20.2368126, -11.5787582, 11.5850754
35: -33.0739441, -16.7727146, -33.0736465, -16.7754154, -11.3380928, 11.3503952
36: -31.1256161, -13.4960461, -31.1262207, -13.4984341, -12.6317444, 12.6501083
37: -50.2976913, -32.2502632, -50.2968140, -32.2503586, -9.9484138, 9.9495277
38: -38.7985268, -20.2275257, -38.7997169, -20.2292595, -11.3843155, 11.4195290
39: -42.5793533, -23.5565586, -42.5764313, -23.5577106, -10.6948433, 10.6851444
40: -38.0382309, -24.5430355, -38.0376778, -24.5402718, -7.9469948, 7.9774246
41: -24.8283424, -8.8756580, -24.8271751, -8.8745098, -13.1749496, 13.1937218
42: -15.1525087, -4.8141007, -15.1501303, -4.8121314, -8.1287460, 8.1399651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 971

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5245462, upper bound: 4.5685449
time: 29.28 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5545600, upper bound: 4.5685449
time: 41.23 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 72.55 seconds
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.55
Output dim: 3, lower bound: -4.5245462, upper bound: 4.5680603
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.55
Output dim: 3, lower bound: -4.5245462, upper bound: 4.5680603
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.55
Output dim: 3, lower bound: -4.5423198, upper bound: 4.5685449
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.55
Output dim: 3, lower bound: -4.5423198, upper bound: 4.5685449
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.55
Output dim: 3, lower bound: -4.5367916, upper bound: 4.5680603
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.55
Output dim: 3, lower bound: -4.5245462, upper bound: 4.5680603
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 72.55
Output dim: 3, lower bound: -4.5245462, upper bound: 4.5506488
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 72.55
Output dim: 3, lower bound: -4.5245462, upper bound: 4.5506488
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 72.55
Output dim: 3, lower bound: -4.5245462, upper bound: 4.5685449
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 72.55
Output dim: 3, lower bound: -4.5545600, upper bound: 4.5685449

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.3400097, 4.3032012, -13.3611383, 4.3026681, -15.6241074, 15.6401520
1: 0.4338624, 12.3290243, 0.4364412, 12.3262014, -9.2468071, 9.2284698
2: 2.0749655, 13.4736128, 2.0760067, 13.4673729, -9.0010452, 8.9849434
3: 1.5907402, 14.1629181, 1.6008711, 14.1435547, -8.9642448, 8.9465942
4: -4.2046199, 10.4442873, -4.1800861, 10.4177485, -12.5651016, 12.5564880
5: 2.1131530, 13.7554655, 2.1136811, 13.7404642, -8.3650894, 8.3642311
6: -25.1638660, -8.7833157, -25.1529102, -8.7892342, -13.3374214, 13.3629951
7: 2.5624914, 15.2875404, 2.5725350, 15.2774982, -9.3310890, 9.3077888
8: -4.4690695, 14.2330980, -4.4409933, 14.2120266, -15.6344604, 15.5985641
9: 0.5915008, 13.5894308, 0.6082923, 13.5678349, -9.2384262, 9.2172089
10: -4.3897076, 11.2935181, -4.3801351, 11.2685261, -12.0205460, 12.0212479
11: -4.4579558, 6.9068351, -4.4383912, 6.8899221, -8.5635834, 8.5774803
12: -26.2286644, -11.1525965, -26.2095680, -11.1829042, -10.8969765, 10.9156303
13: -14.1655846, 4.6724548, -14.1472616, 4.6512041, -13.3643188, 13.3397293
14: -24.1472435, -5.2074003, -24.1395702, -5.2224674, -16.3016663, 16.2957268
15: -7.6042514, 4.6923079, -7.5920911, 4.6718335, -11.2775345, 11.2652588
16: -7.6570902, 5.0023170, -7.6535273, 4.9874563, -9.3866768, 9.4371243
17: -26.7188740, -11.0883560, -26.7018700, -11.1280041, -10.9508171, 10.9870033
18: -17.6868286, -2.0215092, -17.6611996, -2.0690584, -10.5326004, 10.5979843
19: -10.4771414, -0.0796072, -10.4708290, -0.0736763, -6.8327103, 6.8470802
20: -5.8809462, 4.6946378, -5.8808694, 4.7147932, -7.5132065, 7.5128231
21: -8.5788050, 3.8054838, -8.5726576, 3.8202605, -9.6688881, 9.6724014
22: -10.7895737, 0.8237939, -10.7820835, 0.8275840, -7.7909775, 7.8172894
23: -4.6382942, 6.9062119, -4.6194143, 6.8958969, -8.7586441, 8.7713623
24: -8.0890608, 5.2102919, -8.0635080, 5.1903853, -10.5104485, 10.5442772
25: -8.3553858, 4.8436441, -8.3350067, 4.8409100, -8.2724953, 8.3007278
26: -16.5761528, 0.1465055, -16.5711670, 0.1468420, -11.6186943, 11.6485977
27: -7.8307915, 6.1928878, -7.8175693, 6.1968904, -12.1722641, 12.1817780
28: -6.5285015, 6.3118839, -6.5144672, 6.3145213, -10.1162643, 10.1213226
29: -7.7624807, 2.8732331, -7.7423635, 2.8594177, -8.6259956, 8.6446609
30: -3.8498514, 10.3039351, -3.8287790, 10.3041687, -12.2621231, 12.2617874
31: -14.8522625, -0.1696930, -14.8382473, -0.1804142, -10.7411652, 10.7828026
32: -20.8134918, -5.8452959, -20.8100319, -5.8508711, -12.0180054, 12.0424805
33: -38.5857506, -20.0208740, -38.5773430, -20.0163612, -10.6049614, 10.5986900
34: -35.7563477, -20.2554035, -35.7437820, -20.2697315, -11.5326195, 11.5432205
35: -33.0619011, -16.7926121, -33.0425720, -16.8079090, -11.3043060, 11.3065453
36: -31.1112633, -13.5211678, -31.0971222, -13.5274839, -12.5971336, 12.6010437
37: -50.2841797, -32.2625847, -50.2571945, -32.2901459, -9.8977203, 9.9115219
38: -38.7841911, -20.2487793, -38.7742767, -20.2539577, -11.3672218, 11.3731480
39: -42.5724258, -23.5652809, -42.5508537, -23.5735321, -10.6737061, 10.6590843
40: -38.0310440, -24.5506439, -38.0296173, -24.5643902, -7.9200497, 7.9463634
41: -24.8198071, -8.8893738, -24.8101921, -8.8958187, -13.1285400, 13.1607208
42: -15.1456966, -4.8180556, -15.1315517, -4.8262234, -8.1052055, 8.1195641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=66, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1663

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5066237, upper bound: 4.5501626
time: 57.46 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5066237, upper bound: 4.5680604
time: 24.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3873272, 4.3191328, -13.3611383, 4.3026681, -15.6566010, 15.6411133
1: 0.4096611, 12.3374281, 0.4364412, 12.3262014, -9.2669563, 9.2327080
2: 2.0479109, 13.4835072, 2.0760067, 13.4673729, -9.0199776, 8.9864960
3: 1.5648832, 14.1721859, 1.6008711, 14.1435547, -8.9793892, 8.9456215
4: -4.2307959, 10.4533443, -4.1800861, 10.4177485, -12.5884552, 12.5622253
5: 2.0869026, 13.7646856, 2.1136811, 13.7404642, -8.3792839, 8.3610382
6: -25.1684227, -8.7668028, -25.1529102, -8.7892342, -13.3513603, 13.3894844
7: 2.5378754, 15.2955132, 2.5725350, 15.2774982, -9.3477440, 9.3063393
8: -4.5048289, 14.2436380, -4.4409933, 14.2120266, -15.6641541, 15.6032639
9: 0.5713000, 13.5937557, 0.6082923, 13.5678349, -9.2533455, 9.2166519
10: -4.4103012, 11.2976274, -4.3801351, 11.2685261, -12.0356598, 12.0203209
11: -4.4648066, 6.9206161, -4.4383912, 6.8899221, -8.5687599, 8.5898705
12: -26.2330837, -11.1403599, -26.2095680, -11.1829042, -10.8999977, 10.9262352
13: -14.1837368, 4.6841197, -14.1472616, 4.6512041, -13.3783264, 13.3464928
14: -24.1738682, -5.1996098, -24.1395702, -5.2224674, -16.3270416, 16.3032532
15: -7.6213045, 4.6993494, -7.5920911, 4.6718335, -11.3003616, 11.2767372
16: -7.6774206, 5.0076947, -7.6535273, 4.9874563, -9.3947601, 9.4352150
17: -26.7405262, -11.0802383, -26.7018700, -11.1280041, -10.9620132, 10.9883270
18: -17.6926899, -2.0074244, -17.6611996, -2.0690584, -10.5379753, 10.6117363
19: -10.4869003, -0.0594358, -10.4708290, -0.0736763, -6.8344536, 6.8600502
20: -5.8907928, 4.7211943, -5.8808694, 4.7147932, -7.5094471, 7.5256119
21: -8.5909615, 3.8331549, -8.5726576, 3.8202605, -9.6699486, 9.6890602
22: -10.7982826, 0.8474367, -10.7820835, 0.8275840, -7.7826614, 7.8240681
23: -4.6479015, 6.9265308, -4.6194143, 6.8958969, -8.7596474, 8.7831192
24: -8.0972300, 5.2298946, -8.0635080, 5.1903853, -10.5076675, 10.5512314
25: -8.3641052, 4.8675084, -8.3350067, 4.8409100, -8.2666168, 8.3099861
26: -16.5885468, 0.1756943, -16.5711670, 0.1468420, -11.6113052, 11.6577301
27: -7.8420196, 6.2170086, -7.8175693, 6.1968904, -12.1761322, 12.1986732
28: -6.5404582, 6.3413439, -6.5144672, 6.3145213, -10.1148643, 10.1378403
29: -7.7694139, 2.8913887, -7.7423635, 2.8594177, -8.6231804, 8.6533508
30: -3.8607914, 10.3333015, -3.8287790, 10.3041687, -12.2603226, 12.2788887
31: -14.8610210, -0.1516662, -14.8382473, -0.1804142, -10.7491493, 10.7996216
32: -20.8181534, -5.8337555, -20.8100319, -5.8508711, -12.0260506, 12.0585518
33: -38.5894890, -20.0035553, -38.5773430, -20.0163612, -10.6024895, 10.6096497
34: -35.7637482, -20.2358303, -35.7437820, -20.2697315, -11.5301590, 11.5539169
35: -33.0666275, -16.7742672, -33.0425720, -16.8079090, -11.2993279, 11.3164711
36: -31.1173038, -13.4974995, -31.0971222, -13.5274839, -12.5932083, 12.6152458
37: -50.2863884, -32.2516365, -50.2571945, -32.2901459, -9.8971176, 9.9201279
38: -38.7913742, -20.2287655, -38.7742767, -20.2539577, -11.3594627, 11.3766975
39: -42.5739365, -23.5587521, -42.5508537, -23.5735321, -10.6736870, 10.6625385
40: -38.0373001, -24.5461731, -38.0296173, -24.5643902, -7.9376068, 7.9640007
41: -24.8253250, -8.8769617, -24.8101921, -8.8958187, -13.1357803, 13.1760292
42: -15.1472273, -4.8150167, -15.1315517, -4.8262234, -8.1137657, 8.1296082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1663

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5066237, upper bound: 4.5541047
time: 27.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5066237, upper bound: 4.5541050
time: 26.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.3409348, 4.3066411, -13.3704414, 4.3122935, -15.6309891, 15.6502380
1: 0.4333875, 12.3309555, 0.4303634, 12.3316774, -9.2510185, 9.2348900
2: 2.0747542, 13.4768448, 2.0707278, 13.4767485, -9.0071106, 8.9929085
3: 1.5905979, 14.1716909, 1.5907533, 14.1683245, -8.9808235, 8.9659462
4: -4.2049050, 10.4543171, -4.1950655, 10.4460516, -12.5802689, 12.5816383
5: 2.1129670, 13.7641430, 2.1033106, 13.7647934, -8.3813820, 8.3837852
6: -25.1655235, -8.7827072, -25.1587753, -8.7864199, -13.3511696, 13.3728676
7: 2.5622365, 15.2919693, 2.5646806, 15.2900887, -9.3401031, 9.3202362
8: -4.4696627, 14.2366838, -4.4471788, 14.2225981, -15.6461716, 15.6097260
9: 0.5912232, 13.5982990, 0.5929110, 13.5928516, -9.2489243, 9.2412605
10: -4.3900647, 11.3052320, -4.4017644, 11.3013067, -12.0372772, 12.0539055
11: -4.4626398, 6.9069624, -4.4520202, 6.8962750, -8.5741997, 8.5829697
12: -26.2359943, -11.1523399, -26.2299805, -11.1733608, -10.9143829, 10.9310493
13: -14.1695595, 4.6729479, -14.1584358, 4.6620417, -13.3656235, 13.3553963
14: -24.1540375, -5.2073946, -24.1618519, -5.2195749, -16.3022079, 16.3160629
15: -7.6046844, 4.7008057, -7.6040468, 4.6968160, -11.2931671, 11.2861176
16: -7.6572342, 5.0076699, -7.6661091, 5.0022802, -9.3927155, 9.4568481
17: -26.7330856, -11.0883265, -26.7419395, -11.1069841, -10.9842911, 10.9958458
18: -17.6870384, -2.0163074, -17.6698036, -2.0539126, -10.5393105, 10.6069794
19: -10.4818974, -0.0794580, -10.4846420, -0.0666709, -6.8438721, 6.8523617
20: -5.8835154, 4.6948118, -5.8881869, 4.7174830, -7.5189037, 7.5198021
21: -8.5812273, 3.8058081, -8.5801783, 3.8218000, -9.6744232, 9.6791916
22: -10.7933044, 0.8239245, -10.7932158, 0.8338771, -7.7997894, 7.8257675
23: -4.6455073, 6.9063635, -4.6404800, 6.9060450, -8.7759666, 8.7848854
24: -8.0965786, 5.2103539, -8.0856609, 5.2004371, -10.5281448, 10.5585136
25: -8.3665619, 4.8437262, -8.3663855, 4.8550792, -8.2972260, 8.3137150
26: -16.5783081, 0.1467997, -16.5779419, 0.1517366, -11.6273499, 11.6560326
27: -7.8341131, 6.1931210, -7.8284855, 6.2004337, -12.1860657, 12.1931992
28: -6.5368657, 6.3121142, -6.5386105, 6.3267031, -10.1368217, 10.1348267
29: -7.7692661, 2.8733244, -7.7623935, 2.8713276, -8.6445808, 8.6532631
30: -3.8571796, 10.3041286, -3.8499541, 10.3112488, -12.2749557, 12.2760353
31: -14.8553581, -0.1691489, -14.8480339, -0.1767886, -10.7507782, 10.7887421
32: -20.8140373, -5.8446331, -20.8120003, -5.8483143, -12.0380173, 12.0443268
33: -38.5900040, -20.0205650, -38.5888519, -20.0094471, -10.6148987, 10.6044846
34: -35.7568130, -20.2548923, -35.7459641, -20.2677612, -11.5467682, 11.5450935
35: -33.0665855, -16.7923679, -33.0560989, -16.7985497, -11.3172951, 11.3129082
36: -31.1178112, -13.5208311, -31.1159019, -13.5144167, -12.6170387, 12.6141357
37: -50.2928314, -32.2624092, -50.2812538, -32.2770538, -9.9195099, 9.9219685
38: -38.7895317, -20.2485542, -38.7898712, -20.2458973, -11.3821259, 11.3840961
39: -42.5778046, -23.5650291, -42.5660477, -23.5649891, -10.6881371, 10.6659451
40: -38.0311165, -24.5478134, -38.0332794, -24.5553818, -7.9258595, 7.9538689
41: -24.8213120, -8.8888636, -24.8160973, -8.8923683, -13.1457443, 13.1675110
42: -15.1508846, -4.8178072, -15.1461468, -4.8182740, -8.1186028, 8.1268139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=66, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1663

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5243994, upper bound: 4.5685448
time: 30.00 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5066237, upper bound: 4.5680604
time: 30.17 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3882351, 4.3225822, -13.3704414, 4.3122935, -15.6634979, 15.6511841
1: 0.4092035, 12.3393669, 0.4303634, 12.3316774, -9.2711601, 9.2391205
2: 2.0476985, 13.4867449, 2.0707278, 13.4767485, -9.0260582, 8.9944763
3: 1.5647848, 14.1809521, 1.5907533, 14.1683245, -8.9959641, 8.9649773
4: -4.2310715, 10.4633875, -4.1950655, 10.4460516, -12.6036148, 12.5873337
5: 2.0867090, 13.7733850, 2.1033106, 13.7647934, -8.3955574, 8.3806038
6: -25.1700459, -8.7662106, -25.1587753, -8.7864199, -13.3651123, 13.3993912
7: 2.5376124, 15.2999334, 2.5646806, 15.2900887, -9.3567734, 9.3187752
8: -4.5054302, 14.2472687, -4.4471788, 14.2225981, -15.6758499, 15.6144104
9: 0.5710056, 13.6026545, 0.5929110, 13.5928516, -9.2638397, 9.2407150
10: -4.4106970, 11.3093395, -4.4017644, 11.3013067, -12.0523872, 12.0529938
11: -4.4694643, 6.9207697, -4.4520202, 6.8962750, -8.5793686, 8.5953827
12: -26.2403717, -11.1400661, -26.2299805, -11.1733608, -10.9174042, 10.9416656
13: -14.1876698, 4.6845636, -14.1584358, 4.6620417, -13.3796158, 13.3621788
14: -24.1806602, -5.1995649, -24.1618519, -5.2195749, -16.3276062, 16.3236160
15: -7.6217279, 4.7078466, -7.6040468, 4.6968160, -11.3159866, 11.2975845
16: -7.6775646, 5.0130391, -7.6661091, 5.0022802, -9.4008026, 9.4549294
17: -26.7547150, -11.0802088, -26.7419395, -11.1069841, -10.9954987, 10.9971619
18: -17.6928921, -2.0022202, -17.6698036, -2.0539126, -10.5446854, 10.6207275
19: -10.4916534, -0.0592589, -10.4846420, -0.0666709, -6.8456211, 6.8653221
20: -5.8933692, 4.7213440, -5.8881869, 4.7174830, -7.5151520, 7.5325851
21: -8.5933809, 3.8334680, -8.5801783, 3.8218000, -9.6754761, 9.6958504
22: -10.8020420, 0.8475671, -10.7932158, 0.8338771, -7.7915001, 7.8325367
23: -4.6550875, 6.9266863, -4.6404800, 6.9060450, -8.7769775, 8.7966461
24: -8.1047287, 5.2299323, -8.0856609, 5.2004371, -10.5253716, 10.5654755
25: -8.3752785, 4.8675971, -8.3663855, 4.8550792, -8.2913513, 8.3229637
26: -16.5907211, 0.1760015, -16.5779419, 0.1517366, -11.6199760, 11.6651840
27: -7.8453445, 6.2172360, -7.8284855, 6.2004337, -12.1899338, 12.2100983
28: -6.5488510, 6.3416014, -6.5386105, 6.3267031, -10.1354294, 10.1513290
29: -7.7762103, 2.8914406, -7.7623935, 2.8713276, -8.6417618, 8.6619568
30: -3.8681700, 10.3334532, -3.8499541, 10.3112488, -12.2731552, 12.2931404
31: -14.8641090, -0.1510975, -14.8480339, -0.1767886, -10.7587624, 10.8055344
32: -20.8186989, -5.8331003, -20.8120003, -5.8483143, -12.0460663, 12.0604095
33: -38.5936508, -20.0032959, -38.5888519, -20.0094471, -10.6124268, 10.6154480
34: -35.7642097, -20.2353325, -35.7459641, -20.2677612, -11.5443115, 11.5557671
35: -33.0713005, -16.7740192, -33.0560989, -16.7985497, -11.3123245, 11.3228416
36: -31.1238346, -13.4972296, -31.1159019, -13.5144167, -12.6131096, 12.6283646
37: -50.2949905, -32.2515221, -50.2812538, -32.2770538, -9.9188919, 9.9305840
38: -38.7967148, -20.2285500, -38.7898712, -20.2458973, -11.3743935, 11.3876286
39: -42.5792732, -23.5585136, -42.5660477, -23.5649891, -10.6881294, 10.6693974
40: -38.0373497, -24.5433350, -38.0332794, -24.5553818, -7.9434090, 7.9715042
41: -24.8268738, -8.8764515, -24.8160973, -8.8923683, -13.1529999, 13.1828346
42: -15.1524153, -4.8147869, -15.1461468, -4.8182740, -8.1271591, 8.1368599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1663

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5243994, upper bound: 4.5545756
time: 23.53 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5066237, upper bound: 4.5545758
time: 25.93 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 51.50 seconds
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 51.50
Output dim: 3, lower bound: -4.5066237, upper bound: 4.5501626
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 51.50
Output dim: 3, lower bound: -4.5066237, upper bound: 4.5680604
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 51.50
Output dim: 3, lower bound: -4.5066237, upper bound: 4.5541047
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 51.50
Output dim: 3, lower bound: -4.5066237, upper bound: 4.5541050
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 51.50
Output dim: 3, lower bound: -4.5243994, upper bound: 4.5685448
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 51.50
Output dim: 3, lower bound: -4.5066237, upper bound: 4.5680604
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 51.50
Output dim: 3, lower bound: -4.5243994, upper bound: 4.5545756
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 51.50
Output dim: 3, lower bound: -4.5066237, upper bound: 4.5545758
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 51.50
Output dim: 3, lower bound: -4.5367916, upper bound: 4.5680603
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 51.50
Output dim: 3, lower bound: -4.5245462, upper bound: 4.5680603
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 51.50
Output dim: 3, lower bound: -4.5245462, upper bound: 4.5685449
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 51.50
Output dim: 3, lower bound: -4.5545600, upper bound: 4.5685449

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 36.87 + 1787.51 = 1824.38 seconds
