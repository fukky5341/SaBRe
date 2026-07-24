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
execution time: IAR + RelationalAnalysis = 2.30 + 35.12 = 37.42 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -4.5715991, upper bound: 4.5715991

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 717

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5712715, upper bound: 4.5507615
time: 33.16 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5712715, upper bound: 4.5712714
time: 25.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 58.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 58.69
Output dim: 3, lower bound: -4.5712715, upper bound: 4.5507615
IS_A2, status: Status.UNKNOWN, split count: 1, time: 58.69
Output dim: 3, lower bound: -4.5712715, upper bound: 4.5712714

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.3650694, 4.2953181, -13.3793478, 4.3078289, -15.6571198, 15.6591797
1: 0.4413812, 12.3225784, 0.4289610, 12.3355589, -9.2492065, 9.2488899
2: 2.0841241, 13.4660711, 2.0696812, 13.4819040, -9.0061722, 9.0049362
3: 1.6036375, 14.1640472, 1.5896964, 14.1775742, -8.9849548, 8.9854202
4: -4.1817684, 10.4441557, -4.1964192, 10.4604273, -12.5846939, 12.5833168
5: 2.1129842, 13.7600403, 2.1023581, 13.7714901, -8.3959808, 8.3953056
6: -25.1625938, -8.7900648, -25.1682777, -8.7847157, -13.3589478, 13.3533249
7: 2.5791521, 15.2807446, 2.5637560, 15.2971725, -9.3358765, 9.3348122
8: -4.4246407, 14.2176933, -4.4493227, 14.2436657, -15.6224213, 15.6211166
9: 0.6077113, 13.5815325, 0.5909445, 13.5976448, -9.2529678, 9.2532921
10: -4.3925567, 11.2883282, -4.4043922, 11.3013754, -12.0533524, 12.0527802
11: -4.4521856, 6.8867016, -4.4650393, 6.8971920, -8.5796204, 8.5812149
12: -26.2431984, -11.1767960, -26.2448139, -11.1720123, -10.9297066, 10.9262657
13: -14.1440048, 4.6554976, -14.1598988, 4.6741829, -13.3528366, 13.3502274
14: -24.1606331, -5.2327471, -24.1725559, -5.2227430, -16.3057938, 16.3100739
15: -7.5991254, 4.6969881, -7.6060138, 4.7059045, -11.2891541, 11.2928810
16: -7.6651835, 4.9928370, -7.6731515, 5.0000186, -9.4293594, 9.4261513
17: -26.7505550, -11.1083584, -26.7545242, -11.1066952, -11.0150871, 11.0170898
18: -17.6705132, -2.0708199, -17.6911430, -2.0519171, -10.5733261, 10.5737228
19: -10.4674444, -0.0808370, -10.4844704, -0.0657670, -6.8586502, 6.8603687
20: -5.8742013, 4.7088060, -5.8851004, 4.7189527, -7.5337353, 7.5343742
21: -8.5648689, 3.8074358, -8.5840511, 3.8229022, -9.6848145, 9.6881676
22: -10.7743607, 0.8181353, -10.7937908, 0.8349051, -7.8216858, 7.8237743
23: -4.6329064, 6.8936443, -4.6482205, 6.9072204, -8.7851295, 8.7868309
24: -8.0754566, 5.1829600, -8.0949593, 5.2015676, -10.5440369, 10.5458298
25: -8.3408909, 4.8347311, -8.3638525, 4.8562374, -8.3184013, 8.3193932
26: -16.5708656, 0.1416142, -16.5849476, 0.1531909, -11.6564293, 11.6583405
27: -7.8127151, 6.1858654, -7.8328142, 6.2020350, -12.1831512, 12.1868935
28: -6.5217443, 6.3119531, -6.5399222, 6.3280706, -10.1451225, 10.1469994
29: -7.7503557, 2.8581963, -7.7680264, 2.8723969, -8.6516190, 8.6548729
30: -3.8461244, 10.3009796, -3.8602021, 10.3128242, -12.2830734, 12.2850761
31: -14.8273544, -0.1998706, -14.8546181, -0.1752036, -10.7683334, 10.7709465
32: -20.8166943, -5.8505440, -20.8182755, -5.8462172, -12.0346146, 12.0275269
33: -38.5875854, -20.0118675, -38.5912399, -20.0077400, -10.6199760, 10.6162090
34: -35.7491379, -20.2731419, -35.7611542, -20.2661228, -11.5366669, 11.5386848
35: -33.0518875, -16.8071041, -33.0650978, -16.7969246, -11.3135910, 11.3148193
36: -31.1054764, -13.5245895, -31.1196156, -13.5130997, -12.6173935, 12.6174965
37: -50.2965736, -32.2783127, -50.2982788, -32.2759132, -9.9339676, 9.9293938
38: -38.7710953, -20.2586899, -38.7909927, -20.2446404, -11.3920326, 11.4052258
39: -42.5657806, -23.5594997, -42.5668640, -23.5582466, -10.6843643, 10.6850929
40: -38.0381126, -24.5655861, -38.0391464, -24.5568790, -7.9485855, 7.9348545
41: -24.8229294, -8.8947668, -24.8249435, -8.8912029, -13.1511536, 13.1403427
42: -15.1494522, -4.8198128, -15.1471767, -4.8158021, -8.1298599, 8.1217766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=68, inp2_unstable=69, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1682

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5529369, upper bound: 4.5497289
time: 34.63 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5707281, upper bound: 4.5502182
time: 49.10 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -13.3807030, 4.3140168, -13.3808918, 4.3144445, -15.6794510, 15.6732864
1: 0.4286797, 12.3420496, 0.4286237, 12.3424530, -9.2689056, 9.2592850
2: 2.0693645, 13.4895992, 2.0693321, 13.4900951, -9.0292358, 9.0155563
3: 1.5895557, 14.1844959, 1.5895138, 14.1849489, -9.0064507, 8.9928627
4: -4.1967621, 10.4684782, -4.1968045, 10.4689999, -12.6084366, 12.6034889
5: 2.1021028, 13.7772627, 2.1020751, 13.7776327, -8.4129066, 8.4060822
6: -25.1703892, -8.7840357, -25.1709251, -8.7839708, -13.3669357, 13.3748283
7: 2.5634692, 15.3054743, 2.5634384, 15.3060150, -9.3602486, 9.3461761
8: -4.4498386, 14.2567949, -4.4498806, 14.2576218, -15.6613617, 15.6443481
9: 0.5905745, 13.6059151, 0.5905347, 13.6064634, -9.2785454, 9.2629433
10: -4.4051709, 11.3075790, -4.4052210, 11.3079996, -12.0732651, 12.0687752
11: -4.4710245, 6.8974233, -4.4714327, 6.8974547, -8.5912476, 8.5989685
12: -26.2453823, -11.1716347, -26.2456360, -11.1712170, -10.9325638, 10.9332657
13: -14.1602116, 4.6829643, -14.1602411, 4.6835117, -13.3784027, 13.3665504
14: -24.1744423, -5.2179165, -24.1746483, -5.2176275, -16.3296432, 16.3262939
15: -7.6068439, 4.7095137, -7.6069112, 4.7101603, -11.3124619, 11.3059311
16: -7.6732435, 5.0034432, -7.6732955, 5.0038347, -9.4306602, 9.4385757
17: -26.7555275, -11.1059484, -26.7556915, -11.1058311, -11.0226669, 11.0218811
18: -17.7007561, -2.0515766, -17.7017174, -2.0515666, -10.5794296, 10.6048317
19: -10.4929523, -0.0656705, -10.4934921, -0.0656762, -6.8682442, 6.8845863
20: -5.8903494, 4.7191644, -5.8907752, 4.7191834, -7.5327950, 7.5514946
21: -8.5933638, 3.8232183, -8.5940104, 3.8232408, -9.6964645, 9.7138672
22: -10.8034801, 0.8349423, -10.8041000, 0.8349617, -7.8273392, 7.8513374
23: -4.6558018, 6.9074440, -4.6563015, 6.9074678, -8.7984390, 8.8086128
24: -8.1051388, 5.2016592, -8.1058655, 5.2016859, -10.5549545, 10.5756149
25: -8.3755274, 4.8564301, -8.3763037, 4.8564568, -8.3241348, 8.3541069
26: -16.5917282, 0.1532948, -16.5923939, 0.1533505, -11.6581306, 11.6791611
27: -7.8422241, 6.2023826, -7.8428631, 6.2024059, -12.2026291, 12.2134972
28: -6.5486212, 6.3283720, -6.5491810, 6.3284278, -10.1606140, 10.1726456
29: -7.7765074, 2.8724406, -7.7770958, 2.8724525, -8.6641769, 8.6782875
30: -3.8672309, 10.3132458, -3.8676906, 10.3132849, -12.2991257, 12.3043365
31: -14.8684721, -0.1749315, -14.8694324, -0.1749201, -10.7820053, 10.8107414
32: -20.8184357, -5.8458214, -20.8185692, -5.8453560, -12.0372696, 12.0471725
33: -38.5923042, -20.0072365, -38.5927010, -20.0071754, -10.6253510, 10.6273556
34: -35.7649155, -20.2659531, -35.7660408, -20.2659721, -11.5537491, 11.5545273
35: -33.0708237, -16.7967110, -33.0713196, -16.7966843, -11.3296585, 11.3331337
36: -31.1258717, -13.5127544, -31.1268272, -13.5126905, -12.6300507, 12.6398964
37: -50.2986488, -32.2753754, -50.2988663, -32.2753410, -9.9362831, 9.9403267
38: -38.7995148, -20.2444458, -38.8010025, -20.2444839, -11.4033775, 11.4217224
39: -42.5663795, -23.5579834, -42.5670776, -23.5579281, -10.6847458, 10.6853905
40: -38.0392532, -24.5549011, -38.0392685, -24.5532455, -7.9423923, 7.9534454
41: -24.8252220, -8.8913441, -24.8253937, -8.8906507, -13.1527023, 13.1690712
42: -15.1472149, -4.8176775, -15.1472025, -4.8152313, -8.1252365, 8.1338310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=68, inp2_unstable=69, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1682

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5529369, upper bound: 4.5702400
time: 25.22 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5707281, upper bound: 4.5707281
time: 24.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 51.51 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 51.51
Output dim: 3, lower bound: -4.5529369, upper bound: 4.5497289
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 51.51
Output dim: 3, lower bound: -4.5707281, upper bound: 4.5502182
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 51.51
Output dim: 3, lower bound: -4.5529369, upper bound: 4.5702400
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 51.51
Output dim: 3, lower bound: -4.5707281, upper bound: 4.5707281

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.3650246, 4.2952151, -13.3791571, 4.3073740, -15.6545258, 15.6587105
1: 0.4413960, 12.3224907, 0.4290693, 12.3352013, -9.2477875, 9.2485809
2: 2.0841322, 13.4659176, 2.0697513, 13.4813137, -9.0022888, 9.0046768
3: 1.6036353, 14.1637688, 1.5897391, 14.1764183, -8.9750175, 8.9850693
4: -4.1817265, 10.4438381, -4.1963387, 10.4590883, -12.5697784, 12.5829124
5: 2.1129830, 13.7597446, 2.1024182, 13.7702818, -8.3863411, 8.3949776
6: -25.1623039, -8.7901287, -25.1671219, -8.7849426, -13.3613052, 13.3524475
7: 2.5791769, 15.2806225, 2.5637965, 15.2966013, -9.3311844, 9.3345909
8: -4.4245949, 14.2174549, -4.4491501, 14.2427731, -15.6215744, 15.6208572
9: 0.6077156, 13.5812635, 0.5910149, 13.5964975, -9.2369080, 9.2529526
10: -4.3925400, 11.2879801, -4.4043250, 11.2998428, -12.0355682, 12.0523491
11: -4.4519234, 6.8866854, -4.4639587, 6.8971643, -8.5792770, 8.5719719
12: -26.2429714, -11.1768093, -26.2438316, -11.1721020, -10.9293709, 10.9193954
13: -14.1438293, 4.6554575, -14.1592531, 4.6741185, -13.3510628, 13.3583984
14: -24.1603165, -5.2327681, -24.1712532, -5.2227650, -16.3046112, 16.3121223
15: -7.5990963, 4.6967425, -7.6059184, 4.7047949, -11.2776947, 11.2925339
16: -7.6651859, 4.9926548, -7.6731291, 4.9992580, -9.4185791, 9.4259224
17: -26.7500648, -11.1083632, -26.7524338, -11.1066961, -11.0145874, 10.9870224
18: -17.6705132, -2.0710020, -17.6910973, -2.0526142, -10.5656662, 10.5734367
19: -10.4673042, -0.0808535, -10.4838600, -0.0658185, -6.8584290, 6.8512726
20: -5.8740292, 4.7088089, -5.8843918, 4.7189450, -7.5334930, 7.5326309
21: -8.5646553, 3.8074317, -8.5830746, 3.8228257, -9.6845589, 9.6854439
22: -10.7742615, 0.8181124, -10.7933149, 0.8348846, -7.8215065, 7.8211212
23: -4.6326537, 6.8936357, -4.6472316, 6.9071894, -8.7848511, 8.7783470
24: -8.0752039, 5.1829629, -8.0939579, 5.2015672, -10.5437813, 10.5369263
25: -8.3405437, 4.8347325, -8.3623295, 4.8562055, -8.3180161, 8.3001328
26: -16.5707474, 0.1416180, -16.5844917, 0.1531534, -11.6563416, 11.6572800
27: -7.8125596, 6.1858587, -7.8321409, 6.2019701, -12.1859436, 12.1856842
28: -6.5214639, 6.3119297, -6.5387516, 6.3280482, -10.1448174, 10.1357231
29: -7.7501488, 2.8581903, -7.7671595, 2.8723683, -8.6513596, 8.6427345
30: -3.8457720, 10.3009796, -3.8587530, 10.3128023, -12.2825165, 12.2775345
31: -14.8269768, -0.1999192, -14.8532066, -0.1753933, -10.7679710, 10.7633705
32: -20.8166656, -5.8508253, -20.8182030, -5.8473206, -12.0439987, 12.0259171
33: -38.5874367, -20.0118942, -38.5906258, -20.0077858, -10.6196060, 10.6099014
34: -35.7491150, -20.2732716, -35.7610970, -20.2666759, -11.5424957, 11.5371017
35: -33.0517654, -16.8071327, -33.0644989, -16.7969666, -11.3133125, 11.3073578
36: -31.1052818, -13.5245943, -31.1187496, -13.5131626, -12.6171188, 12.6108513
37: -50.2962875, -32.2783241, -50.2971268, -32.2759285, -9.9336433, 9.9143562
38: -38.7709122, -20.2586880, -38.7902451, -20.2446461, -11.3917809, 11.3987865
39: -42.5656166, -23.5594997, -42.5661545, -23.5583286, -10.6841354, 10.6755791
40: -38.0381203, -24.5657997, -38.0391388, -24.5577583, -7.9432640, 7.9347095
41: -24.8228741, -8.8948174, -24.8246422, -8.8913364, -13.1564331, 13.1391830
42: -15.1492825, -4.8198233, -15.1464672, -4.8158216, -8.1296463, 8.1131973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=68, inp2_unstable=68, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1784

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5577086, upper bound: 4.5496902
time: 28.96 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5701983, upper bound: 4.5496902
time: 26.82 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.3797522, 4.3104796, -13.3714304, 4.3044224, -15.6699982, 15.6627502
1: 0.4291813, 12.3400230, 0.4348109, 12.3366261, -9.2633018, 9.2525673
2: 2.0695913, 13.4861984, 2.0746536, 13.4800901, -9.0192719, 9.0073395
3: 1.5896428, 14.1754227, 1.5996907, 14.1589766, -8.9799232, 8.9731712
4: -4.1964302, 10.4581413, -4.1817656, 10.4393730, -12.5783386, 12.5779152
5: 2.1022935, 13.7682753, 2.1124847, 13.7521009, -8.3869820, 8.3861961
6: -25.1684799, -8.7846851, -25.1639519, -8.7869968, -13.3555450, 13.3640785
7: 2.5637536, 15.3009396, 2.5713549, 15.2928772, -9.3465195, 9.3335228
8: -4.4492426, 14.2529001, -4.4435444, 14.2461281, -15.6487503, 15.6329269
9: 0.5908933, 13.5967827, 0.6059649, 13.5803089, -9.2519951, 9.2385674
10: -4.4048195, 11.2955151, -4.3835125, 11.2736549, -12.0387344, 12.0356750
11: -4.4661036, 6.8972807, -4.4566827, 6.8910842, -8.5803032, 8.5842209
12: -26.2378254, -11.1719112, -26.2242241, -11.1808176, -10.9148140, 10.9109726
13: -14.1560516, 4.6824818, -14.1484013, 4.6725988, -13.3753395, 13.3590431
14: -24.1673470, -5.2179756, -24.1511097, -5.2205620, -16.3279343, 16.3079720
15: -7.6063876, 4.7007341, -7.5948672, 4.6840534, -11.2853966, 11.2847290
16: -7.6730690, 4.9979162, -7.6606703, 4.9882336, -9.4138336, 9.4186249
17: -26.7408180, -11.1059790, -26.7134933, -11.1268167, -10.9887009, 10.9829826
18: -17.7005672, -2.0569987, -17.6930923, -2.0674305, -10.5650711, 10.5955124
19: -10.4880714, -0.0658467, -10.4790907, -0.0727270, -6.8568649, 6.8701935
20: -5.8875914, 4.7189684, -5.8827553, 4.7164783, -7.5268745, 7.5427914
21: -8.5907116, 3.8228760, -8.5855036, 3.8216195, -9.6906624, 9.7043419
22: -10.7996025, 0.8348594, -10.7924900, 0.8286319, -7.8183517, 7.8402138
23: -4.6483684, 6.9072542, -4.6342630, 6.8972750, -8.7808266, 8.7866173
24: -8.0973883, 5.2015953, -8.0826807, 5.1916399, -10.5369720, 10.5524750
25: -8.3639698, 4.8563070, -8.3433752, 4.8422618, -8.2990112, 8.3218517
26: -16.5894547, 0.1529787, -16.5851154, 0.1484240, -11.6493263, 11.6706734
27: -7.8387465, 6.2021298, -7.8312702, 6.1988063, -12.1916580, 12.2008743
28: -6.5399370, 6.3280764, -6.5238838, 6.3161836, -10.1397514, 10.1478577
29: -7.7694893, 2.8723698, -7.7561193, 2.8605213, -8.6453285, 8.6575089
30: -3.8595200, 10.3130617, -3.8450613, 10.3061752, -12.2857285, 12.2825012
31: -14.8650436, -0.1755567, -14.8582306, -0.1787350, -10.7720261, 10.7972527
32: -20.8179016, -5.8467350, -20.8165836, -5.8489838, -12.0266304, 12.0437202
33: -38.5879631, -20.0074558, -38.5805893, -20.0142021, -10.6150513, 10.6152554
34: -35.7644157, -20.2665977, -35.7637787, -20.2684479, -11.5454788, 11.5510941
35: -33.0660362, -16.7969799, -33.0571899, -16.8061028, -11.3163795, 11.3192749
36: -31.1191521, -13.5130196, -31.1071587, -13.5258532, -12.6098824, 12.6201439
37: -50.2897415, -32.2756195, -50.2736435, -32.2885132, -9.9141808, 9.9148350
38: -38.7940292, -20.2447433, -38.7847366, -20.2525902, -11.3881760, 11.4043503
39: -42.5608826, -23.5582447, -42.5511780, -23.5665531, -10.6700706, 10.6689987
40: -38.0391769, -24.5579185, -38.0355911, -24.5630798, -7.9312649, 7.9457951
41: -24.8236675, -8.8918896, -24.8191833, -8.8942871, -13.1407623, 13.1611252
42: -15.1418419, -4.8179150, -15.1318827, -4.8232174, -8.1116104, 8.1180000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=68, inp2_unstable=68, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1784

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5524078, upper bound: 4.5572193
time: 26.31 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5524078, upper bound: 4.5697113
time: 24.71 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.3806801, 4.3139215, -13.3806915, 4.3139968, -15.6768723, 15.6727905
1: 0.4287016, 12.3419542, 0.4287486, 12.3420801, -9.2674980, 9.2589836
2: 2.0693798, 13.4894381, 2.0693727, 13.4894733, -9.0253639, 9.0153122
3: 1.5895529, 14.1842031, 1.5895727, 14.1837692, -8.9965134, 8.9925308
4: -4.1967306, 10.4681683, -4.1967392, 10.4676676, -12.5934906, 12.6030922
5: 2.1021161, 13.7769833, 2.1021276, 13.7764225, -8.4032631, 8.4057541
6: -25.1701126, -8.7840853, -25.1697845, -8.7841644, -13.3693085, 13.3739738
7: 2.5634921, 15.3053455, 2.5634871, 15.3054428, -9.3555450, 9.3459663
8: -4.4498091, 14.2565556, -4.4497452, 14.2567282, -15.6604919, 15.6441345
9: 0.5905857, 13.6056471, 0.5905974, 13.6053238, -9.2624893, 9.2626038
10: -4.4051323, 11.3072081, -4.4051428, 11.3064575, -12.0554962, 12.0683479
11: -4.4707623, 6.8974228, -4.4703503, 6.8974342, -8.5908966, 8.5897179
12: -26.2451477, -11.1716261, -26.2446251, -11.1712923, -10.9322243, 10.9263802
13: -14.1600666, 4.6829309, -14.1595716, 4.6834545, -13.3766327, 13.3747063
14: -24.1741352, -5.2179155, -24.1733761, -5.2176142, -16.3284683, 16.3283119
15: -7.6068077, 4.7092528, -7.6068144, 4.7090425, -11.3009949, 11.3055916
16: -7.6732235, 5.0032611, -7.6732659, 5.0031042, -9.4198761, 9.4383469
17: -26.7550316, -11.1059380, -26.7535782, -11.1058474, -11.0221634, 10.9918060
18: -17.7007542, -2.0517449, -17.7017059, -2.0522418, -10.5717964, 10.6045151
19: -10.4927979, -0.0656717, -10.4928885, -0.0657277, -6.8680153, 6.8754883
20: -5.8901696, 4.7191448, -5.8900671, 4.7191677, -7.5325546, 7.5497627
21: -8.5931282, 3.8231766, -8.5930176, 3.8231549, -9.6962128, 9.7111397
22: -10.8033810, 0.8349502, -10.8036404, 0.8349404, -7.8271599, 7.8486767
23: -4.6555786, 6.9074364, -4.6552963, 6.9074430, -8.7981796, 8.8001404
24: -8.1049185, 5.2016392, -8.1048489, 5.2016487, -10.5546722, 10.5666962
25: -8.3751574, 4.8564157, -8.3747730, 4.8564320, -8.3237495, 8.3348579
26: -16.5916023, 0.1532934, -16.5918922, 0.1533135, -11.6580124, 11.6781006
27: -7.8420706, 6.2023606, -7.8422060, 6.2023630, -12.2054520, 12.2123032
28: -6.5483336, 6.3283639, -6.5480013, 6.3283739, -10.1602745, 10.1613731
29: -7.7763014, 2.8724353, -7.7761974, 2.8724396, -8.6639328, 8.6661491
30: -3.8668790, 10.3132305, -3.8662412, 10.3132391, -12.2985535, 12.2967873
31: -14.8681030, -0.1749868, -14.8680124, -0.1751106, -10.7816505, 10.8031769
32: -20.8184166, -5.8460674, -20.8185234, -5.8464375, -12.0466461, 12.0455589
33: -38.5921555, -20.0072479, -38.5920868, -20.0072231, -10.6249847, 10.6210442
34: -35.7648926, -20.2660885, -35.7659683, -20.2664700, -11.5595970, 11.5529404
35: -33.0706902, -16.7967262, -33.0707169, -16.7967529, -11.3293839, 11.3256721
36: -31.1256714, -13.5127563, -31.1259766, -13.5127697, -12.6297989, 12.6332664
37: -50.2983780, -32.2753983, -50.2977142, -32.2753601, -9.9359665, 9.9252911
38: -38.7993546, -20.2444801, -38.8002586, -20.2444706, -11.4031143, 11.4153156
39: -42.5662346, -23.5579987, -42.5663834, -23.5579910, -10.6845131, 10.6758804
40: -38.0392532, -24.5550938, -38.0392494, -24.5540886, -7.9370689, 7.9532986
41: -24.8251534, -8.8913918, -24.8250771, -8.8908396, -13.1580048, 13.1679306
42: -15.1470299, -4.8176708, -15.1464853, -4.8152823, -8.1250191, 8.1252441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=68, inp2_unstable=68, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1784

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5524078, upper bound: 4.5577086
time: 30.33 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5524078, upper bound: 4.5701984
time: 34.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 66.54 seconds
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 66.54
Output dim: 3, lower bound: -4.5577086, upper bound: 4.5496902
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 66.54
Output dim: 3, lower bound: -4.5701983, upper bound: 4.5496902
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 66.54
Output dim: 3, lower bound: -4.5524078, upper bound: 4.5572193
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 66.54
Output dim: 3, lower bound: -4.5524078, upper bound: 4.5697113
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 66.54
Output dim: 3, lower bound: -4.5524078, upper bound: 4.5577086
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 66.54
Output dim: 3, lower bound: -4.5524078, upper bound: 4.5701984

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -13.3644981, 4.2949905, -13.3886967, 4.3133664, -15.6605148, 15.6662178
1: 0.4414492, 12.3219900, 0.4075882, 12.3353004, -9.2465515, 9.2691956
2: 2.0841515, 13.4653225, 2.0460787, 13.4816208, -9.0010490, 9.0271873
3: 1.6036544, 14.1630459, 1.5631654, 14.1767063, -8.9738350, 9.0105209
4: -4.1816835, 10.4428797, -4.2297297, 10.4593506, -12.5687408, 12.6155205
5: 2.1130095, 13.7592010, 2.0838585, 13.7694035, -8.3847580, 8.4131737
6: -25.1617012, -8.7901821, -25.1691437, -8.7656145, -13.3851357, 13.3503990
7: 2.5792289, 15.2799301, 2.5391715, 15.2950935, -9.3282280, 9.3586922
8: -4.4245615, 14.2159519, -4.5019274, 14.2414055, -15.6181488, 15.6723251
9: 0.6077545, 13.5806866, 0.5696788, 13.5961399, -9.2353134, 9.2737274
10: -4.3925133, 11.2876997, -4.4118476, 11.3021193, -12.0324326, 12.0676880
11: -4.4510937, 6.8866649, -4.4662552, 6.9255180, -8.6069183, 8.5714417
12: -26.2423382, -11.1769114, -26.2427139, -11.1465654, -10.9542084, 10.9168854
13: -14.1437759, 4.6544390, -14.1922674, 4.6789408, -13.3524475, 13.3906059
14: -24.1598663, -5.2329311, -24.1800079, -5.2118702, -16.3119965, 16.3198891
15: -7.5990257, 4.6961570, -7.6247988, 4.7058573, -11.2755051, 11.3157120
16: -7.6648703, 4.9925423, -7.6800718, 5.0079179, -9.4432220, 9.4172173
17: -26.7495747, -11.1084614, -26.7553940, -11.0906773, -11.0302658, 10.9881439
18: -17.6690788, -2.0710588, -17.6905499, -2.0017509, -10.6148376, 10.5683098
19: -10.4667645, -0.0808735, -10.4848223, -0.0524044, -6.8715553, 6.8506603
20: -5.8737288, 4.7087860, -5.8865557, 4.7245164, -7.5335064, 7.5386162
21: -8.5640278, 3.8074102, -8.5857000, 3.8402529, -9.7013054, 9.6867218
22: -10.7737589, 0.8181281, -10.7944355, 0.8488052, -7.8351326, 7.8206825
23: -4.6319895, 6.8936067, -4.6492767, 6.9303575, -8.8073959, 8.7778091
24: -8.0743303, 5.1829586, -8.0975857, 5.2327633, -10.5741959, 10.5376205
25: -8.3401279, 4.8347154, -8.3646088, 4.8692808, -8.3312645, 8.3006458
26: -16.5700779, 0.1415825, -16.5864010, 0.1776626, -11.6811447, 11.6566429
27: -7.8118777, 6.1858411, -7.8370337, 6.2213879, -12.2090530, 12.1866722
28: -6.5210099, 6.3118792, -6.5411329, 6.3457718, -10.1622124, 10.1365433
29: -7.7495246, 2.8581820, -7.7698498, 2.8921287, -8.6704597, 8.6434669
30: -3.8450019, 10.3009357, -3.8632178, 10.3367395, -12.3068695, 12.2782249
31: -14.8260393, -0.1999559, -14.8533669, -0.1457543, -10.7968636, 10.7615585
32: -20.8161964, -5.8508577, -20.8198013, -5.8337860, -12.0582161, 12.0272064
33: -38.5872955, -20.0119438, -38.5934143, -20.0007362, -10.6268501, 10.6111870
34: -35.7482147, -20.2732964, -35.7646179, -20.2360802, -11.5721893, 11.5373917
35: -33.0511169, -16.8071842, -33.0682220, -16.7742653, -11.3352470, 11.3085861
36: -31.1048279, -13.5245972, -31.1196251, -13.4976130, -12.6331177, 12.6107483
37: -50.2955666, -32.2783279, -50.2970810, -32.2495422, -9.9591217, 9.9125729
38: -38.7704239, -20.2587109, -38.7903748, -20.2283859, -11.3939972, 11.3986721
39: -42.5655975, -23.5597935, -42.5764008, -23.5566349, -10.6842842, 10.6861420
40: -38.0374374, -24.5658321, -38.0383797, -24.5437279, -7.9569340, 7.9331074
41: -24.8223915, -8.8948584, -24.8276367, -8.8742371, -13.1764755, 13.1387024
42: -15.1492500, -4.8199415, -15.1502428, -4.8122373, -8.1332245, 8.1146393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=68, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1663

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5696654, upper bound: 4.5312591
time: 41.21 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5393833, upper bound: 4.5491574
time: 23.29 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.3892975, 4.3164353, -13.3709126, 4.3041906, -15.6774750, 15.6687622
1: 0.4076874, 12.3401241, 0.4348571, 12.3361320, -9.2838936, 9.2513351
2: 2.0459492, 13.4865398, 2.0746884, 13.4794788, -9.0417900, 9.0061150
3: 1.5630798, 14.1757021, 1.5997193, 14.1582718, -9.0053787, 8.9719810
4: -4.2298107, 10.4584169, -4.1817207, 10.4384060, -12.6109238, 12.5769119
5: 2.0837440, 13.7674065, 2.1125343, 13.7515659, -8.4051819, 8.3846245
6: -25.1704845, -8.7653437, -25.1633377, -8.7870207, -13.3534813, 13.3878784
7: 2.5391221, 15.2993975, 2.5713778, 15.2922010, -9.3706093, 9.3305893
8: -4.5020142, 14.2515278, -4.4434814, 14.2446098, -15.7002182, 15.6295166
9: 0.5695674, 13.5964108, 0.6059978, 13.5797234, -9.2727509, 9.2369728
10: -4.4122906, 11.2978268, -4.3834639, 11.2733793, -12.0540619, 12.0325394
11: -4.4683800, 6.9256506, -4.4558744, 6.8910537, -8.5797729, 8.6118584
12: -26.2367058, -11.1464090, -26.2235756, -11.1808815, -10.9122963, 10.9358063
13: -14.1891232, 4.6872931, -14.1484013, 4.6715870, -13.4075470, 13.3604126
14: -24.1760254, -5.2071133, -24.1506577, -5.2207317, -16.3356857, 16.3153419
15: -7.6252689, 4.7017846, -7.5947890, 4.6834745, -11.3085556, 11.2825508
16: -7.6800327, 5.0065312, -7.6603775, 4.9881201, -9.4051323, 9.4432507
17: -26.7438488, -11.0899544, -26.7130280, -11.1269531, -10.9898338, 10.9986610
18: -17.6999912, -2.0061145, -17.6916466, -2.0674486, -10.5599442, 10.6446838
19: -10.4890108, -0.0524282, -10.4785509, -0.0727320, -6.8562565, 6.8833141
20: -5.8897419, 4.7245321, -5.8824744, 4.7164712, -7.5328445, 7.5428009
21: -8.5933418, 3.8402567, -8.5848827, 3.8215811, -9.6919594, 9.7210884
22: -10.8007298, 0.8487744, -10.7920313, 0.8286319, -7.8179092, 7.8538208
23: -4.6503973, 6.9304214, -4.6335983, 6.8972464, -8.7802963, 8.8091507
24: -8.1010303, 5.2328081, -8.0818024, 5.1916199, -10.5376740, 10.5828972
25: -8.3662701, 4.8693714, -8.3429747, 4.8422441, -8.2995300, 8.3351097
26: -16.5913448, 0.1775707, -16.5844288, 0.1483728, -11.6487007, 11.6954613
27: -7.8436460, 6.2215557, -7.8306084, 6.1987729, -12.1926727, 12.2240067
28: -6.5423417, 6.3458424, -6.5234179, 6.3161325, -10.1405563, 10.1652679
29: -7.7721863, 2.8921020, -7.7555118, 2.8605042, -8.6460724, 8.6766129
30: -3.8639641, 10.3370047, -3.8442934, 10.3061161, -12.2864227, 12.3069000
31: -14.8652420, -0.1459148, -14.8573046, -0.1787810, -10.7702408, 10.8261375
32: -20.8195152, -5.8332148, -20.8161201, -5.8490610, -12.0279083, 12.0579109
33: -38.5907745, -20.0004101, -38.5804405, -20.0142059, -10.6163559, 10.6224861
34: -35.7679443, -20.2360287, -35.7629089, -20.2684917, -11.5457382, 11.5807762
35: -33.0697479, -16.7742977, -33.0565414, -16.8061733, -11.3176079, 11.3412476
36: -31.1199856, -13.4975090, -31.1067085, -13.5258369, -12.6097794, 12.6361237
37: -50.2896957, -32.2492561, -50.2728958, -32.2884979, -9.9123688, 9.9403095
38: -38.7941895, -20.2284317, -38.7841988, -20.2526207, -11.3880920, 11.4065914
39: -42.5710983, -23.5565529, -42.5511513, -23.5668163, -10.6806183, 10.6691494
40: -38.0384140, -24.5439091, -38.0349274, -24.5631256, -7.9296589, 7.9594784
41: -24.8266220, -8.8747978, -24.8187256, -8.8943233, -13.1402855, 13.1811943
42: -15.1456137, -4.8143282, -15.1318493, -4.8233585, -8.1130562, 8.1215744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=68, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1663

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5339766, upper bound: 4.5691783
time: 29.34 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5518748, upper bound: 4.5691783
time: 23.29 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3902035, 4.3198915, -13.3801937, 4.3137631, -15.6843719, 15.6787872
1: 0.4072435, 12.3420620, 0.4287934, 12.3415852, -9.2880859, 9.2577515
2: 2.0457277, 13.4897852, 2.0694056, 13.4888744, -9.0478783, 9.0140724
3: 1.5629718, 14.1844778, 1.5895765, 14.1830835, -9.0219498, 8.9913406
4: -4.2301102, 10.4684381, -4.1967120, 10.4667053, -12.6260910, 12.6020546
5: 2.0835419, 13.7760916, 2.1021457, 13.7758865, -8.4214630, 8.4041748
6: -25.1721172, -8.7647419, -25.1691456, -8.7842188, -13.3672371, 13.3978043
7: 2.5388737, 15.3038397, 2.5635245, 15.3047514, -9.3796349, 9.3430252
8: -4.5025549, 14.2551613, -4.4497194, 14.2552080, -15.7119522, 15.6406479
9: 0.5692410, 13.6052952, 0.5906520, 13.6047316, -9.2832527, 9.2610283
10: -4.4126506, 11.3095093, -4.4050865, 11.3061628, -12.0708275, 12.0652161
11: -4.4730687, 6.9257832, -4.4695086, 6.8974133, -8.5903893, 8.6173515
12: -26.2440033, -11.1461678, -26.2439899, -11.1713581, -10.9297104, 10.9512444
13: -14.1931410, 4.6877766, -14.1595497, 4.6824427, -13.4088211, 13.3760529
14: -24.1828499, -5.2070704, -24.1728897, -5.2178011, -16.3362503, 16.3357010
15: -7.6257010, 4.7102985, -7.6067514, 4.7084470, -11.3241920, 11.3034096
16: -7.6801920, 5.0118980, -7.6729527, 5.0029840, -9.4111633, 9.4629593
17: -26.7580376, -11.0899124, -26.7531357, -11.1059780, -11.0233078, 11.0074730
18: -17.7001762, -2.0008869, -17.7002659, -2.0522757, -10.5666809, 10.6536674
19: -10.4937553, -0.0522501, -10.4923620, -0.0657246, -6.8674183, 6.8886051
20: -5.8923244, 4.7247062, -5.8897767, 4.7191305, -7.5385437, 7.5497723
21: -8.5957613, 3.8405931, -8.5923920, 3.8231416, -9.6974792, 9.7278786
22: -10.8044968, 0.8488820, -10.8031626, 0.8349483, -7.8267097, 7.8622990
23: -4.6576080, 6.9306002, -4.6546268, 6.9074135, -8.7976379, 8.8226624
24: -8.1085272, 5.2328548, -8.1039925, 5.2016516, -10.5553665, 10.5971222
25: -8.3774614, 4.8694677, -8.3743544, 4.8564157, -8.3242798, 8.3481007
26: -16.5935040, 0.1778866, -16.5912552, 0.1532687, -11.6573601, 11.7029076
27: -7.8469796, 6.2217846, -7.8415060, 6.2023230, -12.2064362, 12.2354050
28: -6.5507331, 6.3461008, -6.5475473, 6.3283300, -10.1610985, 10.1787872
29: -7.7789774, 2.8921843, -7.7755828, 2.8724387, -8.6646500, 8.6852379
30: -3.8713288, 10.3372021, -3.8654654, 10.3131876, -12.2992630, 12.3211823
31: -14.8683271, -0.1453600, -14.8670902, -0.1751328, -10.7798538, 10.8320694
32: -20.8200378, -5.8325424, -20.8180561, -5.8464699, -12.0479050, 12.0597649
33: -38.5949516, -20.0001640, -38.5919647, -20.0072289, -10.6262703, 10.6282845
34: -35.7683945, -20.2355461, -35.7650681, -20.2665062, -11.5598793, 11.5826416
35: -33.0744057, -16.7740021, -33.0700760, -16.7967892, -11.3306160, 11.3475990
36: -31.1265202, -13.4972086, -31.1255245, -13.5127974, -12.6296844, 12.6492577
37: -50.2983322, -32.2490387, -50.2969818, -32.2754211, -9.9341736, 9.9507523
38: -38.7995453, -20.2282104, -38.7997818, -20.2445297, -11.4029999, 11.4175396
39: -42.5764465, -23.5563316, -42.5663605, -23.5582695, -10.6950684, 10.6760254
40: -38.0384827, -24.5410805, -38.0385704, -24.5541801, -7.9354706, 7.9669800
41: -24.8281670, -8.8742857, -24.8246155, -8.8908653, -13.1574974, 13.1879807
42: -15.1508236, -4.8140821, -15.1464605, -4.8153872, -8.1264496, 8.1288376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=68, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1663

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5517684, upper bound: 4.5696654
time: 27.04 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5696654, upper bound: 4.5696654
time: 34.45 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 63.43 seconds
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 63.43
Output dim: 3, lower bound: -4.5696654, upper bound: 4.5312591
IS_A1_B2_B2_A2, status: Status.VERIFIED, split count: 4, time: 63.43
Output dim: 3, lower bound: -4.5393833, upper bound: 4.5491574
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 63.43
Output dim: 3, lower bound: -4.5339766, upper bound: 4.5691783
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 63.43
Output dim: 3, lower bound: -4.5518748, upper bound: 4.5691783
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 63.43
Output dim: 3, lower bound: -4.5517684, upper bound: 4.5696654
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 63.43
Output dim: 3, lower bound: -4.5696654, upper bound: 4.5696654

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -13.3523483, 4.2733731, -13.3873835, 4.3058510, -15.6400604, 15.6420021
1: 0.4503794, 12.2998161, 0.4078543, 12.3275719, -9.2295227, 9.2464485
2: 2.0907145, 13.4477577, 2.0464964, 13.4755344, -8.9880257, 9.0090065
3: 1.6126230, 14.1364021, 1.5632892, 14.1673756, -8.9555740, 8.9838142
4: -4.1724687, 10.4184322, -4.2294130, 10.4508553, -12.5510254, 12.5906448
5: 2.1209347, 13.7372532, 2.0840936, 13.7617350, -8.3692436, 8.3909492
6: -25.1596794, -8.7951355, -25.1685982, -8.7669744, -13.3802185, 13.3413811
7: 2.5880830, 15.2590141, 2.5393982, 15.2877769, -9.3120880, 9.3375702
8: -4.4111891, 14.1866760, -4.5014582, 14.2312059, -15.5947342, 15.6426315
9: 0.6171486, 13.5615225, 0.5700104, 13.5894499, -9.2194786, 9.2542610
10: -4.3865671, 11.2862082, -4.4103642, 11.3017025, -12.0244675, 12.0624390
11: -4.4304361, 6.8807898, -4.4591751, 6.9253502, -8.5861816, 8.5585365
12: -26.2295799, -11.1822767, -26.2382793, -11.1470184, -10.9410553, 10.9070663
13: -14.1414366, 4.6329136, -14.1919298, 4.6715589, -13.3410492, 13.3685341
14: -24.1414299, -5.2330961, -24.1742821, -5.2119236, -16.2931824, 16.3131104
15: -7.5921679, 4.6800070, -7.6241207, 4.7002959, -11.2631645, 11.2990608
16: -7.6583347, 4.9834795, -7.6798019, 5.0047379, -9.4381790, 9.4101791
17: -26.7282600, -11.1156940, -26.7483501, -11.0907049, -11.0089684, 10.9734001
18: -17.6449909, -2.0796690, -17.6822166, -2.0022411, -10.5905647, 10.5512619
19: -10.4451618, -0.0875030, -10.4772882, -0.0524702, -6.8499184, 6.8365231
20: -5.8565955, 4.7036080, -5.8806467, 4.7242723, -7.5154858, 7.5270920
21: -8.5452995, 3.8029814, -8.5793648, 3.8399448, -9.6824150, 9.6755676
22: -10.7556639, 0.8118181, -10.7881680, 0.8486996, -7.8169365, 7.8075829
23: -4.6076880, 6.8861647, -4.6408691, 6.9301605, -8.7829132, 8.7619057
24: -8.0452604, 5.1737251, -8.0875130, 5.2326469, -10.5446663, 10.5180893
25: -8.3112383, 4.8248610, -8.3546276, 4.8690739, -8.3020821, 8.2806816
26: -16.5463543, 0.1336646, -16.5782223, 0.1774976, -11.6570473, 11.6403580
27: -7.7964230, 6.1831522, -7.8318062, 6.2210722, -12.1933136, 12.1787949
28: -6.4976664, 6.3036327, -6.5330639, 6.3454928, -10.1386681, 10.1201401
29: -7.7334685, 2.8534989, -7.7643037, 2.8920691, -8.6539383, 8.6326752
30: -3.8190358, 10.2936878, -3.8543839, 10.3363743, -12.2807541, 12.2620621
31: -14.7956686, -0.2087283, -14.8427773, -0.1460109, -10.7662582, 10.7422256
32: -20.8147850, -5.8552294, -20.8193703, -5.8348351, -12.0549965, 12.0206985
33: -38.5819817, -20.0148888, -38.5915833, -20.0014954, -10.6202393, 10.6061211
34: -35.7375641, -20.2773991, -35.7610474, -20.2362633, -11.5608444, 11.5293007
35: -33.0441437, -16.8114319, -33.0659485, -16.7747135, -11.3262444, 11.3006935
36: -31.1004505, -13.5283699, -31.1181316, -13.4982147, -12.6265945, 12.6046600
37: -50.2833176, -32.2842102, -50.2928543, -32.2501755, -9.9452591, 9.9040260
38: -38.7640839, -20.2615166, -38.7882233, -20.2289696, -11.3863373, 11.3933907
39: -42.5635986, -23.5612011, -42.5756760, -23.5571461, -10.6816635, 10.6843319
40: -38.0357437, -24.5691299, -38.0380363, -24.5447121, -7.9537621, 7.9273911
41: -24.8210068, -8.9010315, -24.8273468, -8.8759689, -13.1733589, 13.1310997
42: -15.1474257, -4.8232775, -15.1496792, -4.8131828, -8.1303215, 8.1093845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1785

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5571985, upper bound: 4.5307004
time: 37.68 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5691036, upper bound: 4.5307004
time: 25.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.3879833, 4.3089428, -13.3587456, 4.2825418, -15.6532593, 15.6483040
1: 0.4079692, 12.3324108, 0.4437532, 12.3139400, -9.2611465, 9.2342949
2: 2.0463676, 13.4804459, 2.0812299, 13.4619455, -9.0236244, 8.9931068
3: 1.5632191, 14.1663780, 1.6086702, 14.1316242, -8.9786530, 8.9537125
4: -4.2294788, 10.4499416, -4.1725073, 10.4139786, -12.5860901, 12.5592117
5: 2.0839720, 13.7597446, 2.1204281, 13.7296247, -8.3829651, 8.3690834
6: -25.1699524, -8.7667303, -25.1613178, -8.7919931, -13.3444862, 13.3829842
7: 2.5393791, 15.2920847, 2.5802498, 15.2712688, -9.3494873, 9.3144455
8: -4.5015364, 14.2413282, -4.4301529, 14.2153378, -15.6705322, 15.6060944
9: 0.5698783, 13.5896969, 0.6153955, 13.5605583, -9.2532768, 9.2211609
10: -4.4108377, 11.2973957, -4.3775344, 11.2718792, -12.0488243, 12.0245743
11: -4.4613123, 6.9254742, -4.4352093, 6.8851900, -8.5668678, 8.5911331
12: -26.2322979, -11.1468573, -26.2108021, -11.1862698, -10.9024925, 10.9226685
13: -14.1887922, 4.6799397, -14.1460333, 4.6500883, -13.3854675, 13.3490410
14: -24.1703548, -5.2071686, -24.1322002, -5.2209024, -16.3289490, 16.2965431
15: -7.6245975, 4.6962538, -7.5879388, 4.6673293, -11.2919197, 11.2702141
16: -7.6797562, 5.0033870, -7.6538415, 4.9790583, -9.3980980, 9.4381886
17: -26.7367268, -11.0899820, -26.6917400, -11.1342182, -10.9751015, 10.9773750
18: -17.6916885, -2.0065746, -17.6675434, -2.0760550, -10.5429382, 10.6203880
19: -10.4814892, -0.0524790, -10.4569349, -0.0793769, -6.8421173, 6.8616848
20: -5.8838706, 4.7243023, -5.8653045, 4.7112832, -7.5213242, 7.5247917
21: -8.5870028, 3.8399639, -8.5661497, 3.8171937, -9.6808090, 9.7021790
22: -10.7945004, 0.8486834, -10.7739134, 0.8223896, -7.8048134, 7.8356552
23: -4.6419706, 6.9302316, -4.6092901, 6.8897977, -8.7644005, 8.7846642
24: -8.0909233, 5.2326937, -8.0527449, 5.1823859, -10.5181503, 10.5533600
25: -8.3562756, 4.8691778, -8.3140955, 4.8323941, -8.2795525, 8.3059349
26: -16.5832176, 0.1773523, -16.5606651, 0.1404638, -11.6324196, 11.6713715
27: -7.8384328, 6.2212396, -7.8151355, 6.1960897, -12.1847458, 12.2082520
28: -6.5342550, 6.3455558, -6.5000906, 6.3078775, -10.1241570, 10.1417122
29: -7.7666264, 2.8920727, -7.7394781, 2.8558350, -8.6352959, 8.6600838
30: -3.8550935, 10.3366308, -3.8183341, 10.2988844, -12.2702637, 12.2807846
31: -14.8546047, -0.1461415, -14.8268948, -0.1875343, -10.7508850, 10.7955399
32: -20.8190575, -5.8342915, -20.8146782, -5.8534184, -12.0214081, 12.0547180
33: -38.5889168, -20.0012054, -38.5751839, -20.0171795, -10.6112900, 10.6158867
34: -35.7644272, -20.2361698, -35.7522163, -20.2726135, -11.5376587, 11.5694008
35: -33.0674744, -16.7747459, -33.0495644, -16.8104076, -11.3097420, 11.3322525
36: -31.1184597, -13.4981003, -31.1023483, -13.5296001, -12.6036949, 12.6295929
37: -50.2854347, -32.2498894, -50.2606354, -32.2943764, -9.9038658, 9.9264507
38: -38.7920303, -20.2290382, -38.7779083, -20.2554226, -11.3827972, 11.3989220
39: -42.5704041, -23.5570469, -42.5491753, -23.5682201, -10.6788101, 10.6665268
40: -38.0380630, -24.5449295, -38.0332184, -24.5664482, -7.9239273, 7.9563046
41: -24.8263550, -8.8765278, -24.8173542, -8.9004860, -13.1326332, 13.1780968
42: -15.1450500, -4.8152795, -15.1300335, -4.8267002, -8.1077957, 8.1186752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1785

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5334173, upper bound: 4.5567117
time: 16.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5334173, upper bound: 4.5686170
time: 34.92 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.3892937, 4.3163447, -13.3707571, 4.3034806, -15.6702194, 15.6684647
1: 0.4077163, 12.3400326, 0.4348905, 12.3353901, -9.2743721, 9.2512054
2: 2.0459435, 13.4864721, 2.0747447, 13.4789047, -9.0347786, 9.0059776
3: 1.5630777, 14.1755829, 1.5997329, 14.1573982, -8.9939499, 8.9718475
4: -4.2297902, 10.4583092, -4.1816931, 10.4375944, -12.6056061, 12.5767937
5: 2.0837371, 13.7673111, 2.1125340, 13.7508383, -8.3970718, 8.3845139
6: -25.1704655, -8.7653675, -25.1632423, -8.7873116, -13.3601837, 13.3869057
7: 2.5391259, 15.2993040, 2.5713897, 15.2915154, -9.3617859, 9.3304825
8: -4.5019941, 14.2514191, -4.4434605, 14.2436104, -15.6910477, 15.6293564
9: 0.5695603, 13.5963383, 0.6060619, 13.5790939, -9.2637444, 9.2368622
10: -4.4122620, 11.2977877, -4.3832912, 11.2730503, -12.0537491, 12.0323906
11: -4.4682913, 6.9256549, -4.4552202, 6.8910537, -8.5796661, 8.6058884
12: -26.2366600, -11.1464090, -26.2231445, -11.1809263, -10.9122200, 10.9320984
13: -14.1891098, 4.6872196, -14.1483393, 4.6708937, -13.4026833, 13.3602905
14: -24.1759682, -5.2071152, -24.1501141, -5.2208614, -16.3348770, 16.3153305
15: -7.6252551, 4.7017164, -7.5947304, 4.6829467, -11.3079834, 11.2827682
16: -7.6800289, 5.0064974, -7.6603317, 4.9878235, -9.4102440, 9.4418144
17: -26.7437515, -11.0899582, -26.7123947, -11.1269684, -10.9897232, 10.9858856
18: -17.6998825, -2.0060983, -17.6908989, -2.0675097, -10.5598183, 10.6345482
19: -10.4889078, -0.0524309, -10.4778290, -0.0727451, -6.8561592, 6.8719311
20: -5.8896799, 4.7245426, -5.8819189, 4.7164288, -7.5327206, 7.5344276
21: -8.5932503, 3.8402665, -8.5842762, 3.8215547, -9.6918221, 9.7144318
22: -10.8006763, 0.8487620, -10.7914314, 0.8286195, -7.8178139, 7.8416424
23: -4.6503077, 6.9304242, -4.6328115, 6.8972325, -8.7801819, 8.8012733
24: -8.1009026, 5.2328062, -8.0808620, 5.1916037, -10.5375290, 10.5702400
25: -8.3661509, 4.8693595, -8.3420677, 4.8422346, -8.2993851, 8.3205452
26: -16.5912647, 0.1775341, -16.5836658, 0.1483488, -11.6485634, 11.6827621
27: -7.8435960, 6.2215662, -7.8300939, 6.1987548, -12.1925812, 12.2222290
28: -6.5422316, 6.3458414, -6.5226717, 6.3160977, -10.1404266, 10.1559067
29: -7.7721348, 2.8921070, -7.7549663, 2.8605134, -8.6460075, 8.6678581
30: -3.8638549, 10.3369923, -3.8434668, 10.3060999, -12.2862701, 12.3005600
31: -14.8651085, -0.1459055, -14.8563061, -0.1787877, -10.7701073, 10.8129272
32: -20.8194885, -5.8332481, -20.8160591, -5.8492632, -12.0316315, 12.0574074
33: -38.5906754, -20.0004292, -38.5798912, -20.0143242, -10.6161728, 10.6209946
34: -35.7678986, -20.2360306, -35.7625427, -20.2685356, -11.5456734, 11.5802002
35: -33.0696983, -16.7743053, -33.0561867, -16.8062077, -11.3175125, 11.3393021
36: -31.1199074, -13.4975204, -31.1060123, -13.5259457, -12.6096573, 12.6351357
37: -50.2896118, -32.2492676, -50.2723389, -32.2886009, -9.9122543, 9.9350853
38: -38.7940292, -20.2284412, -38.7831345, -20.2526474, -11.3875084, 11.4083652
39: -42.5710068, -23.5565624, -42.5504303, -23.5668716, -10.6803818, 10.6690578
40: -38.0383987, -24.5439301, -38.0348587, -24.5632896, -7.9328671, 7.9591370
41: -24.8266258, -8.8748369, -24.8186607, -8.8945007, -13.1439018, 13.1802292
42: -15.1455965, -4.8143620, -15.1317883, -4.8236728, -8.1170273, 8.1211891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1785

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5334173, upper bound: 4.5567117
time: 33.15 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5334173, upper bound: 4.5686170
time: 8.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.3889122, 4.3123765, -13.3680534, 4.2921529, -15.6601868, 15.6583557
1: 0.4074857, 12.3343410, 0.4377036, 12.3194160, -9.2653618, 9.2407036
2: 2.0461531, 13.4836922, 2.0759499, 13.4713268, -9.0296936, 9.0010605
3: 1.5631099, 14.1751604, 1.5985129, 14.1564369, -8.9952278, 8.9730682
4: -4.2297955, 10.4599485, -4.1874990, 10.4422588, -12.6012268, 12.5843506
5: 2.0837684, 13.7684441, 2.1100559, 13.7539196, -8.3992500, 8.3886566
6: -25.1715679, -8.7661591, -25.1671600, -8.7891798, -13.3582535, 13.3928986
7: 2.5391054, 15.2965202, 2.5724027, 15.2838535, -9.3585091, 9.3268814
8: -4.5020814, 14.2449760, -4.4363699, 14.2259502, -15.6822662, 15.6172943
9: 0.5695729, 13.5986042, 0.6000292, 13.5855665, -9.2637749, 9.2451859
10: -4.4111910, 11.3090715, -4.3991480, 11.3046703, -12.0655670, 12.0572319
11: -4.4659734, 6.9256072, -4.4488244, 6.8915458, -8.5774727, 8.5966492
12: -26.2395744, -11.1465607, -26.2312260, -11.1767426, -10.9198952, 10.9380798
13: -14.1927643, 4.6803713, -14.1571503, 4.6609602, -13.3867531, 13.3646965
14: -24.1771507, -5.2071228, -24.1544247, -5.2179403, -16.3294754, 16.3168907
15: -7.6250477, 4.7047682, -7.5999317, 4.6923165, -11.3075447, 11.2910652
16: -7.6799068, 5.0087399, -7.6664152, 4.9938927, -9.4041367, 9.4579124
17: -26.7509518, -11.0899506, -26.7318077, -11.1131964, -11.0085564, 10.9862061
18: -17.6918755, -2.0013771, -17.6761761, -2.0608826, -10.5496445, 10.6293869
19: -10.4862280, -0.0523386, -10.4707479, -0.0723596, -6.8532753, 6.8669739
20: -5.8864245, 4.7244682, -5.8726301, 4.7139645, -7.5270081, 7.5317631
21: -8.5894413, 3.8402934, -8.5736704, 3.8187394, -9.6863251, 9.7089844
22: -10.7982445, 0.8487980, -10.7850504, 0.8286824, -7.8136215, 7.8441200
23: -4.6491675, 6.9304276, -4.6303196, 6.8999496, -8.7817307, 8.7981834
24: -8.0984755, 5.2327623, -8.0749111, 5.1924076, -10.5358505, 10.5675888
25: -8.3674736, 4.8692522, -8.3454599, 4.8465471, -8.3043137, 8.3189240
26: -16.5853786, 0.1776434, -16.5675182, 0.1453544, -11.6411133, 11.6787910
27: -7.8417740, 6.2214851, -7.8260550, 6.1996446, -12.1985245, 12.2196655
28: -6.5426550, 6.3458138, -6.5242186, 6.3200655, -10.1447067, 10.1552429
29: -7.7734418, 2.8921475, -7.7595453, 2.8677356, -8.6538658, 8.6687088
30: -3.8624532, 10.3368063, -3.8394992, 10.3059540, -12.2830887, 12.2950401
31: -14.8577127, -0.1455624, -14.8366947, -0.1838989, -10.7604790, 10.8014603
32: -20.8195992, -5.8335733, -20.8166466, -5.8508720, -12.0414352, 12.0565872
33: -38.5931091, -20.0009594, -38.5866623, -20.0101738, -10.6212196, 10.6216850
34: -35.7649078, -20.2356682, -35.7544250, -20.2705822, -11.5517921, 11.5712547
35: -33.0721359, -16.7744808, -33.0631218, -16.8010635, -11.3227196, 11.3386154
36: -31.1249733, -13.4978304, -31.1211224, -13.5164948, -12.6235847, 12.6427155
37: -50.2941017, -32.2497139, -50.2847061, -32.2812500, -9.9256401, 9.9368916
38: -38.7973671, -20.2287712, -38.7934265, -20.2473526, -11.3977203, 11.4098606
39: -42.5757370, -23.5568123, -42.5643616, -23.5596752, -10.6932602, 10.6733837
40: -38.0381317, -24.5421219, -38.0368767, -24.5574760, -7.9297371, 7.9638100
41: -24.8278770, -8.8760424, -24.8231907, -8.8970518, -13.1499023, 13.1848679
42: -15.1502371, -4.8150091, -15.1446438, -4.8187499, -8.1211929, 8.1259289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1785

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5512085, upper bound: 4.5571988
time: 25.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5512085, upper bound: 4.5691037
time: 36.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.3901978, 4.3198023, -13.3800468, 4.3130679, -15.6771011, 15.6785278
1: 0.4072506, 12.3419781, 0.4288137, 12.3408508, -9.2785721, 9.2576180
2: 2.0457368, 13.4896946, 2.0694561, 13.4882860, -9.0408669, 9.0139465
3: 1.5629778, 14.1843653, 1.5896006, 14.1821976, -9.0105209, 8.9911957
4: -4.2300887, 10.4683466, -4.1966658, 10.4658861, -12.6207733, 12.6019363
5: 2.0835509, 13.7759876, 2.1021709, 13.7751675, -8.4133606, 8.4040680
6: -25.1721058, -8.7647762, -25.1690464, -8.7845087, -13.3739243, 13.3968391
7: 2.5388787, 15.3037624, 2.5635462, 15.3040667, -9.3708115, 9.3429108
8: -4.5025616, 14.2550545, -4.4496436, 14.2542686, -15.7028122, 15.6405258
9: 0.5692430, 13.6051998, 0.5906808, 13.6041021, -9.2742271, 9.2609138
10: -4.4126596, 11.3094616, -4.4049444, 11.3058186, -12.0705223, 12.0650368
11: -4.4729819, 6.9257870, -4.4688549, 6.8974028, -8.5902901, 8.6113892
12: -26.2439499, -11.1461363, -26.2435665, -11.1714115, -10.9296455, 10.9475060
13: -14.1931190, 4.6876860, -14.1595316, 4.6817436, -13.4039574, 13.3759346
14: -24.1827850, -5.2070780, -24.1723804, -5.2179232, -16.3354263, 16.3356628
15: -7.6257010, 4.7102299, -7.6066842, 4.7079263, -11.3236122, 11.3036232
16: -7.6801729, 5.0118570, -7.6728983, 5.0026875, -9.4162788, 9.4615326
17: -26.7579594, -11.0899086, -26.7524242, -11.1059704, -11.0231934, 10.9947128
18: -17.7000771, -2.0008845, -17.6994877, -2.0523448, -10.5665321, 10.6435623
19: -10.4936695, -0.0522454, -10.4916344, -0.0657229, -6.8673153, 6.8772259
20: -5.8922472, 4.7246938, -5.8892374, 4.7191105, -7.5384140, 7.5413914
21: -8.5956764, 3.8405962, -8.5917883, 3.8230982, -9.6973763, 9.7212334
22: -10.8044205, 0.8488853, -10.8025789, 0.8349435, -7.8266411, 7.8501091
23: -4.6574969, 6.9306040, -4.6538496, 6.9074078, -8.7975197, 8.8147774
24: -8.1084232, 5.2328601, -8.1030302, 5.2016344, -10.5552216, 10.5844765
25: -8.3773403, 4.8694792, -8.3734264, 4.8563824, -8.3241196, 8.3335419
26: -16.5934010, 0.1778952, -16.5904789, 0.1532514, -11.6572189, 11.6902084
27: -7.8468981, 6.2217717, -7.8410230, 6.2022953, -12.2063599, 12.2336731
28: -6.5506392, 6.3461146, -6.5467858, 6.3282795, -10.1609726, 10.1694145
29: -7.7789335, 2.8921738, -7.7750473, 2.8724313, -8.6645660, 8.6764717
30: -3.8712268, 10.3371897, -3.8646243, 10.3131695, -12.2991104, 12.3148041
31: -14.8682137, -0.1453509, -14.8661089, -0.1751714, -10.7797089, 10.8188477
32: -20.8200417, -5.8325615, -20.8180046, -5.8467102, -12.0516548, 12.0592842
33: -38.5949173, -20.0001678, -38.5914154, -20.0073128, -10.6260948, 10.6267815
34: -35.7683525, -20.2355270, -35.7646980, -20.2665119, -11.5598221, 11.5820770
35: -33.0743561, -16.7740593, -33.0697441, -16.7968426, -11.3305206, 11.3456688
36: -31.1264305, -13.4972305, -31.1248360, -13.5128393, -12.6295853, 12.6482506
37: -50.2982445, -32.2490616, -50.2964058, -32.2754593, -9.9340515, 9.9455223
38: -38.7993851, -20.2282314, -38.7987213, -20.2445660, -11.4024239, 11.4193077
39: -42.5763474, -23.5563297, -42.5656471, -23.5583611, -10.6948280, 10.6759224
40: -38.0384827, -24.5411129, -38.0385170, -24.5543194, -7.9386864, 7.9666405
41: -24.8281403, -8.8743114, -24.8245621, -8.8910379, -13.1611404, 13.1869888
42: -15.1508064, -4.8141170, -15.1463842, -4.8157072, -8.1304054, 8.1284466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1785

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5691038, upper bound: 4.5571988
time: 26.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5691038, upper bound: 4.5691037
time: 20.41 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 49.22 seconds
IS_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 49.22
Output dim: 3, lower bound: -4.5571985, upper bound: 4.5307004
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 49.22
Output dim: 3, lower bound: -4.5691036, upper bound: 4.5307004
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 49.22
Output dim: 3, lower bound: -4.5334173, upper bound: 4.5567117
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 49.22
Output dim: 3, lower bound: -4.5334173, upper bound: 4.5686170
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 49.22
Output dim: 3, lower bound: -4.5334173, upper bound: 4.5567117
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 49.22
Output dim: 3, lower bound: -4.5334173, upper bound: 4.5686170
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 49.22
Output dim: 3, lower bound: -4.5512085, upper bound: 4.5571988
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 49.22
Output dim: 3, lower bound: -4.5512085, upper bound: 4.5691037
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 49.22
Output dim: 3, lower bound: -4.5691038, upper bound: 4.5571988
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 49.22
Output dim: 3, lower bound: -4.5691038, upper bound: 4.5691037

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.3517389, 4.2730937, -13.3978424, 4.3153696, -15.6520691, 15.6489143
1: 0.4504366, 12.2993231, 0.3872933, 12.3279743, -9.2279739, 9.2659760
2: 2.0907664, 13.4471855, 2.0237577, 13.4762383, -8.9867859, 9.0318985
3: 1.6126544, 14.1357803, 1.5375905, 14.1683922, -8.9544640, 9.0094452
4: -4.1724381, 10.4174595, -4.2649035, 10.4516869, -12.5496902, 12.6255188
5: 2.1209836, 13.7367744, 2.0679388, 13.7614574, -8.3680458, 8.4074249
6: -25.1587448, -8.7952070, -25.1707344, -8.7476034, -13.4073105, 13.3371201
7: 2.5881176, 15.2583218, 2.5127416, 15.2864256, -9.3080254, 9.3637924
8: -4.4111319, 14.1851835, -4.5588722, 14.2304535, -15.5902710, 15.6988449
9: 0.6171818, 13.5609312, 0.5487638, 13.5897579, -9.2186356, 9.2764816
10: -4.3865066, 11.2859097, -4.4184523, 11.3052883, -12.0217056, 12.0774269
11: -4.4296885, 6.8807778, -4.4626217, 6.9494023, -8.6115150, 8.5593948
12: -26.2288742, -11.1823654, -26.2372246, -11.1142578, -10.9734154, 10.9029655
13: -14.1413918, 4.6319942, -14.2203178, 4.6786623, -13.3439560, 13.3978996
14: -24.1408176, -5.2332573, -24.1847572, -5.1929798, -16.3050385, 16.3216515
15: -7.5920720, 4.6795101, -7.6405234, 4.7021537, -11.2572212, 11.3254318
16: -7.6576018, 4.9833369, -7.6867385, 5.0150228, -9.4668045, 9.4006386
17: -26.7276840, -11.1158371, -26.7518177, -11.0646200, -11.0354042, 10.9736633
18: -17.6436043, -2.0796905, -17.6821976, -1.9515104, -10.6396103, 10.5444679
19: -10.4445410, -0.0874984, -10.4786139, -0.0456316, -6.8573761, 6.8368950
20: -5.8564429, 4.7032471, -5.8848248, 4.7275405, -7.5164280, 7.5368347
21: -8.5445004, 3.8029706, -8.5829630, 3.8512805, -9.6934471, 9.6781387
22: -10.7551718, 0.8118110, -10.7897253, 0.8623197, -7.8302574, 7.8066940
23: -4.6070642, 6.8861403, -4.6445284, 6.9504938, -8.8044548, 8.7622986
24: -8.0444689, 5.1737070, -8.0925608, 5.2620153, -10.5759201, 10.5198898
25: -8.3109207, 4.8248491, -8.3575401, 4.8813677, -8.3148460, 8.2816925
26: -16.5457096, 0.1336348, -16.5812168, 0.2013466, -11.6786156, 11.6410370
27: -7.7958326, 6.1831245, -7.8386097, 6.2372856, -12.2162170, 12.1784286
28: -6.4970665, 6.3035975, -6.5364246, 6.3598614, -10.1527596, 10.1222572
29: -7.7328215, 2.8534930, -7.7678137, 2.9121103, -8.6737900, 8.6335030
30: -3.8183475, 10.2936487, -3.8600671, 10.3579931, -12.3078156, 12.2609100
31: -14.7948112, -0.2087681, -14.8440266, -0.1211729, -10.7915840, 10.7412376
32: -20.8141556, -5.8552866, -20.8210297, -5.8205528, -12.0714722, 12.0213699
33: -38.5817375, -20.0149174, -38.5943718, -19.9958134, -10.6286888, 10.6063652
34: -35.7366219, -20.2774067, -35.7651367, -20.2044773, -11.5928650, 11.5294418
35: -33.0434761, -16.8114948, -33.0701675, -16.7505322, -11.3500290, 11.3014603
36: -31.0999584, -13.5283813, -31.1192265, -13.4813662, -12.6437912, 12.6040649
37: -50.2825623, -32.2842102, -50.2934380, -32.2253571, -9.9713478, 9.9013195
38: -38.7636147, -20.2615414, -38.7889404, -20.2118168, -11.3920555, 11.3922710
39: -42.5635719, -23.5615177, -42.5896149, -23.5552864, -10.6807175, 10.6980190
40: -38.0353508, -24.5692024, -38.0377388, -24.5326996, -7.9657669, 7.9263783
41: -24.8202877, -8.9010534, -24.8309402, -8.8603258, -13.1948471, 13.1294708
42: -15.1473713, -4.8234625, -15.1552429, -4.8091960, -8.1351013, 8.1106949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 908

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5266689, upper bound: 4.5161167
time: 32.26 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5266689, upper bound: 4.5301400
time: 30.20 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.3984299, 4.3184676, -13.3581448, 4.2822700, -15.6602097, 15.6603394
1: 0.3874159, 12.3328075, 0.4438279, 12.3134441, -9.2806854, 9.2327538
2: 2.0236404, 13.4811535, 2.0812914, 13.4613724, -9.0464935, 8.9918442
3: 1.5375338, 14.1674004, 1.6086731, 14.1310129, -9.0042839, 8.9526176
4: -4.2649937, 10.4506998, -4.1724825, 10.4130030, -12.6209335, 12.5579071
5: 2.0678082, 13.7594404, 2.1204557, 13.7291527, -8.3994370, 8.3678932
6: -25.1720505, -8.7473555, -25.1604023, -8.7920351, -13.3401871, 13.4100800
7: 2.5127220, 15.2907677, 2.5802860, 15.2705593, -9.3756981, 9.3103790
8: -4.5589600, 14.2405968, -4.4301081, 14.2138596, -15.7267456, 15.6016312
9: 0.5486515, 13.5900307, 0.6154373, 13.5599747, -9.2754974, 9.2202873
10: -4.4189234, 11.3010082, -4.3774662, 11.2716141, -12.0638199, 12.0218163
11: -4.4647608, 6.9495516, -4.4344597, 6.8851848, -8.5677299, 8.6164551
12: -26.2311878, -11.1140814, -26.2100849, -11.1863537, -10.8983917, 10.9550476
13: -14.2170925, 4.6870070, -14.1459866, 4.6491871, -13.4148445, 13.3519325
14: -24.1808224, -5.1882229, -24.1316395, -5.2210979, -16.3374710, 16.3084030
15: -7.6409664, 4.6981049, -7.5878644, 4.6668315, -11.3182831, 11.2642822
16: -7.6866980, 5.0136948, -7.6531162, 4.9789162, -9.3885460, 9.4668159
17: -26.7402916, -11.0639009, -26.6911354, -11.1343603, -10.9753685, 11.0037994
18: -17.6916428, -1.9558902, -17.6661797, -2.0760927, -10.5361023, 10.6694374
19: -10.4827909, -0.0456426, -10.4563360, -0.0793889, -6.8424873, 6.8691406
20: -5.8880186, 4.7275620, -5.8651514, 4.7108946, -7.5310631, 7.5257359
21: -8.5906029, 3.8513014, -8.5653172, 3.8171887, -9.6833687, 9.7132301
22: -10.7960396, 0.8622675, -10.7734203, 0.8223825, -7.8039131, 7.8489685
23: -4.6456299, 6.9505630, -4.6086712, 6.8897724, -8.7647972, 8.8061905
24: -8.0959997, 5.2620950, -8.0519390, 5.1823893, -10.5199432, 10.5846138
25: -8.3591757, 4.8814535, -8.3137541, 4.8323822, -8.2805672, 8.3187046
26: -16.5861626, 0.2012370, -16.5600662, 0.1404176, -11.6330833, 11.6929474
27: -7.8452516, 6.2374492, -7.8145342, 6.1960764, -12.1844254, 12.2311363
28: -6.5375996, 6.3599138, -6.4994779, 6.3078346, -10.1262894, 10.1557884
29: -7.7701416, 2.9120941, -7.7388163, 2.8558292, -8.6361122, 8.6799278
30: -3.8607810, 10.3582602, -3.8176374, 10.2988300, -12.2691498, 12.3078651
31: -14.8558664, -0.1213136, -14.8260555, -0.1875460, -10.7499123, 10.8208466
32: -20.8207188, -5.8199749, -20.8140621, -5.8534622, -12.0220833, 12.0711746
33: -38.5917091, -19.9954948, -38.5749207, -20.0172195, -10.6115379, 10.6243362
34: -35.7685165, -20.2044106, -35.7512932, -20.2726154, -11.5378036, 11.6014366
35: -33.0716972, -16.7505512, -33.0488892, -16.8104668, -11.3105011, 11.3560104
36: -31.1195698, -13.4812107, -31.1018105, -13.5296288, -12.6031075, 12.6467934
37: -50.2860069, -32.2250366, -50.2599258, -32.2943878, -9.9011421, 9.9525452
38: -38.7926941, -20.2119007, -38.7774200, -20.2554512, -11.3816948, 11.4046631
39: -42.5843391, -23.5552025, -42.5491409, -23.5685825, -10.6925087, 10.6655846
40: -38.0377655, -24.5328903, -38.0328331, -24.5665131, -7.9229279, 7.9683189
41: -24.8299446, -8.8608799, -24.8166618, -8.9005547, -13.1310730, 13.1995926
42: -15.1506491, -4.8112779, -15.1299629, -4.8268876, -8.1091118, 8.1234818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 908

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5188733, upper bound: 4.5680603
time: 35.67 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5328593, upper bound: 4.5680603
time: 38.05 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3997421, 4.3258967, -13.3701458, 4.3031998, -15.6771469, 15.6804810
1: 0.3871610, 12.3404484, 0.4349515, 12.3349094, -9.2938995, 9.2496681
2: 2.0232320, 13.4871616, 2.0747542, 13.4783382, -9.0576782, 9.0047417
3: 1.5373988, 14.1765928, 1.5997598, 14.1567421, -9.0195847, 8.9707451
4: -4.2652788, 10.4590702, -4.1816578, 10.4366322, -12.6404648, 12.5754585
5: 2.0676010, 13.7670145, 2.1125708, 13.7503729, -8.4135513, 8.3833199
6: -25.1725922, -8.7460041, -25.1623039, -8.7873650, -13.3558502, 13.4139977
7: 2.5124955, 15.2979946, 2.5714283, 15.2908144, -9.3880157, 9.3264160
8: -4.5594292, 14.2506666, -4.4434128, 14.2421637, -15.7473221, 15.6248856
9: 0.5483177, 13.5966625, 0.6060832, 13.5785255, -9.2859612, 9.2360191
10: -4.4203606, 11.3013659, -4.3832669, 11.2727604, -12.0687218, 12.0296288
11: -4.4717426, 6.9497566, -4.4544520, 6.8910098, -8.5805283, 8.6312065
12: -26.2355556, -11.1136627, -26.2223969, -11.1810265, -10.9081421, 10.9644585
13: -14.2174549, 4.6942668, -14.1483135, 4.6699963, -13.4320526, 13.3631706
14: -24.1864586, -5.1881676, -24.1495266, -5.2210617, -16.3434067, 16.3271790
15: -7.6416292, 4.7035847, -7.5946531, 4.6824408, -11.3343315, 11.2768440
16: -7.6869731, 5.0167828, -7.6595740, 4.9876652, -9.4006615, 9.4704399
17: -26.7472954, -11.0638714, -26.7117920, -11.1271124, -10.9899979, 11.0123062
18: -17.6998253, -1.9553957, -17.6895332, -2.0675478, -10.5530014, 10.6835823
19: -10.4902382, -0.0455737, -10.4772148, -0.0727575, -6.8565254, 6.8793697
20: -5.8938446, 4.7278061, -5.8817730, 4.7160497, -7.5424767, 7.5353642
21: -8.5968456, 3.8515909, -8.5834570, 3.8215570, -9.6943893, 9.7254601
22: -10.8022194, 0.8623714, -10.7909441, 0.8286269, -7.8169250, 7.8549728
23: -4.6539588, 6.9507408, -4.6321983, 6.8972292, -8.7805634, 8.8228035
24: -8.1059608, 5.2621808, -8.0800762, 5.1915770, -10.5393372, 10.6015129
25: -8.3690434, 4.8816776, -8.3417320, 4.8421998, -8.3003883, 8.3333015
26: -16.5941963, 0.2013743, -16.5830402, 0.1482942, -11.6492310, 11.7043228
27: -7.8504009, 6.2377625, -7.8295126, 6.1987252, -12.1922226, 12.2451553
28: -6.5455914, 6.3602109, -6.5220785, 6.3160691, -10.1425552, 10.1699524
29: -7.7756433, 2.9121318, -7.7543211, 2.8604996, -8.6468239, 8.6877251
30: -3.8695481, 10.3586378, -3.8427474, 10.3060598, -12.2851562, 12.3276024
31: -14.8663616, -0.1210778, -14.8554535, -0.1788154, -10.7691154, 10.8382301
32: -20.8211536, -5.8189306, -20.8154144, -5.8493156, -12.0322838, 12.0738487
33: -38.5934830, -19.9946918, -38.5796585, -20.0143681, -10.6164131, 10.6294365
34: -35.7719955, -20.2042427, -35.7616005, -20.2685261, -11.5458221, 11.6122513
35: -33.0739059, -16.7500782, -33.0555420, -16.8063030, -11.3182831, 11.3630753
36: -31.1209888, -13.4806299, -31.1055450, -13.5259428, -12.6090813, 12.6523323
37: -50.2901688, -32.2244186, -50.2716217, -32.2886200, -9.9095497, 9.9611759
38: -38.7947121, -20.2113152, -38.7826614, -20.2526970, -11.3863792, 11.4140759
39: -42.5849686, -23.5547009, -42.5504112, -23.5672245, -10.6940765, 10.6681137
40: -38.0381050, -24.5318680, -38.0344696, -24.5633698, -7.9318771, 7.9711475
41: -24.8302097, -8.8591738, -24.8179512, -8.8945246, -13.1423187, 13.2017136
42: -15.1511869, -4.8103819, -15.1317186, -4.8238306, -8.1183357, 8.1259918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5367917, upper bound: 4.5680603
time: 23.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5367917, upper bound: 4.5680603
time: 25.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.3993645, 4.3219008, -13.3674335, 4.2918816, -15.6670837, 15.6703644
1: 0.3869619, 12.3347530, 0.4377828, 12.3189211, -9.2848816, 9.2391624
2: 2.0234246, 13.4843884, 2.0760016, 13.4707518, -9.0525742, 8.9998093
3: 1.5374262, 14.1761761, 1.5985603, 14.1558084, -9.0208664, 8.9719658
4: -4.2652826, 10.4607401, -4.1874542, 10.4412670, -12.6360474, 12.5830307
5: 2.0676274, 13.7681322, 2.1100893, 13.7534409, -8.4157143, 8.3874626
6: -25.1737061, -8.7467594, -25.1662483, -8.7892418, -13.3539543, 13.4199829
7: 2.5124474, 15.2951946, 2.5724294, 15.2831593, -9.3847237, 9.3228188
8: -4.5595121, 14.2442074, -4.4363441, 14.2244720, -15.7384720, 15.6128082
9: 0.5483537, 13.5989113, 0.6000540, 13.5849829, -9.2860107, 9.2443428
10: -4.4192829, 11.3126936, -4.3991032, 11.3044033, -12.0805550, 12.0544968
11: -4.4694190, 6.9497027, -4.4480820, 6.8915119, -8.5783386, 8.6219482
12: -26.2385139, -11.1137848, -26.2305012, -11.1768303, -10.9158058, 10.9704552
13: -14.2211132, 4.6874566, -14.1571350, 4.6600604, -13.4161034, 13.3675690
14: -24.1876450, -5.1881857, -24.1538658, -5.2181492, -16.3380051, 16.3287430
15: -7.6414032, 4.7066145, -7.5998068, 4.6918106, -11.3338966, 11.2851334
16: -7.6868482, 5.0190325, -7.6656790, 4.9937739, -9.3945808, 9.4865417
17: -26.7544556, -11.0638781, -26.7312145, -11.1133223, -11.0088692, 11.0126190
18: -17.6917915, -1.9506454, -17.6748066, -2.0609035, -10.5428276, 10.6784058
19: -10.4875412, -0.0455074, -10.4701252, -0.0723546, -6.8536415, 6.8744221
20: -5.8905931, 4.7277479, -5.8724923, 4.7135739, -7.5367661, 7.5327091
21: -8.5930176, 3.8516197, -8.5728531, 3.8187284, -9.6888962, 9.7200165
22: -10.7997856, 0.8623760, -10.7845783, 0.8286679, -7.8127251, 7.8574295
23: -4.6528335, 6.9507160, -4.6296911, 6.8999553, -8.7821312, 8.8197365
24: -8.1035233, 5.2621422, -8.0741158, 5.1924038, -10.5376282, 10.5988312
25: -8.3703527, 4.8815517, -8.3451328, 4.8465290, -8.3053131, 8.3316803
26: -16.5883541, 0.2015376, -16.5668831, 0.1453016, -11.6417770, 11.7003593
27: -7.8485718, 6.2376976, -7.8254719, 6.1996026, -12.1982117, 12.2425804
28: -6.5460100, 6.3601799, -6.5236268, 6.3200150, -10.1468315, 10.1692963
29: -7.7769423, 2.9121475, -7.7588792, 2.8677373, -8.6547050, 8.6885719
30: -3.8681715, 10.3584251, -3.8387952, 10.3059072, -12.2819748, 12.3221245
31: -14.8589745, -0.1207750, -14.8358297, -0.1839390, -10.7594872, 10.8268089
32: -20.8212719, -5.8192883, -20.8160172, -5.8509054, -12.0420837, 12.0730171
33: -38.5959320, -19.9952278, -38.5863991, -20.0102463, -10.6214714, 10.6301346
34: -35.7689667, -20.2039032, -35.7534676, -20.2706070, -11.5519295, 11.6032829
35: -33.0763435, -16.7502918, -33.0624619, -16.8011169, -11.3234863, 11.3623734
36: -31.1261139, -13.4809332, -31.1206131, -13.5165615, -12.6230011, 12.6599083
37: -50.2946587, -32.2248611, -50.2839813, -32.2812958, -9.9229317, 9.9629917
38: -38.7980537, -20.2116051, -38.7929535, -20.2473679, -11.3966026, 11.4155884
39: -42.5896912, -23.5549755, -42.5643234, -23.5600567, -10.7069588, 10.6724510
40: -38.0378342, -24.5300694, -38.0365105, -24.5575066, -7.9287357, 7.9758148
41: -24.8314285, -8.8603382, -24.8224983, -8.8970957, -13.1483078, 13.2063484
42: -15.1558342, -4.8110361, -15.1445732, -4.8189354, -8.1225014, 8.1307392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 908

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5366441, upper bound: 4.5685449
time: 44.22 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5188733, upper bound: 4.5685449
time: 35.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.3785744, 4.3176045, -13.3764830, 4.3124523, -15.6634216, 15.6721191
1: 0.4082072, 12.3316326, 0.4291067, 12.3378582, -9.2742729, 9.2464981
2: 2.0463107, 13.4762297, 2.0696154, 13.4843683, -9.0359001, 8.9992867
3: 1.5635180, 14.1685600, 1.5897377, 14.1776781, -9.0054131, 8.9752083
4: -4.2291160, 10.4431095, -4.1963940, 10.4587078, -12.6128845, 12.5764923
5: 2.0842409, 13.7643795, 2.1023638, 13.7718792, -8.4095573, 8.3923302
6: -25.1605721, -8.7660027, -25.1657600, -8.7848358, -13.3566818, 13.3909683
7: 2.5396085, 15.2853012, 2.5637553, 15.2988510, -9.3649750, 9.3243828
8: -4.5013032, 14.2161255, -4.4492884, 14.2431812, -15.6907730, 15.6013336
9: 0.5702922, 13.5908413, 0.5909882, 13.6000290, -9.2694817, 9.2464981
10: -4.4112926, 11.3040895, -4.4045672, 11.3043089, -12.0663300, 12.0549049
11: -4.4546957, 6.9251938, -4.4635482, 6.8972197, -8.5714836, 8.6055260
12: -26.2243843, -11.1474867, -26.2379417, -11.1718082, -10.9099693, 10.9410324
13: -14.1923313, 4.6664009, -14.1593027, 4.6756620, -13.3968506, 13.3535233
14: -24.1675148, -5.2094479, -24.1680298, -5.2185946, -16.3215790, 16.3282356
15: -7.6235600, 4.6990213, -7.6060624, 4.7046514, -11.3175240, 11.2864838
16: -7.6769943, 5.0107989, -7.6719484, 5.0023556, -9.4021492, 9.4559517
17: -26.7418327, -11.0913029, -26.7478561, -11.1063700, -11.0062828, 10.9879112
18: -17.6643372, -2.0016303, -17.6893616, -2.0525532, -10.5306320, 10.6326103
19: -10.4886055, -0.0524828, -10.4901171, -0.0657971, -6.8614368, 6.8753300
20: -5.8909111, 4.7238560, -5.8888588, 4.7188387, -7.5364952, 7.5382423
21: -8.5858917, 3.8401003, -8.5889320, 3.8229628, -9.6858521, 9.7172356
22: -10.7925568, 0.8487804, -10.7991590, 0.8349123, -7.8150978, 7.8465843
23: -4.6423903, 6.9300947, -4.6494832, 6.9072661, -8.7810707, 8.8094788
24: -8.0885134, 5.2325058, -8.0973740, 5.2015328, -10.5350227, 10.5780830
25: -8.3696527, 4.8687696, -8.3712358, 4.8562155, -8.3165283, 8.3307076
26: -16.5787468, 0.1771164, -16.5862751, 0.1530293, -11.6439133, 11.6854744
27: -7.8338857, 6.2209163, -7.8372722, 6.2020411, -12.1878281, 12.2273674
28: -6.5434804, 6.3452024, -6.5447283, 6.3280458, -10.1532860, 10.1664085
29: -7.7623925, 2.8920438, -7.7702727, 2.8723967, -8.6477737, 8.6714401
30: -3.8546181, 10.3361416, -3.8599033, 10.3128777, -12.2781448, 12.3076973
31: -14.8498726, -0.1461778, -14.8608665, -0.1753759, -10.7606316, 10.8126678
32: -20.8123035, -5.8337793, -20.8158150, -5.8470564, -12.0422478, 12.0557060
33: -38.5939178, -20.0013542, -38.5911446, -20.0077057, -10.6218643, 10.6238670
34: -35.7439041, -20.2359524, -35.7577286, -20.2666473, -11.5351715, 11.5745773
35: -33.0571632, -16.7746162, -33.0648270, -16.7969894, -11.3125114, 11.3398933
36: -31.1139774, -13.4977598, -31.1212540, -13.5129881, -12.6170578, 12.6440773
37: -50.2815170, -32.2495384, -50.2916031, -32.2755928, -9.9160843, 9.9398785
38: -38.7877731, -20.2286434, -38.7954178, -20.2447090, -11.3891907, 11.4111595
39: -42.5761223, -23.5658875, -42.5655594, -23.5610561, -10.6914215, 10.6653824
40: -38.0337296, -24.5423756, -38.0371056, -24.5546951, -7.9321747, 7.9637775
41: -24.8187675, -8.8752422, -24.8218575, -8.8912802, -13.1472626, 13.1819763
42: -15.1502752, -4.8169622, -15.1462278, -4.8165140, -8.1285172, 8.1258755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5367917, upper bound: 4.5566397
time: 52.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5685449, upper bound: 4.5566397
time: 28.13 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.4006701, 4.3293376, -13.3794422, 4.3128037, -15.6840134, 15.6905670
1: 0.3866916, 12.3423862, 0.4288578, 12.3403702, -9.2980995, 9.2560692
2: 2.0230002, 13.4903812, 2.0694799, 13.4877100, -9.0637703, 9.0127068
3: 1.5373058, 14.1853924, 1.5896039, 14.1815691, -9.0361595, 8.9901047
4: -4.2655754, 10.4691095, -4.1966252, 10.4649382, -12.6556244, 12.6006317
5: 2.0674050, 13.7757053, 2.1021826, 13.7747030, -8.4298325, 8.4028778
6: -25.1742439, -8.7454166, -25.1681423, -8.7845678, -13.3696136, 13.4239082
7: 2.5122292, 15.3024216, 2.5635598, 15.3033724, -9.3970375, 9.3388519
8: -4.5600047, 14.2542610, -4.4496369, 14.2527590, -15.7590332, 15.6360703
9: 0.5480280, 13.6055584, 0.5907304, 13.6035175, -9.2964554, 9.2600594
10: -4.4207134, 11.3130817, -4.4049120, 11.3055649, -12.0854912, 12.0622978
11: -4.4764085, 6.9498854, -4.4680924, 6.8973675, -8.5911598, 8.6367149
12: -26.2428360, -11.1133451, -26.2428246, -11.1714802, -10.9255333, 10.9798737
13: -14.2214680, 4.6947317, -14.1594925, 4.6808643, -13.4333267, 13.3788185
14: -24.1932602, -5.1881237, -24.1717911, -5.2181015, -16.3439407, 16.3475266
15: -7.6420665, 4.7120929, -7.6066313, 4.7074261, -11.3499565, 11.2976837
16: -7.6871309, 5.0221429, -7.6721745, 5.0025368, -9.4066925, 9.4901543
17: -26.7614670, -11.0638466, -26.7518368, -11.1061392, -11.0234947, 11.0211296
18: -17.6999969, -1.9501734, -17.6981125, -2.0523443, -10.5597267, 10.6925850
19: -10.4949732, -0.0454400, -10.4910259, -0.0657499, -6.8676987, 6.8846645
20: -5.8963957, 4.7279916, -5.8890834, 4.7187204, -7.5481758, 7.5423317
21: -8.5992699, 3.8519316, -8.5909758, 3.8230789, -9.6999207, 9.7322617
22: -10.8059521, 0.8624949, -10.8021059, 0.8349514, -7.8257446, 7.8634491
23: -4.6611681, 6.9509010, -4.6532259, 6.9073668, -8.7979012, 8.8363266
24: -8.1134901, 5.2622437, -8.1022396, 5.2016459, -10.5570183, 10.6157227
25: -8.3802214, 4.8817830, -8.3730736, 4.8563709, -8.3251305, 8.3462925
26: -16.5963745, 0.2016976, -16.5898457, 0.1531903, -11.6579018, 11.7117615
27: -7.8537140, 6.2379961, -7.8404446, 6.2022438, -12.2060165, 12.2565880
28: -6.5539846, 6.3604698, -6.5461679, 6.3282280, -10.1630859, 10.1834908
29: -7.7824364, 2.9121926, -7.7743959, 2.8724258, -8.6654015, 8.6963272
30: -3.8769174, 10.3587856, -3.8639359, 10.3131208, -12.2979889, 12.3418884
31: -14.8694715, -0.1205168, -14.8652439, -0.1751904, -10.7787018, 10.8441772
32: -20.8216934, -5.8182793, -20.8173485, -5.8467369, -12.0523148, 12.0757370
33: -38.5976868, -19.9944267, -38.5911674, -20.0073891, -10.6263390, 10.6352215
34: -35.7724457, -20.2037640, -35.7637291, -20.2665348, -11.5599556, 11.6141167
35: -33.0785751, -16.7498322, -33.0690765, -16.7968826, -11.3312721, 11.3694496
36: -31.1275520, -13.4803295, -31.1243172, -13.5128832, -12.6289825, 12.6654396
37: -50.2988052, -32.2242279, -50.2957001, -32.2754784, -9.9313316, 9.9716282
38: -38.8000755, -20.2110806, -38.7982101, -20.2446346, -11.4012985, 11.4250298
39: -42.5902939, -23.5544777, -42.5655899, -23.5586605, -10.7085152, 10.6749840
40: -38.0381813, -24.5290985, -38.0381393, -24.5543861, -7.9376812, 7.9786434
41: -24.8317413, -8.8586330, -24.8238411, -8.8910694, -13.1595535, 13.2084999
42: -15.1563883, -4.8101249, -15.1463375, -4.8158712, -8.1317215, 8.1332474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5545601, upper bound: 4.5685449
time: 41.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5367917, upper bound: 4.5685449
time: 25.80 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 69.41 seconds
IS_A1_B2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 69.41
Output dim: 3, lower bound: -4.5266689, upper bound: 4.5161167
IS_A1_B2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 69.41
Output dim: 3, lower bound: -4.5266689, upper bound: 4.5301400
IS_A2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 69.41
Output dim: 3, lower bound: -4.5188733, upper bound: 4.5680603
IS_A2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 69.41
Output dim: 3, lower bound: -4.5328593, upper bound: 4.5680603
IS_A2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 69.41
Output dim: 3, lower bound: -4.5367917, upper bound: 4.5680603
IS_A2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 69.41
Output dim: 3, lower bound: -4.5367917, upper bound: 4.5680603
IS_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 69.41
Output dim: 3, lower bound: -4.5366441, upper bound: 4.5685449
IS_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 69.41
Output dim: 3, lower bound: -4.5188733, upper bound: 4.5685449
IS_A2_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 69.41
Output dim: 3, lower bound: -4.5367917, upper bound: 4.5566397
IS_A2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 69.41
Output dim: 3, lower bound: -4.5685449, upper bound: 4.5566397
IS_A2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 69.41
Output dim: 3, lower bound: -4.5545601, upper bound: 4.5685449
IS_A2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 69.41
Output dim: 3, lower bound: -4.5367917, upper bound: 4.5685449

## BFS IS instance: IS_A2_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -13.3503380, 4.3024864, -13.3409576, 4.2820482, -15.6113892, 15.6267090
1: 0.4124126, 12.3243885, 0.4528081, 12.3131866, -9.2546806, 9.2145576
2: 2.0515747, 13.4712086, 2.0911570, 13.4610176, -9.0181808, 8.9720421
3: 1.5641754, 14.1581402, 1.6178703, 14.1307354, -8.9775124, 8.9340439
4: -4.2378745, 10.4415903, -4.1632490, 10.4124985, -12.5931931, 12.5396957
5: 2.0946584, 13.7501678, 2.1298838, 13.7288017, -8.3720436, 8.3489265
6: -25.1674461, -8.7650423, -25.1599903, -8.7978334, -13.3233566, 13.3899727
7: 2.5379536, 15.2827206, 2.5891242, 15.2703190, -9.3481522, 9.2921562
8: -4.5216293, 14.2299633, -4.4173250, 14.2136307, -15.6887512, 15.5770111
9: 0.5702448, 13.5856838, 0.6228769, 13.5597172, -9.2515411, 9.2043800
10: -4.3961458, 11.2967882, -4.3698864, 11.2710009, -12.0390320, 12.0054436
11: -4.4578352, 6.9351645, -4.4338193, 6.8802290, -8.5560875, 8.6016464
12: -26.2267914, -11.1271782, -26.2099266, -11.1902723, -10.8891106, 10.9411240
13: -14.1984901, 4.6752796, -14.1393833, 4.6484127, -13.3950539, 13.3337669
14: -24.1529713, -5.1960192, -24.1217232, -5.2214689, -16.3068161, 16.2866402
15: -7.6229849, 4.6910291, -7.5819969, 4.6663380, -11.2985992, 11.2483940
16: -7.6648760, 5.0082541, -7.6454940, 4.9786634, -9.3757210, 9.4617958
17: -26.7177486, -11.0720501, -26.6834717, -11.1345387, -10.9532700, 10.9865036
18: -17.6857300, -1.9709525, -17.6656837, -2.0811815, -10.5230255, 10.6528549
19: -10.4729643, -0.0666022, -10.4556847, -0.0868132, -6.8243713, 6.8465309
20: -5.8781614, 4.7001534, -5.8648539, 4.7010899, -7.5094528, 7.4967880
21: -8.5783558, 3.8227024, -8.5646715, 3.8071849, -9.6598015, 9.6829796
22: -10.7872620, 0.8376727, -10.7730370, 0.8137901, -7.7823410, 7.8216667
23: -4.6359062, 6.9293127, -4.6079187, 6.8824000, -8.7471237, 8.7840729
24: -8.0876904, 5.2414699, -8.0513058, 5.1752434, -10.5023575, 10.5631180
25: -8.3503914, 4.8565845, -8.3134117, 4.8236732, -8.2612839, 8.2925663
26: -16.5736485, 0.1710558, -16.5593643, 0.1295605, -11.6066437, 11.6610184
27: -7.8338900, 6.2122164, -7.8137503, 6.1872315, -12.1632690, 12.2045746
28: -6.5255666, 6.3295383, -6.4987049, 6.2970638, -10.1020737, 10.1239014
29: -7.7631121, 2.8929343, -7.7382145, 2.8492754, -8.6214790, 8.6598244
30: -3.8497431, 10.3277416, -3.8170216, 10.2883472, -12.2481384, 12.2772408
31: -14.8470535, -0.1404417, -14.8253460, -0.1940441, -10.7317696, 10.7995224
32: -20.8160210, -5.8323336, -20.8134556, -5.8574266, -12.0107651, 12.0567322
33: -38.5879669, -20.0142307, -38.5742607, -20.0236282, -10.5984306, 10.6037979
34: -35.7609863, -20.2248669, -35.7504387, -20.2797966, -11.5237465, 11.5802078
35: -33.0668297, -16.7702312, -33.0481987, -16.8171940, -11.2965126, 11.3335686
36: -31.1134758, -13.5060520, -31.1014099, -13.5383558, -12.5860596, 12.6199417
37: -50.2837906, -32.2373047, -50.2594452, -32.2984123, -9.8924370, 9.9386425
38: -38.7854767, -20.2329159, -38.7769241, -20.2629128, -11.3602943, 11.3796425
39: -42.5827637, -23.5627556, -42.5483894, -23.5710430, -10.6876717, 10.6581650
40: -38.0311737, -24.5375214, -38.0305634, -24.5674171, -7.9157314, 7.9651585
41: -24.8243332, -8.8740587, -24.8158417, -8.9048786, -13.1199722, 13.1848564
42: -15.1490250, -4.8145266, -15.1292381, -4.8277879, -8.1025848, 8.1191177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 638

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5180440, upper bound: 4.5436863
time: 36.17 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5171484, upper bound: 4.5665641
time: 32.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -13.3976154, 4.3184347, -13.3580685, 4.2822437, -15.6441650, 15.6602631
1: 0.3882303, 12.3327885, 0.4438739, 12.3134527, -9.2750473, 9.2326775
2: 2.0245504, 13.4811029, 2.0813541, 13.4613628, -9.0373955, 8.9917526
3: 1.5383368, 14.1673651, 1.6087418, 14.1310139, -8.9930153, 8.9525185
4: -4.2640648, 10.4506159, -4.1723976, 10.4129982, -12.6169815, 12.5577431
5: 2.0684075, 13.7593889, 2.1205235, 13.7291603, -8.3865013, 8.3678055
6: -25.1720161, -8.7485218, -25.1604042, -8.7921181, -13.3456421, 13.4091568
7: 2.5133243, 15.2907333, 2.5803242, 15.2705622, -9.3651199, 9.3102646
8: -4.5574131, 14.2405357, -4.4300108, 14.2138748, -15.7187119, 15.6015015
9: 0.5500369, 13.5900126, 0.6155267, 13.5599623, -9.2666702, 9.2202148
10: -4.4167852, 11.3009272, -4.3773174, 11.2715979, -12.0545044, 12.0216827
11: -4.4646726, 6.9489436, -4.4344578, 6.8851256, -8.5675888, 8.6145210
12: -26.2311497, -11.1149101, -26.2100639, -11.1864109, -10.8982773, 10.9519882
13: -14.2166309, 4.6869106, -14.1459713, 4.6491804, -13.4096031, 13.3517952
14: -24.1796093, -5.1882572, -24.1315422, -5.2210703, -16.3328400, 16.3082924
15: -7.6400294, 4.6980429, -7.5878015, 4.6668181, -11.3175125, 11.2673378
16: -7.6852350, 5.0136395, -7.6529942, 4.9789181, -9.3842812, 9.4665642
17: -26.7394066, -11.0639114, -26.6910629, -11.1343603, -10.9652748, 11.0036469
18: -17.6915302, -1.9568701, -17.6661854, -2.0761743, -10.5360565, 10.6668816
19: -10.4827137, -0.0464015, -10.4563351, -0.0794420, -6.8423939, 6.8598118
20: -5.8879900, 4.7266860, -5.8651648, 4.7108440, -7.5309753, 7.5098495
21: -8.5905190, 3.8503704, -8.5653172, 3.8171182, -9.6832275, 9.7001839
22: -10.7959900, 0.8612959, -10.7734356, 0.8222597, -7.8038445, 7.8287468
23: -4.6455088, 6.9496198, -4.6086712, 6.8897042, -8.7646179, 8.7965355
24: -8.0958815, 5.2610602, -8.0519352, 5.1823244, -10.5197601, 10.5709000
25: -8.3591270, 4.8804379, -8.3137465, 4.8322916, -8.2804642, 8.3020897
26: -16.5860596, 0.2002857, -16.5600395, 0.1403290, -11.6329956, 11.6707993
27: -7.8451662, 6.2363296, -7.8145337, 6.1959844, -12.1842651, 12.2221069
28: -6.5375099, 6.3590169, -6.4994740, 6.3077335, -10.1261444, 10.1409683
29: -7.7700586, 2.9111018, -7.7388096, 2.8557489, -8.6359596, 8.6689873
30: -3.8607211, 10.3570986, -3.8176329, 10.2987432, -12.2689896, 12.2946854
31: -14.8558159, -0.1223795, -14.8260460, -0.1876326, -10.7497826, 10.8171234
32: -20.8206787, -5.8207712, -20.8140602, -5.8535128, -12.0239372, 12.0706100
33: -38.5915985, -19.9969006, -38.5749321, -20.0173321, -10.6114388, 10.6150475
34: -35.7683792, -20.2052860, -35.7512474, -20.2726803, -11.5376663, 11.5913200
35: -33.0715904, -16.7518787, -33.0488968, -16.8105717, -11.3103561, 11.3439026
36: -31.1195183, -13.4824247, -31.1018200, -13.5296936, -12.6030006, 12.6343918
37: -50.2859230, -32.2263184, -50.2599258, -32.2944870, -9.9010620, 9.9474545
38: -38.7926712, -20.2129059, -38.7774124, -20.2555256, -11.3816071, 11.3840103
39: -42.5842819, -23.5562172, -42.5491409, -23.5686588, -10.6923904, 10.6621304
40: -38.0373573, -24.5330181, -38.0328064, -24.5665169, -7.9334564, 7.9659138
41: -24.8298607, -8.8616228, -24.8166180, -8.9006138, -13.1322479, 13.1990242
42: -15.1505871, -4.8114996, -15.1299648, -4.8269014, -8.1135941, 8.1230125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 908

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 638

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5180440, upper bound: 4.5436863
time: 32.16 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5311536, upper bound: 4.5665641
time: 24.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -13.3516378, 4.3099332, -13.3529720, 4.3029585, -15.6283264, 15.6469154
1: 0.4121382, 12.3320236, 0.4439468, 12.3346462, -9.2678986, 9.2314529
2: 2.0511527, 13.4772329, 2.0846338, 13.4779778, -9.0293350, 8.9849319
3: 1.5640390, 14.1673317, 1.6089156, 14.1564655, -8.9928131, 8.9522018
4: -4.2381816, 10.4499521, -4.1724663, 10.4361525, -12.6127243, 12.5572357
5: 2.0944345, 13.7577248, 2.1219945, 13.7500362, -8.3861732, 8.3643417
6: -25.1679764, -8.7636890, -25.1618786, -8.7931385, -13.3390465, 13.3939285
7: 2.5377269, 15.2899685, 2.5802693, 15.2905664, -9.3604698, 9.3081856
8: -4.5221257, 14.2400036, -4.4305868, 14.2419214, -15.7092972, 15.6002121
9: 0.5699477, 13.5923138, 0.6135249, 13.5782385, -9.2620010, 9.2201004
10: -4.3976121, 11.2971754, -4.3756657, 11.2721634, -12.0439835, 12.0132637
11: -4.4648104, 6.9353304, -4.4538088, 6.8860769, -8.5688972, 8.6163902
12: -26.2311382, -11.1267958, -26.2222137, -11.1848974, -10.8988419, 10.9505424
13: -14.1987972, 4.6825800, -14.1417332, 4.6691904, -13.4122658, 13.3449974
14: -24.1585636, -5.1959801, -24.1396751, -5.2214499, -16.3127594, 16.3054237
15: -7.6236362, 4.6964941, -7.5888000, 4.6819191, -11.3146896, 11.2609558
16: -7.6651583, 5.0113721, -7.6519871, 4.9874430, -9.3878326, 9.4654121
17: -26.7247524, -11.0720034, -26.7040920, -11.1272898, -10.9679108, 10.9950066
18: -17.6939163, -1.9704752, -17.6890182, -2.0726223, -10.5399055, 10.6669998
19: -10.4804173, -0.0665078, -10.4765797, -0.0801930, -6.8384171, 6.8567696
20: -5.8839512, 4.7003813, -5.8814640, 4.7062235, -7.5208588, 7.5064144
21: -8.5846138, 3.8230040, -8.5828009, 3.8115363, -9.6708412, 9.6952286
22: -10.7934532, 0.8377471, -10.7905331, 0.8200743, -7.7953529, 7.8276768
23: -4.6442380, 6.9294786, -4.6314259, 6.8898544, -8.7629013, 8.8006706
24: -8.0976343, 5.2415800, -8.0794344, 5.1844778, -10.5217438, 10.5799980
25: -8.3602705, 4.8567858, -8.3413782, 4.8334999, -8.2811012, 8.3071709
26: -16.5817108, 0.1712633, -16.5823593, 0.1374252, -11.6227608, 11.6724091
27: -7.8390818, 6.2125134, -7.8287125, 6.1898575, -12.1710892, 12.2185783
28: -6.5334978, 6.3298321, -6.5212793, 6.3052673, -10.1183357, 10.1380997
29: -7.7686119, 2.8929889, -7.7537270, 2.8539324, -8.6321754, 8.6676178
30: -3.8584886, 10.3281088, -3.8421521, 10.2955303, -12.2641830, 12.2970047
31: -14.8575411, -0.1401913, -14.8547516, -0.1852884, -10.7509766, 10.8168869
32: -20.8164444, -5.8312769, -20.8147545, -5.8533106, -12.0209885, 12.0594330
33: -38.5897408, -20.0134392, -38.5789795, -20.0207558, -10.6033173, 10.6089058
34: -35.7645149, -20.2247162, -35.7607269, -20.2757092, -11.5317421, 11.5909882
35: -33.0690765, -16.7697697, -33.0548019, -16.8129921, -11.3043060, 11.3406296
36: -31.1149235, -13.5054445, -31.1051121, -13.5346813, -12.5920563, 12.6254768
37: -50.2879524, -32.2366447, -50.2711449, -32.2925758, -9.9008636, 9.9472733
38: -38.7874832, -20.2323647, -38.7822113, -20.2601719, -11.3649826, 11.3890839
39: -42.5833817, -23.5622692, -42.5496788, -23.5696774, -10.6892395, 10.6606979
40: -38.0315094, -24.5365086, -38.0322113, -24.5643044, -7.9246655, 7.9679756
41: -24.8246269, -8.8723125, -24.8171558, -8.8988466, -13.1312027, 13.1870461
42: -15.1495943, -4.8136396, -15.1309719, -4.8247232, -8.1118355, 8.1216469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 638

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5124062, upper bound: 4.5674474
time: 37.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5352724, upper bound: 4.5665644
time: 29.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -13.3989410, 4.3258791, -13.3700962, 4.3031898, -15.6610947, 15.6804123
1: 0.3879468, 12.3404236, 0.4350054, 12.3349018, -9.2882614, 9.2495689
2: 2.0241199, 13.4871178, 2.0748177, 13.4783440, -9.0485878, 9.0046463
3: 1.5382175, 14.1765747, 1.5998068, 14.1567507, -9.0083046, 8.9706497
4: -4.2643538, 10.4589863, -4.1815805, 10.4366360, -12.6365356, 12.5753174
5: 2.0681565, 13.7669630, 2.1126242, 13.7503796, -8.4006271, 8.3832283
6: -25.1725521, -8.7471561, -25.1623001, -8.7874384, -13.3613052, 13.4130630
7: 2.5130847, 15.2979450, 2.5714736, 15.2908249, -9.3774071, 9.3263168
8: -4.5579157, 14.2506056, -4.4433007, 14.2421455, -15.7392654, 15.6247559
9: 0.5497205, 13.5966291, 0.6061873, 13.5785141, -9.2771339, 9.2359390
10: -4.4182224, 11.3013077, -4.3831038, 11.2727594, -12.0594292, 12.0295105
11: -4.4716635, 6.9491329, -4.4544363, 6.8909650, -8.5803909, 8.6292801
12: -26.2355347, -11.1145020, -26.2223911, -11.1810780, -10.9080276, 10.9613953
13: -14.2169476, 4.6941853, -14.1482687, 4.6699848, -13.4268417, 13.3630562
14: -24.1852207, -5.1882010, -24.1494217, -5.2210445, -16.3387756, 16.3270721
15: -7.6406584, 4.7035222, -7.5945816, 4.6824431, -11.3335991, 11.2798691
16: -7.6855183, 5.0167575, -7.6594601, 4.9876595, -9.3964195, 9.4701729
17: -26.7464256, -11.0638819, -26.7117290, -11.1271124, -10.9799194, 11.0121574
18: -17.6997604, -1.9563770, -17.6895008, -2.0676079, -10.5529099, 10.6810341
19: -10.4901676, -0.0463333, -10.4772291, -0.0728261, -6.8564377, 6.8700504
20: -5.8937931, 4.7269144, -5.8817644, 4.7159982, -7.5423775, 7.5194817
21: -8.5967855, 3.8506813, -8.5834351, 3.8214850, -9.6942482, 9.7124290
22: -10.8021851, 0.8613918, -10.7909451, 0.8285041, -7.8168564, 7.8347454
23: -4.6538382, 6.9498119, -4.6321831, 6.8971539, -8.7803917, 8.8131142
24: -8.1058512, 5.2611532, -8.0800505, 5.1915107, -10.5391464, 10.5877800
25: -8.3689947, 4.8806529, -8.3417253, 4.8421245, -8.3002968, 8.3167000
26: -16.5941410, 0.2004389, -16.5830326, 0.1482282, -11.6491394, 11.6821747
27: -7.8503075, 6.2366271, -7.8295207, 6.1986551, -12.1920776, 12.2361221
28: -6.5454798, 6.3592973, -6.5220594, 6.3159680, -10.1424179, 10.1551743
29: -7.7755380, 2.9111464, -7.7543221, 2.8604269, -8.6466713, 8.6767502
30: -3.8694520, 10.3574820, -3.8427532, 10.3059578, -12.2850266, 12.3144608
31: -14.8663034, -0.1221383, -14.8554287, -0.1789017, -10.7689972, 10.8345070
32: -20.8210983, -5.8197498, -20.8154259, -5.8493671, -12.0341644, 12.0733223
33: -38.5933723, -19.9961166, -38.5796432, -20.0144634, -10.6163101, 10.6201515
34: -35.7718735, -20.2051296, -35.7615891, -20.2685814, -11.5456696, 11.6021271
35: -33.0738068, -16.7514057, -33.0555420, -16.8063965, -11.3181458, 11.3509712
36: -31.1209583, -13.4818115, -31.1055260, -13.5260181, -12.6089859, 12.6399078
37: -50.2901154, -32.2256927, -50.2716141, -32.2887001, -9.9094810, 9.9560890
38: -38.7946930, -20.2123375, -38.7826500, -20.2527657, -11.3863029, 11.3934383
39: -42.5849075, -23.5557442, -42.5503922, -23.5672836, -10.6939468, 10.6646404
40: -38.0376968, -24.5320168, -38.0344505, -24.5633965, -7.9423866, 7.9687462
41: -24.8301315, -8.8598804, -24.8179283, -8.8945961, -13.1435013, 13.2011566
42: -15.1511555, -4.8106108, -15.1317129, -4.8238516, -8.1228104, 8.1255379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 638

## Relational analysis of IS_A2_B1_A2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5124062, upper bound: 4.5674474
time: 32.07 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5124062, upper bound: 4.5665644
time: 24.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -13.3512516, 4.3059587, -13.3502674, 4.2916384, -15.6182556, 15.6367950
1: 0.4119368, 12.3263187, 0.4467731, 12.3186569, -9.2588844, 9.2209740
2: 2.0513635, 13.4744606, 2.0858858, 13.4703903, -9.0242538, 8.9800072
3: 1.5640860, 14.1669083, 1.6077158, 14.1555157, -8.9941025, 8.9534187
4: -4.2381802, 10.4516325, -4.1782417, 10.4407873, -12.6083603, 12.5648308
5: 2.0944500, 13.7588539, 2.1194992, 13.7531052, -8.3883438, 8.3684883
6: -25.1691151, -8.7644348, -25.1658230, -8.7949867, -13.3371048, 13.3998604
7: 2.5376658, 15.2871571, 2.5812626, 15.2828741, -9.3571815, 9.3045921
8: -4.5222178, 14.2335892, -4.4235015, 14.2242136, -15.7004776, 15.5881729
9: 0.5699496, 13.5945721, 0.6075048, 13.5847340, -9.2620201, 9.2284203
10: -4.3965254, 11.3084898, -4.3915281, 11.3037853, -12.0557785, 12.0381050
11: -4.4624805, 6.9352884, -4.4474354, 6.8865666, -8.5667000, 8.6071510
12: -26.2341022, -11.1269608, -26.2303314, -11.1807070, -10.9065247, 10.9565277
13: -14.2024689, 4.6757455, -14.1505890, 4.6592665, -13.3963470, 13.3494110
14: -24.1598015, -5.1959763, -24.1440029, -5.2185755, -16.3073730, 16.3069305
15: -7.6233916, 4.6995454, -7.5939507, 4.6913419, -11.3142471, 11.2692337
16: -7.6650405, 5.0136242, -7.6580672, 4.9935198, -9.3817558, 9.4815178
17: -26.7319469, -11.0720034, -26.7235603, -11.1135178, -10.9867592, 10.9953232
18: -17.6859016, -1.9657264, -17.6743259, -2.0659819, -10.5297394, 10.6618462
19: -10.4777012, -0.0664287, -10.4694901, -0.0798235, -6.8355255, 6.8518219
20: -5.8806987, 4.7003388, -5.8721890, 4.7037482, -7.5151558, 7.5037479
21: -8.5807676, 3.8230541, -8.5721874, 3.8087192, -9.6653214, 9.6897621
22: -10.7910280, 0.8377814, -10.7841663, 0.8200614, -7.7911758, 7.8301315
23: -4.6430974, 6.9294844, -4.6289663, 6.8925624, -8.7644653, 8.7975769
24: -8.0952148, 5.2415171, -8.0734854, 5.1852818, -10.5200424, 10.5773201
25: -8.3615856, 4.8566604, -8.3447838, 4.8378439, -8.2860260, 8.3055477
26: -16.5758076, 0.1713578, -16.5661774, 0.1344686, -11.6153221, 11.6684570
27: -7.8372235, 6.2124457, -7.8247013, 6.1907678, -12.1770248, 12.2160110
28: -6.5339499, 6.3298082, -6.5228500, 6.3092318, -10.1226311, 10.1374054
29: -7.7699180, 2.8930271, -7.7582502, 2.8611834, -8.6400757, 8.6684685
30: -3.8571057, 10.3279009, -3.8381999, 10.2953911, -12.2609863, 12.2914925
31: -14.8501396, -0.1399055, -14.8351345, -0.1903963, -10.7413483, 10.8054314
32: -20.8165512, -5.8316717, -20.8154144, -5.8549042, -12.0307732, 12.0585747
33: -38.5921860, -20.0139618, -38.5857239, -20.0166664, -10.6083641, 10.6096058
34: -35.7614822, -20.2243767, -35.7526016, -20.2778244, -11.5378914, 11.5820122
35: -33.0715065, -16.7699680, -33.0617943, -16.8078442, -11.3094940, 11.3399467
36: -31.1200104, -13.5057621, -31.1201763, -13.5252781, -12.6059570, 12.6330605
37: -50.2924347, -32.2370834, -50.2835007, -32.2853317, -9.9142227, 9.9490967
38: -38.7908173, -20.2326641, -38.7924995, -20.2548752, -11.3751945, 11.3905907
39: -42.5880775, -23.5625172, -42.5635910, -23.5624771, -10.7021141, 10.6650372
40: -38.0312424, -24.5346966, -38.0342827, -24.5584202, -7.9215317, 7.9726524
41: -24.8258476, -8.8735342, -24.8216896, -8.9014244, -13.1371994, 13.1916656
42: -15.1542196, -4.8142738, -15.1438570, -4.8198175, -8.1159897, 8.1263752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 638

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5358175, upper bound: 4.5441715
time: 33.33 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5171484, upper bound: 4.5665641
time: 33.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -13.3985624, 4.3218837, -13.3673697, 4.2918777, -15.6510468, 15.6703453
1: 0.3877518, 12.3347273, 0.4378293, 12.3189220, -9.2792511, 9.2390823
2: 2.0243170, 13.4843531, 2.0760880, 13.4707413, -9.0434952, 8.9997101
3: 1.5382357, 14.1761494, 1.5986059, 14.1557941, -9.0095940, 8.9718742
4: -4.2643557, 10.4606724, -4.1873779, 10.4412699, -12.6321335, 12.5828857
5: 2.0681958, 13.7680883, 2.1101511, 13.7534380, -8.4027863, 8.3873596
6: -25.1736431, -8.7479095, -25.1662426, -8.7893324, -13.3593788, 13.4190369
7: 2.5130572, 15.2951612, 2.5724826, 15.2831535, -9.3741264, 9.3227234
8: -4.5579901, 14.2441463, -4.4362326, 14.2244701, -15.7304459, 15.6126938
9: 0.5497305, 13.5988865, 0.6001353, 13.5849857, -9.2771759, 9.2442665
10: -4.4171252, 11.3126202, -4.3989468, 11.3044033, -12.0712585, 12.0543594
11: -4.4692936, 6.9490809, -4.4480791, 6.8914833, -8.5782089, 8.6200333
12: -26.2384624, -11.1146727, -26.2304878, -11.1769028, -10.9156990, 10.9673996
13: -14.2206059, 4.6873374, -14.1570978, 4.6600695, -13.4108963, 13.3674583
14: -24.1864471, -5.1882200, -24.1537781, -5.2181330, -16.3333817, 16.3286362
15: -7.6404657, 4.7065516, -7.5997348, 4.6918120, -11.3331528, 11.2881546
16: -7.6853690, 5.0189934, -7.6655722, 4.9937592, -9.3903084, 9.4862633
17: -26.7536030, -11.0638800, -26.7311554, -11.1133327, -10.9987602, 11.0124741
18: -17.6917706, -1.9516010, -17.6747932, -2.0609932, -10.5427551, 10.6758766
19: -10.4874630, -0.0462596, -10.4701300, -0.0724239, -6.8535538, 6.8650990
20: -5.8905334, 4.7268729, -5.8724895, 4.7135158, -7.5366631, 7.5168247
21: -8.5929337, 3.8506894, -8.5728464, 3.8186562, -9.6887627, 9.7069702
22: -10.7997360, 0.8614192, -10.7845783, 0.8285451, -7.8126602, 7.8372192
23: -4.6527138, 6.9497805, -4.6296988, 6.8998947, -8.7819519, 8.8100357
24: -8.1033993, 5.2611303, -8.0741014, 5.1923380, -10.5374374, 10.5851097
25: -8.3703175, 4.8805385, -8.3451328, 4.8464522, -8.3052101, 8.3150768
26: -16.5882378, 0.2005724, -16.5668488, 0.1452355, -11.6416740, 11.6782341
27: -7.8484707, 6.2365642, -7.8254642, 6.1995397, -12.1980896, 12.2335434
28: -6.5458984, 6.3592682, -6.5236149, 6.3199134, -10.1467056, 10.1544800
29: -7.7768373, 2.9111707, -7.7588568, 2.8676419, -8.6545486, 8.6776199
30: -3.8680801, 10.3572721, -3.8387976, 10.3058214, -12.2818222, 12.3089485
31: -14.8589001, -0.1218171, -14.8358440, -0.1840258, -10.7593765, 10.8230629
32: -20.8212128, -5.8200760, -20.8160172, -5.8509722, -12.0439529, 12.0724640
33: -38.5958061, -19.9966164, -38.5863953, -20.0103588, -10.6213684, 10.6208324
34: -35.7688560, -20.2047634, -35.7534599, -20.2706776, -11.5518036, 11.5931931
35: -33.0762558, -16.7516193, -33.0624390, -16.8012142, -11.3233490, 11.3502846
36: -31.1260529, -13.4821224, -31.1206188, -13.5166569, -12.6229095, 12.6475105
37: -50.2945824, -32.2261429, -50.2839928, -32.2814255, -9.9228477, 9.9578972
38: -38.7980194, -20.2126617, -38.7929459, -20.2474403, -11.3965263, 11.3949490
39: -42.5896301, -23.5559769, -42.5643120, -23.5601196, -10.7068291, 10.6690102
40: -38.0374374, -24.5301952, -38.0364761, -24.5575294, -7.9392586, 7.9734173
41: -24.8313904, -8.8610964, -24.8225021, -8.8971472, -13.1494827, 13.2057915
42: -15.1557732, -4.8112736, -15.1445751, -4.8189425, -8.1269836, 8.1302605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 638

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5498349, upper bound: 4.5441715
time: 47.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5489454, upper bound: 4.5670508
time: 27.54 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -13.3777952, 4.3175554, -13.3764038, 4.3124642, -15.6473465, 15.6720657
1: 0.4090285, 12.3316059, 0.4291761, 12.3378592, -9.2686424, 9.2464104
2: 2.0471802, 13.4761772, 2.0696864, 13.4843731, -9.0267982, 8.9991951
3: 1.5643234, 14.1685333, 1.5897918, 14.1776781, -8.9941444, 8.9751091
4: -4.2281857, 10.4430323, -4.1963100, 10.4587030, -12.6089478, 12.5763359
5: 2.0848255, 13.7643433, 2.1024323, 13.7718611, -8.3966293, 8.3922348
6: -25.1605339, -8.7671432, -25.1657372, -8.7849188, -13.3621483, 13.3900452
7: 2.5402062, 15.2852669, 2.5638239, 15.2988386, -9.3543739, 9.3242798
8: -4.4997988, 14.2160788, -4.4491615, 14.2431793, -15.6827393, 15.6012344
9: 0.5716808, 13.5908241, 0.5910773, 13.6000080, -9.2606392, 9.2464218
10: -4.4091315, 11.3040447, -4.4044266, 11.3043232, -12.0570412, 12.0548019
11: -4.4545832, 6.9245787, -4.4635458, 6.8971834, -8.5713615, 8.6036148
12: -26.2243385, -11.1483498, -26.2379436, -11.1718597, -10.9098625, 10.9379616
13: -14.1918707, 4.6663399, -14.1592426, 4.6756577, -13.3916435, 13.3533974
14: -24.1663094, -5.2094593, -24.1679115, -5.2185850, -16.3169632, 16.3281021
15: -7.6225901, 4.6989427, -7.6060138, 4.7046380, -11.3167877, 11.2895088
16: -7.6755095, 5.0107732, -7.6718330, 5.0023499, -9.3979187, 9.4556885
17: -26.7409821, -11.0913200, -26.7478065, -11.1063700, -10.9961929, 10.9877663
18: -17.6643162, -2.0025949, -17.6893578, -2.0526195, -10.5305634, 10.6300468
19: -10.4885187, -0.0532422, -10.4901161, -0.0658512, -6.8613491, 6.8659992
20: -5.8908496, 4.7229738, -5.8888574, 4.7187877, -7.5364037, 7.5223656
21: -8.5857964, 3.8391771, -8.5889244, 3.8228908, -9.6857071, 9.7041855
22: -10.7925129, 0.8477931, -10.7991524, 0.8347819, -7.8150177, 7.8263626
23: -4.6422758, 6.9291658, -4.6494598, 6.9071970, -8.7808990, 8.7998047
24: -8.0883942, 5.2314835, -8.0973549, 5.2014661, -10.5348358, 10.5643768
25: -8.3696098, 4.8677602, -8.3712215, 4.8561239, -8.3164330, 8.3140869
26: -16.5786648, 0.1761760, -16.5862503, 0.1529330, -11.6438179, 11.6633148
27: -7.8337507, 6.2197967, -7.8372645, 6.2019553, -12.1876907, 12.2183266
28: -6.5433898, 6.3443208, -6.5447006, 6.3279605, -10.1531563, 10.1515923
29: -7.7623005, 2.8910670, -7.7702594, 2.8723238, -8.6476097, 8.6604881
30: -3.8545315, 10.3349962, -3.8599141, 10.3127766, -12.2779999, 12.2945213
31: -14.8498354, -0.1472368, -14.8608513, -0.1754713, -10.7605324, 10.8089333
32: -20.8122406, -5.8345718, -20.8157921, -5.8470984, -12.0441170, 12.0551338
33: -38.5938492, -20.0027676, -38.5911407, -20.0077858, -10.6217575, 10.6145840
34: -35.7437820, -20.2368450, -35.7577248, -20.2667160, -11.5350342, 11.5644646
35: -33.0570145, -16.7759304, -33.0648193, -16.7970848, -11.3123894, 11.3278008
36: -31.1139202, -13.4989491, -31.1212521, -13.5130758, -12.6169434, 12.6316986
37: -50.2814293, -32.2508469, -50.2915955, -32.2756958, -9.9160271, 9.9347725
38: -38.7877083, -20.2296906, -38.7954178, -20.2447624, -11.3891144, 11.3905354
39: -42.5760803, -23.5669098, -42.5655441, -23.5611324, -10.6913033, 10.6619377
40: -38.0333481, -24.5424862, -38.0370636, -24.5546989, -7.9426880, 7.9613628
41: -24.8186989, -8.8759813, -24.8218613, -8.8913536, -13.1484375, 13.1814270
42: -15.1501980, -4.8171992, -15.1462297, -4.8165278, -8.1329880, 8.1253853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 638

## Relational analysis of IS_A2_B2_A2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -4.5441716, upper bound: 4.5560265
time: 37.50 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5670508, upper bound: 4.5551436
time: 54.30 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -13.3525524, 4.3133898, -13.3622608, 4.3125544, -15.6352310, 15.6569672
1: 0.4116712, 12.3339539, 0.4378927, 12.3401375, -9.2720985, 9.2378731
2: 2.0509479, 13.4804544, 2.0793614, 13.4873505, -9.0354118, 8.9929047
3: 1.5639582, 14.1761150, 1.5987689, 14.1812763, -9.0093918, 8.9715614
4: -4.2384796, 10.4599552, -4.1874180, 10.4644442, -12.6278915, 12.5823860
5: 2.0942409, 13.7664242, 2.1116087, 13.7743521, -8.4024506, 8.3838959
6: -25.1696281, -8.7630787, -25.1677151, -8.7903128, -13.3527603, 13.4038124
7: 2.5374551, 15.2943821, 2.5724106, 15.3031330, -9.3694992, 9.3206253
8: -4.5226812, 14.2436428, -4.4367909, 14.2525167, -15.7210388, 15.6113968
9: 0.5696301, 13.6011963, 0.5981596, 13.6032696, -9.2724915, 9.2441330
10: -4.3979769, 11.3088779, -4.3973150, 11.3049555, -12.0607109, 12.0459442
11: -4.4695020, 6.9354844, -4.4673986, 6.8924155, -8.5795059, 8.6218872
12: -26.2384529, -11.1265106, -26.2426186, -11.1753216, -10.9162521, 10.9659462
13: -14.2027521, 4.6830301, -14.1528683, 4.6800752, -13.4135437, 13.3606453
14: -24.1654205, -5.1959286, -24.1619053, -5.2185335, -16.3133087, 16.3257446
15: -7.6240573, 4.7050104, -7.6007547, 4.7069440, -11.3303070, 11.2817917
16: -7.6653233, 5.0167332, -7.6645660, 5.0023141, -9.3938675, 9.4851418
17: -26.7389507, -11.0719786, -26.7441807, -11.1063137, -11.0013924, 11.0038300
18: -17.6941185, -1.9652414, -17.6976547, -2.0574465, -10.5466194, 10.6759872
19: -10.4851475, -0.0663490, -10.4903917, -0.0731919, -6.8495827, 6.8620510
20: -5.8865194, 4.7005601, -5.8887758, 4.7088957, -7.5265541, 7.5133820
21: -8.5870399, 3.8233325, -8.5903606, 3.8130987, -9.6763687, 9.7020187
22: -10.7971954, 0.8378866, -10.8016729, 0.8263612, -7.8041763, 7.8361340
23: -4.6514506, 6.9296417, -4.6524801, 6.9000092, -8.7802277, 8.8141785
24: -8.1051750, 5.2416039, -8.1015997, 5.1944947, -10.5394287, 10.5942307
25: -8.3714333, 4.8568850, -8.3727283, 4.8476725, -8.3058472, 8.3201675
26: -16.5838394, 0.1715407, -16.5891762, 0.1423528, -11.6314468, 11.6798248
27: -7.8423853, 6.2127576, -7.8396492, 6.1934180, -12.1848755, 12.2300301
28: -6.5419111, 6.3300934, -6.5453873, 6.3174677, -10.1388779, 10.1516075
29: -7.7754040, 2.8930664, -7.7737761, 2.8658671, -8.6507721, 8.6762390
30: -3.8658588, 10.3283224, -3.8633208, 10.3026037, -12.2769852, 12.3112602
31: -14.8606358, -0.1396475, -14.8645191, -0.1816826, -10.7605476, 10.8228226
32: -20.8169823, -5.8305864, -20.8167667, -5.8507710, -12.0410309, 12.0612755
33: -38.5939445, -20.0131569, -38.5904922, -20.0137558, -10.6132317, 10.6147022
34: -35.7649803, -20.2242146, -35.7629280, -20.2737179, -11.5458908, 11.5928345
35: -33.0737228, -16.7695007, -33.0683784, -16.8036118, -11.3172722, 11.3470230
36: -31.1214561, -13.5051565, -31.1238880, -13.5215540, -12.6119385, 12.6385727
37: -50.2966156, -32.2364807, -50.2952271, -32.2795258, -9.9226379, 9.9577293
38: -38.7928314, -20.2320938, -38.7977371, -20.2521000, -11.3799057, 11.4000225
39: -42.5886841, -23.5620308, -42.5648499, -23.5611172, -10.7036858, 10.6675663
40: -38.0315895, -24.5337067, -38.0359192, -24.5553017, -7.9304790, 7.9754734
41: -24.8261356, -8.8718119, -24.8230171, -8.8953781, -13.1484451, 13.1938133
42: -15.1547804, -4.8133802, -15.1455984, -4.8167696, -8.1252251, 8.1289062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=201, inp2_unstable=201, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=19, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 908

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 638

## Relational analysis of IS_A2_B2_A2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5301759, upper bound: 4.5679339
time: 25.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4.5124062, upper bound: 4.5670511
time: 40.03 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 67.07 seconds
IS_A2_B1_A2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5180440, upper bound: 4.5436863
IS_A2_B1_A2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5171484, upper bound: 4.5665641
IS_A2_B1_A2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5180440, upper bound: 4.5436863
IS_A2_B1_A2_B1_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5311536, upper bound: 4.5665641
IS_A2_B1_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5124062, upper bound: 4.5674474
IS_A2_B1_A2_B2_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5352724, upper bound: 4.5665644
IS_A2_B1_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5124062, upper bound: 4.5674474
IS_A2_B1_A2_B2_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5124062, upper bound: 4.5665644
IS_A2_B2_A2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5358175, upper bound: 4.5441715
IS_A2_B2_A2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5171484, upper bound: 4.5665641
IS_A2_B2_A2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5498349, upper bound: 4.5441715
IS_A2_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5489454, upper bound: 4.5670508
IS_A2_B2_A2_B2_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5441716, upper bound: 4.5560265
IS_A2_B2_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5670508, upper bound: 4.5551436
IS_A2_B2_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5301759, upper bound: 4.5679339
IS_A2_B2_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 67.07
Output dim: 3, lower bound: -4.5124062, upper bound: 4.5670511
IS_A2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 67.07
Output dim: 3, lower bound: -4.5367917, upper bound: 4.5685449

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 37.42 + 1823.72 = 1861.14 seconds
