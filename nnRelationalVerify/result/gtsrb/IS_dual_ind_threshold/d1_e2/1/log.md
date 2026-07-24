## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 1800 seconds
Split limit: 100
Threshold: 7.170150672


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3245811, 17.3245811)
1: (-12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4212303, 11.4212303)
2: (-12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3746300, 10.3746262)
3: (-12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6396389, 11.6396370)
4: (-20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8510933, 12.8510933)
5: (-15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5319290, 15.5319290)
6: (2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5436325, 11.5436325)
7: (-15.3206863, 6.3509693, -15.3206863, 6.3509693, -15.0055122, 15.0055122)
8: (-21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6041107, 14.6041107)
9: (-8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8073616, 14.8073616)
10: (-20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7961006, 21.7961044)
11: (-10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2985764, 12.2985764)
12: (-13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0251045, 17.0251083)
13: (-18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0499268, 21.0499268)
14: (-55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4156799, 19.4156799)
15: (-24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9223652, 12.9223671)
16: (-11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4667168, 21.4667168)
17: (-55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6371155, 24.6371193)
18: (-21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6875534, 16.6875572)
19: (-10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000)
20: (-9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3799438, 14.3799438)
21: (-15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2695618, 17.2695656)
22: (-25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016)
23: (-7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9153214, 12.9153214)
24: (-13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0194473, 17.0194435)
25: (-12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8247719, 15.8247681)
26: (-28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4676285, 20.4676323)
27: (-13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5181122, 17.5181160)
28: (-6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1677132, 14.1677132)
29: (-22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1082382, 18.1082382)
30: (-11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4285049, 16.4285088)
31: (-12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202)
32: (-0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0283813, 13.0283813)
33: (-14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2129059, 24.2129059)
34: (-12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1320915, 16.1320915)
35: (-14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.6067352, 18.6067314)
36: (-13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3266144, 19.3266144)
37: (-17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.5008392, 20.5008430)
38: (-18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2407837, 24.2407837)
39: (-21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2652740, 28.2652740)
40: (-8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7143326, 19.7143326)
41: (3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3277054, 10.3277035)
42: (2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.81 + 55.66 = 58.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 41, lower bound: -7.1773280, upper bound: 7.1773280

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 641

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 731

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1755466, upper bound: 7.1730935
time: 33.65 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1755466, upper bound: 7.1755466
time: 30.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 64.71 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 64.71
Output dim: 41, lower bound: -7.1755466, upper bound: 7.1730935
IS_A2, status: Status.UNKNOWN, split count: 1, time: 64.71
Output dim: 41, lower bound: -7.1755466, upper bound: 7.1755466

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -23.8091660, -0.3022079, -23.8241577, -0.2955494, -17.3027649, 17.3066273
1: -12.3236799, 4.7406235, -12.3327112, 4.7453732, -11.4037476, 11.4092293
2: -12.0541573, 2.7347665, -12.0626907, 2.7431381, -10.3573799, 10.3570728
3: -12.2955790, 4.8773131, -12.3029852, 4.8877821, -11.6277752, 11.6264896
4: -20.5792370, -2.1410851, -20.5851746, -2.1349449, -12.8375359, 12.8341084
5: -15.6136284, 4.8211641, -15.6180897, 4.8279061, -15.5226631, 15.5191612
6: 2.2600002, 15.6327496, 2.2509890, 15.6384373, -11.5266781, 11.5319614
7: -15.3084536, 6.3432570, -15.3166475, 6.3491364, -14.9924431, 14.9931183
8: -21.3627110, 0.0907168, -21.3788605, 0.1060147, -14.5764847, 14.5781097
9: -8.8703690, 8.9512177, -8.8792992, 8.9559040, -14.7936764, 14.7967949
10: -20.8298225, 5.0473189, -20.8444920, 5.0640960, -21.7708740, 21.7679749
11: -10.9263945, 6.3878431, -10.9309101, 6.3926210, -12.2893295, 12.2881203
12: -13.6056681, 9.2651443, -13.6281986, 9.2875681, -16.9885101, 16.9867783
13: -18.2631340, 4.8481565, -18.2724075, 4.8637576, -21.0309219, 21.0250969
14: -55.3304558, -25.9107647, -55.3410645, -25.9048500, -19.3878593, 19.3923950
15: -24.2663059, -9.2244625, -24.2767563, -9.2124767, -12.9018097, 12.9027615
16: -11.7607679, 12.8292046, -11.7709484, 12.8326120, -21.4482040, 21.4584961
17: -55.9846954, -21.7587891, -55.9949036, -21.7386589, -24.6094627, 24.6052361
18: -21.0151348, 0.8159885, -21.0232544, 0.8238058, -16.6744423, 16.6745300
19: -10.6270485, 1.5412639, -10.6342030, 1.5447612, -12.1718102, 12.1754665
20: -9.6791239, 4.7704725, -9.6856709, 4.7735162, -14.3678093, 14.3664017
21: -15.6640301, 2.7075782, -15.6731148, 2.7120383, -17.2528458, 17.2589874
22: -25.0532570, -5.8782692, -25.0628853, -5.8733845, -19.1798725, 19.1846161
23: -7.8676009, 6.5133038, -7.8769712, 6.5165062, -12.9001808, 12.9058952
24: -13.4196014, 3.7609940, -13.4347916, 3.7760758, -16.9938736, 16.9938431
25: -12.3420429, 3.6734281, -12.3515444, 3.6816816, -15.8076210, 15.8093681
26: -28.1919174, -3.0559173, -28.2123260, -3.0365314, -20.4471054, 20.4447212
27: -13.3590660, 4.7110238, -13.3788452, 4.7257032, -17.4860954, 17.4915237
28: -6.8845015, 9.2403908, -6.8975534, 9.2470255, -14.1471634, 14.1530190
29: -22.1263733, -2.5806675, -22.1336384, -2.5758963, -18.0956268, 18.0875702
30: -11.3887997, 7.9651079, -11.3977671, 7.9753733, -16.4110794, 16.4099159
31: -12.0842247, 2.5985379, -12.0969219, 2.6072237, -14.6914482, 14.6954594
32: -0.5662880, 14.1513586, -0.5756269, 14.1535778, -13.0132065, 13.0153332
33: -14.5528545, 14.1843185, -14.5648527, 14.1887226, -24.1946335, 24.2026367
34: -12.9177542, 8.7421560, -12.9303722, 8.7479458, -16.1119995, 16.1189613
35: -14.2610779, 10.7256432, -14.2728395, 10.7319641, -18.5883827, 18.5933228
36: -13.3487043, 10.9297943, -13.3566074, 10.9337626, -19.3156738, 19.3099480
37: -17.5316658, 7.9445300, -17.5523109, 7.9517131, -20.4718094, 20.4845276
38: -18.2952042, 10.2749329, -18.3067989, 10.2812843, -24.2213287, 24.2225647
39: -21.6757240, 10.0233097, -21.6923904, 10.0349083, -28.2405396, 28.2462463
40: -8.4278259, 14.9499178, -8.4422207, 14.9511604, -19.6929703, 19.7069855
41: 3.2081537, 15.4769945, 3.1933336, 15.4831734, -10.3023643, 10.3097019
42: 2.8877511, 13.6351223, 2.8734698, 13.6399231, -10.7521725, 10.7616520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=81, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 641

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1686

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1739030, upper bound: 7.1639513
time: 22.39 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1739030, upper bound: 7.1714564
time: 23.02 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -23.8256111, -0.2948475, -23.8287582, -0.2943945, -17.3122864, 17.3225994
1: -12.3348045, 4.7466469, -12.3364182, 4.7469931, -11.4125595, 11.4203110
2: -12.0657635, 2.7440109, -12.0667257, 2.7444890, -10.3641434, 10.3736649
3: -12.3044844, 4.8894949, -12.3056259, 4.8899856, -11.6358109, 11.6380768
4: -20.5848198, -2.1336575, -20.5865917, -2.1330776, -12.8404198, 12.8498077
5: -15.6177683, 4.8303728, -15.6192341, 4.8308358, -15.5294609, 15.5305824
6: 2.2501764, 15.6376877, 2.2490907, 15.6396732, -11.5415230, 11.5354977
7: -15.3173790, 6.3505096, -15.3197985, 6.3508401, -14.9978523, 15.0040474
8: -21.3862610, 0.1064186, -21.3868504, 0.1068947, -14.5771217, 14.6024742
9: -8.8805237, 8.9570637, -8.8807125, 8.9579258, -14.8066559, 14.8044128
10: -20.8459625, 5.0706763, -20.8462601, 5.0719047, -21.7949905, 21.7891922
11: -10.9324322, 6.3931456, -10.9329491, 6.3936901, -12.2975006, 12.2973442
12: -13.6281967, 9.2975616, -13.6289539, 9.2984095, -17.0235596, 17.0014343
13: -18.2728539, 4.8693552, -18.2733650, 4.8709855, -21.0470428, 21.0402756
14: -55.3393250, -25.9041271, -55.3441658, -25.9029655, -19.3983650, 19.4129448
15: -24.2774029, -9.2076578, -24.2776337, -9.2065525, -12.9202576, 12.9064732
16: -11.7719145, 12.8343163, -11.7739687, 12.8347969, -21.4714203, 21.4636917
17: -55.9948120, -21.7311516, -55.9952583, -21.7298183, -24.6404495, 24.6339951
18: -21.0244312, 0.8261318, -21.0250263, 0.8266907, -16.6863632, 16.6848106
19: -10.6360512, 1.5450751, -10.6369591, 1.5452148, -12.1812658, 12.1820345
20: -9.6874590, 4.7740483, -9.6881714, 4.7742424, -14.3780022, 14.3865471
21: -15.6754036, 2.7124648, -15.6766348, 2.7125878, -17.2678108, 17.2687531
22: -25.0645905, -5.8730955, -25.0651855, -5.8722191, -19.1923714, 19.1920891
23: -7.8801541, 6.5169954, -7.8811474, 6.5171547, -12.9116898, 12.9141617
24: -13.4414845, 3.7763968, -13.4426460, 3.7768068, -17.0084114, 17.0183716
25: -12.3555784, 3.6821699, -12.3561630, 3.6826231, -15.8195038, 15.8237724
26: -28.2139282, -3.0279603, -28.2144012, -3.0271916, -20.4653702, 20.4579010
27: -13.3870926, 4.7256975, -13.3882761, 4.7261500, -17.5120087, 17.5168266
28: -6.9019098, 9.2471619, -6.9034758, 9.2475624, -14.1613464, 14.1665916
29: -22.1358013, -2.5754700, -22.1365547, -2.5751562, -18.1027069, 18.1192703
30: -11.4008427, 7.9758277, -11.4018135, 7.9763842, -16.4192657, 16.4272346
31: -12.1014032, 2.6075993, -12.1025505, 2.6077259, -14.7091293, 14.7101498
32: -0.5775995, 14.1532822, -0.5785675, 14.1538210, -13.0221481, 13.0329304
33: -14.5677147, 14.1893959, -14.5686064, 14.1897154, -24.2093048, 24.2108231
34: -12.9339657, 8.7486420, -12.9353294, 8.7490616, -16.1275520, 16.1290817
35: -14.2763996, 10.7325897, -14.2775717, 10.7328529, -18.6000862, 18.6059036
36: -13.3585739, 10.9337740, -13.3594790, 10.9351397, -19.3213806, 19.3419228
37: -17.5560379, 7.9531336, -17.5570107, 7.9548960, -20.4984550, 20.4921570
38: -18.3081818, 10.2824917, -18.3103237, 10.2827110, -24.2380142, 24.2414551
39: -21.6962776, 10.0355663, -21.6980095, 10.0359697, -28.2593918, 28.2641373
40: -8.4460745, 14.9511633, -8.4469070, 14.9513865, -19.7128220, 19.7132759
41: 3.1915889, 15.4853544, 3.1901641, 15.4865913, -10.3256340, 10.3137817
42: 2.8720136, 13.6387691, 2.8712821, 13.6412315, -10.7692184, 10.7674866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=81, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 641

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1686

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1739030, upper bound: 7.1663929
time: 42.40 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1739030, upper bound: 7.1739029
time: 37.84 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 82.60 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 82.60
Output dim: 41, lower bound: -7.1739030, upper bound: 7.1639513
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 82.60
Output dim: 41, lower bound: -7.1739030, upper bound: 7.1714564
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 82.60
Output dim: 41, lower bound: -7.1739030, upper bound: 7.1663929
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 82.60
Output dim: 41, lower bound: -7.1739030, upper bound: 7.1739029

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -23.8062458, -0.3037548, -23.8119144, -0.3011246, -17.2928009, 17.2893124
1: -12.3232136, 4.7385111, -12.3289261, 4.7372856, -11.3952942, 11.4034843
2: -12.0527067, 2.7328079, -12.0573101, 2.7336996, -10.3433380, 10.3491096
3: -12.2943268, 4.8752699, -12.2985439, 4.8773561, -11.6146469, 11.6191597
4: -20.5773029, -2.1437349, -20.5753479, -2.1473308, -12.8189278, 12.8205891
5: -15.6116467, 4.8192487, -15.6121683, 4.8200197, -15.5117569, 15.5074768
6: 2.2677822, 15.6320658, 2.2809820, 15.6278858, -11.5070953, 11.4993439
7: -15.3072624, 6.3403854, -15.3141632, 6.3396301, -14.9825058, 14.9894333
8: -21.3558311, 0.0889947, -21.3571815, 0.0894039, -14.5515518, 14.5556870
9: -8.8685598, 8.9378290, -8.8550835, 8.9171734, -14.7531490, 14.7588692
10: -20.8281784, 5.0193844, -20.7982197, 4.9837437, -21.6889954, 21.6939316
11: -10.9255276, 6.3794680, -10.9094028, 6.3677912, -12.2650909, 12.2576427
12: -13.6046410, 9.2543869, -13.6061859, 9.2546482, -16.9544754, 16.9533882
13: -18.2619896, 4.8432570, -18.2658577, 4.8467412, -21.0134888, 21.0126762
14: -55.3295288, -25.9311295, -55.3154449, -25.9645958, -19.3383865, 19.3495026
15: -24.2651482, -9.2297764, -24.2647743, -9.2327194, -12.8817558, 12.8870316
16: -11.7580051, 12.8229771, -11.7445755, 12.8139572, -21.4237404, 21.4199333
17: -55.9844894, -21.7837715, -55.9633713, -21.8137264, -24.5348549, 24.5485153
18: -21.0121555, 0.8073206, -21.0023403, 0.7968407, -16.6436882, 16.6436691
19: -10.6240368, 1.5402571, -10.6127644, 1.5379509, -12.1619873, 12.1530218
20: -9.6772203, 4.7699404, -9.6759586, 4.7685061, -14.3536072, 14.3567314
21: -15.6609602, 2.7064342, -15.6518736, 2.7094889, -17.2382812, 17.2345009
22: -25.0494423, -5.8804932, -25.0457611, -5.8798599, -19.1695824, 19.1652679
23: -7.8661585, 6.5118623, -7.8683577, 6.5106788, -12.8887444, 12.8987999
24: -13.4133644, 3.7598362, -13.4155836, 3.7591600, -16.9642868, 16.9719505
25: -12.3370705, 3.6703396, -12.3345146, 3.6651835, -15.7851486, 15.7888107
26: -28.1898499, -3.0608406, -28.1946468, -3.0534158, -20.4265137, 20.4229851
27: -13.3467827, 4.7101049, -13.3402481, 4.7109637, -17.4491806, 17.4477043
28: -6.8794012, 9.2392921, -6.8808250, 9.2361670, -14.1256523, 14.1348763
29: -22.1217194, -2.5822744, -22.1152420, -2.5791273, -18.0748291, 18.0630493
30: -11.3872728, 7.9632802, -11.3905182, 7.9630165, -16.3929253, 16.4000626
31: -12.0817032, 2.5976286, -12.0799904, 2.6003079, -14.6820107, 14.6776190
32: -0.5601552, 14.1511955, -0.5528541, 14.1520720, -12.9958687, 12.9910011
33: -14.5347576, 14.1834641, -14.5112438, 14.1651402, -24.1536255, 24.1489868
34: -12.9084988, 8.7414131, -12.9011793, 8.7307138, -16.0853195, 16.0890312
35: -14.2444477, 10.7252569, -14.2244873, 10.7090912, -18.5487442, 18.5445061
36: -13.3316374, 10.9294376, -13.3054028, 10.9136391, -19.2784996, 19.2587433
37: -17.5106621, 7.9443216, -17.4869900, 7.9295063, -20.4283447, 20.4264717
38: -18.2804432, 10.2741537, -18.2593689, 10.2607918, -24.1846619, 24.1726456
39: -21.6554756, 10.0226431, -21.6306114, 10.0077391, -28.1963959, 28.1918259
40: -8.4081507, 14.9498568, -8.3811169, 14.9304113, -19.6511536, 19.6496086
41: 3.2204924, 15.4767179, 3.2341995, 15.4741020, -10.2824821, 10.2732010
42: 2.8911147, 13.6338367, 2.8977900, 13.6310253, -10.7399101, 10.7360468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 641

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 725

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1618716
time: 38.84 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1623983
time: 37.66 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -23.8090401, -0.3024845, -23.8235207, -0.2968159, -17.2973900, 17.3054962
1: -12.3236418, 4.7403088, -12.3325386, 4.7439566, -11.3997726, 11.4086494
2: -12.0540485, 2.7347288, -12.0621729, 2.7429018, -10.3587952, 10.3557377
3: -12.2955551, 4.8772449, -12.3028126, 4.8874502, -11.6271286, 11.6260509
4: -20.5791473, -2.1411748, -20.5847206, -2.1353006, -12.8374805, 12.8322639
5: -15.6135826, 4.8209715, -15.6178169, 4.8270535, -15.5207329, 15.5178833
6: 2.2602186, 15.6327248, 2.2519860, 15.6383095, -11.5256882, 11.5259628
7: -15.3084145, 6.3427992, -15.3164644, 6.3469682, -14.9926491, 14.9924126
8: -21.3625088, 0.0906723, -21.3778496, 0.1057837, -14.5760002, 14.5709724
9: -8.8701820, 8.9508934, -8.8784323, 8.9543419, -14.7778130, 14.7955322
10: -20.8297615, 5.0466332, -20.8443012, 5.0607986, -21.7382812, 21.7671013
11: -10.9263668, 6.3875327, -10.9307833, 6.3911605, -12.2727585, 12.2877293
12: -13.6056194, 9.2647696, -13.6279449, 9.2857647, -16.9778786, 16.9862022
13: -18.2630005, 4.8478017, -18.2717590, 4.8620229, -21.0283051, 21.0238075
14: -55.3304253, -25.9112225, -55.3409042, -25.9071712, -19.3645248, 19.3908615
15: -24.2662849, -9.2246742, -24.2765980, -9.2134867, -12.8978539, 12.9018440
16: -11.7605982, 12.8283539, -11.7701664, 12.8286247, -21.4376144, 21.4572716
17: -55.9846878, -21.7593937, -55.9948883, -21.7415810, -24.5837708, 24.6046295
18: -21.0150738, 0.8157215, -21.0229206, 0.8225951, -16.6613350, 16.6739159
19: -10.6269674, 1.5412327, -10.6337996, 1.5446442, -12.1716118, 12.1750326
20: -9.6790552, 4.7704196, -9.6853848, 4.7732925, -14.3767319, 14.3642387
21: -15.6639614, 2.7075353, -15.6727009, 2.7118216, -17.2501678, 17.2575111
22: -25.0531597, -5.8783550, -25.0624466, -5.8737764, -19.1793823, 19.1840916
23: -7.8675642, 6.5130730, -7.8767514, 6.5155010, -12.9051971, 12.9042645
24: -13.4193268, 3.7609158, -13.4335079, 3.7755814, -16.9986191, 16.9920731
25: -12.3418159, 3.6732886, -12.3504887, 3.6810451, -15.8076782, 15.8084106
26: -28.1918392, -3.0561490, -28.2120743, -3.0377460, -20.4375076, 20.4434357
27: -13.3584156, 4.7109323, -13.3756676, 4.7252903, -17.4884377, 17.4893913
28: -6.8843508, 9.2403584, -6.8968687, 9.2468691, -14.1503983, 14.1515961
29: -22.1262436, -2.5807323, -22.1331043, -2.5763721, -18.0966949, 18.0818405
30: -11.3887815, 7.9650068, -11.3975706, 7.9748206, -16.4134750, 16.4082680
31: -12.0841465, 2.5984688, -12.0965643, 2.6069272, -14.6910734, 14.6950331
32: -0.5660117, 14.1513529, -0.5742335, 14.1535511, -13.0173149, 13.0135975
33: -14.5522795, 14.1842880, -14.5620975, 14.1886091, -24.1938782, 24.1858177
34: -12.9175491, 8.7421227, -12.9292984, 8.7477913, -16.1116447, 16.1148415
35: -14.2606411, 10.7256441, -14.2706680, 10.7319298, -18.5878830, 18.5745087
36: -13.3482809, 10.9297981, -13.3545637, 10.9337082, -19.3151932, 19.2974892
37: -17.5310574, 7.9445086, -17.5493736, 7.9516702, -20.4706421, 20.4604683
38: -18.2947140, 10.2748756, -18.3047447, 10.2809782, -24.2206421, 24.2120438
39: -21.6750813, 10.0232630, -21.6892796, 10.0347881, -28.2391281, 28.2300186
40: -8.4272633, 14.9499187, -8.4394703, 14.9511690, -19.6921120, 19.6843185
41: 3.2085013, 15.4769878, 3.1950336, 15.4831123, -10.3018303, 10.3034763
42: 2.8878436, 13.6350899, 2.8739114, 13.6397495, -10.7519054, 10.7611790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 641

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 725

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1693698
time: 34.13 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1698963
time: 20.71 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -23.8226490, -0.2963829, -23.8165607, -0.2999420, -17.3022995, 17.3052788
1: -12.3343878, 4.7445602, -12.3326483, 4.7389240, -11.4040871, 11.4145641
2: -12.0643101, 2.7419996, -12.0613499, 2.7350576, -10.3500786, 10.3657074
3: -12.3032389, 4.8874626, -12.3011494, 4.8795381, -11.6226730, 11.6307755
4: -20.5828934, -2.1362705, -20.5767708, -2.1454816, -12.8217945, 12.8362980
5: -15.6157904, 4.8284330, -15.6133194, 4.8229518, -15.5185814, 15.5188904
6: 2.2579513, 15.6370068, 2.2790751, 15.6291237, -11.5219479, 11.5028763
7: -15.3161926, 6.3476486, -15.3172855, 6.3412800, -14.9879112, 15.0003586
8: -21.3793964, 0.1047258, -21.3651733, 0.0902815, -14.5521965, 14.5800400
9: -8.8787079, 8.9436703, -8.8565083, 8.9191809, -14.7661057, 14.7664871
10: -20.8443184, 5.0427299, -20.7999859, 4.9915566, -21.7130928, 21.7151527
11: -10.9315395, 6.3847589, -10.9114199, 6.3688622, -12.2732353, 12.2668743
12: -13.6271524, 9.2867756, -13.6069508, 9.2655144, -16.9895401, 16.9680214
13: -18.2716808, 4.8644524, -18.2668171, 4.8539815, -21.0296097, 21.0278702
14: -55.3384132, -25.9244843, -55.3185577, -25.9627304, -19.3488731, 19.3700695
15: -24.2762566, -9.2129545, -24.2656479, -9.2267981, -12.9001865, 12.8907413
16: -11.7691574, 12.8281116, -11.7476139, 12.8161554, -21.4469528, 21.4251366
17: -55.9946136, -21.7561207, -55.9637260, -21.8048782, -24.5658417, 24.5772476
18: -21.0214310, 0.8174629, -21.0041065, 0.7997069, -16.6555786, 16.6539650
19: -10.6330528, 1.5440664, -10.6155052, 1.5384114, -12.1714640, 12.1595716
20: -9.6855640, 4.7735128, -9.6784821, 4.7692256, -14.3638077, 14.3768654
21: -15.6723289, 2.7113354, -15.6553879, 2.7100158, -17.2532425, 17.2442741
22: -25.0608025, -5.8753099, -25.0480366, -5.8786860, -19.1821175, 19.1727257
23: -7.8787007, 6.5155530, -7.8725133, 6.5113487, -12.9002838, 12.9070644
24: -13.4352531, 3.7752438, -13.4234171, 3.7598877, -16.9788094, 16.9964409
25: -12.3505926, 3.6790862, -12.3391285, 3.6661263, -15.7970581, 15.8032150
26: -28.2118797, -3.0329514, -28.1966972, -3.0440311, -20.4447937, 20.4361572
27: -13.3748217, 4.7248077, -13.3496847, 4.7113905, -17.4750710, 17.4730110
28: -6.8968215, 9.2460327, -6.8867302, 9.2367325, -14.1398163, 14.1484375
29: -22.1311626, -2.5770788, -22.1181583, -2.5783825, -18.0818748, 18.0947456
30: -11.3993139, 7.9740067, -11.3945522, 7.9640179, -16.4010925, 16.4174004
31: -12.0988712, 2.6066835, -12.0856228, 2.6007991, -14.6996708, 14.6923065
32: -0.5714693, 14.1531010, -0.5557976, 14.1523228, -13.0048065, 13.0085869
33: -14.5496082, 14.1885576, -14.5150299, 14.1661510, -24.1682892, 24.1571732
34: -12.9247284, 8.7478495, -12.9061594, 8.7318277, -16.1008987, 16.0991402
35: -14.2597675, 10.7322235, -14.2292137, 10.7099848, -18.5604553, 18.5570908
36: -13.3415346, 10.9333820, -13.3083229, 10.9149809, -19.2842178, 19.2907257
37: -17.5350647, 7.9529171, -17.4916935, 7.9326334, -20.4549637, 20.4341011
38: -18.2934189, 10.2817097, -18.2628765, 10.2622452, -24.2013168, 24.1915054
39: -21.6760349, 10.0349083, -21.6362095, 10.0087395, -28.2152328, 28.2097015
40: -8.4263754, 14.9511042, -8.3857822, 14.9306469, -19.6710396, 19.6558876
41: 3.2039146, 15.4850721, 3.2310195, 15.4775162, -10.3057442, 10.2772751
42: 2.8753624, 13.6374922, 2.8956008, 13.6323175, -10.7569551, 10.7418919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 641

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 725

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1643226
time: 31.41 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1648442
time: 22.37 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -23.8254623, -0.2951088, -23.8281441, -0.2956305, -17.3069000, 17.3214645
1: -12.3347979, 4.7463322, -12.3362617, 4.7455668, -11.4085884, 11.4197330
2: -12.0656776, 2.7439520, -12.0662384, 2.7442653, -10.3655453, 10.3723354
3: -12.3044596, 4.8894191, -12.3054390, 4.8896389, -11.6351509, 11.6376514
4: -20.5847454, -2.1337109, -20.5861702, -2.1334090, -12.8403625, 12.8479595
5: -15.6176987, 4.8302116, -15.6189556, 4.8299761, -15.5275383, 15.5292892
6: 2.2503862, 15.6376591, 2.2500858, 15.6395493, -11.5405426, 11.5294991
7: -15.3173409, 6.3500681, -15.3195992, 6.3486981, -14.9980659, 15.0033226
8: -21.3860588, 0.1063795, -21.3858566, 0.1066930, -14.5766525, 14.5953331
9: -8.8803291, 8.9567394, -8.8798389, 8.9563484, -14.7907906, 14.8031578
10: -20.8459225, 5.0700030, -20.8460903, 5.0686002, -21.7624130, 21.7883492
11: -10.9324045, 6.3928671, -10.9328308, 6.3922462, -12.2809143, 12.2969551
12: -13.6281319, 9.2971792, -13.6287165, 9.2966318, -17.0129318, 17.0008430
13: -18.2727070, 4.8689756, -18.2726936, 4.8692598, -21.0444107, 21.0390015
14: -55.3392677, -25.9045944, -55.3439941, -25.9052773, -19.3750381, 19.4114075
15: -24.2773876, -9.2078619, -24.2774811, -9.2075768, -12.9162922, 12.9055672
16: -11.7717781, 12.8334818, -11.7732067, 12.8308105, -21.4608078, 21.4624405
17: -55.9948196, -21.7317162, -55.9952164, -21.7327385, -24.6147461, 24.6333160
18: -21.0243721, 0.8258801, -21.0246811, 0.8254404, -16.6732292, 16.6841965
19: -10.6359615, 1.5450501, -10.6365328, 1.5451014, -12.1810627, 12.1815834
20: -9.6873932, 4.7740006, -9.6878834, 4.7740145, -14.3869362, 14.3843803
21: -15.6753235, 2.7124324, -15.6762199, 2.7123566, -17.2651367, 17.2672729
22: -25.0644951, -5.8731794, -25.0647297, -5.8726130, -19.1918831, 19.1915512
23: -7.8800993, 6.5167847, -7.8809147, 6.5161452, -12.9167175, 12.9125214
24: -13.4412184, 3.7762947, -13.4413548, 3.7763276, -17.0131378, 17.0165939
25: -12.3553696, 3.6820209, -12.3551016, 3.6819770, -15.8195419, 15.8228188
26: -28.2138634, -3.0282149, -28.2141190, -3.0284081, -20.4557571, 20.4565926
27: -13.3864365, 4.7256055, -13.3851137, 4.7257376, -17.5143700, 17.5146980
28: -6.9017649, 9.2471361, -6.9027872, 9.2474442, -14.1645622, 14.1651802
29: -22.1357002, -2.5755882, -22.1360340, -2.5756416, -18.1037903, 18.1135368
30: -11.4007816, 7.9757032, -11.4016094, 7.9758081, -16.4216614, 16.4255753
31: -12.1013222, 2.6075301, -12.1021967, 2.6074159, -14.7087383, 14.7097263
32: -0.5773149, 14.1532612, -0.5771704, 14.1537952, -13.0262489, 13.0311985
33: -14.5671568, 14.1893892, -14.5658550, 14.1895723, -24.2085419, 24.1939926
34: -12.9337463, 8.7485790, -12.9342651, 8.7489157, -16.1271896, 16.1249542
35: -14.2759609, 10.7325821, -14.2754030, 10.7327795, -18.5996017, 18.5870895
36: -13.3581963, 10.9337626, -13.3574820, 10.9350958, -19.3209305, 19.3294830
37: -17.5554295, 7.9531269, -17.5540581, 7.9548311, -20.4972534, 20.4681015
38: -18.3076515, 10.2824593, -18.3082695, 10.2824211, -24.2373199, 24.2309265
39: -21.6956310, 10.0355377, -21.6948738, 10.0358257, -28.2579498, 28.2479324
40: -8.4455042, 14.9511499, -8.4441357, 14.9513741, -19.7120209, 19.6905899
41: 3.1919346, 15.4853373, 3.1918483, 15.4865236, -10.3250961, 10.3075523
42: 2.8721023, 13.6387377, 2.8717175, 13.6410503, -10.7689476, 10.7670202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 641

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 725

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1718305
time: 31.35 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1723519
time: 37.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 70.97 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 70.97
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1618716
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 70.97
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1623983
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 70.97
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1693698
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 70.97
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1698963
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 70.97
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1643226
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 70.97
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1648442
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 70.97
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1718305
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 70.97
Output dim: 41, lower bound: -7.1723520, upper bound: 7.1723519

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -23.7876587, -0.3193512, -23.8079166, -0.3031759, -17.2719841, 17.2686539
1: -12.3068638, 4.7252502, -12.3237438, 4.7354894, -11.3780117, 11.3848209
2: -12.0469913, 2.7277808, -12.0559721, 2.7323945, -10.3356667, 10.3391037
3: -12.2868776, 4.8641729, -12.2963886, 4.8750033, -11.6046581, 11.6036701
4: -20.5703144, -2.1537218, -20.5734348, -2.1498833, -12.8043823, 12.8081856
5: -15.6077919, 4.8045845, -15.6106987, 4.8163381, -15.4970016, 15.4918747
6: 2.2847991, 15.6282444, 2.2837563, 15.6268835, -11.4878521, 11.4916916
7: -15.2888603, 6.3253241, -15.3077908, 6.3378220, -14.9635201, 14.9667511
8: -21.3531418, 0.0823634, -21.3558731, 0.0876291, -14.5451164, 14.5425911
9: -8.8490696, 8.9282742, -8.8528872, 8.9146690, -14.7210884, 14.7443047
10: -20.8148632, 4.9951730, -20.7966728, 4.9764023, -21.6584778, 21.6651955
11: -10.8820219, 6.3479295, -10.8939877, 6.3666487, -12.2217560, 12.2110958
12: -13.5838709, 9.2312965, -13.6046791, 9.2471123, -16.9254608, 16.9282494
13: -18.2393341, 4.8141613, -18.2646275, 4.8372250, -20.9775543, 20.9803391
14: -55.3212204, -25.9463673, -55.3136444, -25.9691372, -19.3218765, 19.3266277
15: -24.2609367, -9.2419167, -24.2637405, -9.2360821, -12.8655357, 12.8719025
16: -11.7403059, 12.8043823, -11.7402573, 12.8107901, -21.3918610, 21.3984070
17: -55.9810677, -21.8053875, -55.9627228, -21.8195438, -24.5113411, 24.5265808
18: -21.0029411, 0.8008518, -21.0000210, 0.7952757, -16.6309509, 16.6338615
19: -10.5979004, 1.5246152, -10.6044044, 1.5370831, -12.1349831, 12.1290197
20: -9.6474495, 4.7538643, -9.6657591, 4.7679348, -14.3228416, 14.3285103
21: -15.6022015, 2.6760621, -15.6318064, 2.7087593, -17.1784439, 17.1846352
22: -24.9991970, -5.9112902, -25.0287437, -5.8809013, -19.1182957, 19.1174545
23: -7.8378134, 6.4922099, -7.8585291, 6.5096703, -12.8596115, 12.8692245
24: -13.3705196, 3.7285261, -13.4008036, 3.7580671, -16.9195023, 16.9249992
25: -12.3048763, 3.6468935, -12.3232555, 3.6638379, -15.7505226, 15.7520332
26: -28.1771450, -3.0735097, -28.1906662, -3.0551977, -20.4108276, 20.4055862
27: -13.2943497, 4.6779518, -13.3225021, 4.7102146, -17.3957100, 17.3970833
28: -6.8469372, 9.2231646, -6.8695121, 9.2354774, -14.0913467, 14.1063538
29: -22.0632362, -2.6188898, -22.0952797, -2.5799837, -18.0142822, 18.0050964
30: -11.3344193, 7.9244909, -11.3718910, 7.9616680, -16.3383865, 16.3420601
31: -12.0528870, 2.5792298, -12.0704336, 2.5994785, -14.6523657, 14.6496639
32: -0.5387366, 14.1531315, -0.5479875, 14.1517916, -12.9734344, 12.9743481
33: -14.5101404, 14.1765804, -14.5056553, 14.1642771, -24.1257935, 24.1316528
34: -12.8964539, 8.7316113, -12.8992109, 8.7279663, -16.0707359, 16.0775146
35: -14.2278252, 10.7205544, -14.2216873, 10.7077255, -18.5295830, 18.5319557
36: -13.3094702, 10.9177647, -13.3019447, 10.9097233, -19.2575378, 19.2428131
37: -17.4793682, 7.9417400, -17.4813633, 7.9287624, -20.3913116, 20.4095612
38: -18.2499275, 10.2423887, -18.2567024, 10.2504988, -24.1456299, 24.1453018
39: -21.6194801, 10.0195923, -21.6249828, 10.0065575, -28.1614456, 28.1705933
40: -8.3712854, 14.9492569, -8.3748722, 14.9296570, -19.6110802, 19.6331100
41: 3.2385960, 15.4724369, 3.2376361, 15.4728527, -10.2647266, 10.2660465
42: 2.9053755, 13.6251163, 2.9013815, 13.6302738, -10.7248983, 10.7237349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=216, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 641

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 710

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1581744
time: 39.69 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1577000
time: 39.59 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -23.8053989, -0.3041620, -23.8117256, -0.3012180, -17.2913170, 17.2884521
1: -12.3223181, 4.7378335, -12.3287210, 4.7371459, -11.3858376, 11.4025650
2: -12.0523920, 2.7324712, -12.0572309, 2.7336411, -10.3421135, 10.3496742
3: -12.2934303, 4.8746991, -12.2983150, 4.8772211, -11.6101379, 11.6180534
4: -20.5769253, -2.1443958, -20.5752602, -2.1474862, -12.8182640, 12.8198280
5: -15.6112623, 4.8187885, -15.6120815, 4.8199244, -15.5111885, 15.5065117
6: 2.2684016, 15.6319151, 2.2811327, 15.6278467, -11.5064354, 11.4989395
7: -15.3057814, 6.3400173, -15.3138332, 6.3395219, -14.9698639, 14.9884567
8: -21.3554974, 0.0883889, -21.3570824, 0.0892515, -14.5498734, 14.5586700
9: -8.8681402, 8.9372864, -8.8550158, 8.9170609, -14.7586403, 14.7557411
10: -20.8276062, 5.0177169, -20.7981071, 4.9833598, -21.6954498, 21.6902351
11: -10.9240208, 6.3789511, -10.9090786, 6.3676815, -12.2372627, 12.2568798
12: -13.6043739, 9.2533960, -13.6061096, 9.2544546, -16.9537354, 16.9445457
13: -18.2614746, 4.8416934, -18.2657413, 4.8463411, -21.0127563, 20.9980011
14: -55.3285217, -25.9318485, -55.3152237, -25.9647675, -19.3346062, 19.3459206
15: -24.2648697, -9.2328911, -24.2647095, -9.2334328, -12.8810444, 12.8827667
16: -11.7572498, 12.8223495, -11.7443714, 12.8138142, -21.4226761, 21.4171677
17: -55.9843063, -21.7844658, -55.9633331, -21.8139133, -24.5408249, 24.5468254
18: -21.0114613, 0.8069134, -21.0022011, 0.7967310, -16.6429443, 16.6428528
19: -10.6228619, 1.5400777, -10.6124573, 1.5379130, -12.1607752, 12.1525345
20: -9.6753559, 4.7697563, -9.6754684, 4.7684612, -14.3457756, 14.3561401
21: -15.6590157, 2.7060742, -15.6514664, 2.7094154, -17.2121391, 17.2336845
22: -25.0479832, -5.8810749, -25.0454254, -5.8800049, -19.1679783, 19.1643505
23: -7.8650227, 6.5115690, -7.8681021, 6.5106120, -12.8807640, 12.8982677
24: -13.4117451, 3.7595873, -13.4152088, 3.7591081, -16.9553070, 16.9713058
25: -12.3352356, 3.6700010, -12.3340092, 3.6651144, -15.7762299, 15.7882271
26: -28.1893234, -3.0615244, -28.1945038, -3.0535564, -20.4256859, 20.4219818
27: -13.3452988, 4.7098742, -13.3399067, 4.7108850, -17.4363289, 17.4470139
28: -6.8778872, 9.2390213, -6.8804979, 9.2361097, -14.1177177, 14.1342888
29: -22.1201000, -2.5827532, -22.1149025, -2.5792580, -18.0572052, 18.0621452
30: -11.3857250, 7.9627666, -11.3901787, 7.9628782, -16.3637962, 16.3992500
31: -12.0778885, 2.5974450, -12.0790215, 2.6002724, -14.6781607, 14.6764660
32: -0.5593953, 14.1511145, -0.5526838, 14.1520452, -12.9941406, 12.9908619
33: -14.5338478, 14.1832190, -14.5110474, 14.1650925, -24.1505814, 24.1482468
34: -12.9080219, 8.7403946, -12.9010658, 8.7305164, -16.0842819, 16.0863724
35: -14.2438202, 10.7250099, -14.2243843, 10.7090425, -18.5473022, 18.5419350
36: -13.3310223, 10.9280491, -13.3052788, 10.9133291, -19.2762299, 19.2559395
37: -17.5097771, 7.9441686, -17.4867916, 7.9294415, -20.4263229, 20.4224434
38: -18.2796173, 10.2729607, -18.2591782, 10.2605553, -24.1822205, 24.1584930
39: -21.6544456, 10.0223560, -21.6303577, 10.0076456, -28.1936111, 28.1911087
40: -8.4069195, 14.9496183, -8.3808165, 14.9303741, -19.6488228, 19.6463737
41: 3.2210755, 15.4752874, 3.2343273, 15.4737711, -10.2814293, 10.2720814
42: 2.8918505, 13.6333618, 2.8979535, 13.6309204, -10.7390699, 10.7354088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 641

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 710

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1586976
time: 33.08 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1582108
time: 40.42 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -23.7904472, -0.3181133, -23.8195381, -0.2988605, -17.2765617, 17.2848167
1: -12.3072901, 4.7270527, -12.3273487, 4.7421379, -11.3825054, 11.3899803
2: -12.0483646, 2.7297213, -12.0608406, 2.7415910, -10.3511219, 10.3457413
3: -12.2880926, 4.8661089, -12.3006802, 4.8851023, -11.6171360, 11.6105309
4: -20.5721664, -2.1511483, -20.5828209, -2.1378126, -12.8229599, 12.8198643
5: -15.6096926, 4.8063402, -15.6163568, 4.8233790, -15.5059624, 15.5022926
6: 2.2772231, 15.6289062, 2.2547741, 15.6373215, -11.5064449, 11.5183144
7: -15.2900305, 6.3277540, -15.3101082, 6.3451958, -14.9736748, 14.9697380
8: -21.3598080, 0.0840068, -21.3765392, 0.1040387, -14.5695801, 14.5578766
9: -8.8506908, 8.9413271, -8.8762341, 8.9518261, -14.7457657, 14.7809525
10: -20.8164444, 5.0224066, -20.8427925, 5.0534301, -21.7077866, 21.7383652
11: -10.8828497, 6.3559976, -10.9153786, 6.3900185, -12.2294235, 12.2411728
12: -13.5848551, 9.2416534, -13.6264257, 9.2782097, -16.9489021, 16.9610825
13: -18.2403507, 4.8187160, -18.2704983, 4.8525147, -20.9923630, 20.9914894
14: -55.3220825, -25.9264965, -55.3390961, -25.9116726, -19.3480186, 19.3679924
15: -24.2620773, -9.2368231, -24.2755928, -9.2168446, -12.8816338, 12.8867416
16: -11.7429276, 12.8097954, -11.7658472, 12.8254423, -21.4057541, 21.4357147
17: -55.9812889, -21.7810307, -55.9941483, -21.7474022, -24.5602341, 24.5826302
18: -21.0059090, 0.8092742, -21.0206184, 0.8209867, -16.6486053, 16.6641083
19: -10.6008205, 1.5255878, -10.6254330, 1.5437903, -12.1446114, 12.1510210
20: -9.6492901, 4.7543259, -9.6751757, 4.7727098, -14.3459473, 14.3360252
21: -15.6051922, 2.6771576, -15.6526470, 2.7111187, -17.1903152, 17.2076607
22: -25.0029030, -5.9091568, -25.0454521, -5.8748512, -19.1280518, 19.1362953
23: -7.8392267, 6.4934387, -7.8669534, 6.5144958, -12.8760681, 12.8746586
24: -13.3764973, 3.7295904, -13.4187469, 3.7744999, -16.9538269, 16.9451370
25: -12.3096371, 3.6498523, -12.3392353, 3.6796885, -15.7730179, 15.7716179
26: -28.1791725, -3.0688219, -28.2081127, -3.0395417, -20.4218216, 20.4260559
27: -13.3059483, 4.6787767, -13.3579206, 4.7245522, -17.4349823, 17.4387550
28: -6.8518715, 9.2242422, -6.8855782, 9.2461472, -14.1160774, 14.1230698
29: -22.0677567, -2.6173849, -22.1131115, -2.5772457, -18.0361214, 18.0239258
30: -11.3358898, 7.9261951, -11.3789492, 7.9734783, -16.3589172, 16.3502388
31: -12.0553303, 2.5801082, -12.0870218, 2.6061025, -14.6614323, 14.6671295
32: -0.5446026, 14.1532784, -0.5693765, 14.1532593, -12.9949036, 12.9969578
33: -14.5276852, 14.1774387, -14.5564499, 14.1877317, -24.1660690, 24.1684875
34: -12.9055281, 8.7323246, -12.9273224, 8.7450790, -16.0970383, 16.1033211
35: -14.2440529, 10.7209473, -14.2678356, 10.7305565, -18.5687408, 18.5619507
36: -13.3261318, 10.9181461, -13.3511162, 10.9297972, -19.2942390, 19.2815742
37: -17.4997349, 7.9419084, -17.5437202, 7.9509621, -20.4335861, 20.4435539
38: -18.2641983, 10.2431259, -18.3020630, 10.2706547, -24.1816483, 24.1846848
39: -21.6390953, 10.0202427, -21.6836700, 10.0336246, -28.2042160, 28.2088165
40: -8.3903999, 14.9493179, -8.4332218, 14.9504204, -19.6520157, 19.6678123
41: 3.2266011, 15.4726963, 3.1984754, 15.4818668, -10.2840500, 10.2963142
42: 2.9021130, 13.6263599, 2.8775005, 13.6389942, -10.7368813, 10.7488594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=216, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 641

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 710

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1656543
time: 40.43 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1651695
time: 34.07 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -23.8081818, -0.3028898, -23.8233414, -0.2969189, -17.2959137, 17.3046112
1: -12.3227453, 4.7396564, -12.3323326, 4.7438011, -11.3903294, 11.4077358
2: -12.0537395, 2.7344277, -12.0621109, 2.7428241, -10.3575668, 10.3563023
3: -12.2946491, 4.8766823, -12.3026152, 4.8873148, -11.6226044, 11.6249371
4: -20.5787678, -2.1418285, -20.5846443, -2.1354265, -12.8368111, 12.8314934
5: -15.6131535, 4.8205180, -15.6177216, 4.8269272, -15.5201645, 15.5169258
6: 2.2608194, 15.6325865, 2.2521338, 15.6382771, -11.5250320, 11.5255604
7: -15.3069391, 6.3424139, -15.3161364, 6.3468981, -14.9800034, 14.9914474
8: -21.3622189, 0.0900593, -21.3777771, 0.1056459, -14.5743370, 14.5739670
9: -8.8697519, 8.9503441, -8.8783598, 8.9542246, -14.7833099, 14.7924004
10: -20.8292027, 5.0449567, -20.8441696, 5.0603948, -21.7447548, 21.7634430
11: -10.9248657, 6.3870230, -10.9304638, 6.3910532, -12.2449341, 12.2869587
12: -13.6053219, 9.2637672, -13.6278934, 9.2855234, -16.9771652, 16.9773445
13: -18.2624931, 4.8462062, -18.2716484, 4.8616257, -21.0275421, 21.0091438
14: -55.3294067, -25.9119644, -55.3406639, -25.9073009, -19.3607216, 19.3873024
15: -24.2659836, -9.2278013, -24.2765427, -9.2142019, -12.8971367, 12.8975868
16: -11.7598743, 12.8277245, -11.7700014, 12.8284788, -21.4365387, 21.4544754
17: -55.9844666, -21.7601185, -55.9947891, -21.7417374, -24.5897293, 24.6029015
18: -21.0144081, 0.8153219, -21.0227661, 0.8225012, -16.6605759, 16.6730804
19: -10.6257772, 1.5410542, -10.6334934, 1.5446050, -12.1703825, 12.1745472
20: -9.6772079, 4.7702284, -9.6848946, 4.7732592, -14.3688850, 14.3636436
21: -15.6619854, 2.7071609, -15.6722813, 2.7117393, -17.2240219, 17.2566986
22: -25.0517120, -5.8789215, -25.0621452, -5.8739300, -19.1777821, 19.1832237
23: -7.8664322, 6.5127954, -7.8765011, 6.5154266, -12.8972206, 12.9037094
24: -13.4177141, 3.7606826, -13.4331398, 3.7755370, -16.9896393, 16.9914360
25: -12.3400021, 3.6729581, -12.3500051, 3.6809621, -15.7987442, 15.8078423
26: -28.1913376, -3.0568304, -28.2119617, -3.0379114, -20.4366608, 20.4424133
27: -13.3569212, 4.7106705, -13.3753300, 4.7252340, -17.4755859, 17.4887009
28: -6.8828411, 9.2401037, -6.8965359, 9.2468128, -14.1424561, 14.1510239
29: -22.1246338, -2.5812435, -22.1327477, -2.5764856, -18.0790710, 18.0809250
30: -11.3871918, 7.9644632, -11.3972301, 7.9746594, -16.3843346, 16.4074326
31: -12.0803213, 2.5983088, -12.0956116, 2.6068838, -14.6872053, 14.6939201
32: -0.5652471, 14.1512699, -0.5740566, 14.1535234, -13.0155945, 13.0134659
33: -14.5513706, 14.1840820, -14.5618820, 14.1885605, -24.1908417, 24.1850510
34: -12.9170561, 8.7411060, -12.9291840, 8.7475843, -16.1105919, 16.1121902
35: -14.2600203, 10.7253847, -14.2705336, 10.7318544, -18.5864182, 18.5719299
36: -13.3476524, 10.9284019, -13.3544350, 10.9334011, -19.3129272, 19.2946968
37: -17.5301628, 7.9443693, -17.5491734, 7.9516430, -20.4686012, 20.4564514
38: -18.2938881, 10.2736893, -18.3045540, 10.2807140, -24.2182083, 24.1979065
39: -21.6740379, 10.0229931, -21.6890526, 10.0347195, -28.2363892, 28.2293091
40: -8.4260330, 14.9496813, -8.4391899, 14.9511013, -19.6897888, 19.6810760
41: 3.2090721, 15.4755478, 3.1951575, 15.4827871, -10.3007774, 10.3023472
42: 2.8885779, 13.6346111, 2.8740811, 13.6396446, -10.7510662, 10.7605305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 641

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 710

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1661770
time: 41.01 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1656790
time: 39.76 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -23.8040543, -0.3119926, -23.8125305, -0.3019896, -17.2814827, 17.2846069
1: -12.3180199, 4.7312975, -12.3274651, 4.7371387, -11.3868103, 11.3958855
2: -12.0586252, 2.7369840, -12.0600138, 2.7337508, -10.3424149, 10.3556976
3: -12.2957783, 4.8763409, -12.2990112, 4.8771787, -11.6126823, 11.6152763
4: -20.5759010, -2.1462507, -20.5748444, -2.1480293, -12.8072701, 12.8238869
5: -15.6119328, 4.8137922, -15.6118555, 4.8192563, -15.5038185, 15.5033188
6: 2.2749529, 15.6331825, 2.2818518, 15.6281338, -11.5027008, 11.4952183
7: -15.2978058, 6.3325672, -15.3109293, 6.3395147, -14.9689484, 14.9777031
8: -21.3766785, 0.0980852, -21.3638554, 0.0885096, -14.5457458, 14.5669327
9: -8.8592062, 8.9341125, -8.8543005, 8.9166689, -14.7340622, 14.7519112
10: -20.8309879, 5.0185246, -20.7984676, 4.9841700, -21.6826019, 21.6863976
11: -10.8880501, 6.3532348, -10.8960152, 6.3677173, -12.2299156, 12.2203159
12: -13.6064100, 9.2636585, -13.6054678, 9.2579746, -16.9605293, 16.9429054
13: -18.2490616, 4.8353715, -18.2655869, 4.8444929, -20.9936829, 20.9955139
14: -55.3300705, -25.9397354, -55.3167953, -25.9672222, -19.3323631, 19.3471832
15: -24.2720718, -9.2251196, -24.2646255, -9.2301474, -12.8839703, 12.8756332
16: -11.7514601, 12.8094826, -11.7432899, 12.8129520, -21.4150772, 21.4035683
17: -55.9912300, -21.7777252, -55.9630890, -21.8107395, -24.5423012, 24.5552826
18: -21.0122528, 0.8110056, -21.0018120, 0.7981310, -16.6428680, 16.6441383
19: -10.6068878, 1.5284190, -10.6071444, 1.5375651, -12.1444530, 12.1355629
20: -9.6557922, 4.7574167, -9.6682758, 4.7686405, -14.3330383, 14.3486671
21: -15.6135492, 2.6809826, -15.6353359, 2.7093129, -17.1934013, 17.1944275
22: -25.0105343, -5.9061279, -25.0310249, -5.8797445, -19.1307907, 19.1248970
23: -7.8503656, 6.4959049, -7.8627110, 6.5103340, -12.8711319, 12.8774834
24: -13.3924274, 3.7439260, -13.4086514, 3.7587957, -16.9340210, 16.9495010
25: -12.3184004, 3.6556439, -12.3278675, 3.6647975, -15.7624054, 15.7664146
26: -28.1991444, -3.0455866, -28.1927376, -3.0458460, -20.4290810, 20.4187698
27: -13.3223381, 4.6926384, -13.3319368, 4.7106724, -17.4216003, 17.4224167
28: -6.8643360, 9.2299423, -6.8754120, 9.2360210, -14.1055222, 14.1199226
29: -22.0726852, -2.6136827, -22.0981903, -2.5792761, -18.0213394, 18.0368004
30: -11.3464203, 7.9352164, -11.3759289, 7.9626942, -16.3465805, 16.3593979
31: -12.0700607, 2.5882864, -12.0760670, 2.5999801, -14.6700411, 14.6643534
32: -0.5500531, 14.1550407, -0.5509427, 14.1520414, -12.9823799, 12.9919205
33: -14.5249825, 14.1816540, -14.5094185, 14.1652584, -24.1404800, 24.1398468
34: -12.9126835, 8.7381020, -12.9041605, 8.7290812, -16.0863152, 16.0876350
35: -14.2431850, 10.7275352, -14.2263861, 10.7086287, -18.5413437, 18.5445480
36: -13.3193960, 10.9217520, -13.3048477, 10.9110699, -19.2632561, 19.2748146
37: -17.5037708, 7.9502935, -17.4860668, 7.9319420, -20.4179306, 20.4171867
38: -18.2629108, 10.2499990, -18.2602119, 10.2519560, -24.1623230, 24.1641541
39: -21.6400471, 10.0318813, -21.6305923, 10.0075788, -28.1802902, 28.1884613
40: -8.3895397, 14.9505043, -8.3795605, 14.9299059, -19.6309814, 19.6393776
41: 3.2220173, 15.4807854, 3.2344594, 15.4762669, -10.2879734, 10.2701168
42: 2.8896222, 13.6287689, 2.8991785, 13.6315708, -10.7419491, 10.7295904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=216, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 641

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 710

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1606356
time: 34.28 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1601654
time: 26.95 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -23.8217850, -0.2967825, -23.8163891, -0.3000536, -17.3008385, 17.3044186
1: -12.3334713, 4.7438612, -12.3324347, 4.7387829, -11.3946381, 11.4136486
2: -12.0640020, 2.7417049, -12.0612917, 2.7349820, -10.3488789, 10.3662643
3: -12.3023710, 4.8868885, -12.3009377, 4.8794079, -11.6181698, 11.6296616
4: -20.5824890, -2.1369166, -20.5766907, -2.1456280, -12.8211365, 12.8355103
5: -15.6153870, 4.8280096, -15.6132259, 4.8228412, -15.5180092, 15.5179367
6: 2.2585692, 15.6368675, 2.2792053, 15.6290865, -11.5212803, 11.5024738
7: -15.3146830, 6.3472567, -15.3169565, 6.3411970, -14.9752541, 14.9994011
8: -21.3790760, 0.1041052, -21.3650990, 0.0901482, -14.5504990, 14.5830269
9: -8.8782921, 8.9431305, -8.8564196, 8.9190578, -14.7715912, 14.7633667
10: -20.8437271, 5.0410442, -20.7998714, 4.9911542, -21.7195854, 21.7114563
11: -10.9300318, 6.3842626, -10.9110947, 6.3687525, -12.2454147, 12.2661037
12: -13.6268902, 9.2857962, -13.6068974, 9.2652740, -16.9888115, 16.9591827
13: -18.2712002, 4.8628807, -18.2667141, 4.8536010, -21.0288429, 21.0131760
14: -55.3373795, -25.9251862, -55.3183327, -25.9628754, -19.3450928, 19.3665028
15: -24.2759781, -9.2161007, -24.2655983, -9.2275114, -12.8994713, 12.8864746
16: -11.7683439, 12.8274374, -11.7474422, 12.8160038, -21.4458237, 21.4223366
17: -55.9944077, -21.7568016, -55.9636612, -21.8050842, -24.5718307, 24.5755577
18: -21.0207481, 0.8170586, -21.0039635, 0.7996292, -16.6548462, 16.6531448
19: -10.6318560, 1.5438908, -10.6152000, 1.5383666, -12.1702223, 12.1590910
20: -9.6837044, 4.7733412, -9.6779881, 4.7691936, -14.3559647, 14.3763046
21: -15.6703568, 2.7109857, -15.6549654, 2.7099204, -17.2271118, 17.2434692
22: -25.0593472, -5.8759022, -25.0477200, -5.8788214, -19.1805267, 19.1718178
23: -7.8775721, 6.5152555, -7.8722634, 6.5112720, -12.8922729, 12.9065323
24: -13.4336376, 3.7750120, -13.4230595, 3.7598295, -16.9698410, 16.9958191
25: -12.3487835, 3.6787560, -12.3386374, 3.6660600, -15.7881126, 15.8026314
26: -28.2113552, -3.0336657, -28.1965752, -3.0441942, -20.4439430, 20.4351501
27: -13.3733149, 4.7245450, -13.3493557, 4.7113485, -17.4622307, 17.4723358
28: -6.8953133, 9.2458000, -6.8863878, 9.2366695, -14.1318703, 14.1478653
29: -22.1295280, -2.5775833, -22.1178093, -2.5784941, -18.0642624, 18.0938683
30: -11.3977280, 7.9734879, -11.3942204, 7.9638872, -16.3719635, 16.4165840
31: -12.0950432, 2.6065102, -12.0846462, 2.6007600, -14.6958027, 14.6911564
32: -0.5707011, 14.1530209, -0.5556231, 14.1523123, -13.0030708, 13.0084629
33: -14.5486937, 14.1883097, -14.5148230, 14.1660671, -24.1652298, 24.1564331
34: -12.9242496, 8.7468529, -12.9060478, 8.7316370, -16.0998459, 16.0964928
35: -14.2591457, 10.7319784, -14.2290897, 10.7099333, -18.5590019, 18.5545158
36: -13.3409395, 10.9319849, -13.3081856, 10.9146748, -19.2819366, 19.2879562
37: -17.5341930, 7.9527221, -17.4915237, 7.9326067, -20.4529648, 20.4300613
38: -18.2925701, 10.2805395, -18.2626953, 10.2620077, -24.1988983, 24.1773834
39: -21.6749992, 10.0346317, -21.6359653, 10.0086956, -28.2124481, 28.2089844
40: -8.4251528, 14.9508657, -8.3855085, 14.9305859, -19.6687202, 19.6526375
41: 3.2045069, 15.4836292, 3.2311511, 15.4771872, -10.3046894, 10.2761631
42: 2.8760948, 13.6370277, 2.8957629, 13.6322289, -10.7561340, 10.7412643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 641

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 710

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1611441
time: 43.20 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1606578
time: 72.52 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -23.8068504, -0.3107443, -23.8241596, -0.2977047, -17.2860718, 17.3007736
1: -12.3184357, 4.7331028, -12.3310652, 4.7437663, -11.3913078, 11.4010620
2: -12.0599794, 2.7389340, -12.0648851, 2.7429569, -10.3578911, 10.3623371
3: -12.2969866, 4.8783045, -12.3033028, 4.8872757, -11.6251698, 11.6221542
4: -20.5777550, -2.1436915, -20.5842476, -2.1359711, -12.8258400, 12.8355656
5: -15.6138468, 4.8155432, -15.6175146, 4.8262968, -15.5127678, 15.5137367
6: 2.2673960, 15.6338415, 2.2528715, 15.6385670, -11.5212955, 11.5218487
7: -15.2989483, 6.3349905, -15.3132420, 6.3468647, -14.9790878, 14.9806633
8: -21.3833447, 0.0997264, -21.3845596, 0.1049409, -14.5702209, 14.5822258
9: -8.8608341, 8.9471788, -8.8776360, 8.9538288, -14.7587471, 14.7885742
10: -20.8326054, 5.0457616, -20.8445435, 5.0612268, -21.7319298, 21.7595711
11: -10.8888893, 6.3613319, -10.9174232, 6.3910885, -12.2375717, 12.2503967
12: -13.6073828, 9.2740736, -13.6271448, 9.2890949, -16.9839439, 16.9757385
13: -18.2500877, 4.8398867, -18.2714481, 4.8597565, -21.0084686, 21.0066452
14: -55.3309746, -25.9198627, -55.3422279, -25.9097996, -19.3585167, 19.3885536
15: -24.2731876, -9.2200279, -24.2764683, -9.2109165, -12.9000702, 12.8904591
16: -11.7541027, 12.8148823, -11.7688770, 12.8276253, -21.4289551, 21.4409142
17: -55.9914246, -21.7533665, -55.9945335, -21.7385826, -24.5912323, 24.6113510
18: -21.0151958, 0.8194599, -21.0223465, 0.8238354, -16.6604881, 16.6743927
19: -10.6098194, 1.5294036, -10.6281786, 1.5442570, -12.1540766, 12.1575823
20: -9.6576281, 4.7579002, -9.6776953, 4.7734251, -14.3561516, 14.3561859
21: -15.6165476, 2.6820421, -15.6561623, 2.7116768, -17.2052879, 17.2174492
22: -25.0142307, -5.9039869, -25.0477295, -5.8736610, -19.1405697, 19.1437416
23: -7.8517609, 6.4971223, -7.8711119, 6.5151591, -12.8875847, 12.8829384
24: -13.3983812, 3.7449908, -13.4265871, 3.7752328, -16.9683609, 16.9696388
25: -12.3231697, 3.6585813, -12.3438559, 3.6806350, -15.7849083, 15.7860298
26: -28.2011547, -3.0408626, -28.2101784, -3.0302305, -20.4401169, 20.4391823
27: -13.3339653, 4.6934500, -13.3673649, 4.7249994, -17.4608727, 17.4640923
28: -6.8693008, 9.2310171, -6.8914890, 9.2467070, -14.1302452, 14.1366348
29: -22.0771751, -2.6121836, -22.1160431, -2.5765038, -18.0431976, 18.0556221
30: -11.3479271, 7.9369116, -11.3829803, 7.9744897, -16.3671036, 16.3675728
31: -12.0725031, 2.5891521, -12.0926552, 2.6065855, -14.6790886, 14.6818075
32: -0.5559006, 14.1551800, -0.5723221, 14.1535149, -13.0038300, 13.0145321
33: -14.5425110, 14.1824856, -14.5602436, 14.1887207, -24.1807480, 24.1766739
34: -12.9217415, 8.7388067, -12.9322805, 8.7461796, -16.1126175, 16.1134491
35: -14.2593603, 10.7278728, -14.2725735, 10.7314396, -18.5804596, 18.5745239
36: -13.3360329, 10.9220943, -13.3540316, 10.9311934, -19.2999649, 19.3135681
37: -17.5241318, 7.9504962, -17.5484009, 7.9541073, -20.4601898, 20.4511719
38: -18.2771740, 10.2506838, -18.3055649, 10.2721424, -24.1983109, 24.2035446
39: -21.6596622, 10.0325537, -21.6892815, 10.0346527, -28.2230530, 28.2266617
40: -8.4086199, 14.9505587, -8.4379139, 14.9506865, -19.6718979, 19.6741219
41: 3.2100396, 15.4810534, 3.1952958, 15.4852810, -10.3073254, 10.3003941
42: 2.8863611, 13.6300192, 2.8753023, 13.6402988, -10.7539377, 10.7547169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=216, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 641

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 710

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1681226
time: 37.44 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1676407
time: 42.95 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -23.8246002, -0.2955360, -23.8279476, -0.2957468, -17.3054199, 17.3205910
1: -12.3338938, 4.7456560, -12.3360481, 4.7454138, -11.3991470, 11.4188042
2: -12.0653610, 2.7436514, -12.0661449, 2.7441854, -10.3643341, 10.3729019
3: -12.3035460, 4.8888679, -12.3052578, 4.8895125, -11.6306419, 11.6365318
4: -20.5843639, -2.1343575, -20.5860672, -2.1335611, -12.8396912, 12.8471889
5: -15.6172981, 4.8297243, -15.6188593, 4.8298845, -15.5269890, 15.5283470
6: 2.2510023, 15.6375198, 2.2502365, 15.6395111, -11.5398808, 11.5291023
7: -15.3158293, 6.3496513, -15.3192806, 6.3486080, -14.9853973, 15.0023804
8: -21.3857174, 0.1057441, -21.3857517, 0.1065431, -14.5749588, 14.5983353
9: -8.8799114, 8.9561892, -8.8797512, 8.9562254, -14.7962952, 14.8000183
10: -20.8453331, 5.0683103, -20.8459473, 5.0682125, -21.7688599, 21.7846451
11: -10.9308796, 6.3923316, -10.9324932, 6.3921385, -12.2530861, 12.2961788
12: -13.6278572, 9.2961960, -13.6286469, 9.2964230, -17.0122490, 16.9919930
13: -18.2722244, 4.8673887, -18.2725735, 4.8688717, -21.0436401, 21.0243149
14: -55.3382759, -25.9053459, -55.3437767, -25.9054604, -19.3712463, 19.4078369
15: -24.2771034, -9.2110014, -24.2774162, -9.2082710, -12.9155712, 12.9013023
16: -11.7710199, 12.8328285, -11.7730446, 12.8306541, -21.4597588, 21.4596519
17: -55.9946022, -21.7324619, -55.9951668, -21.7329235, -24.6207390, 24.6316261
18: -21.0236683, 0.8254843, -21.0245132, 0.8253255, -16.6724701, 16.6833878
19: -10.6347780, 1.5448787, -10.6362324, 1.5450556, -12.1798334, 12.1811113
20: -9.6855402, 4.7738047, -9.6874037, 4.7739697, -14.3790855, 14.3837929
21: -15.6733513, 2.7120500, -15.6757860, 2.7122602, -17.2389832, 17.2664795
22: -25.0630341, -5.8737726, -25.0644169, -5.8727579, -19.1902771, 19.1906433
23: -7.8789787, 6.5164909, -7.8806705, 6.5160875, -12.9087486, 12.9119835
24: -13.4396086, 3.7760668, -13.4409924, 3.7762613, -17.0041618, 17.0159531
25: -12.3535385, 3.6816807, -12.3546143, 3.6819122, -15.8106308, 15.8222351
26: -28.2133255, -3.0289268, -28.2140121, -3.0285268, -20.4549522, 20.4555779
27: -13.3849354, 4.7253428, -13.3847742, 4.7256727, -17.5014992, 17.5140266
28: -6.9002666, 9.2468729, -6.9024563, 9.2473803, -14.1566200, 14.1646004
29: -22.1340466, -2.5760469, -22.1356831, -2.5757618, -18.0861435, 18.1126404
30: -11.3992300, 7.9751444, -11.4012747, 7.9756780, -16.3925133, 16.4247780
31: -12.0974951, 2.6073556, -12.1012373, 2.6073778, -14.7048731, 14.7085934
32: -0.5765531, 14.1531754, -0.5769997, 14.1537733, -13.0245171, 13.0310783
33: -14.5662203, 14.1891279, -14.5656376, 14.1895313, -24.2054901, 24.1932602
34: -12.9332895, 8.7475843, -12.9341545, 8.7486982, -16.1261444, 16.1222916
35: -14.2753916, 10.7323599, -14.2752666, 10.7327461, -18.5981293, 18.5845070
36: -13.3575802, 10.9323597, -13.3573475, 10.9347916, -19.3186493, 19.3267097
37: -17.5545368, 7.9529400, -17.5538940, 7.9548049, -20.4952316, 20.4640694
38: -18.3068542, 10.2812433, -18.3081055, 10.2821598, -24.2348862, 24.2168045
39: -21.6945915, 10.0352917, -21.6946621, 10.0357876, -28.2552261, 28.2471695
40: -8.4442606, 14.9509201, -8.4438763, 14.9513206, -19.7096672, 19.6873474
41: 3.1925154, 15.4839106, 3.1919746, 15.4861956, -10.3240395, 10.3064308
42: 2.8728385, 13.6382627, 2.8718815, 13.6409492, -10.7681103, 10.7663813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 641

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 710

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1686306
time: 41.66 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1681311
time: 51.48 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 95.48 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1581744
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1577000
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1586976
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1582108
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1656543
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1651695
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1661770
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1656790
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1606356
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1601654
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1611441
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1606578
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1681226
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1676407
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1628458, upper bound: 7.1686306
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 95.48
Output dim: 41, lower bound: -7.1681311, upper bound: 7.1681311

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 58.47 + 1135.34 = 1193.81 seconds
