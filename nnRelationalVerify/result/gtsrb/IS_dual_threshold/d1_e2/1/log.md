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
execution time: IAR + RelationalAnalysis = 2.67 + 55.35 = 58.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 41, lower bound: -7.1773280, upper bound: 7.1773280

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1686

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1756940, upper bound: 7.1681806
time: 43.95 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1756940, upper bound: 7.1756940
time: 44.98 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 89.06 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 89.06
Output dim: 41, lower bound: -7.1756940, upper bound: 7.1681806
IS_B2, status: Status.UNKNOWN, split count: 1, time: 89.06
Output dim: 41, lower bound: -7.1756940, upper bound: 7.1756940

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -23.8272762, -0.2957058, -23.8180046, -0.2997284, -17.3145981, 17.3072758
1: -12.3367691, 4.7450628, -12.3334513, 4.7391109, -11.4127693, 11.4154930
2: -12.0656242, 2.7427137, -12.0616970, 2.7352610, -10.3605747, 10.3666668
3: -12.3049660, 4.8881540, -12.3017378, 4.8797369, -11.6264896, 11.6323280
4: -20.5854683, -2.1354799, -20.5775852, -2.1452470, -12.8324661, 12.8375778
5: -15.6178284, 4.8291059, -15.6138735, 4.8231549, -15.5210114, 15.5202293
6: 2.2564178, 15.6397190, 2.2786140, 15.6298561, -11.5240517, 11.5110226
7: -15.3194971, 6.3481455, -15.3181763, 6.3414326, -14.9956017, 15.0018539
8: -21.3802242, 0.1053920, -21.3654060, 0.0904791, -14.5791664, 14.5816689
9: -8.8790140, 8.9448662, -8.8565845, 8.9195290, -14.7667999, 14.7694283
10: -20.8447342, 5.0444756, -20.8001099, 4.9920421, -21.7141953, 21.7220497
11: -10.9322968, 6.3855028, -10.9116745, 6.3690648, -12.2743187, 12.2681160
12: -13.6282139, 9.2879896, -13.6072721, 9.2658577, -16.9910660, 16.9917107
13: -18.2724476, 4.8667841, -18.2670383, 4.8546762, -21.0325012, 21.0375290
14: -55.3450317, -25.9228287, -55.3203621, -25.9622421, -19.3661995, 19.3728008
15: -24.2765617, -9.2113895, -24.2657299, -9.2263432, -12.9023094, 12.9066334
16: -11.7719841, 12.8287888, -11.7483988, 12.8163738, -21.4422379, 21.4281464
17: -55.9951744, -21.7542572, -55.9638901, -21.8043671, -24.5624962, 24.5804214
18: -21.0222645, 0.8182163, -21.0043488, 0.7999048, -16.6568031, 16.6566963
19: -10.6343269, 1.5442681, -10.6158915, 1.5384594, -12.1727867, 12.1601601
20: -9.6865807, 4.7737885, -9.6787891, 4.7692971, -14.3657684, 14.3702660
21: -15.6741428, 2.7115030, -15.6559868, 2.7100687, -17.2549820, 17.2451019
22: -25.0616665, -5.8740683, -25.0482845, -5.8783274, -19.1833382, 19.1742172
23: -7.8801041, 6.5157990, -7.8729239, 6.5114183, -12.9039268, 12.9082279
24: -13.4368801, 3.7758050, -13.4238920, 3.7600679, -16.9898415, 16.9975204
25: -12.3514166, 3.6797357, -12.3393698, 3.6663320, -15.8023071, 15.8042221
26: -28.2125721, -3.0318255, -28.1968994, -3.0437460, -20.4470749, 20.4459267
27: -13.3764744, 4.7254429, -13.3501587, 4.7115817, -17.4811630, 17.4743004
28: -6.8990021, 9.2466059, -6.8873367, 9.2368965, -14.1462021, 14.1495857
29: -22.1322174, -2.5766277, -22.1184673, -2.5782461, -18.0874405, 18.0837555
30: -11.4006767, 7.9747829, -11.3949604, 7.9642277, -16.4103241, 16.4186516
31: -12.1005249, 2.6068418, -12.0861073, 2.6008434, -14.7013683, 14.6929493
32: -0.5728333, 14.1538734, -0.5562015, 14.1525297, -13.0110435, 13.0040302
33: -14.5508747, 14.1889725, -14.5154076, 14.1662636, -24.1718445, 24.1592407
34: -12.9267082, 8.7485018, -12.9067688, 8.7320175, -16.1054382, 16.1021385
35: -14.2613697, 10.7325497, -14.2296715, 10.7100830, -18.5671082, 18.5579185
36: -13.3428106, 10.9352665, -13.3086700, 10.9155264, -19.2894440, 19.2754059
37: -17.5364456, 7.9553776, -17.4921227, 7.9333625, -20.4573517, 20.4427643
38: -18.2964191, 10.2820435, -18.2637596, 10.2623367, -24.2040939, 24.1908340
39: -21.6784763, 10.0354233, -21.6369324, 10.0088854, -28.2211075, 28.2108688
40: -8.4275599, 14.9514360, -8.3861475, 14.9307404, -19.6725388, 19.6569595
41: 3.2019100, 15.4867792, 3.2304301, 15.4779882, -10.3078156, 10.2911892
42: 2.8743391, 13.6409531, 2.8952913, 13.6333351, -10.7589960, 10.7456617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 725

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1741414, upper bound: 7.1661091
time: 32.21 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1741414, upper bound: 7.1666301
time: 33.75 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -23.8300953, -0.2944698, -23.8295898, -0.2954512, -17.3191948, 17.3234291
1: -12.3371773, 4.7468491, -12.3370571, 4.7457380, -11.4172707, 11.4206543
2: -12.0669556, 2.7446518, -12.0665760, 2.7444701, -10.3760319, 10.3732948
3: -12.3061905, 4.8900952, -12.3060646, 4.8898239, -11.6389694, 11.6392078
4: -20.5873165, -2.1329203, -20.5869675, -2.1331739, -12.8510323, 12.8492508
5: -15.6197262, 4.8308396, -15.6195135, 4.8301563, -15.5299950, 15.5306396
6: 2.2488394, 15.6403780, 2.2496405, 15.6402912, -11.5426521, 11.5376453
7: -15.3206406, 6.3505478, -15.3204937, 6.3488255, -15.0057373, 15.0048065
8: -21.3868752, 0.1070597, -21.3860989, 0.1068583, -14.6036148, 14.5969658
9: -8.8806000, 8.9579353, -8.8799276, 8.9566965, -14.7915001, 14.8060837
10: -20.8463497, 5.0717168, -20.8462276, 5.0691023, -21.7635193, 21.7952576
11: -10.9331465, 6.3935714, -10.9330778, 6.3924289, -12.2819939, 12.2981949
12: -13.6292191, 9.2984066, -13.6290359, 9.2969656, -17.0144997, 17.0245247
13: -18.2734356, 4.8713088, -18.2728901, 4.8699369, -21.0472908, 21.0486488
14: -55.3459167, -25.9029598, -55.3457985, -25.9048023, -19.3923378, 19.4141541
15: -24.2776871, -9.2063093, -24.2775688, -9.2071018, -12.9184055, 12.9214554
16: -11.7746105, 12.8341722, -11.7739992, 12.8310242, -21.4561157, 21.4654922
17: -55.9953995, -21.7299175, -55.9953651, -21.7322521, -24.6114311, 24.6364975
18: -21.0251999, 0.8266191, -21.0249195, 0.8256502, -16.6744423, 16.6869545
19: -10.6372519, 1.5452534, -10.6369247, 1.5451604, -12.1824121, 12.1821785
20: -9.6884193, 4.7742682, -9.6881809, 4.7740698, -14.3888817, 14.3777809
21: -15.6771297, 2.7125921, -15.6768179, 2.7123942, -17.2668839, 17.2681046
22: -25.0653534, -5.8719482, -25.0650024, -5.8722572, -19.1930962, 19.1930542
23: -7.8815088, 6.5170226, -7.8813338, 6.5162301, -12.9203796, 12.9136887
24: -13.4428549, 3.7768683, -13.4418278, 3.7764750, -17.0241776, 17.0176773
25: -12.3561802, 3.6826854, -12.3553429, 3.6821580, -15.8247833, 15.8238220
26: -28.2145405, -3.0271330, -28.2143021, -3.0281029, -20.4580536, 20.4663391
27: -13.3881025, 4.7262573, -13.3855944, 4.7259283, -17.5204620, 17.5159836
28: -6.9039450, 9.2476788, -6.9034181, 9.2475863, -14.1709480, 14.1663170
29: -22.1367569, -2.5751057, -22.1363144, -2.5755024, -18.1093178, 18.1025085
30: -11.4021645, 7.9764929, -11.4020071, 7.9760408, -16.4308701, 16.4268570
31: -12.1029749, 2.6077065, -12.1026955, 2.6074793, -14.7104540, 14.7104015
32: -0.5786674, 14.1540241, -0.5775781, 14.1540051, -13.0324783, 13.0266418
33: -14.5684233, 14.1897831, -14.5662365, 14.1896992, -24.2121124, 24.1960983
34: -12.9357433, 8.7492237, -12.9348936, 8.7491188, -16.1317329, 16.1279640
35: -14.2775536, 10.7329063, -14.2758522, 10.7328939, -18.6062546, 18.5879173
36: -13.3594408, 10.9356632, -13.3578625, 10.9356098, -19.3261337, 19.3141556
37: -17.5568256, 7.9555717, -17.5544834, 7.9555435, -20.4996109, 20.4767609
38: -18.3106880, 10.2827473, -18.3091469, 10.2825203, -24.2400665, 24.2302322
39: -21.6980801, 10.0361004, -21.6956062, 10.0359879, -28.2638397, 28.2490463
40: -8.4466724, 14.9514828, -8.4444599, 14.9514685, -19.7135086, 19.6916542
41: 3.1899199, 15.4870472, 3.1912613, 15.4869957, -10.3271675, 10.3214645
42: 2.8710737, 13.6421909, 2.8714180, 13.6420603, -10.7709866, 10.7707729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 678

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 725

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1741414, upper bound: 7.1736165
time: 40.90 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1741414, upper bound: 7.1741413
time: 35.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 78.41 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 78.41
Output dim: 41, lower bound: -7.1741414, upper bound: 7.1661091
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 78.41
Output dim: 41, lower bound: -7.1741414, upper bound: 7.1666301
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 78.41
Output dim: 41, lower bound: -7.1741414, upper bound: 7.1736165
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 78.41
Output dim: 41, lower bound: -7.1741414, upper bound: 7.1741413

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -23.8087101, -0.3113461, -23.8140144, -0.3017883, -17.2937660, 17.2865906
1: -12.3204193, 4.7318373, -12.3282623, 4.7372999, -11.3954887, 11.3968029
2: -12.0599098, 2.7376959, -12.0603752, 2.7339697, -10.3529053, 10.3566608
3: -12.2974977, 4.8770247, -12.2996168, 4.8773832, -11.6165104, 11.6168270
4: -20.5784798, -2.1454687, -20.5756912, -2.1477833, -12.8179283, 12.8251839
5: -15.6139574, 4.8144631, -15.6124067, 4.8194599, -15.5062485, 15.5046654
6: 2.2734175, 15.6359043, 2.2813978, 15.6288719, -11.5048027, 11.5033627
7: -15.3011026, 6.3330631, -15.3118229, 6.3396425, -14.9766426, 14.9791718
8: -21.3775063, 0.0987718, -21.3640900, 0.0887277, -14.5727196, 14.5685463
9: -8.8595066, 8.9353180, -8.8543987, 8.9170208, -14.7347488, 14.7548752
10: -20.8314247, 5.0202522, -20.7985802, 4.9846787, -21.6837006, 21.6933022
11: -10.8888149, 6.3539677, -10.8962574, 6.3679032, -12.2309952, 12.2215481
12: -13.6074858, 9.2648830, -13.6057663, 9.2583075, -16.9620705, 16.9665794
13: -18.2497864, 4.8377218, -18.2657967, 4.8451562, -20.9965897, 21.0051804
14: -55.3367195, -25.9381065, -55.3185768, -25.9667568, -19.3496857, 19.3499165
15: -24.2723770, -9.2235556, -24.2647171, -9.2296886, -12.8860950, 12.8915119
16: -11.7543087, 12.8101921, -11.7440662, 12.8131905, -21.4103699, 21.4066277
17: -55.9918404, -21.7758808, -55.9632683, -21.8101921, -24.5389709, 24.5584259
18: -21.0130730, 0.8117547, -21.0020542, 0.7983580, -16.6440659, 16.6468887
19: -10.6081800, 1.5286127, -10.6075230, 1.5376132, -12.1457930, 12.1361361
20: -9.6568060, 4.7576847, -9.6685715, 4.7687254, -14.3349876, 14.3420563
21: -15.6153793, 2.6811509, -15.6359186, 2.7093558, -17.1951675, 17.1952591
22: -25.0114231, -5.9048948, -25.0312767, -5.8793802, -19.1320419, 19.1263809
23: -7.8517642, 6.4961529, -7.8631096, 6.5104103, -12.8747864, 12.8786430
24: -13.3940439, 3.7444787, -13.4091206, 3.7589445, -16.9451141, 16.9506035
25: -12.3192282, 3.6563053, -12.3281002, 3.6649802, -15.7676773, 15.7674370
26: -28.1998062, -3.0444708, -28.1929169, -3.0455618, -20.4313812, 20.4285164
27: -13.3239994, 4.6932945, -13.3324089, 4.7108421, -17.4277344, 17.4236794
28: -6.8665323, 9.2305241, -6.8760495, 9.2361784, -14.1118927, 14.1210632
29: -22.0737457, -2.6132669, -22.0984802, -2.5791292, -18.0268936, 18.0257721
30: -11.3478117, 7.9359965, -11.3763351, 7.9629059, -16.3557892, 16.3606606
31: -12.0717182, 2.5884628, -12.0765724, 2.6000209, -14.6717396, 14.6650352
32: -0.5514226, 14.1558056, -0.5513487, 14.1522598, -12.9886284, 12.9873791
33: -14.5262747, 14.1821012, -14.5097761, 14.1653614, -24.1440506, 24.1419258
34: -12.9146414, 8.7387238, -12.9047956, 8.7292671, -16.0908585, 16.0906105
35: -14.2447577, 10.7278347, -14.2268286, 10.7087183, -18.5479660, 18.5453873
36: -13.3206730, 10.9236364, -13.3052120, 10.9116030, -19.2684517, 19.2594795
37: -17.5051365, 7.9527559, -17.4864693, 7.9326410, -20.4203033, 20.4258423
38: -18.2658939, 10.2503271, -18.2610664, 10.2520199, -24.1650848, 24.1634598
39: -21.6424942, 10.0323830, -21.6313248, 10.0077343, -28.1861725, 28.1896133
40: -8.3907013, 14.9508677, -8.3799162, 14.9300089, -19.6324692, 19.6404381
41: 3.2200084, 15.4824944, 3.2338777, 15.4767418, -10.2900391, 10.2840347
42: 2.8885984, 13.6322298, 2.8988833, 13.6325836, -10.7439852, 10.7333469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 692

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1691039, upper bound: 7.1649858
time: 28.24 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1691039, upper bound: 7.1648644
time: 33.89 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -23.8264427, -0.2961226, -23.8178139, -0.2998362, -17.3131256, 17.3063965
1: -12.3358593, 4.7444077, -12.3332529, 4.7389317, -11.4033127, 11.4145622
2: -12.0652962, 2.7423997, -12.0616417, 2.7351856, -10.3593731, 10.3672256
3: -12.3040695, 4.8875484, -12.3015490, 4.8796158, -11.6219978, 11.6312141
4: -20.5850983, -2.1361275, -20.5774975, -2.1453867, -12.8318043, 12.8368053
5: -15.6173925, 4.8286591, -15.6137924, 4.8230381, -15.5204506, 15.5192871
6: 2.2570157, 15.6395779, 2.2787724, 15.6298294, -11.5233879, 11.5106220
7: -15.3180017, 6.3477144, -15.3178682, 6.3413267, -14.9829597, 15.0008965
8: -21.3798904, 0.1047776, -21.3653259, 0.0903652, -14.5774727, 14.5846252
9: -8.8785763, 8.9443207, -8.8565083, 8.9194126, -14.7722931, 14.7662888
10: -20.8441658, 5.0428185, -20.7999878, 4.9916682, -21.7206650, 21.7183838
11: -10.9308023, 6.3849697, -10.9113178, 6.3689570, -12.2464981, 12.2673359
12: -13.6279459, 9.2869930, -13.6071892, 9.2656307, -16.9903488, 16.9828835
13: -18.2719460, 4.8652081, -18.2669201, 4.8542695, -21.0317535, 21.0228615
14: -55.3439865, -25.9235802, -55.3201180, -25.9624214, -19.3624115, 19.3692322
15: -24.2762833, -9.2145185, -24.2656708, -9.2270517, -12.9015865, 12.9023628
16: -11.7711802, 12.8281355, -11.7482290, 12.8162308, -21.4411621, 21.4253578
17: -55.9949989, -21.7550011, -55.9638672, -21.8045483, -24.5684967, 24.5786972
18: -21.0215664, 0.8178129, -21.0041714, 0.7998171, -16.6560440, 16.6558914
19: -10.6331453, 1.5440873, -10.6155872, 1.5384214, -12.1715670, 12.1596746
20: -9.6847191, 4.7736216, -9.6782913, 4.7692599, -14.3579140, 14.3696899
21: -15.6721888, 2.7111447, -15.6555653, 2.7099738, -17.2288513, 17.2442932
22: -25.0602150, -5.8746576, -25.0479813, -5.8784714, -19.1817436, 19.1733246
23: -7.8789668, 6.5155163, -7.8726630, 6.5113454, -12.8959274, 12.9076958
24: -13.4352579, 3.7755699, -13.4235239, 3.7599983, -16.9809074, 16.9968948
25: -12.3495998, 3.6794062, -12.3388691, 3.6662641, -15.7933731, 15.8036156
26: -28.2120228, -3.0325489, -28.1967716, -3.0439129, -20.4462166, 20.4449120
27: -13.3749771, 4.7251801, -13.3498316, 4.7115240, -17.4683304, 17.4736176
28: -6.8974743, 9.2463655, -6.8870239, 9.2368488, -14.1382561, 14.1489983
29: -22.1305923, -2.5771170, -22.1181145, -2.5783653, -18.0698204, 18.0828285
30: -11.3991241, 7.9742475, -11.3946056, 7.9641132, -16.3812332, 16.4178505
31: -12.0966997, 2.6066766, -12.0851555, 2.6008120, -14.6975117, 14.6918316
32: -0.5720711, 14.1537895, -0.5560193, 14.1525164, -13.0093193, 13.0039062
33: -14.5499649, 14.1887426, -14.5151958, 14.1661930, -24.1688004, 24.1584892
34: -12.9262323, 8.7475119, -12.9066620, 8.7317953, -16.1043930, 16.0994835
35: -14.2607565, 10.7323227, -14.2295303, 10.7100334, -18.5656509, 18.5553398
36: -13.3422070, 10.9338779, -13.3085165, 10.9152241, -19.2871437, 19.2726059
37: -17.5356102, 7.9552159, -17.4919224, 7.9333358, -20.4553299, 20.4387360
38: -18.2955551, 10.2808456, -18.2635555, 10.2621040, -24.2017059, 24.1766968
39: -21.6774368, 10.0351591, -21.6366997, 10.0088329, -28.2183838, 28.2101669
40: -8.4263325, 14.9511909, -8.3858519, 14.9306726, -19.6702194, 19.6536827
41: 3.2025008, 15.4853363, 3.2305613, 15.4776669, -10.3067551, 10.2900791
42: 2.8750763, 13.6404848, 2.8954592, 13.6332302, -10.7581539, 10.7450256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 678

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 692

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1691039, upper bound: 7.1654851
time: 29.55 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1728745, upper bound: 7.1653677
time: 30.95 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -23.8115234, -0.3100891, -23.8255939, -0.2975073, -17.2983551, 17.3027477
1: -12.3208084, 4.7336273, -12.3318691, 4.7439322, -11.3999672, 11.4019814
2: -12.0612707, 2.7396457, -12.0652037, 2.7431443, -10.3683662, 10.3633003
3: -12.2987080, 4.8789883, -12.3039074, 4.8874660, -11.6290016, 11.6237068
4: -20.5803375, -2.1428881, -20.5850525, -2.1357479, -12.8364964, 12.8368416
5: -15.6158695, 4.8161974, -15.6180525, 4.8264904, -15.5152130, 15.5150795
6: 2.2658434, 15.6365509, 2.2524118, 15.6392937, -11.5234051, 11.5299873
7: -15.3022499, 6.3354759, -15.3141441, 6.3470240, -14.9867706, 14.9821510
8: -21.3841610, 0.1003985, -21.3847866, 0.1051505, -14.5971832, 14.5838470
9: -8.8611202, 8.9483833, -8.8777390, 8.9541740, -14.7594490, 14.7915192
10: -20.8330116, 5.0475020, -20.8446350, 5.0617270, -21.7330017, 21.7665138
11: -10.8896456, 6.3620462, -10.9176559, 6.3912816, -12.2386589, 12.2516384
12: -13.6084280, 9.2752619, -13.6274567, 9.2894287, -16.9855003, 16.9994164
13: -18.2507935, 4.8422203, -18.2716827, 4.8604231, -21.0113907, 21.0163193
14: -55.3375664, -25.9182262, -55.3440285, -25.9093456, -19.3758316, 19.3912964
15: -24.2734909, -9.2184563, -24.2765598, -9.2104540, -12.9021912, 12.9063263
16: -11.7569218, 12.8155823, -11.7696676, 12.8278389, -21.4242783, 21.4439659
17: -55.9920158, -21.7515221, -55.9947128, -21.7380333, -24.5879021, 24.6145020
18: -21.0160065, 0.8201694, -21.0226231, 0.8240600, -16.6617050, 16.6771278
19: -10.6110945, 1.5295980, -10.6285686, 1.5443091, -12.1554031, 12.1581669
20: -9.6586533, 4.7581558, -9.6779842, 4.7735071, -14.3580856, 14.3495560
21: -15.6183538, 2.6822081, -15.6567459, 2.7117062, -17.2070465, 17.2182770
22: -25.0150871, -5.9027576, -25.0479813, -5.8733282, -19.1417580, 19.1452236
23: -7.8531733, 6.4973741, -7.8715205, 6.5152173, -12.8912277, 12.8840904
24: -13.4000111, 3.7455444, -13.4270506, 3.7753892, -16.9794197, 16.9707260
25: -12.3239927, 3.6592140, -12.3440790, 3.6808217, -15.7901726, 15.7870102
26: -28.2018566, -3.0397658, -28.2103863, -3.0298934, -20.4423752, 20.4489594
27: -13.3356314, 4.6940913, -13.3678379, 4.7251973, -17.4669952, 17.4653778
28: -6.8714800, 9.2315931, -6.8920937, 9.2468796, -14.1366348, 14.1377792
29: -22.0782604, -2.6117458, -22.1163254, -2.5763874, -18.0487747, 18.0445786
30: -11.3492870, 7.9376831, -11.3833599, 7.9747009, -16.3763428, 16.3688240
31: -12.0741491, 2.5893126, -12.0931530, 2.6066360, -14.6807852, 14.6824656
32: -0.5572770, 14.1559696, -0.5727229, 14.1537266, -13.0100822, 13.0100021
33: -14.5437994, 14.1829481, -14.5606108, 14.1888371, -24.1843109, 24.1787720
34: -12.9237309, 8.7394590, -12.9329004, 8.7463789, -16.1171494, 16.1164513
35: -14.2609415, 10.7282362, -14.2730227, 10.7315617, -18.5870972, 18.5753555
36: -13.3373203, 10.9239950, -13.3543959, 10.9316807, -19.3051605, 19.2982216
37: -17.5255203, 7.9529800, -17.5488129, 7.9548435, -20.4625778, 20.4598579
38: -18.2801743, 10.2509995, -18.3064499, 10.2722130, -24.2010880, 24.2028732
39: -21.6620750, 10.0330467, -21.6899986, 10.0348186, -28.2288895, 28.2278519
40: -8.4098167, 14.9509068, -8.4382563, 14.9507732, -19.6734161, 19.6751556
41: 3.2080150, 15.4827595, 3.1947141, 15.4857597, -10.3093948, 10.3143101
42: 2.8853345, 13.6334705, 2.8750000, 13.6413002, -10.7559662, 10.7584705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 678

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 692

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1691039, upper bound: 7.1724864
time: 40.50 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1728745, upper bound: 7.1723687
time: 36.98 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -23.8292427, -0.2948694, -23.8294086, -0.2955446, -17.3177261, 17.3225784
1: -12.3362789, 4.7461867, -12.3368511, 4.7455964, -11.4077950, 11.4197330
2: -12.0666437, 2.7443449, -12.0665054, 2.7443993, -10.3748112, 10.3738594
3: -12.3052912, 4.8895435, -12.3058538, 4.8897181, -11.6344604, 11.6380920
4: -20.5869522, -2.1335602, -20.5868893, -2.1333237, -12.8503723, 12.8484650
5: -15.6193256, 4.8304124, -15.6194344, 4.8300629, -15.5294342, 15.5296936
6: 2.2494502, 15.6402416, 2.2497778, 15.6402445, -11.5419922, 11.5372391
7: -15.3191671, 6.3501406, -15.3201609, 6.3487353, -14.9930840, 15.0038376
8: -21.3865738, 0.1064231, -21.3860168, 0.1067209, -14.6019287, 14.5999565
9: -8.8802128, 8.9573860, -8.8798475, 8.9565697, -14.7969856, 14.8029594
10: -20.8457565, 5.0700564, -20.8460808, 5.0687170, -21.7699776, 21.7915573
11: -10.9316578, 6.3930669, -10.9327335, 6.3923206, -12.2541733, 12.2973938
12: -13.6289186, 9.2973633, -13.6289520, 9.2967815, -17.0137787, 17.0156898
13: -18.2729492, 4.8697567, -18.2728024, 4.8695445, -21.0465469, 21.0339775
14: -55.3449097, -25.9037075, -55.3455582, -25.9049778, -19.3885498, 19.4105835
15: -24.2774086, -9.2094297, -24.2775002, -9.2078066, -12.9176941, 12.9171829
16: -11.7738180, 12.8335295, -11.7738018, 12.8308849, -21.4550552, 21.4626961
17: -55.9951935, -21.7306232, -55.9953384, -21.7323799, -24.6174126, 24.6347656
18: -21.0245113, 0.8262391, -21.0247746, 0.8255558, -16.6736832, 16.6861420
19: -10.6360569, 1.5450692, -10.6366110, 1.5451071, -12.1811638, 12.1816807
20: -9.6865635, 4.7740803, -9.6877041, 4.7740335, -14.3810310, 14.3771858
21: -15.6751547, 2.7122312, -15.6763916, 2.7123265, -17.2407379, 17.2672806
22: -25.0638943, -5.8725176, -25.0646667, -5.8723936, -19.1915016, 19.1921501
23: -7.8803759, 6.5167198, -7.8810811, 6.5161490, -12.9123802, 12.9131355
24: -13.4412270, 3.7766390, -13.4414539, 3.7764316, -17.0152245, 17.0170364
25: -12.3543682, 3.6823449, -12.3548517, 3.6820757, -15.8158722, 15.8232231
26: -28.2139854, -3.0278368, -28.2141972, -3.0282545, -20.4572220, 20.4653358
27: -13.3865900, 4.7260013, -13.3852577, 4.7258797, -17.5076294, 17.5153198
28: -6.9024258, 9.2474308, -6.9030876, 9.2475376, -14.1630058, 14.1657181
29: -22.1351357, -2.5756102, -22.1359806, -2.5755939, -18.0917282, 18.1015816
30: -11.4005909, 7.9759417, -11.4016438, 7.9759207, -16.4017601, 16.4260254
31: -12.0991468, 2.6075375, -12.1017303, 2.6074333, -14.7065802, 14.7092676
32: -0.5779181, 14.1539431, -0.5773916, 14.1539850, -13.0307655, 13.0265198
33: -14.5675182, 14.1895447, -14.5660105, 14.1896524, -24.2090607, 24.1953316
34: -12.9352722, 8.7482214, -12.9347954, 8.7489119, -16.1306763, 16.1253014
35: -14.2769537, 10.7326984, -14.2757378, 10.7328510, -18.6047821, 18.5853348
36: -13.3588476, 10.9342499, -13.3577194, 10.9352779, -19.3238220, 19.3113747
37: -17.5559502, 7.9554024, -17.5542831, 7.9554801, -20.4975967, 20.4727287
38: -18.3098679, 10.2815714, -18.3089294, 10.2822418, -24.2376480, 24.2161102
39: -21.6970329, 10.0357971, -21.6953964, 10.0359058, -28.2610931, 28.2483292
40: -8.4454556, 14.9512682, -8.4441786, 14.9514294, -19.7111626, 19.6884193
41: 3.1904936, 15.4856062, 3.1913943, 15.4866629, -10.3261185, 10.3203468
42: 2.8718038, 13.6417294, 2.8715849, 13.6419506, -10.7701473, 10.7701445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 695

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 692

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1691039, upper bound: 7.1729863
time: 27.72 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1728745, upper bound: 7.1728745
time: 31.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 61.70 seconds
IS_B1_A1_B1, status: Status.VERIFIED, split count: 3, time: 61.70
Output dim: 41, lower bound: -7.1691039, upper bound: 7.1649858
IS_B1_A1_B2, status: Status.VERIFIED, split count: 3, time: 61.70
Output dim: 41, lower bound: -7.1691039, upper bound: 7.1648644
IS_B1_A2_B1, status: Status.VERIFIED, split count: 3, time: 61.70
Output dim: 41, lower bound: -7.1691039, upper bound: 7.1654851
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 61.70
Output dim: 41, lower bound: -7.1728745, upper bound: 7.1653677
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 61.70
Output dim: 41, lower bound: -7.1691039, upper bound: 7.1724864
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 61.70
Output dim: 41, lower bound: -7.1728745, upper bound: 7.1723687
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 61.70
Output dim: 41, lower bound: -7.1691039, upper bound: 7.1729863
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 61.70
Output dim: 41, lower bound: -7.1728745, upper bound: 7.1728745

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -23.8260593, -0.2962599, -23.8166084, -0.3002453, -17.3130913, 17.3045082
1: -12.3357363, 4.7442112, -12.3327646, 4.7383165, -11.3997135, 11.4025726
2: -12.0651999, 2.7418838, -12.0612631, 2.7334628, -10.3590546, 10.3660316
3: -12.3038950, 4.8866787, -12.3008423, 4.8767738, -11.6187744, 11.6300278
4: -20.5849991, -2.1367788, -20.5771904, -2.1475854, -12.8294754, 12.8358917
5: -15.6172352, 4.8278103, -15.6132326, 4.8202209, -15.5168228, 15.5176659
6: 2.2573609, 15.6395063, 2.2798166, 15.6295776, -11.5225697, 11.5089264
7: -15.3177471, 6.3476644, -15.3169889, 6.3411136, -14.9782448, 14.9835167
8: -21.3797741, 0.1042988, -21.3649445, 0.0886528, -14.5734215, 14.5835762
9: -8.8782864, 8.9441509, -8.8555050, 8.9188490, -14.7677040, 14.7618408
10: -20.8437920, 5.0424428, -20.7987404, 4.9905462, -21.7186699, 21.7133446
11: -10.9300747, 6.3848324, -10.9088383, 6.3685155, -12.2454262, 12.2313786
12: -13.6278391, 9.2863827, -13.6068897, 9.2637501, -16.9793205, 16.9822655
13: -18.2717133, 4.8645706, -18.2661572, 4.8521628, -21.0038528, 21.0214539
14: -55.3438034, -25.9238777, -55.3195190, -25.9634037, -19.3602562, 19.3666153
15: -24.2762222, -9.2155161, -24.2654266, -9.2303677, -12.8923035, 12.9007759
16: -11.7705650, 12.8279228, -11.7461004, 12.8154850, -21.4394531, 21.3997726
17: -55.9949493, -21.7564564, -55.9636383, -21.8095093, -24.5659103, 24.5780411
18: -21.0212078, 0.8175006, -21.0029678, 0.7988100, -16.6546021, 16.6394539
19: -10.6324987, 1.5439547, -10.6134453, 1.5380043, -12.1705027, 12.1574001
20: -9.6841431, 4.7734175, -9.6763973, 4.7686381, -14.3567162, 14.3548164
21: -15.6712923, 2.7109632, -15.6526241, 2.7094102, -17.2272263, 17.2068939
22: -25.0597744, -5.8748713, -25.0465450, -5.8791685, -19.1806068, 19.1716728
23: -7.8783770, 6.5153570, -7.8707190, 6.5108743, -12.8945541, 12.8940678
24: -13.4348240, 3.7752972, -13.4220152, 3.7591405, -16.9795609, 16.9820061
25: -12.3491278, 3.6792259, -12.3373146, 3.6656296, -15.7922821, 15.7896652
26: -28.2114525, -3.0329494, -28.1949825, -3.0452037, -20.4442291, 20.4394608
27: -13.3744087, 4.7250185, -13.3479614, 4.7109270, -17.4670715, 17.4461670
28: -6.8969703, 9.2462368, -6.8853421, 9.2364054, -14.1373100, 14.1384354
29: -22.1300983, -2.5772543, -22.1164551, -2.5787354, -18.0688248, 18.0720634
30: -11.3985205, 7.9740739, -11.3926258, 7.9635286, -16.3798409, 16.3841667
31: -12.0959234, 2.6065071, -12.0828199, 2.6002593, -14.6961823, 14.6893272
32: -0.5717778, 14.1537619, -0.5551009, 14.1524773, -13.0064926, 13.0004807
33: -14.5496321, 14.1885128, -14.5141048, 14.1654673, -24.1593475, 24.1570129
34: -12.9259815, 8.7469721, -12.9059410, 8.7301979, -16.0802307, 16.0983086
35: -14.2606201, 10.7319298, -14.2289610, 10.7087822, -18.5382385, 18.5544281
36: -13.3420315, 10.9334908, -13.3080301, 10.9139500, -19.2561493, 19.2715683
37: -17.5351772, 7.9550562, -17.4905891, 7.9329205, -20.4421082, 20.4312553
38: -18.2954235, 10.2802868, -18.2630043, 10.2601242, -24.1653290, 24.1755371
39: -21.6771183, 10.0347595, -21.6355686, 10.0075731, -28.1968460, 28.2084885
40: -8.4257660, 14.9511471, -8.3839912, 14.9304686, -19.6732483, 19.6439476
41: 3.2028122, 15.4852142, 3.2316060, 15.4772472, -10.2962685, 10.2885075
42: 2.8756814, 13.6403885, 2.8973370, 13.6329288, -10.7572479, 10.7430515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 678

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 731

## Relational analysis of IS_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1686258, upper bound: 7.1635817
time: 42.00 seconds

## Relational analysis of IS_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1648407, upper bound: 7.1635817
time: 121.37 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -23.8091660, -0.3106194, -23.8118591, -0.3084917, -17.2832565, 17.2880821
1: -12.3183384, 4.7323666, -12.3230581, 4.7342720, -11.3836937, 11.3914642
2: -12.0606995, 2.7377264, -12.0632095, 2.7379036, -10.3623619, 10.3598747
3: -12.2981434, 4.8755798, -12.3030987, 4.8750286, -11.6150169, 11.6159248
4: -20.5794907, -2.1457882, -20.5778160, -2.1443434, -12.8263588, 12.8244991
5: -15.6153936, 4.8132935, -15.6132269, 4.8151937, -15.5044632, 15.5078506
6: 2.2679601, 15.6358051, 2.2616539, 15.6362457, -11.5167961, 11.5194721
7: -15.2981300, 6.3345904, -15.3018198, 6.3334956, -14.9641304, 14.9687805
8: -21.3836479, 0.0959954, -21.3775330, 0.0901506, -14.5831108, 14.5742226
9: -8.8586330, 8.9474974, -8.8656864, 8.9464178, -14.7462044, 14.7678452
10: -20.8281288, 5.0453668, -20.8287010, 5.0473185, -21.7145920, 21.7485771
11: -10.8725281, 6.3612304, -10.8689384, 6.3528404, -12.1821899, 12.2017136
12: -13.6071835, 9.2724609, -13.6149683, 9.2808132, -16.9738617, 16.9809761
13: -18.2491226, 4.8255773, -18.2240906, 4.8119588, -20.9613037, 20.9521713
14: -55.3366089, -25.9196758, -55.3416824, -25.9162579, -19.3662415, 19.3870430
15: -24.2728729, -9.2246857, -24.2672272, -9.2307014, -12.8820820, 12.8915977
16: -11.7467384, 12.8140383, -11.7351809, 12.8052626, -21.3904114, 21.4082031
17: -55.9917755, -21.7542171, -56.0008430, -21.7508411, -24.5705757, 24.6010818
18: -21.0054359, 0.8183827, -20.9895000, 0.7921839, -16.6186104, 16.6420364
19: -10.6024685, 1.5287707, -10.6029167, 1.5281818, -12.1306505, 12.1316872
20: -9.6504030, 4.7575803, -9.6537075, 4.7538700, -14.3300705, 14.3249092
21: -15.6008987, 2.6815257, -15.6057606, 2.6769261, -17.1538544, 17.1661644
22: -25.0045357, -5.9038143, -25.0162888, -5.9013367, -19.1031990, 19.1124744
23: -7.8445826, 6.4961891, -7.8475008, 6.4918361, -12.8597031, 12.8605747
24: -13.3883572, 3.7446480, -13.3923893, 3.7432866, -16.9348373, 16.9349098
25: -12.3164024, 3.6583505, -12.3212767, 3.6582477, -15.7599754, 15.7633514
26: -28.1947746, -3.0418262, -28.1895428, -3.0505180, -20.4134369, 20.4270935
27: -13.3205709, 4.6929903, -13.3238487, 4.6852741, -17.4107971, 17.4196243
28: -6.8652267, 9.2306423, -6.8750262, 9.2313204, -14.1143532, 14.1197510
29: -22.0659924, -2.6123114, -22.0802021, -2.6106377, -18.0012512, 18.0074539
30: -11.3348866, 7.9365826, -11.3410645, 7.9342017, -16.3208389, 16.3243065
31: -12.0610018, 2.5885696, -12.0541344, 2.5797641, -14.6407661, 14.6427040
32: -0.5543585, 14.1558800, -0.5594633, 14.1533222, -13.0004883, 12.9943542
33: -14.5411882, 14.1821156, -14.5360985, 14.1867580, -24.1790390, 24.1508408
34: -12.9226971, 8.7303734, -12.9132414, 8.7187862, -16.0878448, 16.0869293
35: -14.2596016, 10.7174988, -14.2417755, 10.7001152, -18.5542221, 18.5330429
36: -13.3359356, 10.9127388, -13.3231506, 10.8987656, -19.2703018, 19.2556915
37: -17.5213375, 7.9521527, -17.5165901, 7.9526424, -20.4469376, 20.4204903
38: -18.2789879, 10.2350588, -18.2664070, 10.2238474, -24.1511612, 24.1470642
39: -21.6581516, 10.0267582, -21.6469746, 10.0170679, -28.2070160, 28.1779480
40: -8.4053755, 14.9501600, -8.4155521, 14.9549522, -19.6443634, 19.6448326
41: 3.2098494, 15.4810772, 3.2097325, 15.4804611, -10.3020782, 10.2960129
42: 2.8912301, 13.6329679, 2.8918214, 13.6301003, -10.7388706, 10.7411461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 678

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 731

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1673253, upper bound: 7.1682362
time: 30.61 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1673253, upper bound: 7.1707022
time: 41.87 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -23.8111420, -0.3101988, -23.8244019, -0.2978840, -17.2983398, 17.3008785
1: -12.3206692, 4.7334309, -12.3313808, 4.7433023, -11.3963947, 11.3899841
2: -12.0611725, 2.7391160, -12.0648594, 2.7414243, -10.3680534, 10.3620987
3: -12.2984953, 4.8781137, -12.3032389, 4.8846283, -11.6257782, 11.6225281
4: -20.5802288, -2.1435571, -20.5847530, -2.1379175, -12.8341618, 12.8359222
5: -15.6156988, 4.8153524, -15.6175175, 4.8236375, -15.5115929, 15.5134621
6: 2.2661715, 15.6364861, 2.2534585, 15.6390600, -11.5225849, 11.5283051
7: -15.3019924, 6.3354349, -15.3132830, 6.3468051, -14.9820824, 14.9648018
8: -21.3840523, 0.0998864, -21.3843594, 0.1034458, -14.5931091, 14.5827789
9: -8.8608246, 8.9481888, -8.8767176, 8.9536190, -14.7548409, 14.7870789
10: -20.8326454, 5.0471644, -20.8434124, 5.0605822, -21.7310028, 21.7614594
11: -10.8889475, 6.3618989, -10.9151869, 6.3908291, -12.2376137, 12.2156658
12: -13.6083403, 9.2746611, -13.6272049, 9.2875347, -16.9744682, 16.9988098
13: -18.2505836, 4.8416123, -18.2708874, 4.8583269, -20.9835129, 21.0149155
14: -55.3373947, -25.9185600, -55.3433647, -25.9102898, -19.3737030, 19.3886681
15: -24.2734108, -9.2194500, -24.2763100, -9.2137718, -12.8929062, 12.9047585
16: -11.7562532, 12.8153400, -11.7675705, 12.8270893, -21.4225769, 21.4183617
17: -55.9919853, -21.7530174, -55.9944687, -21.7429981, -24.5853233, 24.6138458
18: -21.0156326, 0.8198433, -21.0214119, 0.8230524, -16.6602745, 16.6606979
19: -10.6104450, 1.5294712, -10.6264257, 1.5438982, -12.1543436, 12.1558971
20: -9.6580582, 4.7579832, -9.6761055, 4.7728772, -14.3569145, 14.3346901
21: -15.6174927, 2.6820335, -15.6538286, 2.7110980, -17.2054062, 17.1808472
22: -25.0146713, -5.9029627, -25.0465355, -5.8739929, -19.1406784, 19.1435738
23: -7.8525848, 6.4972239, -7.8695717, 6.5147400, -12.8898659, 12.8704624
24: -13.3995743, 3.7452793, -13.4255505, 3.7745161, -16.9781036, 16.9558144
25: -12.3235245, 3.6590564, -12.3425245, 3.6801934, -15.7890625, 15.7730598
26: -28.2012787, -3.0401564, -28.2086010, -3.0311923, -20.4403725, 20.4434967
27: -13.3350582, 4.6939068, -13.3659744, 4.7246046, -17.4657364, 17.4379349
28: -6.8709536, 9.2314634, -6.8904252, 9.2464285, -14.1356850, 14.1272163
29: -22.0777512, -2.6118383, -22.1146755, -2.5767374, -18.0477600, 18.0338058
30: -11.3487310, 7.9374881, -11.3813858, 7.9741287, -16.3749847, 16.3351364
31: -12.0733852, 2.5891478, -12.0908213, 2.6060905, -14.6794758, 14.6799688
32: -0.5569921, 14.1559515, -0.5718071, 14.1536674, -13.0072594, 13.0065804
33: -14.5434513, 14.1827164, -14.5594864, 14.1881027, -24.1748810, 24.1772881
34: -12.9234896, 8.7389355, -12.9322090, 8.7447453, -16.0929680, 16.1152687
35: -14.2607965, 10.7278490, -14.2724743, 10.7302847, -18.5596962, 18.5744476
36: -13.3371506, 10.9236364, -13.3538828, 10.9304495, -19.2742081, 19.2972031
37: -17.5251274, 7.9528646, -17.5475121, 7.9544249, -20.4493637, 20.4524307
38: -18.2800064, 10.2504253, -18.3059273, 10.2702808, -24.1647110, 24.2017136
39: -21.6617184, 10.0326633, -21.6888371, 10.0335484, -28.2073822, 28.2261887
40: -8.4092674, 14.9508457, -8.4364223, 14.9505234, -19.6763916, 19.6653748
41: 3.2083359, 15.4826326, 3.1957588, 15.4853363, -10.2989159, 10.3127441
42: 2.8859434, 13.6333714, 2.8768897, 13.6410122, -10.7550688, 10.7564812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 731

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1710858, upper bound: 7.1681183
time: 35.95 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1710858, upper bound: 7.1705850
time: 32.54 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -23.8268700, -0.2953920, -23.8156452, -0.3065481, -17.3026085, 17.3078537
1: -12.3337994, 4.7449226, -12.3280392, 4.7359309, -11.3915138, 11.4092293
2: -12.0660725, 2.7424109, -12.0644655, 2.7391500, -10.3688240, 10.3704548
3: -12.3046942, 4.8861289, -12.3050251, 4.8772736, -11.6204815, 11.6303120
4: -20.5860920, -2.1364517, -20.5796452, -2.1419425, -12.8402176, 12.8361320
5: -15.6188784, 4.8275042, -15.6145916, 4.8187585, -15.5186539, 15.5224609
6: 2.2515678, 15.6394901, 2.2590170, 15.6372185, -11.5353794, 11.5267143
7: -15.3150129, 6.3492403, -15.3078175, 6.3351998, -14.9704475, 14.9904785
8: -21.3860283, 0.1020226, -21.3787727, 0.0917747, -14.5878677, 14.5903282
9: -8.8776989, 8.9565067, -8.8677864, 8.9488201, -14.7837620, 14.7792740
10: -20.8408661, 5.0679274, -20.8301048, 5.0543175, -21.7515945, 21.7736168
11: -10.9145126, 6.3922434, -10.8840275, 6.3538699, -12.1976814, 12.2474880
12: -13.6276741, 9.2945747, -13.6164370, 9.2881184, -17.0021362, 16.9972458
13: -18.2712402, 4.8530817, -18.2252293, 4.8211069, -20.9964294, 20.9698448
14: -55.3439331, -25.9051323, -55.3432350, -25.9119053, -19.3789787, 19.4063530
15: -24.2767792, -9.2156630, -24.2681770, -9.2280645, -12.8975697, 12.9024372
16: -11.7636576, 12.8319874, -11.7393446, 12.8083038, -21.4212608, 21.4269142
17: -55.9950104, -21.7333450, -56.0014725, -21.7451649, -24.6000443, 24.6213417
18: -21.0139427, 0.8244514, -20.9916420, 0.7936378, -16.6305923, 16.6510239
19: -10.6274252, 1.5442334, -10.6109638, 1.5289899, -12.1564150, 12.1551971
20: -9.6782970, 4.7735062, -9.6633987, 4.7543960, -14.3529968, 14.3525352
21: -15.6576633, 2.7115340, -15.6253853, 2.6775410, -17.1875458, 17.2151947
22: -25.0533276, -5.8735590, -25.0329590, -5.9003963, -19.1529312, 19.1594009
23: -7.8717837, 6.5155458, -7.8570480, 6.4927564, -12.8808479, 12.8896255
24: -13.4295826, 3.7757130, -13.4067955, 3.7443261, -16.9706116, 16.9812355
25: -12.3467741, 3.6814799, -12.3320379, 3.6595259, -15.7856636, 15.7995834
26: -28.2069397, -3.0298691, -28.1933956, -3.0488844, -20.4282799, 20.4434891
27: -13.3715286, 4.7248850, -13.3412724, 4.6859503, -17.4514198, 17.4695511
28: -6.8961811, 9.2464962, -6.8860016, 9.2319498, -14.1407280, 14.1477051
29: -22.1228676, -2.5761757, -22.0998211, -2.6098995, -18.0442276, 18.0644684
30: -11.3861685, 7.9748535, -11.3593349, 7.9354105, -16.3462334, 16.3815079
31: -12.0859938, 2.6067793, -12.0627060, 2.5805502, -14.6665440, 14.6694851
32: -0.5750122, 14.1538782, -0.5641561, 14.1535749, -13.0211830, 13.0108776
33: -14.5648594, 14.1887627, -14.5415020, 14.1875496, -24.2037888, 24.1674500
34: -12.9342480, 8.7391157, -12.9151144, 8.7213154, -16.1013794, 16.0957832
35: -14.2755871, 10.7219591, -14.2445059, 10.7013988, -18.5719032, 18.5430298
36: -13.3574915, 10.9229794, -13.3264961, 10.9023533, -19.2889748, 19.2688560
37: -17.5517464, 7.9545450, -17.5220680, 7.9533129, -20.4819641, 20.4333649
38: -18.3086796, 10.2655945, -18.2689037, 10.2338572, -24.1877213, 24.1602936
39: -21.6931057, 10.0294867, -21.6523743, 10.0181875, -28.2391663, 28.1985016
40: -8.4410028, 14.9505568, -8.4214993, 14.9556217, -19.6821213, 19.6581230
41: 3.1923370, 15.4839211, 3.2064147, 15.4813786, -10.3187923, 10.3020439
42: 2.8776922, 13.6412239, 2.8884163, 13.6307535, -10.7530613, 10.7528076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 695

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 731

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1673253, upper bound: 7.1687378
time: 42.13 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1673253, upper bound: 7.1711972
time: 32.06 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -23.8288498, -0.2949848, -23.8282127, -0.2959347, -17.3177071, 17.3206787
1: -12.3361340, 4.7459893, -12.3363857, 4.7449574, -11.4041977, 11.4077339
2: -12.0665388, 2.7438374, -12.0661287, 2.7426529, -10.3745079, 10.3726501
3: -12.3050556, 4.8886557, -12.3051710, 4.8868465, -11.6312523, 11.6369076
4: -20.5868607, -2.1342201, -20.5865898, -2.1355104, -12.8480301, 12.8475418
5: -15.6191483, 4.8295636, -15.6188669, 4.8272305, -15.5258102, 15.5280724
6: 2.2497864, 15.6401682, 2.2508316, 15.6400070, -11.5411682, 11.5355473
7: -15.3189125, 6.3500667, -15.3193111, 6.3484998, -14.9883842, 14.9864883
8: -21.3864555, 0.1059411, -21.3855972, 0.1050687, -14.5978699, 14.5988846
9: -8.8798971, 8.9572010, -8.8788319, 8.9560194, -14.7923851, 14.7985229
10: -20.8453979, 5.0696840, -20.8448067, 5.0675926, -21.7679749, 21.7865295
11: -10.9309034, 6.3929367, -10.9302483, 6.3918762, -12.2531128, 12.2614594
12: -13.6288128, 9.2967567, -13.6286306, 9.2948647, -17.0027580, 17.0150948
13: -18.2726860, 4.8690977, -18.2720299, 4.8674402, -21.0186691, 21.0326080
14: -55.3446693, -25.9040184, -55.3449326, -25.9059601, -19.3864098, 19.4079666
15: -24.2773342, -9.2104340, -24.2772598, -9.2111187, -12.9084129, 12.9156151
16: -11.7732038, 12.8333149, -11.7717295, 12.8301163, -21.4533653, 21.4370880
17: -55.9951553, -21.7320976, -55.9951134, -21.7373543, -24.6148300, 24.6341171
18: -21.0241528, 0.8259354, -21.0235386, 0.8245544, -16.6722488, 16.6696892
19: -10.6354008, 1.5449355, -10.6344814, 1.5446990, -12.1800995, 12.1794167
20: -9.6859722, 4.7738767, -9.6858120, 4.7734189, -14.3798523, 14.3623085
21: -15.6742764, 2.7120285, -15.6734514, 2.7117338, -17.2391014, 17.2298813
22: -25.0634499, -5.8727379, -25.0632324, -5.8730879, -19.1903610, 19.1904945
23: -7.8797903, 6.5165772, -7.8791342, 6.5156665, -12.9109917, 12.8995209
24: -13.4407816, 3.7763658, -13.4399509, 3.7755494, -17.0138931, 17.0021286
25: -12.3538876, 3.6821580, -12.3532887, 3.6814616, -15.8147354, 15.8092804
26: -28.2134304, -3.0282087, -28.2124176, -3.0295610, -20.4552040, 20.4598923
27: -13.3860340, 4.7258024, -13.3833847, 4.7252693, -17.5063248, 17.4878464
28: -6.9019156, 9.2473087, -6.9014053, 9.2470942, -14.1620522, 14.1551781
29: -22.1346245, -2.5757236, -22.1342964, -2.5759573, -18.0906906, 18.0908356
30: -11.4000082, 7.9757776, -11.3996716, 7.9753399, -16.4003944, 16.3923378
31: -12.0983763, 2.6073594, -12.0994081, 2.6068618, -14.7052383, 14.7067680
32: -0.5776370, 14.1539173, -0.5764768, 14.1539383, -13.0279388, 13.0230923
33: -14.5671616, 14.1893511, -14.5649204, 14.1889353, -24.1996231, 24.1938400
34: -12.9350433, 8.7477179, -12.9340687, 8.7472858, -16.1065063, 16.1241341
35: -14.2767849, 10.7323074, -14.2751436, 10.7316132, -18.5773849, 18.5844383
36: -13.3587036, 10.9338884, -13.3572025, 10.9340153, -19.2928429, 19.3103523
37: -17.5555267, 7.9552808, -17.5529556, 7.9550867, -20.4843826, 20.4652786
38: -18.3097210, 10.2809868, -18.3084221, 10.2803259, -24.2012939, 24.2149429
39: -21.6966820, 10.0354271, -21.6942101, 10.0346651, -28.2395630, 28.2466965
40: -8.4448624, 14.9512014, -8.4423370, 14.9512491, -19.7141571, 19.6786308
41: 3.1908240, 15.4854679, 3.1924415, 15.4862576, -10.3156261, 10.3187790
42: 2.8724241, 13.6416330, 2.8734508, 13.6416626, -10.7692385, 10.7681828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 678

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 731

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1710858, upper bound: 7.1686258
time: 39.57 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1673253, upper bound: 7.1710857
time: 52.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 94.40 seconds
IS_B1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 94.40
Output dim: 41, lower bound: -7.1686258, upper bound: 7.1635817
IS_B1_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 94.40
Output dim: 41, lower bound: -7.1648407, upper bound: 7.1635817
IS_B2_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 94.40
Output dim: 41, lower bound: -7.1673253, upper bound: 7.1682362
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 94.40
Output dim: 41, lower bound: -7.1673253, upper bound: 7.1707022
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 94.40
Output dim: 41, lower bound: -7.1710858, upper bound: 7.1681183
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 94.40
Output dim: 41, lower bound: -7.1710858, upper bound: 7.1705850
IS_B2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 94.40
Output dim: 41, lower bound: -7.1673253, upper bound: 7.1687378
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 94.40
Output dim: 41, lower bound: -7.1673253, upper bound: 7.1711972
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 94.40
Output dim: 41, lower bound: -7.1710858, upper bound: 7.1686258
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 94.40
Output dim: 41, lower bound: -7.1673253, upper bound: 7.1710857

## BFS IS instance: IS_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -23.8044987, -0.3112488, -23.8104000, -0.3087006, -17.2709694, 17.2860794
1: -12.3159485, 4.7318182, -12.3222389, 4.7341032, -11.3750172, 11.3905506
2: -12.0594101, 2.7370017, -12.0628548, 2.7377143, -10.3518963, 10.3589211
3: -12.2964087, 4.8749104, -12.3024788, 4.8748331, -11.6111927, 11.6143684
4: -20.5769119, -2.1465702, -20.5770092, -2.1445847, -12.8156929, 12.8232079
5: -15.6133575, 4.8126531, -15.6126566, 4.8150001, -15.5020294, 15.5064888
6: 2.2695160, 15.6330996, 2.2620983, 15.6355305, -11.5146942, 11.5113335
7: -15.2948093, 6.3341098, -15.3009014, 6.3333354, -14.9564438, 14.9673119
8: -21.3828163, 0.0953085, -21.3772831, 0.0899556, -14.5561371, 14.5726166
9: -8.8583450, 8.9463224, -8.8655834, 8.9460783, -14.7455139, 14.7648888
10: -20.8276978, 5.0436320, -20.8285942, 5.0467992, -21.7135010, 21.7416534
11: -10.8717403, 6.3605251, -10.8687267, 6.3526258, -12.1810951, 12.2004890
12: -13.6061373, 9.2712526, -13.6146631, 9.2804585, -16.9723206, 16.9572792
13: -18.2484055, 4.8232412, -18.2238712, 4.8112931, -20.9583893, 20.9424973
14: -55.3299866, -25.9212952, -55.3399048, -25.9167023, -19.3489265, 19.3843155
15: -24.2725658, -9.2262344, -24.2671318, -9.2311516, -12.8799572, 12.8757210
16: -11.7439165, 12.8133135, -11.7344227, 12.8050423, -21.3951187, 21.4051552
17: -55.9911804, -21.7561035, -56.0007095, -21.7514191, -24.5739441, 24.5979385
18: -21.0045891, 0.8176279, -20.9892616, 0.7919569, -16.6173782, 16.6392899
19: -10.6011753, 1.5285810, -10.6025267, 1.5281203, -12.1292953, 12.1311073
20: -9.6493797, 4.7573256, -9.6533985, 4.7537923, -14.3281174, 14.3315201
21: -15.5990763, 2.6813517, -15.6052074, 2.6768832, -17.1521111, 17.1653481
22: -25.0036774, -5.9050779, -25.0160522, -5.9017010, -19.1019764, 19.1109734
23: -7.8431892, 6.4959459, -7.8470936, 6.4917822, -12.8560677, 12.8594055
24: -13.3867207, 3.7440805, -13.3918991, 3.7431207, -16.9237595, 16.9338379
25: -12.3155737, 3.6576991, -12.3210430, 3.6580696, -15.7547417, 15.7623749
26: -28.1941261, -3.0429211, -28.1893330, -3.0508080, -20.4111671, 20.4173470
27: -13.3188992, 4.6923256, -13.3233805, 4.6850677, -17.4046860, 17.4183311
28: -6.8630590, 9.2300758, -6.8744164, 9.2311478, -14.1080017, 14.1186066
29: -22.0649223, -2.6127510, -22.0798988, -2.6107674, -17.9957008, 18.0184746
30: -11.3334885, 7.9358163, -11.3406773, 7.9339523, -16.3115997, 16.3230400
31: -12.0593491, 2.5884011, -12.0536366, 2.5797000, -14.6390495, 14.6420374
32: -0.5529838, 14.1551113, -0.5590763, 14.1531096, -12.9942436, 12.9988823
33: -14.5399084, 14.1816750, -14.5357113, 14.1866598, -24.1754608, 24.1488037
34: -12.9207544, 8.7297430, -12.9125929, 8.7186012, -16.0833168, 16.0839272
35: -14.2580109, 10.7171602, -14.2413368, 10.7000027, -18.5475693, 18.5322037
36: -13.3346558, 10.9108582, -13.3227711, 10.8982487, -19.2651215, 19.2710571
37: -17.5199680, 7.9496613, -17.5161629, 7.9519448, -20.4445572, 20.4118195
38: -18.2759743, 10.2347212, -18.2654877, 10.2237463, -24.1483765, 24.1477356
39: -21.6557198, 10.0262194, -21.6462593, 10.0168848, -28.2011337, 28.1767654
40: -8.4041824, 14.9498110, -8.4151983, 14.9548492, -19.6428719, 19.6437683
41: 3.2118635, 15.4793692, 3.2103181, 15.4799881, -10.3000107, 10.2820892
42: 2.8922396, 13.6295090, 2.8921361, 13.6290998, -10.7368603, 10.7373734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=216, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 678

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 710

## Relational analysis of IS_B2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1578003, upper bound: 7.1669686
time: 37.76 seconds

## Relational analysis of IS_B2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1631185, upper bound: 7.1664648
time: 44.25 seconds

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -23.7900658, -0.3182344, -23.8182869, -0.2992706, -17.2765617, 17.2829361
1: -12.3071270, 4.7268481, -12.3268709, 4.7415066, -11.3789062, 11.3779945
2: -12.0482435, 2.7291863, -12.0604753, 2.7398560, -10.3508224, 10.3445358
3: -12.2878628, 4.8652372, -12.3000441, 4.8822002, -11.6139221, 11.6093521
4: -20.5720749, -2.1518192, -20.5825005, -2.1399965, -12.8206367, 12.8189430
5: -15.6095257, 4.8054790, -15.6158047, 4.8205132, -15.5023499, 15.5006714
6: 2.2775517, 15.6288309, 2.2558308, 15.6370840, -11.5056324, 11.5166245
7: -15.2898026, 6.3276734, -15.3092537, 6.3449678, -14.9689903, 14.9523888
8: -21.3596764, 0.0835056, -21.3761311, 0.1023557, -14.5655174, 14.5568085
9: -8.8503723, 8.9411516, -8.8752174, 8.9512730, -14.7411804, 14.7765083
10: -20.8160763, 5.0220685, -20.8415031, 5.0522780, -21.7057533, 21.7333298
11: -10.8821316, 6.3558602, -10.9129028, 6.3895884, -12.2283630, 12.2052174
12: -13.5847540, 9.2410183, -13.6261044, 9.2763300, -16.9378662, 16.9604645
13: -18.2401352, 4.8180943, -18.2697392, 4.8504066, -20.9645157, 20.9900742
14: -55.3218689, -25.9267578, -55.3384705, -25.9126644, -19.3458786, 19.3653774
15: -24.2620010, -9.2378168, -24.2753220, -9.2201576, -12.8723526, 12.8851643
16: -11.7422848, 12.8095512, -11.7637300, 12.8247051, -21.4040718, 21.4101372
17: -55.9812241, -21.7824764, -55.9939651, -21.7523613, -24.5576706, 24.5819778
18: -21.0055161, 0.8089767, -21.0193844, 0.8200064, -16.6471481, 16.6476555
19: -10.6001625, 1.5254565, -10.6232948, 1.5433800, -12.1435423, 12.1487513
20: -9.6487017, 4.7541375, -9.6732960, 4.7720885, -14.3447723, 14.3211365
21: -15.6043091, 2.6769669, -15.6497192, 2.7105255, -17.1886902, 17.1702461
22: -25.0024567, -5.9093685, -25.0440083, -5.8755722, -19.1268845, 19.1346397
23: -7.8386354, 6.4932961, -7.8650093, 6.5140266, -12.8746834, 12.8610344
24: -13.3760309, 3.7293262, -13.4172153, 3.7736111, -16.9525299, 16.9302177
25: -12.3091526, 3.6496582, -12.3376770, 3.6790657, -15.7719307, 15.7576599
26: -28.1785793, -3.0691929, -28.2063141, -3.0408263, -20.4198189, 20.4205780
27: -13.3053904, 4.6785917, -13.3560457, 4.7239423, -17.4337158, 17.4113007
28: -6.8513708, 9.2241058, -6.8839016, 9.2457056, -14.1151390, 14.1124992
29: -22.0672569, -2.6174860, -22.1114578, -2.5775928, -18.0350914, 18.0131531
30: -11.3353062, 7.9260063, -11.3769503, 7.9728832, -16.3575516, 16.3165283
31: -12.0545559, 2.5799170, -12.0847082, 2.6055551, -14.6601105, 14.6646252
32: -0.5443125, 14.1532593, -0.5684652, 14.1532078, -12.9920731, 12.9935398
33: -14.5273209, 14.1772213, -14.5553761, 14.1869936, -24.1566315, 24.1669998
34: -12.9052763, 8.7318411, -12.9265871, 8.7434559, -16.0728531, 16.1021500
35: -14.2438602, 10.7205601, -14.2672844, 10.7293243, -18.5413361, 18.5610313
36: -13.3259926, 10.9177780, -13.3505993, 10.9285088, -19.2632675, 19.2805290
37: -17.4993286, 7.9417782, -17.5423717, 7.9505434, -20.4203720, 20.4360962
38: -18.2640305, 10.2425184, -18.3015022, 10.2687378, -24.1452866, 24.1835098
39: -21.6387558, 10.0198851, -21.6824989, 10.0323515, -28.1826782, 28.2071762
40: -8.3898363, 14.9492559, -8.4313812, 14.9502163, -19.6549950, 19.6580238
41: 3.2269278, 15.4725695, 3.1995158, 15.4814453, -10.2735672, 10.2947483
42: 2.9027247, 13.6262665, 2.8793807, 13.6387014, -10.7359772, 10.7468853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=216, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 710

## Relational analysis of IS_B2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1615458, upper bound: 7.1643846
time: 50.83 seconds

## Relational analysis of IS_B2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1668253, upper bound: 7.1638818
time: 36.19 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -23.8064861, -0.3108606, -23.8229313, -0.2980995, -17.2860565, 17.2988815
1: -12.3182793, 4.7328920, -12.3305683, 4.7431335, -11.3877182, 11.3890648
2: -12.0598660, 2.7384126, -12.0645247, 2.7412016, -10.3575764, 10.3611374
3: -12.2967739, 4.8774452, -12.3026495, 4.8843985, -11.6219654, 11.6209679
4: -20.5776653, -2.1443300, -20.5839386, -2.1381674, -12.8234997, 12.8346558
5: -15.6136684, 4.8146820, -15.6169586, 4.8234406, -15.5091438, 15.5120964
6: 2.2677298, 15.6337690, 2.2539196, 15.6383171, -11.5204849, 11.5201550
7: -15.2986860, 6.3349309, -15.3123970, 6.3466692, -14.9743958, 14.9633293
8: -21.3832302, 0.0992291, -21.3841324, 0.1032660, -14.5661583, 14.5811729
9: -8.8605318, 8.9470148, -8.8766346, 8.9532700, -14.7541447, 14.7841225
10: -20.8322468, 5.0454102, -20.8433056, 5.0600982, -21.7298965, 21.7545242
11: -10.8881378, 6.3612022, -10.9149284, 6.3906355, -12.2365227, 12.2144356
12: -13.6072884, 9.2734432, -13.6268997, 9.2871933, -16.9729233, 16.9751129
13: -18.2498341, 4.8392682, -18.2706757, 4.8576617, -20.9805946, 21.0052605
14: -55.3307571, -25.9201565, -55.3415985, -25.9107800, -19.3563728, 19.3859367
15: -24.2731018, -9.2210264, -24.2762184, -9.2142315, -12.8907757, 12.8888721
16: -11.7534580, 12.8146534, -11.7667923, 12.8268890, -21.4272804, 21.4153290
17: -55.9913559, -21.7548161, -55.9943237, -21.7435760, -24.5886421, 24.6107025
18: -21.0148220, 0.8191109, -21.0211601, 0.8228483, -16.6590614, 16.6579323
19: -10.6091747, 1.5292804, -10.6260319, 1.5438420, -12.1530170, 12.1553125
20: -9.6570568, 4.7577162, -9.6757984, 4.7728033, -14.3549690, 14.3412971
21: -15.6156797, 2.6818826, -15.6532326, 2.7110620, -17.2036476, 17.1800346
22: -25.0137844, -5.9042063, -25.0462837, -5.8743944, -19.1393890, 19.1420784
23: -7.8511753, 6.4969931, -7.8691721, 6.5146618, -12.8862152, 12.8693047
24: -13.3979425, 3.7447100, -13.4250755, 3.7743373, -16.9670486, 16.9547119
25: -12.3226976, 3.6584032, -12.3422871, 3.6800158, -15.7838097, 15.7720566
26: -28.2005920, -3.0412912, -28.2083702, -3.0315046, -20.4380913, 20.4337578
27: -13.3334055, 4.6932650, -13.3654938, 4.7244124, -17.4596405, 17.4366302
28: -6.8687911, 9.2308884, -6.8898134, 9.2462721, -14.1293221, 14.1260757
29: -22.0766869, -2.6123018, -22.1143646, -2.5768518, -18.0421753, 18.0448494
30: -11.3473253, 7.9367161, -11.3810120, 7.9738955, -16.3657455, 16.3338776
31: -12.0717316, 2.5889869, -12.0903273, 2.6060348, -14.6777668, 14.6793137
32: -0.5556273, 14.1551790, -0.5714080, 14.1534538, -13.0010147, 13.0111065
33: -14.5421715, 14.1822777, -14.5591202, 14.1879997, -24.1713104, 24.1751900
34: -12.9214993, 8.7382908, -12.9315510, 8.7445736, -16.0884323, 16.1122665
35: -14.2591705, 10.7275391, -14.2720089, 10.7302170, -18.5530434, 18.5736084
36: -13.3358841, 10.9217129, -13.3534975, 10.9299088, -19.2689819, 19.3125381
37: -17.5237312, 7.9503746, -17.5470963, 7.9537196, -20.4469948, 20.4437332
38: -18.2770061, 10.2501230, -18.3050156, 10.2701931, -24.1619492, 24.2023849
39: -21.6592846, 10.0321293, -21.6880989, 10.0334053, -28.2014923, 28.2250366
40: -8.4080696, 14.9504814, -8.4360466, 14.9504452, -19.6748962, 19.6643105
41: 3.2103491, 15.4809151, 3.1963396, 15.4848690, -10.2968445, 10.2988338
42: 2.8869720, 13.6299276, 2.8771896, 13.6399975, -10.7530251, 10.7527380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=216, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 710

## Relational analysis of IS_B2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1615458, upper bound: 7.1668552
time: 31.75 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1668253, upper bound: 7.1663539
time: 30.08 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -23.8222122, -0.2960682, -23.8142052, -0.3067503, -17.2903061, 17.3058777
1: -12.3313808, 4.7443991, -12.3272438, 4.7357635, -11.3828430, 11.4083061
2: -12.0647850, 2.7417061, -12.0641251, 2.7389464, -10.3583317, 10.3695049
3: -12.3029814, 4.8854475, -12.3044224, 4.8770738, -11.6166573, 11.6287594
4: -20.5835190, -2.1372404, -20.5788307, -2.1421680, -12.8295593, 12.8348408
5: -15.6168232, 4.8268337, -15.6140413, 4.8185711, -15.5162163, 15.5211258
6: 2.2531118, 15.6367836, 2.2594757, 15.6364737, -11.5332718, 11.5185680
7: -15.3116913, 6.3487749, -15.3069401, 6.3350554, -14.9627762, 14.9890137
8: -21.3851929, 0.1013143, -21.3785362, 0.0915668, -14.5608902, 14.5887260
9: -8.8773918, 8.9553242, -8.8677216, 8.9484730, -14.7830658, 14.7763252
10: -20.8404427, 5.0661545, -20.8299789, 5.0538034, -21.7504883, 21.7667046
11: -10.9137516, 6.3915267, -10.8837910, 6.3536730, -12.1966057, 12.2462616
12: -13.6266212, 9.2933865, -13.6161327, 9.2877874, -17.0006065, 16.9735832
13: -18.2705345, 4.8507690, -18.2250156, 4.8204250, -20.9935608, 20.9601746
14: -55.3372841, -25.9067745, -55.3414459, -25.9123573, -19.3616447, 19.4036293
15: -24.2764778, -9.2172194, -24.2680855, -9.2285175, -12.8954334, 12.8865738
16: -11.7608500, 12.8312664, -11.7385292, 12.8081017, -21.4259605, 21.4238815
17: -55.9944191, -21.7351761, -56.0012817, -21.7457409, -24.6033936, 24.6182175
18: -21.0131207, 0.8236861, -20.9914188, 0.7934275, -16.6293793, 16.6482773
19: -10.6261349, 1.5440371, -10.6105881, 1.5289302, -12.1550655, 12.1546249
20: -9.6772785, 4.7732425, -9.6631145, 4.7543297, -14.3510361, 14.3591614
21: -15.6558704, 2.7113652, -15.6247978, 2.6775112, -17.1857986, 17.2143669
22: -25.0525017, -5.8748322, -25.0327072, -5.9007683, -19.1517334, 19.1578751
23: -7.8703699, 6.5153003, -7.8566389, 6.4926996, -12.8772163, 12.8884468
24: -13.4279289, 3.7751670, -13.4063158, 3.7441545, -16.9595604, 16.9801521
25: -12.3459492, 3.6808398, -12.3318071, 3.6593528, -15.7803993, 15.7985878
26: -28.2062588, -3.0309381, -28.1931839, -3.0491567, -20.4260101, 20.4337349
27: -13.3698769, 4.7242351, -13.3407974, 4.6857691, -17.4453087, 17.4682693
28: -6.8939972, 9.2459307, -6.8853855, 9.2317991, -14.1343613, 14.1465530
29: -22.1217995, -2.5766220, -22.0995293, -2.6100388, -18.0386620, 18.0755119
30: -11.3847885, 7.9740782, -11.3589392, 7.9351940, -16.3370056, 16.3802452
31: -12.0843391, 2.6066144, -12.0622091, 2.5805025, -14.6648417, 14.6688232
32: -0.5736532, 14.1531019, -0.5637536, 14.1533804, -13.0149384, 13.0154419
33: -14.5635796, 14.1883249, -14.5411396, 14.1874304, -24.2002029, 24.1653671
34: -12.9322758, 8.7385025, -12.9144859, 8.7211227, -16.0968323, 16.0928001
35: -14.2739878, 10.7216148, -14.2440510, 10.7012882, -18.5652466, 18.5422173
36: -13.3562088, 10.9210949, -13.3261213, 10.9018345, -19.2837753, 19.2842026
37: -17.5503807, 7.9521008, -17.5216484, 7.9526229, -20.4796066, 20.4247055
38: -18.3056278, 10.2652740, -18.2680073, 10.2337494, -24.1849747, 24.1610107
39: -21.6906471, 10.0289602, -21.6516628, 10.0180244, -28.2332993, 28.1973114
40: -8.4398260, 14.9502258, -8.4211550, 14.9555130, -19.6806183, 19.6570435
41: 3.1943502, 15.4822140, 3.2069969, 15.4809122, -10.3167305, 10.2881393
42: 2.8787227, 13.6377659, 2.8887157, 13.6297493, -10.7510262, 10.7490501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 650

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 710

## Relational analysis of IS_B2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1578003, upper bound: 7.1674559
time: 38.44 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1631185, upper bound: 7.1669270
time: 35.44 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -23.8078270, -0.3030348, -23.8221035, -0.2972965, -17.2958870, 17.3027287
1: -12.3226051, 4.7394285, -12.3318567, 4.7431483, -11.3867188, 11.3957291
2: -12.0536184, 2.7338903, -12.0617371, 2.7411039, -10.3572731, 10.3551102
3: -12.2944450, 4.8758068, -12.3019323, 4.8844366, -11.6193810, 11.6237469
4: -20.5787067, -2.1424794, -20.5843430, -2.1376095, -12.8344879, 12.8305721
5: -15.6129904, 4.8196745, -15.6171761, 4.8240967, -15.5165443, 15.5153008
6: 2.2611609, 15.6325150, 2.2531891, 15.6380348, -11.5242062, 11.5238781
7: -15.3066845, 6.3423500, -15.3152657, 6.3466854, -14.9752998, 14.9740715
8: -21.3620815, 0.0895610, -21.3773613, 0.1039882, -14.5702705, 14.5728989
9: -8.8694553, 8.9501801, -8.8773365, 8.9536572, -14.7787151, 14.7879257
10: -20.8288193, 5.0446110, -20.8429127, 5.0592680, -21.7427254, 21.7583771
11: -10.9241343, 6.3868694, -10.9279881, 6.3906164, -12.2438736, 12.2509975
12: -13.6052217, 9.2631531, -13.6275625, 9.2836609, -16.9661560, 16.9767532
13: -18.2622662, 4.8455791, -18.2708702, 4.8595333, -20.9996414, 21.0077362
14: -55.3291855, -25.9122849, -55.3400536, -25.9083061, -19.3585815, 19.3846893
15: -24.2659111, -9.2287855, -24.2762947, -9.2175150, -12.8878517, 12.8960171
16: -11.7592201, 12.8274746, -11.7678690, 12.8277187, -21.4348755, 21.4288788
17: -55.9843903, -21.7616234, -55.9945641, -21.7467365, -24.5871658, 24.6022530
18: -21.0140266, 0.8150272, -21.0215359, 0.8214779, -16.6591492, 16.6566505
19: -10.6251249, 1.5409352, -10.6313496, 1.5441864, -12.1693115, 12.1722851
20: -9.6766176, 4.7700453, -9.6829929, 4.7726302, -14.3677025, 14.3487854
21: -15.6610994, 2.7069876, -15.6693697, 2.7111530, -17.2223663, 17.2192879
22: -25.0512543, -5.8791428, -25.0606651, -5.8746281, -19.1766262, 19.1815224
23: -7.8658438, 6.5126534, -7.8745680, 6.5149479, -12.8958397, 12.8900909
24: -13.4172630, 3.7604017, -13.4316282, 3.7746563, -16.9883156, 16.9765396
25: -12.3395376, 3.6727629, -12.3484459, 3.6803508, -15.7976074, 15.7938881
26: -28.1907959, -3.0572448, -28.2101212, -3.0392118, -20.4346466, 20.4369774
27: -13.3563547, 4.7105026, -13.3734636, 4.7246408, -17.4743347, 17.4612541
28: -6.8823218, 9.2399673, -6.8948698, 9.2463770, -14.1415062, 14.1404610
29: -22.1241474, -2.5813675, -22.1310940, -2.5768414, -18.0780678, 18.0701714
30: -11.3865938, 7.9642754, -11.3952560, 7.9740953, -16.3829765, 16.3737411
31: -12.0795670, 2.5981266, -12.0932808, 2.6063404, -14.6859074, 14.6914072
32: -0.5649724, 14.1512566, -0.5731444, 14.1534691, -13.0127678, 13.0100632
33: -14.5510359, 14.1838312, -14.5607729, 14.1878014, -24.1813812, 24.1835747
34: -12.9168444, 8.7406054, -12.9284658, 8.7460117, -16.0863953, 16.1110191
35: -14.2598619, 10.7250214, -14.2699881, 10.7306137, -18.5590210, 18.5710144
36: -13.3475094, 10.9280319, -13.3539524, 10.9321241, -19.2819443, 19.2936821
37: -17.5297623, 7.9442253, -17.5478420, 7.9512205, -20.4553795, 20.4489937
38: -18.2937317, 10.2730789, -18.3040314, 10.2787876, -24.1818542, 24.1967621
39: -21.6736927, 10.0226431, -21.6878910, 10.0334663, -28.2148514, 28.2276764
40: -8.4254532, 14.9496298, -8.4373169, 14.9508820, -19.6927948, 19.6713028
41: 3.2093902, 15.4754143, 3.1961932, 15.4823723, -10.2902889, 10.3007889
42: 2.8891921, 13.6345234, 2.8759489, 13.6393452, -10.7501526, 10.7585745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 678

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 710

## Relational analysis of IS_B2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1615458, upper bound: 7.1648866
time: 34.71 seconds

## Relational analysis of IS_B2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1668253, upper bound: 7.1643683
time: 31.06 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -23.8242378, -0.2956505, -23.8267345, -0.2961540, -17.3053932, 17.3187065
1: -12.3337622, 4.7454677, -12.3355751, 4.7448039, -11.3955460, 11.4068108
2: -12.0652428, 2.7431297, -12.0657797, 2.7424412, -10.3640232, 10.3717022
3: -12.3033495, 4.8879809, -12.3045673, 4.8866482, -11.6274261, 11.6353455
4: -20.5842743, -2.1350155, -20.5857697, -2.1357470, -12.8373528, 12.8462715
5: -15.6171303, 4.8288836, -15.6183167, 4.8270144, -15.5233536, 15.5267220
6: 2.2513347, 15.6374550, 2.2512817, 15.6392765, -11.5390530, 11.5274086
7: -15.3155832, 6.3495941, -15.3184261, 6.3483715, -14.9807091, 14.9850082
8: -21.3856316, 0.1052544, -21.3853703, 0.1048801, -14.5709381, 14.5972633
9: -8.8796062, 8.9560242, -8.8787498, 8.9556675, -14.7916946, 14.7955704
10: -20.8449936, 5.0679455, -20.8446655, 5.0670815, -21.7668724, 21.7795982
11: -10.9301348, 6.3922024, -10.9300184, 6.3916941, -12.2520332, 12.2602329
12: -13.6277504, 9.2955465, -13.6283321, 9.2945261, -17.0012398, 16.9914093
13: -18.2719803, 4.8667727, -18.2718315, 4.8667717, -21.0157700, 21.0229301
14: -55.3380814, -25.9056778, -55.3431244, -25.9064388, -19.3690872, 19.4052353
15: -24.2770309, -9.2120008, -24.2771645, -9.2115860, -12.9062843, 12.8997326
16: -11.7703562, 12.8325949, -11.7709265, 12.8299303, -21.4580917, 21.4340630
17: -55.9945221, -21.7339535, -55.9949265, -21.7379131, -24.6181641, 24.6309776
18: -21.0233135, 0.8251839, -21.0232964, 0.8243432, -16.6710358, 16.6669312
19: -10.6341352, 1.5447494, -10.6340904, 1.5446361, -12.1787710, 12.1788397
20: -9.6849518, 4.7736273, -9.6855125, 4.7733431, -14.3779106, 14.3689194
21: -15.6724625, 2.7118673, -15.6728535, 2.7116814, -17.2373428, 17.2290764
22: -25.0625877, -5.8739843, -25.0629768, -5.8734684, -19.1891193, 19.1889915
23: -7.8783941, 6.5163431, -7.8787155, 6.5156012, -12.9073715, 12.8983555
24: -13.4391575, 3.7758107, -13.4394722, 3.7753935, -17.0028381, 17.0010376
25: -12.3530684, 3.6815243, -12.3530521, 3.6812823, -15.8095169, 15.8082924
26: -28.2127647, -3.0293193, -28.2121849, -3.0298867, -20.4529152, 20.4501228
27: -13.3843784, 4.7251673, -13.3829079, 4.7250872, -17.5002289, 17.4865570
28: -6.8997478, 9.2467365, -6.9007764, 9.2469435, -14.1556854, 14.1540375
29: -22.1335545, -2.5761805, -22.1340256, -2.5761023, -18.0851517, 18.1018677
30: -11.3986130, 7.9749718, -11.3992968, 7.9750991, -16.3911819, 16.3910789
31: -12.0967264, 2.6071908, -12.0989208, 2.6068339, -14.7035599, 14.7061119
32: -0.5762582, 14.1531639, -0.5760813, 14.1537228, -13.0216942, 13.0276661
33: -14.5658894, 14.1889038, -14.5645275, 14.1887999, -24.1960602, 24.1917572
34: -12.9330635, 8.7470741, -12.9334316, 8.7470980, -16.1019554, 16.1211090
35: -14.2752094, 10.7319851, -14.2746887, 10.7314968, -18.5707169, 18.5835991
36: -13.3574238, 10.9319897, -13.3568392, 10.9334984, -19.2876701, 19.3256950
37: -17.5541363, 7.9527979, -17.5525665, 7.9543810, -20.4820099, 20.4566078
38: -18.3066788, 10.2806902, -18.3075409, 10.2802172, -24.1985016, 24.2156372
39: -21.6942291, 10.0349264, -21.6934929, 10.0345192, -28.2336807, 28.2455597
40: -8.4437008, 14.9508457, -8.4420004, 14.9511471, -19.7126617, 19.6775780
41: 3.1928287, 15.4837675, 3.1930237, 15.4857941, -10.3135586, 10.3048668
42: 2.8734441, 13.6381721, 2.8737593, 13.6406479, -10.7672043, 10.7644129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=217, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 678

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 710

## Relational analysis of IS_B2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1615458, upper bound: 7.1673493
time: 45.12 seconds

## Relational analysis of IS_B2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1668253, upper bound: 7.1668253
time: 19.01 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 66.38 seconds
IS_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 66.38
Output dim: 41, lower bound: -7.1578003, upper bound: 7.1669686
IS_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 66.38
Output dim: 41, lower bound: -7.1631185, upper bound: 7.1664648
IS_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 66.38
Output dim: 41, lower bound: -7.1615458, upper bound: 7.1643846
IS_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 66.38
Output dim: 41, lower bound: -7.1668253, upper bound: 7.1638818
IS_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 66.38
Output dim: 41, lower bound: -7.1615458, upper bound: 7.1668552
IS_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 66.38
Output dim: 41, lower bound: -7.1668253, upper bound: 7.1663539
IS_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 66.38
Output dim: 41, lower bound: -7.1578003, upper bound: 7.1674559
IS_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 66.38
Output dim: 41, lower bound: -7.1631185, upper bound: 7.1669270
IS_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 66.38
Output dim: 41, lower bound: -7.1615458, upper bound: 7.1648866
IS_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 66.38
Output dim: 41, lower bound: -7.1668253, upper bound: 7.1643683
IS_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 66.38
Output dim: 41, lower bound: -7.1615458, upper bound: 7.1673493
IS_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 66.38
Output dim: 41, lower bound: -7.1668253, upper bound: 7.1668253

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 58.02 + 1433.80 = 1491.82 seconds
