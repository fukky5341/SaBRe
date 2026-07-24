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
execution time: IAR + RelationalAnalysis = 2.72 + 56.42 = 59.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 41, lower bound: -7.1773280, upper bound: 7.1773280

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 657

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 635

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1756064, upper bound: 7.1740957
time: 44.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1740957, upper bound: 7.1756064
time: 37.96 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 82.33 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 82.33
Output dim: 41, lower bound: -7.1756064, upper bound: 7.1740957
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 82.33
Output dim: 41, lower bound: -7.1740957, upper bound: 7.1756064

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3236809, 17.3245049
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4179344, 11.4216347
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3724022, 10.3750381
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6378155, 11.6380672
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8501511, 12.8513813
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5269890, 15.5316887
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5392532, 11.5427990
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -15.0009880, 15.0056305
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6038361, 14.6035767
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8075600, 14.8063507
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7945442, 21.7965126
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2928772, 12.2982464
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0207481, 17.0249863
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0453873, 21.0495033
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4154854, 19.4156685
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9206181, 12.9262657
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4656868, 21.4673500
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6249237, 24.6362534
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6813889, 16.6826591
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3811150, 14.3788071
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2651291, 17.2693863
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9151039, 12.9142532
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0202560, 17.0166435
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8253479, 15.8237228
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4637375, 20.4654732
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5168991, 17.5173531
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1702003, 14.1653938
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0958710, 18.1062202
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4302521, 16.4275932
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0272102, 13.0272865
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2124557, 24.2094574
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1292763, 16.1173058
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.6052971, 18.5992279
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3232727, 19.3215637
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4984970, 20.4979286
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2386932, 24.2294922
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2645187, 28.2612991
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7134399, 19.7086601
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3274689, 10.3291435
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1641

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 549

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1719713, upper bound: 7.1738831
time: 34.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1753936, upper bound: 7.1704605
time: 31.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3245049, 17.3236809
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4216347, 11.4179344
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3750420, 10.3724003
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6380672, 11.6378136
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8513794, 12.8501511
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5316887, 15.5269890
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5428009, 11.5392570
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -15.0056305, 15.0009880
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6035767, 14.6038361
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8063507, 14.8075600
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7965126, 21.7945442
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2982483, 12.2928791
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0249825, 17.0207520
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0494995, 21.0453873
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4156685, 19.4154854
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9262676, 12.9206200
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4673500, 21.4656868
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6362534, 24.6249275
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6826553, 16.6813927
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3788071, 14.3811150
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2693863, 17.2651329
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9142532, 12.9151039
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0166397, 17.0202599
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8237228, 15.8253517
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4654770, 20.4637375
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5173492, 17.5168991
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1653938, 14.1702003
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1062164, 18.0958710
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4275894, 16.4302521
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0272865, 13.0272102
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2094574, 24.2124557
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1173058, 16.1292763
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5992241, 18.6053009
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3215637, 19.3232727
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4979324, 20.4984970
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2294922, 24.2386932
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2612991, 28.2645187
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7086563, 19.7134361
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3291473, 10.3274651
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 675

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1703

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1494655, upper bound: 7.1508095
time: 36.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1494668, upper bound: 7.1508084
time: 46.21 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 84.48 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 84.48
Output dim: 41, lower bound: -7.1719713, upper bound: 7.1738831
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 84.48
Output dim: 41, lower bound: -7.1753936, upper bound: 7.1704605
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 84.48
Output dim: 41, lower bound: -7.1494655, upper bound: 7.1508095
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 84.48
Output dim: 41, lower bound: -7.1494668, upper bound: 7.1508084

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3107071, 17.3147526
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4129944, 11.4185143
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3699512, 10.3731556
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6376305, 11.6379185
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8500557, 12.8512993
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5266342, 15.5312996
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5384598, 11.5417614
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -15.0012703, 15.0060272
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6034775, 14.6044197
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8072624, 14.8058395
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7938042, 21.7957344
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2917404, 12.2968826
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0224915, 17.0265045
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0397263, 21.0445862
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4154892, 19.4157791
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9203815, 12.9259758
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4649048, 21.4667587
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6315041, 24.6437454
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6754227, 16.6763611
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3731880, 14.3687401
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2574120, 17.2590141
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9166794, 12.9160557
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0211029, 17.0174789
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8240623, 15.8218269
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4526825, 20.4520035
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5178108, 17.5162544
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1689911, 14.1643066
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0959549, 18.1062546
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4223785, 16.4171295
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0291405, 13.0287590
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2104263, 24.2071457
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1229668, 16.1114807
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5983658, 18.5928268
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3216934, 19.3198700
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4902496, 20.4903030
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2286758, 24.2219925
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2641296, 28.2608643
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7120018, 19.7071609
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3219967, 10.3243961
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1656

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1304

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1717816, upper bound: 7.1736617
time: 21.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1717507, upper bound: 7.1736928
time: 22.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3139305, 17.3115273
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4148140, 11.4166985
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3705158, 10.3725929
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6376610, 11.6378880
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8500710, 12.8512878
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5265999, 15.5313339
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5382195, 11.5420036
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -15.0013847, 15.0059128
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6046753, 14.6032181
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8070488, 14.8060532
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7937660, 21.7957726
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2915115, 12.2971096
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0222702, 17.0267220
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0404739, 21.0438423
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4155960, 19.4156723
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9203281, 12.9260292
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4650955, 21.4665680
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6324196, 24.6428299
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6750946, 16.6766930
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3710442, 14.3708839
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2547569, 17.2616653
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9169083, 12.9158268
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0210876, 17.0174828
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8234596, 15.8224297
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4502640, 20.4544182
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5158043, 17.5182648
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1691132, 14.1641846
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0959091, 18.1063004
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4197845, 16.4197159
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0286827, 13.0292187
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2101440, 24.2074280
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1234474, 16.1109962
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5988998, 18.5922966
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3215790, 19.3199806
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4908752, 20.4896812
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2311935, 24.2194748
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2640839, 28.2609024
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7119408, 19.7072258
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3227139, 10.3236752
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 692

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1370

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1742735, upper bound: 7.1704543
time: 43.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1753873, upper bound: 7.1693407
time: 22.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 68.37 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 68.37
Output dim: 41, lower bound: -7.1717816, upper bound: 7.1736617
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 68.37
Output dim: 41, lower bound: -7.1717507, upper bound: 7.1736928
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 68.37
Output dim: 41, lower bound: -7.1742735, upper bound: 7.1704543
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 68.37
Output dim: 41, lower bound: -7.1753873, upper bound: 7.1693407

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3109627, 17.3150139
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4114361, 11.4169598
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3699570, 10.3731728
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6373062, 11.6376648
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8500252, 12.8512726
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5264206, 15.5311317
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5386276, 11.5420094
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -15.0003510, 15.0054283
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6028328, 14.6038895
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8073254, 14.8060341
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7936630, 21.7956924
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2908325, 12.2958584
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0221443, 17.0260658
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0393982, 21.0443153
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4153290, 19.4155006
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9200554, 12.9257240
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4631767, 21.4650497
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6272659, 24.6394806
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6759491, 16.6767120
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3725662, 14.3679962
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2573967, 17.2589569
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9166832, 12.9160595
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0210571, 17.0174255
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8240547, 15.8217392
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4522629, 20.4515038
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5175781, 17.5159683
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1690331, 14.1643181
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0962067, 18.1064796
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4225807, 16.4171867
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0279922, 13.0277386
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2105408, 24.2072716
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1231499, 16.1116257
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5976982, 18.5921516
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3217163, 19.3198967
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4906120, 20.4905701
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2287750, 24.2221222
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2639694, 28.2607651
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7117195, 19.7068825
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3220367, 10.3245201
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 642

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 738

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1641342, upper bound: 7.1736003
time: 33.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1717203, upper bound: 7.1660128
time: 19.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3109665, 17.3150101
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4114437, 11.4169540
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3699722, 10.3731613
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6373825, 11.6375885
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8500290, 12.8512669
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5264664, 15.5310860
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5387115, 11.5419292
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -15.0006676, 15.0051117
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6029472, 14.6037750
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8074551, 14.8059044
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7937622, 21.7955971
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2907181, 12.2959747
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0220451, 17.0261612
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0394592, 21.0442581
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4152145, 19.4156189
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9201317, 12.9256477
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4631920, 21.4650307
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6272354, 24.6395111
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6757812, 16.6768875
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3724442, 14.3681183
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2573509, 17.2590027
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9166832, 12.9160633
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0210495, 17.0174408
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8239708, 15.8218231
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4521713, 20.4515915
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5175247, 17.5160217
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1690025, 14.1643486
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0961761, 18.1065063
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4224281, 16.4173393
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0281219, 13.0276070
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2105560, 24.2072563
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1231117, 16.1116600
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5976982, 18.5921516
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3217163, 19.3198967
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4905205, 20.4906731
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2288055, 24.2220917
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2640305, 28.2607040
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7117195, 19.7068825
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3221207, 10.3244343
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 739

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 725

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1701989, upper bound: 7.1716146
time: 41.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1696789, upper bound: 7.1721353
time: 42.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3135681, 17.3111248
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4148407, 11.4166775
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3704681, 10.3725319
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6378613, 11.6378975
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8494740, 12.8505974
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5268822, 15.5315132
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5380898, 11.5418549
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -15.0014839, 15.0058250
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6046600, 14.6031532
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8069839, 14.8061333
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7935982, 21.7953949
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2914352, 12.2970123
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0223007, 17.0267372
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0405731, 21.0439339
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4130630, 19.4129257
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9197044, 12.9252396
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4650307, 21.4664040
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6297379, 24.6393394
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6743507, 16.6759682
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3710518, 14.3708954
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2550087, 17.2617912
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9164314, 12.9152603
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0209274, 17.0172462
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8236694, 15.8225822
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4491882, 20.4532890
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5155182, 17.5178986
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1682510, 14.1634903
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0961685, 18.1064758
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4197197, 16.4196167
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0278664, 13.0286541
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2095871, 24.2069473
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1241608, 16.1118813
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5998993, 18.5934792
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3229980, 19.3217735
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4914360, 20.4901276
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2333450, 24.2221222
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2640076, 28.2608414
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7120361, 19.7074165
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3224983, 10.3235149
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 726

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1304

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1740838, upper bound: 7.1702330
time: 33.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1740529, upper bound: 7.1702640
time: 34.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3135262, 17.3111668
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4147949, 11.4167252
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3704605, 10.3725433
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6376781, 11.6380806
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8493824, 12.8506966
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5267792, 15.5316200
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5380707, 11.5418739
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -15.0013008, 15.0060120
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6046104, 14.6032028
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8071289, 14.8059883
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7933922, 21.7956009
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2914200, 12.2970314
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0222855, 17.0267525
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0405655, 21.0439453
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4128532, 19.4131317
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9195404, 12.9254055
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4649315, 21.4664993
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6289291, 24.6401443
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6743660, 16.6759529
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3710556, 14.3708878
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2548866, 17.2619171
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9163437, 12.9153481
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0208588, 17.0173149
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8236084, 15.8226433
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4491425, 20.4533310
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5154419, 17.5179825
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1684189, 14.1633224
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0960846, 18.1065636
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4196892, 16.4196510
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0281143, 13.0284042
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2096710, 24.2068634
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1243362, 16.1117020
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.6000824, 18.5932999
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3233719, 19.3214035
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4913216, 20.4902420
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2338409, 24.2216263
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2640152, 28.2608261
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7121277, 19.7073212
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3225594, 10.3234539
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1718

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 741

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1714532, upper bound: 7.1688642
time: 33.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1749065, upper bound: 7.1654135
time: 31.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 67.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 67.51
Output dim: 41, lower bound: -7.1641342, upper bound: 7.1736003
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 67.51
Output dim: 41, lower bound: -7.1717203, upper bound: 7.1660128
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 67.51
Output dim: 41, lower bound: -7.1701989, upper bound: 7.1716146
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 67.51
Output dim: 41, lower bound: -7.1696789, upper bound: 7.1721353
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 67.51
Output dim: 41, lower bound: -7.1740838, upper bound: 7.1702330
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 67.51
Output dim: 41, lower bound: -7.1740529, upper bound: 7.1702640
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 67.51
Output dim: 41, lower bound: -7.1714532, upper bound: 7.1688642
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 67.51
Output dim: 41, lower bound: -7.1749065, upper bound: 7.1654135

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2991676, 17.2993565
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4005547, 11.4025116
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3649693, 10.3665524
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6361465, 11.6361275
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8415012, 12.8387508
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5223618, 15.5257378
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5330238, 11.5377884
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9911995, 14.9932747
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5921288, 14.5896873
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8045712, 14.8023834
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7808609, 21.7786942
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2904282, 12.2954426
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0246124, 17.0295296
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0398712, 21.0447350
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3894730, 19.3813286
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9143181, 12.9173470
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4604111, 21.4615211
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6002884, 24.6048775
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6770020, 16.6770020
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3722534, 14.3676682
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2616653, 17.2623444
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9168854, 12.9168434
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0196228, 17.0162659
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8250618, 15.8226738
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4562302, 20.4550705
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5188217, 17.5170174
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1634598, 14.1605339
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1021004, 18.1112671
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4140549, 16.4111633
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0283279, 13.0282555
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1941376, 24.1949196
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1078110, 16.1000786
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5769806, 18.5765457
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3093300, 19.3105659
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4797058, 20.4823532
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2290421, 24.2223663
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2587051, 28.2566452
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7017479, 19.6994514
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3187599, 10.3228588
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 740

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 737

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1566149, upper bound: 7.1734926
time: 20.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1640115, upper bound: 7.1630155
time: 36.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2953072, 17.3032207
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3969879, 11.4060745
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3633366, 10.3681850
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6357689, 11.6365051
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8375034, 12.8427486
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5210342, 15.5270691
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5344086, 11.5364037
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9882011, 14.9962730
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5886269, 14.5931892
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8036709, 14.8032837
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7766647, 21.7828865
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2904205, 12.2954521
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0256119, 17.0285339
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0398178, 21.0447884
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3811607, 19.3896408
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9116783, 12.9199867
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4596481, 21.4622841
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5926666, 24.6125031
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6762390, 16.6777611
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3722382, 14.3676834
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2607803, 17.2632256
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9174652, 12.9162636
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0198975, 17.0159912
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8249855, 15.8227501
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4558334, 20.4554634
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5186234, 17.5172157
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1652451, 14.1587448
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1009941, 18.1123734
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4165573, 16.4086533
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0285072, 13.0280762
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1981964, 24.1908760
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1115952, 16.0962982
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5820923, 18.5714340
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3123817, 19.3075066
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4823990, 20.4796600
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2290192, 24.2223892
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2598495, 28.2554932
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7042961, 19.6969032
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3203735, 10.3212433
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 755

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 692

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1705010, upper bound: 7.1610257
time: 49.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1667414, upper bound: 7.1647832
time: 24.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3109550, 17.3150826
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3996048, 11.4081001
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3711281, 10.3746185
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6317902, 11.6335011
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8498611, 12.8511238
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5261955, 15.5312195
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5388718, 11.5419788
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9836082, 14.9930344
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6067085, 14.6075096
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8114204, 14.8121758
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7990646, 21.8031120
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2554665, 12.2694283
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0144768, 17.0169678
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0259056, 21.0310287
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4129066, 19.4119110
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9174175, 12.9234753
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4569702, 21.4651451
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6319199, 24.6459541
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6758423, 16.6768456
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3655396, 14.3620872
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2249298, 17.2344666
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9080925, 12.9091034
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0123444, 17.0099907
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8158417, 15.8142014
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4521141, 20.4515686
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5046082, 17.5047493
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1617317, 14.1578522
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0748291, 18.0904083
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3855972, 16.3895645
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0283852, 13.0261173
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2088394, 24.2052307
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1210442, 16.1085815
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5954971, 18.5885468
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3193283, 19.3118362
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4868889, 20.4853516
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2153778, 24.1980133
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2639694, 28.2557831
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7089386, 19.7002068
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3215332, 10.3228207
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 724

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1692476, upper bound: 7.1716032
time: 16.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1701875, upper bound: 7.1706692
time: 29.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3110390, 17.3149986
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4025879, 11.4051170
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3714256, 10.3743191
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6332932, 11.6319981
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8498878, 12.8510971
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5265961, 15.5308189
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5387611, 11.5420914
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9885902, 14.9880562
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6066818, 14.6075363
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8137245, 14.8098717
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.8012772, 21.8009033
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2641716, 12.2607231
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0128517, 17.0185852
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0262260, 21.0307083
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4115028, 19.4133148
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9179592, 12.9229355
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4633026, 21.4588089
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6336823, 24.6441994
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6757355, 16.6769485
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3664093, 14.3612175
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2328186, 17.2265854
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9097214, 12.9074745
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0135956, 17.0087395
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8163528, 15.8136864
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4521599, 20.4515228
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5062561, 17.5031052
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1625061, 14.1570816
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0800781, 18.0851555
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3946533, 16.3805084
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0266304, 13.0278702
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2085266, 24.2055435
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1200294, 16.1095886
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5940857, 18.5899544
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3136520, 19.3175087
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4851952, 20.4870453
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2047195, 24.2086639
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2591095, 28.2606506
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7050476, 19.7041016
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3205070, 10.3238468
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1370

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 619

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1695526, upper bound: 7.1666670
time: 36.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1634418, upper bound: 7.1720027
time: 28.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3138275, 17.3113880
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4132843, 11.4151268
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3704681, 10.3725433
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6375256, 11.6376438
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8494453, 12.8505726
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5266800, 15.5313530
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5382576, 11.5421066
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -15.0005608, 15.0052147
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6040154, 14.6026230
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8070488, 14.8063278
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7934532, 21.7953453
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2905273, 12.2959862
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0219612, 17.0263023
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0402603, 21.0436707
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4129105, 19.4126549
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9193668, 12.9249763
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4632950, 21.4646873
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6255150, 24.6350861
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6748695, 16.6763191
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3704414, 14.3701591
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2550049, 17.2617340
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9164467, 12.9152718
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0208855, 17.0171928
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8236618, 15.8224907
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4487762, 20.4527931
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5152931, 17.5176163
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1682892, 14.1634979
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0964241, 18.1067009
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4199371, 16.4196739
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0267181, 13.0276394
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2096939, 24.2070694
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1243324, 16.1120224
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5992203, 18.5927925
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3230209, 19.3217888
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4917831, 20.4903793
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2334442, 24.2222519
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2638626, 28.2607574
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7117615, 19.7071381
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3225327, 10.3236408
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 738

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1686

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1649297, upper bound: 7.1686119
time: 17.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1724422, upper bound: 7.1610946
time: 32.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3138313, 17.3113861
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4132919, 11.4151230
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3704796, 10.3725319
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6376019, 11.6375675
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8494530, 12.8505669
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5267220, 15.5313110
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5383415, 11.5420246
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -15.0008736, 15.0049019
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6041298, 14.6025085
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8071823, 14.8061981
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7935448, 21.7952499
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2904129, 12.2961025
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0218697, 17.0263977
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0403137, 21.0436172
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4127884, 19.4127731
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9194431, 12.9249001
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4633102, 21.4646683
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6254845, 24.6351166
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6747017, 16.6764908
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3703194, 14.3702850
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2549591, 17.2617798
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9164429, 12.9152756
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0208702, 17.0172043
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8235855, 15.8225708
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4486847, 20.4528809
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5152397, 17.5176697
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1682587, 14.1635284
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0963936, 18.1067276
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4197845, 16.4198303
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0268517, 13.0275078
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2097092, 24.2070541
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1243019, 16.1120529
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5992203, 18.5927925
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3230209, 19.3217888
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4916840, 20.4904785
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2334747, 24.2222214
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2639160, 28.2606964
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7117615, 19.7071381
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3226242, 10.3235569
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1717

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1622815, upper bound: 7.1699194
time: 55.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1737085, upper bound: 7.1584784
time: 41.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3071098, 17.3046513
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4130630, 11.4148483
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3702660, 10.3725796
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6386509, 11.6395187
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8471184, 12.8488979
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5302505, 15.5362434
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5384979, 11.5422688
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9995384, 15.0045929
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6027336, 14.6013603
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8129787, 14.8139114
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7919540, 21.7943001
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2884407, 12.2947407
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0127525, 17.0148621
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0431213, 21.0473213
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3897934, 19.3828678
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9166088, 12.9240932
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4541016, 21.4555130
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5848770, 24.5899315
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6602859, 16.6581001
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3666420, 14.3674850
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2563400, 17.2638741
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9232750, 12.9214096
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0273972, 17.0225029
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8281059, 15.8261757
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4514275, 20.4557877
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5185699, 17.5207863
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1750107, 14.1686974
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0987930, 18.1103859
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4229355, 16.4233284
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0224075, 13.0227432
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2057571, 24.2027283
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1281548, 16.1141777
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.6007996, 18.5938301
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3192444, 19.3171310
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4922562, 20.4909439
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2353973, 24.2224503
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2571945, 28.2532883
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7081604, 19.7027855
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3204842, 10.3213902
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 723

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1690533, upper bound: 7.1681792
time: 33.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1707807, upper bound: 7.1664863
time: 35.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3070107, 17.3047523
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4129143, 11.4149952
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3704948, 10.3723526
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6391125, 11.6390591
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8475838, 12.8484325
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5314064, 15.5350914
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5384598, 11.5423012
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9998779, 15.0042496
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6027679, 14.6013260
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8150501, 14.8118362
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7920914, 21.7941589
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2891273, 12.2940540
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0103951, 17.0172272
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0439453, 21.0465012
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3825874, 19.3900738
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9182262, 12.9224777
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4539490, 21.4556656
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5787125, 24.5960884
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6565094, 16.6618729
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3676567, 14.3664703
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2568436, 17.2633781
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9224052, 12.9222794
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0260391, 17.0238609
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8271446, 15.8271446
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4515953, 20.4556198
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5182343, 17.5211143
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1737938, 14.1699142
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0999069, 18.1092720
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4233627, 16.4228973
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0224533, 13.0226955
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2055283, 24.2029572
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1268120, 16.1155243
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.6006165, 18.5940170
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3190994, 19.3172684
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4920273, 20.4911766
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2346649, 24.2231827
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2564850, 28.2540054
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7075958, 19.7033501
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3204918, 10.3213825
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 658

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 620

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1745521, upper bound: 7.1643452
time: 32.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1738354, upper bound: 7.1650431
time: 32.91 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 67.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1566149, upper bound: 7.1734926
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1640115, upper bound: 7.1630155
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1705010, upper bound: 7.1610257
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1667414, upper bound: 7.1647832
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1692476, upper bound: 7.1716032
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1701875, upper bound: 7.1706692
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1695526, upper bound: 7.1666670
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1634418, upper bound: 7.1720027
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1649297, upper bound: 7.1686119
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1724422, upper bound: 7.1610946
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1622815, upper bound: 7.1699194
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1737085, upper bound: 7.1584784
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1690533, upper bound: 7.1681792
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1707807, upper bound: 7.1664863
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1745521, upper bound: 7.1643452
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 67.04
Output dim: 41, lower bound: -7.1738354, upper bound: 7.1650431

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2875519, 17.2838745
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3889694, 11.3871403
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3608646, 10.3611050
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6366501, 11.6364594
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8470764, 12.8400116
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5191383, 15.5214653
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5269871, 11.5336208
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9811630, 14.9799652
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5812912, 14.5753021
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8033543, 14.8007050
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7668037, 21.7600479
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2849922, 12.2887859
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0348740, 17.0386810
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0393829, 21.0443192
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3512077, 19.3305645
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9170952, 12.9171295
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4557419, 21.4553947
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5606499, 24.5531387
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6883392, 16.6842728
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3735046, 14.3684158
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2663879, 17.2646790
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9170799, 12.9170990
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0210648, 17.0175018
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8289146, 15.8255386
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4672050, 20.4632950
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5239716, 17.5207481
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1650925, 14.1635895
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1098633, 18.1167717
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4141769, 16.4123726
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0260315, 13.0264854
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1744156, 24.1800537
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0920792, 16.0882149
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5536804, 18.5589828
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2959671, 19.3004951
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4699783, 20.4750137
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2293091, 24.2226181
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2511749, 28.2509689
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6898117, 19.6904526
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3148689, 10.3217545
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 676

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1769

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1537096, upper bound: 7.1733895
time: 31.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1565086, upper bound: 7.1706003
time: 36.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2961349, 17.3043079
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3817730, 11.3952541
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3637676, 10.3682995
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6357422, 11.6367588
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8375397, 12.8427753
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5204048, 15.5263329
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5340767, 11.5359097
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9649391, 14.9802170
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5860672, 14.5889854
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7979355, 14.8005829
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7711716, 21.7793770
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2454224, 12.2614479
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0153694, 17.0177078
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0136070, 21.0112419
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3798714, 19.3882408
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9043083, 12.9092617
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4278717, 21.4385452
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5928383, 24.6128998
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6560631, 16.6624107
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3547440, 14.3544579
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2143669, 17.2281876
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9023056, 12.9050522
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0020752, 17.0025291
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8085480, 15.8102341
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4522018, 20.4525757
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.4848480, 17.4912376
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1532784, 14.1497116
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0889053, 18.1032448
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3738670, 16.3764267
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0265350, 13.0249519
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1898041, 24.1800194
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0885048, 16.0680962
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5553894, 18.5364456
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2821350, 19.2675056
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4709854, 20.4607620
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.1938477, 24.1757965
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2393951, 28.2283096
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7089882, 19.6971779
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3105888, 10.3092861
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1605

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 740

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1650556, upper bound: 7.1609749
time: 31.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1704535, upper bound: 7.1556155
time: 34.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3081245, 17.3112030
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4026165, 11.4089260
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3700104, 10.3733921
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6358337, 11.6376896
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8461285, 12.8468914
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5321808, 15.5366745
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5386848, 11.5419331
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9813652, 14.9896660
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5900688, 14.5923309
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8122025, 14.8129272
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7885437, 21.7892685
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2356339, 12.2433796
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -16.9977036, 16.9946899
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0263977, 21.0313911
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4066010, 19.4022255
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9175301, 12.9235115
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4471130, 21.4514618
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5984077, 24.6014404
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6611862, 16.6572838
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3675804, 14.3637161
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2343674, 17.2408905
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9125443, 12.9115372
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0168076, 17.0137062
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8218460, 15.8190117
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4596024, 20.4557114
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5103760, 17.5093575
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1676331, 14.1632767
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0879173, 18.0983009
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3939056, 16.3965607
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0293808, 13.0274410
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1981583, 24.1971817
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1133118, 16.1023445
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5828438, 18.5784607
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3124390, 19.3055000
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4856224, 20.4841843
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2073746, 24.1910858
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2456436, 28.2420349
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6975212, 19.6923637
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3219376, 10.3237743
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 737

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1587154, upper bound: 7.1714952
time: 37.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1691400, upper bound: 7.1610645
time: 19.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3070755, 17.3122520
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4004269, 11.4111156
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3699036, 10.3735046
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6359787, 11.6375427
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8456249, 12.8473873
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5316505, 15.5372047
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5388298, 11.5417881
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9802437, 14.9907913
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5915298, 14.5908699
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8121758, 14.8129578
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7852249, 21.7925873
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2294159, 12.2495975
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -16.9921951, 17.0002022
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0262680, 21.0315170
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4032211, 19.4056053
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9174538, 12.9235878
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4432907, 21.4552879
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5874062, 24.6124382
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6562805, 16.6621895
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3671722, 14.3641243
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2313614, 17.2439041
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9105263, 12.9135513
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0160599, 17.0144539
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8206406, 15.8202133
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4562531, 20.4590645
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5092163, 17.5105133
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1671524, 14.1637573
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0827217, 18.1034966
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3925934, 16.3978729
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0297089, 13.0271091
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2007980, 24.1945419
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1148071, 16.1008492
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5854149, 18.5758896
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3129959, 19.3049393
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4857216, 20.4840889
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2084503, 24.1900024
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2502213, 28.2374573
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7010994, 19.6887894
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3224869, 10.3232269
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1593

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 668

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1701115, upper bound: 7.1700332
time: 27.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1695628, upper bound: 7.1705916
time: 60.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3033104, 17.3049049
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3924637, 11.3912010
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3641739, 10.3646660
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6176186, 11.6121807
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8557072, 12.8537292
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5126534, 15.5123749
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5376987, 11.5404587
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9706726, 14.9646759
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5944633, 14.5928764
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8087349, 14.8052902
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7986183, 21.7960320
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2593956, 12.2531719
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0074577, 17.0127907
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0266953, 21.0308228
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4173164, 19.4185562
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9285526, 12.9309654
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4597893, 21.4530563
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6345596, 24.6448097
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6488419, 16.6540031
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3656693, 14.3624001
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2338867, 17.2243881
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9065933, 12.9058342
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0090065, 17.0056801
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8143730, 15.8127747
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4310379, 20.4329147
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5026131, 17.4997368
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1583595, 14.1557617
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0774078, 18.0786514
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3938217, 16.3797836
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0254402, 13.0270634
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2097397, 24.2071762
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0987778, 16.0932846
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5810394, 18.5798721
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2875748, 19.2959824
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4647598, 20.4692917
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.1693344, 24.1818695
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2540665, 28.2568207
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6928253, 19.6949272
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3181667, 10.3215637
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 634

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 755

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1544906, upper bound: 7.1719017
time: 23.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1633406, upper bound: 7.1630502
time: 30.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3093338, 17.3062782
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4085274, 11.4113426
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3721008, 10.3743687
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6374607, 11.6373940
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8483734, 12.8510742
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5260048, 15.5305786
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5330658, 11.5337257
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -15.0005341, 15.0057411
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5978851, 14.5950317
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7877140, 14.7916908
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7533379, 21.7649841
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2703247, 12.2807827
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0104980, 17.0167656
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0401459, 21.0435753
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3857727, 19.3928661
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9130669, 12.9214516
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4545822, 21.4558983
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5942764, 24.6114311
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6597595, 16.6643143
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3768501, 14.3794250
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2498703, 17.2593384
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9187012, 12.9208336
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0256805, 17.0226288
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8243332, 15.8232651
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4330292, 20.4436951
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5177116, 17.5207214
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1723289, 14.1670380
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0939064, 18.1086540
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4227600, 16.4226799
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0297241, 13.0320339
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1953964, 24.1880875
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1211929, 16.1081467
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5819550, 18.5699921
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3122253, 19.3077583
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4717178, 20.4633217
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2244568, 24.2109756
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2529907, 28.2456589
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6917648, 19.6807861
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3184528, 10.3187408
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 609

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 641

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1723408, upper bound: 7.1610941
time: 44.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1724417, upper bound: 7.1609903
time: 28.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2853813, 17.2898998
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3876133, 11.3961678
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3539906, 10.3598614
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6168785, 11.6225185
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8471718, 12.8508739
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5127754, 15.5212631
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5362244, 11.5388927
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9693451, 14.9809532
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6040611, 14.6025314
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8045845, 14.8060913
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7878799, 21.7956009
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2476044, 12.2640686
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0135918, 17.0158043
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0443878, 21.0465088
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.4004669, 19.4011765
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9190941, 12.9243546
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4291687, 21.4392014
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6157875, 24.6425629
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6711464, 16.6743431
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3703690, 14.3703461
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2338028, 17.2458038
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9038391, 12.9057503
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0146103, 17.0127640
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8229637, 15.8220558
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4475250, 20.4524307
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5123253, 17.5153847
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1617889, 14.1580544
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0584221, 18.0780525
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3981781, 16.4033966
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0270691, 13.0234833
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2015533, 24.1963882
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1161919, 16.1020241
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5914764, 18.5827637
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3090210, 19.3031158
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4808273, 20.4745636
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2085419, 24.1892242
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2431107, 28.2331390
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.7086678, 19.6951294
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3201733, 10.3203106
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 722

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 710

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1694661, upper bound: 7.1489340
time: 40.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1641911, upper bound: 7.1542525
time: 36.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2638702, 17.2725544
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3875256, 11.3959522
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3361206, 10.3468056
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6167126, 11.6229572
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8121243, 12.8243008
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.4978065, 15.5117569
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5302315, 11.5313187
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9706955, 14.9828224
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5506783, 14.5617752
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8033314, 14.8067284
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7741241, 21.7809677
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2779999, 12.2807484
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0080910, 17.0065117
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0269012, 21.0354614
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3584976, 19.3568573
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.8928375, 12.9058952
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4557266, 21.4566536
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5548019, 24.5619774
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6497307, 16.6441154
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3601761, 14.3591423
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2459106, 17.2500534
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9155273, 12.9102497
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0174866, 17.0097809
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8175697, 15.8122101
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4441605, 20.4461517
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5106430, 17.5102806
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1565247, 14.1442146
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.0976601, 18.1088867
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3865204, 16.3750877
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0201530, 13.0201626
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1826019, 24.1719322
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0990639, 16.0775948
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5784607, 18.5633926
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3113937, 19.3064690
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4747772, 20.4657631
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2398224, 24.2304153
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2580185, 28.2547073
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6983910, 19.6888123
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3218708, 10.3212929
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 676

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 721

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1632916, upper bound: 7.1663614
time: 50.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1706558, upper bound: 7.1590062
time: 33.53 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 86.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1537096, upper bound: 7.1733895
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1565086, upper bound: 7.1706003
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1650556, upper bound: 7.1609749
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1704535, upper bound: 7.1556155
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1587154, upper bound: 7.1714952
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1691400, upper bound: 7.1610645
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1701115, upper bound: 7.1700332
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1695628, upper bound: 7.1705916
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1544906, upper bound: 7.1719017
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1633406, upper bound: 7.1630502
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1723408, upper bound: 7.1610941
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1724417, upper bound: 7.1609903
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1694661, upper bound: 7.1489340
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1641911, upper bound: 7.1542525
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1632916, upper bound: 7.1663614
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 86.16
Output dim: 41, lower bound: -7.1706558, upper bound: 7.1590062
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 86.16
Output dim: 41, lower bound: -7.1745521, upper bound: 7.1643452
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 86.16
Output dim: 41, lower bound: -7.1738354, upper bound: 7.1650431

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 59.14 + 1753.41 = 1812.54 seconds
