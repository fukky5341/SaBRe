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
execution time: IAR + RelationalAnalysis = 2.82 + 54.86 = 57.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 41, lower bound: -7.1773280, upper bound: 7.1773280

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 593

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1708467, upper bound: 7.1773023
time: 32.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1773023, upper bound: 7.1708467
time: 42.80 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 75.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 75.22
Output dim: 41, lower bound: -7.1708467, upper bound: 7.1773023
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 75.22
Output dim: 41, lower bound: -7.1773023, upper bound: 7.1708467

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3278008, 17.3244877
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4098129, 11.4061775
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3727188, 10.3724422
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6391010, 11.6388626
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8486748, 12.8477879
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5330276, 15.5331116
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5300694, 11.5333443
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9936714, 14.9897346
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6017570, 14.6010857
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.8017406, 14.7981567
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7859726, 21.7816544
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2793465, 12.2730331
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0192680, 17.0207253
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0326118, 21.0369492
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3943863, 19.3863716
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9212837, 12.9218864
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4537811, 21.4494934
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6204453, 24.6185608
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6772919, 16.6723747
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3773613, 14.3741455
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2562256, 17.2510643
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9029427, 12.8987808
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0178223, 17.0136490
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8228455, 15.8205261
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4632111, 20.4605904
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5140533, 17.5094986
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1580658, 14.1548309
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1142654, 18.1159172
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4144936, 16.4088440
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0184555, 13.0204468
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1963577, 24.2004623
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1237526, 16.1243515
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5847321, 18.5902176
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3036804, 19.3094025
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4782944, 20.4843597
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2265549, 24.2294998
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2440796, 28.2489853
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6946907, 19.6992226
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3229103, 10.3265476
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 755

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1618871, upper bound: 7.1771996
time: 35.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1707437, upper bound: 7.1683465
time: 37.03 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3244896, 17.3278008
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.4061775, 11.4098129
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3724442, 10.3727226
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6388645, 11.6391029
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8477898, 12.8486767
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5331116, 15.5330276
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5333462, 11.5300694
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9897346, 14.9936714
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.6010857, 14.6017570
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7981586, 14.8017426
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7816544, 21.7859764
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2730370, 12.2793465
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0207253, 17.0192680
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0369530, 21.0326157
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3863716, 19.3943844
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9218864, 12.9212837
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4494934, 21.4537888
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.6185608, 24.6204453
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6723785, 16.6772919
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3741455, 14.3773613
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2510681, 17.2562256
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.8987808, 12.9029427
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0136490, 17.0178185
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8205261, 15.8228493
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4605865, 20.4632187
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5094986, 17.5140495
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1548309, 14.1580658
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1159134, 18.1142693
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4088402, 16.4144936
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0204468, 13.0184555
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.2004623, 24.1963577
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1243477, 16.1237526
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5902176, 18.5847282
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.3094025, 19.3036842
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4843597, 20.4782944
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2294998, 24.2265549
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2489853, 28.2440796
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6992149, 19.6946907
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3265457, 10.3229122
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 755

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1683465, upper bound: 7.1707437
time: 54.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1771996, upper bound: 7.1618871
time: 50.80 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 107.73 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 107.73
Output dim: 41, lower bound: -7.1618871, upper bound: 7.1771996
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 107.73
Output dim: 41, lower bound: -7.1707437, upper bound: 7.1683465
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 107.73
Output dim: 41, lower bound: -7.1683465, upper bound: 7.1707437
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 107.73
Output dim: 41, lower bound: -7.1771996, upper bound: 7.1618871

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3076172, 17.2976341
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3946953, 11.3860645
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3636475, 10.3603668
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6355419, 11.6341286
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8342953, 12.8273735
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5267982, 15.5248222
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5208435, 11.5272427
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9809303, 14.9727783
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5825920, 14.5756035
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7949295, 14.7890968
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7684250, 21.7583008
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2792511, 12.2729588
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0179482, 17.0207138
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0326805, 21.0369720
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3583527, 19.3384285
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9124508, 12.9098129
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4502563, 21.4458961
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5921974, 24.5809822
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6695061, 16.6642303
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3782883, 14.3751984
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2584991, 17.2530823
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9018097, 12.8981094
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0180397, 17.0139427
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8235626, 15.8212547
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4687080, 20.4669075
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5155220, 17.5109253
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1505623, 14.1492844
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1159401, 18.1177063
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3999634, 16.3980293
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0166206, 13.0190659
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1728134, 24.1827698
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1067047, 16.1115417
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5573807, 18.5696678
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2872162, 19.2970238
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4590187, 20.4698677
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2327728, 24.2350922
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2383118, 28.2438812
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6803474, 19.6884346
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3112717, 10.3178043
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1609300, upper bound: 7.1771882
time: 44.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1618757, upper bound: 7.1762482
time: 29.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3009491, 17.3043060
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3896980, 11.3910618
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3606491, 10.3633652
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6343670, 11.6353054
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8282604, 12.8334064
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5247383, 15.5268822
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5239639, 11.5241222
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9767189, 14.9769897
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5762749, 14.5819206
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7926826, 14.7913475
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7626266, 21.7641029
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2792702, 12.2729397
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0192528, 17.0194016
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0326347, 21.0370178
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3464432, 19.3503380
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9092121, 12.9130554
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4501877, 21.4459610
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5828590, 24.5903168
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6691399, 16.6645966
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3784103, 14.3750725
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2582397, 17.2533302
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9022751, 12.8976479
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0181160, 17.0138664
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8235779, 15.8212395
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4695320, 20.4660797
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5154839, 17.5109673
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1525230, 14.1473274
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1160622, 18.1175842
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4036789, 16.3943176
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0170746, 13.0186119
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1786652, 24.1769180
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1109467, 16.1073074
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5641785, 18.5628700
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2913055, 19.2929344
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4638023, 20.4650803
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2321472, 24.2357178
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2389832, 28.2432175
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6839027, 19.6848717
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3141708, 10.3149052
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1697894, upper bound: 7.1683351
time: 42.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1707323, upper bound: 7.1673939
time: 43.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3043060, 17.3009453
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3910637, 11.3897018
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3633652, 10.3606453
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6353054, 11.6343670
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8334064, 12.8282604
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5268822, 15.5247383
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5241241, 11.5239658
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9769897, 14.9767189
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5819206, 14.5762749
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7913475, 14.7926788
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7641068, 21.7626266
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2729378, 12.2792702
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0194054, 17.0192528
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0370140, 21.0326347
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3503380, 19.3464394
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9130573, 12.9092102
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4459610, 21.4501915
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5903130, 24.5828629
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6645927, 16.6691437
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3750725, 14.3784103
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2533264, 17.2582436
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.8976440, 12.9022751
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0138664, 17.0181122
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8212357, 15.8235779
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4660759, 20.4695358
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5109673, 17.5154800
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1473274, 14.1525230
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1175880, 18.1160583
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3943176, 16.4036789
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0186119, 13.0170746
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1769180, 24.1786652
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1073074, 16.1109428
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5628738, 18.5641747
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2929382, 19.2913055
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4650841, 20.4638023
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2357178, 24.2321472
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2432175, 28.2389832
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6848717, 19.6839104
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3149033, 10.3141689
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 747

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1673939, upper bound: 7.1707323
time: 31.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1683351, upper bound: 7.1697894
time: 44.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2976341, 17.3076172
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3860664, 11.3946972
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3603668, 10.3636436
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6341267, 11.6355438
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8273716, 12.8342953
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5248222, 15.5267982
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5272446, 11.5208454
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9727783, 14.9809265
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5756035, 14.5825920
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7890968, 14.7949295
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7583008, 21.7684250
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2729568, 12.2792511
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0207100, 17.0179443
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0369682, 21.0326805
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3384285, 19.3583508
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9098148, 12.9124527
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4459000, 21.4502563
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5809822, 24.5921974
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6642265, 16.6695137
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3751984, 14.3782883
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2530823, 17.2584953
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.8981094, 12.9018097
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0139427, 17.0180359
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8212509, 15.8235626
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4669075, 20.4687080
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5109215, 17.5155220
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1492882, 14.1505623
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1177101, 18.1159363
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3980255, 16.3999672
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0190659, 13.0166206
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1827698, 24.1728134
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1115417, 16.1067085
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5696640, 18.5573845
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2970276, 19.2872124
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4698677, 20.4590149
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2350922, 24.2327728
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2438812, 28.2383118
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6884346, 19.6803436
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3178024, 10.3112698
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 747

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1762482, upper bound: 7.1618756
time: 35.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1771882, upper bound: 7.1609300
time: 44.03 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 81.42 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 81.42
Output dim: 41, lower bound: -7.1609300, upper bound: 7.1771882
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 81.42
Output dim: 41, lower bound: -7.1618757, upper bound: 7.1762482
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 81.42
Output dim: 41, lower bound: -7.1697894, upper bound: 7.1683351
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 81.42
Output dim: 41, lower bound: -7.1707323, upper bound: 7.1673939
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 81.42
Output dim: 41, lower bound: -7.1673939, upper bound: 7.1707323
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 81.42
Output dim: 41, lower bound: -7.1683351, upper bound: 7.1697894
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 81.42
Output dim: 41, lower bound: -7.1762482, upper bound: 7.1618756
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 81.42
Output dim: 41, lower bound: -7.1771882, upper bound: 7.1609300

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3047752, 17.2937412
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3977203, 11.3868942
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3625374, 10.3591423
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6395855, 11.6383152
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8305511, 12.8231354
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5327797, 15.5302773
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5206566, 11.5271969
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9786987, 14.9694290
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5659523, 14.5604248
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7957153, 14.7898560
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7579041, 21.7444611
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2594261, 12.2469101
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0011826, 16.9984322
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0331650, 21.0373268
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3520432, 19.3287392
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9125633, 12.9098492
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4404068, 21.4322319
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5586853, 24.5364761
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6548386, 16.6446533
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3803291, 14.3768311
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2679367, 17.2595100
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9062614, 12.9005470
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0224991, 17.0176506
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8295670, 15.8260612
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4761963, 20.4710388
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5212746, 17.5155258
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1564636, 14.1547089
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1290359, 18.1256027
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4082718, 16.4050255
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0176201, 13.0203972
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1621323, 24.1747169
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0989685, 16.1053047
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5447388, 18.5595894
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2803192, 19.2906876
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4577484, 20.4687004
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2247925, 24.2281876
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2199936, 28.2301407
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6689339, 19.6806030
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3116760, 10.3187580
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1558278, upper bound: 7.1770635
time: 16.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1607925, upper bound: 7.1702118
time: 48.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3037262, 17.2947922
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3955307, 11.3890858
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3624229, 10.3592567
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6397305, 11.6381683
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8300552, 12.8236313
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5322495, 15.5308037
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5208015, 11.5270519
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9775772, 14.9705505
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5674133, 14.5589638
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7956886, 14.7898827
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7545853, 21.7477798
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2532005, 12.2531281
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -16.9956665, 17.0039444
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0330353, 21.0374527
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3486633, 19.3321190
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9124870, 12.9099255
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4365921, 21.4360542
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5476837, 24.5474739
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6499329, 16.6495590
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3799210, 14.3772392
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2649231, 17.2625198
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9042435, 12.9025650
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0217514, 17.0184021
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8283691, 15.8272667
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4728394, 20.4743958
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5201225, 17.5166817
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1559830, 14.1551895
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1238327, 18.1308022
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4069595, 16.4063377
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0179482, 13.0200653
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1647720, 24.1720772
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1004715, 16.1038055
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5473022, 18.5570183
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2808762, 19.2901268
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4578476, 20.4686050
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2258759, 24.2271118
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2245712, 28.2255630
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6725121, 19.6770287
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3122253, 10.3182106
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1567720, upper bound: 7.1761252
time: 40.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1617381, upper bound: 7.1692713
time: 32.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2970543, 17.3014622
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3905334, 11.3940811
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3594246, 10.3622551
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6385555, 11.6393471
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8240204, 12.8296661
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5301895, 15.5328636
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5239220, 11.5239315
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9733658, 14.9747620
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5610962, 14.5652809
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7934380, 14.7921333
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7487869, 21.7535782
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2532234, 12.2531109
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -16.9969788, 17.0026360
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0329895, 21.0374985
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3367538, 19.3440285
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9092445, 12.9131660
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4365158, 21.4361191
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5383530, 24.5568085
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6495667, 16.6499252
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3800468, 14.3771133
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2646713, 17.2627716
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9047089, 12.9020996
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0218201, 17.0183258
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8283844, 15.8272476
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4736710, 20.4735680
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5200844, 17.5167236
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1579437, 14.1532288
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1239548, 18.1306801
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4106750, 16.4026260
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0184059, 13.0196095
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1706161, 24.1662331
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1047058, 16.0995712
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5541000, 18.5502281
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2849655, 19.2860374
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4626312, 20.4638138
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2252426, 24.2277374
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2252426, 28.2248993
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6760750, 19.6734619
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3151245, 10.3153114
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 739

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1637529, upper bound: 7.1672564
time: 37.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1706092, upper bound: 7.1622939
time: 43.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.3014641, 17.2970543
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3940811, 11.3905315
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3622551, 10.3594208
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6393452, 11.6385536
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8296661, 12.8240223
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5328636, 15.5301895
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5239334, 11.5239220
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9747620, 14.9733658
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5652809, 14.5610962
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7921333, 14.7934380
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7535782, 21.7487869
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2531090, 12.2532215
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0026398, 16.9969749
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0374985, 21.0329933
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3440323, 19.3367519
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9131660, 12.9092445
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4361191, 21.4365234
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5568085, 24.5383530
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6499252, 16.6495705
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3771133, 14.3800468
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2627716, 17.2646713
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9020996, 12.9047089
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0183258, 17.0218201
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8272476, 15.8283844
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4735641, 20.4736671
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5167274, 17.5200806
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1532288, 14.1579437
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1306839, 18.1239548
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4026260, 16.4106750
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0196114, 13.0184059
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1662369, 24.1706200
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0995712, 16.1047058
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5502319, 18.5541000
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2860336, 19.2849655
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4638138, 20.4626350
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2277374, 24.2252426
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2248993, 28.2252426
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6734657, 19.6760750
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3153114, 10.3151226
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 739

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1622939, upper bound: 7.1706092
time: 40.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1607925, upper bound: 7.1637529
time: 41.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2947922, 17.3037262
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3890839, 11.3955269
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3592567, 10.3624210
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6381664, 11.6397324
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8236313, 12.8300571
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5308037, 15.5322495
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5270538, 11.5208015
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9705505, 14.9775772
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5589638, 14.5674133
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7898827, 14.7956886
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7477798, 21.7545853
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2531319, 12.2532024
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0039444, 16.9956665
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0374527, 21.0330353
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3321190, 19.3486633
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9099236, 12.9124870
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4360580, 21.4365883
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5474701, 24.5476913
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6495590, 16.6499367
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3772392, 14.3799210
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2625198, 17.2649231
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9025650, 12.9042435
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0184021, 17.0217476
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8272629, 15.8283653
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4743958, 20.4728394
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5166817, 17.5201225
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1551895, 14.1559830
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1308060, 18.1238327
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4063339, 16.4069633
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0200653, 13.0179501
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1720810, 24.1647682
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1038055, 16.1004715
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5570221, 18.5473061
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2901230, 19.2808762
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4686050, 20.4578476
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2271118, 24.2258759
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2255630, 28.2245712
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6770287, 19.6725082
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3182106, 10.3122253
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 739

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1628077, upper bound: 7.1617381
time: 24.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1761252, upper bound: 7.1567720
time: 32.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2937431, 17.3047752
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3868942, 11.3977184
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3591423, 10.3625336
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6383152, 11.6395836
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8231354, 12.8305531
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5302773, 15.5327797
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5271988, 11.5206566
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9694290, 14.9786987
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5604248, 14.5659523
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7898560, 14.7957153
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7444611, 21.7579041
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2469063, 12.2594223
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -16.9984360, 17.0011787
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0373230, 21.0331650
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3287392, 19.3520432
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9098511, 12.9125633
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4322281, 21.4404106
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5364685, 24.5586891
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6446533, 16.6548424
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3768311, 14.3803291
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2595139, 17.2679367
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9005470, 12.9062614
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0176468, 17.0224953
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8260574, 15.8295708
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4710388, 20.4761963
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5155220, 17.5212784
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1547089, 14.1564636
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1256027, 18.1290321
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4050217, 16.4082756
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0203972, 13.0176201
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1747208, 24.1621284
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.1053009, 16.0989723
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5595856, 18.5447350
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2906876, 19.2803154
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4687042, 20.4577484
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2281876, 24.2247925
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2301407, 28.2199936
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6805992, 19.6689377
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3187599, 10.3116779
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 739

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1702118, upper bound: 7.1607925
time: 35.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1770635, upper bound: 7.1558278
time: 35.50 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 73.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 73.13
Output dim: 41, lower bound: -7.1558278, upper bound: 7.1770635
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 73.13
Output dim: 41, lower bound: -7.1607925, upper bound: 7.1702118
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 73.13
Output dim: 41, lower bound: -7.1567720, upper bound: 7.1761252
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 73.13
Output dim: 41, lower bound: -7.1617381, upper bound: 7.1692713
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 73.13
Output dim: 41, lower bound: -7.1637529, upper bound: 7.1672564
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 73.13
Output dim: 41, lower bound: -7.1706092, upper bound: 7.1622939
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 73.13
Output dim: 41, lower bound: -7.1622939, upper bound: 7.1706092
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 73.13
Output dim: 41, lower bound: -7.1607925, upper bound: 7.1637529
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 73.13
Output dim: 41, lower bound: -7.1628077, upper bound: 7.1617381
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 73.13
Output dim: 41, lower bound: -7.1761252, upper bound: 7.1567720
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 73.13
Output dim: 41, lower bound: -7.1702118, upper bound: 7.1607925
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 73.13
Output dim: 41, lower bound: -7.1770635, upper bound: 7.1558278

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2908096, 17.2751675
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3852501, 11.3703156
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3554764, 10.3497562
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6373272, 11.6353168
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8191109, 12.8075085
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5287743, 15.5249481
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5131931, 11.5215836
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9681435, 14.9553947
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5539703, 14.5445709
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7936859, 14.7871590
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7432098, 21.7249222
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2622871, 12.2508945
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0015411, 16.9999657
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0331039, 21.0371552
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3172836, 19.2841320
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9055824, 12.9005623
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4345551, 21.4252701
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5220337, 24.4895248
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6554489, 16.6448975
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3796654, 14.3762894
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2728271, 17.2642097
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9064331, 12.9008999
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0233688, 17.0187759
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8307800, 15.8275452
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4816208, 20.4771309
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5233765, 17.5180168
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1515884, 14.1510849
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1360474, 18.1324120
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3994026, 16.3983765
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0163727, 13.0194130
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1433868, 24.1606293
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0841980, 16.0938148
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5229073, 18.5431709
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2653580, 19.2794342
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4424438, 20.4571953
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2244263, 24.2277908
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2127151, 28.2233582
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6551628, 19.6702118
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3040371, 10.3131447
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1557661, upper bound: 7.1769967
time: 20.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1557637, upper bound: 7.1769996
time: 38.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2862015, 17.2797737
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3811378, 11.3744278
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3531494, 10.3520832
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6365871, 11.6360607
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8149300, 12.8116932
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5274544, 15.5262680
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5150433, 11.5197334
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9646683, 14.9588737
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5500984, 14.5484428
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7930183, 14.7878265
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7383652, 21.7297707
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2634087, 12.2497730
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0027084, 16.9987679
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0329895, 21.0372696
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3074379, 19.2939796
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9032745, 12.9028683
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4334564, 21.4261589
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5117340, 24.4998245
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6550827, 16.6446381
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3797913, 14.3761673
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2726364, 17.2643776
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9066162, 12.9007130
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0236206, 17.0185165
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8310547, 15.8272629
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4822845, 20.4762726
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5237656, 17.5175934
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1528435, 14.1498299
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1358414, 18.1325150
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4016304, 16.3961449
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0166359, 13.0191498
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1480408, 24.1559830
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0874863, 16.0905266
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5283241, 18.5377579
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2690659, 19.2757263
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4462433, 20.4533958
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2243805, 24.2278290
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2131271, 28.2228622
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6585426, 19.6668282
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3060665, 10.3111153
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1607306, upper bound: 7.1701459
time: 30.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1607283, upper bound: 7.1701486
time: 29.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2897568, 17.2762165
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3830605, 11.3725071
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3553619, 10.3498707
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6374798, 11.6351719
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8186150, 12.8080044
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5282440, 15.5254784
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5133381, 11.5214405
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9670219, 14.9565201
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5554314, 14.5431099
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7936592, 14.7871895
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7398911, 21.7282410
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2560692, 12.2571144
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -16.9960327, 17.0054779
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0329819, 21.0372810
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3139038, 19.2875118
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9055061, 12.9006386
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4307251, 21.4290924
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5110321, 24.5005226
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6505432, 16.6498032
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3792572, 14.3766975
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2698135, 17.2672234
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9044113, 12.9029179
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0226212, 17.0195236
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8295746, 15.8287506
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4782715, 20.4804840
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5222168, 17.5191727
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1511078, 14.1515656
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1308517, 18.1376114
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3980827, 16.3996887
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0167046, 13.0190811
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1460266, 24.1579895
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0856934, 16.0923195
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5254707, 18.5406036
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2659149, 19.2788773
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4425430, 20.4570961
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2255096, 24.2267151
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2172928, 28.2187805
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6587334, 19.6666374
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3045788, 10.3125973
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1567099, upper bound: 7.1760600
time: 14.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1567075, upper bound: 7.1760627
time: 27.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2784805, 17.2874947
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3739510, 11.3816147
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3500366, 10.3551960
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6355572, 11.6370907
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8083992, 12.8182240
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5248642, 15.5288582
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5183086, 11.5164680
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9593315, 14.9642067
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5452423, 14.5532990
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7907448, 14.7901039
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7292480, 21.7388878
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2572060, 12.2559757
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -16.9985046, 17.0030022
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0328217, 21.0374413
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.2921448, 19.3092690
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.8999596, 12.9061852
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4295654, 21.4302597
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.4914017, 24.5201569
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6498108, 16.6505356
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3795052, 14.3764534
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2693710, 17.2676620
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9050674, 12.9022655
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0229416, 17.0191994
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8298645, 15.8284531
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4797592, 20.4789963
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5225677, 17.5188217
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1543236, 14.1483536
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1307678, 18.1376991
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.4040260, 16.3937454
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0174217, 13.0183640
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1565247, 24.1474915
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0932159, 16.0847969
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5376778, 18.5283966
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2737198, 19.2710762
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4511261, 20.4485092
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2248459, 24.2273712
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2184525, 28.2176208
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6656837, 19.6596909
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3095074, 10.3076687
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 657

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1705465, upper bound: 7.1622294
time: 35.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1705439, upper bound: 7.1622319
time: 39.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2874947, 17.2784805
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3816147, 11.3739529
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3551941, 10.3500347
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6370907, 11.6355572
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8182259, 12.8083973
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5288582, 15.5248642
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5164700, 11.5183086
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9642067, 14.9593315
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5532990, 14.5452423
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7901039, 14.7907448
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7388840, 21.7292480
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2559776, 12.2572079
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0029984, 16.9985085
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0374374, 21.0328217
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.3092690, 19.2921448
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9061852, 12.8999596
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4302673, 21.4295616
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5201569, 24.4914055
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6505356, 16.6498146
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3764496, 14.3795052
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2676620, 17.2693748
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9022675, 12.9050617
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0192032, 17.0229454
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8284531, 15.8298683
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4789963, 20.4797592
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5188141, 17.5225677
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1483536, 14.1543236
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1376953, 18.1307640
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3937492, 16.4040260
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0183640, 13.0174217
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1474915, 24.1565247
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0847931, 16.0932198
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5284004, 18.5376816
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2710800, 19.2737160
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4485092, 20.4511299
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2273712, 24.2248459
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2176208, 28.2184525
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6596870, 19.6656837
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3076687, 10.3095093
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 657

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1622319, upper bound: 7.1705439
time: 40.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1622294, upper bound: 7.1705465
time: 44.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2762184, 17.2897568
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3725090, 11.3830605
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3498688, 10.3553619
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6351681, 11.6374760
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8080025, 12.8186169
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5254784, 15.5282440
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5214405, 11.5133381
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9565201, 14.9670219
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5431099, 14.5554314
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7871895, 14.7936592
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7282410, 21.7398949
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2571144, 12.2560692
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0054779, 16.9960327
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0372849, 21.0329781
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.2875099, 19.3139038
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9006386, 12.9055061
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4290924, 21.4307289
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5005188, 24.5110359
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6498032, 16.6505432
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3766975, 14.3792572
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2672195, 17.2698135
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9029160, 12.9044132
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0195236, 17.0226212
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8287506, 15.8295708
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4804840, 20.4782677
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5191727, 17.5222168
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1515656, 14.1511078
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1376114, 18.1308517
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3996925, 16.3980827
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0190811, 13.0167046
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1579895, 24.1460342
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0923233, 16.0856934
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5406075, 18.5254745
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2788773, 19.2659187
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4570999, 20.4425430
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2267151, 24.2255096
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2187729, 28.2172928
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6666374, 19.6587372
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3125973, 10.3045807
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 657

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1760627, upper bound: 7.1567075
time: 35.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1760600, upper bound: 7.1567099
time: 32.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2797737, 17.2862015
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3744278, 11.3811398
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3520813, 10.3531475
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6360607, 11.6365871
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8116951, 12.8149281
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5262718, 15.5274544
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5197315, 11.5150433
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9588737, 14.9646683
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5484428, 14.5500984
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7878265, 14.7930183
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7297745, 21.7383652
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2497749, 12.2634087
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -16.9987717, 17.0027084
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0372696, 21.0329895
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.2939796, 19.3074360
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9028702, 12.9032764
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4261627, 21.4334526
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.4998245, 24.5117378
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6446381, 16.6550865
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3761673, 14.3797913
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2643814, 17.2726364
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9007187, 12.9066143
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0185165, 17.0236206
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8272629, 15.8310547
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4762726, 20.4822845
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5175934, 17.5237656
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1498299, 14.1528435
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1325150, 18.1358414
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3961449, 16.4016266
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0191498, 13.0166359
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1559830, 24.1480331
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0905228, 16.0874863
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5377541, 18.5283203
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2757263, 19.2690659
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4533997, 20.4462433
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2278290, 24.2243805
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2228622, 28.2131271
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6668282, 19.6585464
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3111172, 10.3060627
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 657

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1701486, upper bound: 7.1607283
time: 16.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 41, lower bound: -7.1701459, upper bound: 7.1607306
time: 19.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2751694, 17.2908058
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3703156, 11.3852501
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3497543, 10.3554745
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6353207, 11.6373310
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8075066, 12.8191128
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5249519, 15.5287743
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5215855, 11.5131931
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9553947, 14.9681435
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5445709, 14.5539703
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7871590, 14.7936859
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7249222, 21.7432098
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2508965, 12.2622890
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -16.9999619, 17.0015450
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0371552, 21.0331039
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.2841339, 19.3172836
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9005623, 12.9055824
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4252625, 21.4345512
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.4895248, 24.5220375
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6448975, 16.6554489
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3762894, 14.3796692
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2642136, 17.2728271
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.9009018, 12.9064312
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0187759, 17.0233727
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8275452, 15.8307724
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4771271, 20.4816208
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5180130, 17.5233765
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1510849, 14.1515884
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1324158, 18.1360512
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3983803, 16.3993988
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0194130, 13.0163727
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1606293, 24.1433945
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0938187, 16.0841980
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5431709, 18.5229073
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2794342, 19.2653580
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4571991, 20.4424438
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2277908, 24.2244263
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2233582, 28.2127151
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6702080, 19.6551590
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3131466, 10.3040333
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 657

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1769996, upper bound: 7.1557637
time: 28.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1769967, upper bound: 7.1557661
time: 32.57 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 63.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1557661, upper bound: 7.1769967
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1557637, upper bound: 7.1769996
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1607306, upper bound: 7.1701459
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1607283, upper bound: 7.1701486
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1567099, upper bound: 7.1760600
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1567075, upper bound: 7.1760627
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1705465, upper bound: 7.1622294
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1705439, upper bound: 7.1622319
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1622319, upper bound: 7.1705439
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1622294, upper bound: 7.1705465
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1760627, upper bound: 7.1567075
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1760600, upper bound: 7.1567099
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1701486, upper bound: 7.1607283
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1701459, upper bound: 7.1607306
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1769996, upper bound: 7.1557637
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 63.42
Output dim: 41, lower bound: -7.1769967, upper bound: 7.1557661

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2722054, 17.2601738
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3676872, 11.3570499
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3530540, 10.3484745
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6192322, 11.6214561
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8065586, 12.7974205
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5159798, 15.5151787
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5131493, 11.5215569
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9678955, 14.9591827
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5586052, 14.5516472
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7900105, 14.7841911
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7418365, 21.7237892
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2613068, 12.2500839
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0069122, 17.0022202
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0324173, 21.0363998
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.2953720, 19.2582035
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9044590, 12.8994884
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4278107, 21.4209671
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5180359, 24.4858131
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6690292, 16.6541748
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3692017, 14.3629990
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2705612, 17.2622719
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.8982658, 12.8904800
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0169373, 17.0113373
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8273697, 15.8237686
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4550362, 20.4424057
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5115776, 17.5036469
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1455688, 14.1433182
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1359367, 18.1323547
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3983307, 16.3974648
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0164604, 13.0186863
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1431961, 24.1630096
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0841141, 16.0938797
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5212364, 18.5415077
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2637520, 19.2747421
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4433670, 20.4578285
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2259979, 24.2270737
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2121582, 28.2224350
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6554146, 19.6704178
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3036957, 10.3127689
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 757

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1503301, upper bound: 7.1767874
time: 17.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1555494, upper bound: 7.1715782
time: 41.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2758141, 17.2565632
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3719826, 11.3527546
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3541908, 10.3473339
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6234665, 11.6172180
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8090267, 12.7949562
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5190010, 15.5121574
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5131683, 11.5215416
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9719315, 14.9551468
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5610466, 14.5492058
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7907162, 14.7834854
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7420807, 21.7235489
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2614784, 12.2499123
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0037994, 17.0053291
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0323486, 21.0364685
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.2913551, 19.2622185
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9045086, 12.8994408
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4302521, 21.4185257
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5183258, 24.4855194
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6647263, 16.6584816
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3663788, 14.3658257
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2708893, 17.2619438
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.8960114, 12.8927383
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0159302, 17.0123405
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8269958, 15.8241425
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4469032, 20.4505348
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5090065, 17.5062180
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1438217, 14.1450691
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1359901, 18.1322975
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3984833, 16.3973083
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0156479, 13.0194988
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1457748, 24.1604385
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0842590, 16.0937347
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5212440, 18.5415001
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2606621, 19.2778320
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4430771, 20.4581223
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2237091, 24.2293625
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2117996, 28.2227936
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6553688, 19.6704712
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3036575, 10.3128052
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 757

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1503280, upper bound: 7.1767901
time: 17.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1555471, upper bound: 7.1715809
time: 36.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2711525, 17.2612228
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3654976, 11.3592415
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3529396, 10.3485889
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6193771, 11.6213093
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8060627, 12.7979164
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5154495, 15.5157051
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5132942, 11.5214138
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9667740, 14.9603043
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5600662, 14.5501862
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7899837, 14.7842178
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7385178, 21.7271080
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2550850, 12.2563019
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -17.0013962, 17.0077324
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0322952, 21.0365257
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.2919922, 19.2615833
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9043827, 12.8995647
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4239883, 21.4247894
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5070343, 24.4968109
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6641312, 16.6590805
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3687935, 14.3634071
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2675476, 17.2652817
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.8962479, 12.8924980
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0161896, 17.0120850
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8261642, 15.8249702
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4516792, 20.4457588
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5104179, 17.5047989
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1450882, 14.1437988
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1307411, 18.1375542
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3970108, 16.3987770
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0167923, 13.0183544
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1458359, 24.1603699
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0856171, 16.0923805
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5237999, 18.5389404
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2643089, 19.2741852
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4434662, 20.4577332
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2270737, 24.2259979
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2167358, 28.2178574
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6589928, 19.6668434
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3042412, 10.3122196
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 757

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1512699, upper bound: 7.1758478
time: 30.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1564947, upper bound: 7.1706406
time: 49.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.8302288, -0.2941942, -23.8302288, -0.2941942, -17.2747650, 17.2576141
1: -12.3372145, 4.7471724, -12.3372145, 4.7471724, -11.3697929, 11.3549442
2: -12.0670710, 2.7447107, -12.0670710, 2.7447107, -10.3540802, 10.3474483
3: -12.3062305, 4.8901930, -12.3062305, 4.8901930, -11.6236153, 11.6170712
4: -20.5873985, -2.1328430, -20.5873985, -2.1328430, -12.8085270, 12.7954521
5: -15.6197720, 4.8310084, -15.6197720, 4.8310084, -15.5184708, 15.5126839
6: 2.2486405, 15.6404076, 2.2486405, 15.6404076, -11.5133095, 11.5213966
7: -15.3206863, 6.3509693, -15.3206863, 6.3509693, -14.9708099, 14.9562683
8: -21.3870850, 0.1070893, -21.3870850, 0.1070893, -14.5625076, 14.5477448
9: -8.8808041, 8.9582596, -8.8808041, 8.9582596, -14.7906895, 14.7835121
10: -20.8463879, 5.0723829, -20.8463879, 5.0723829, -21.7387619, 21.7268639
11: -10.9331875, 6.3938808, -10.9331875, 6.3938808, -12.2552567, 12.2561302
12: -13.6292591, 9.2987814, -13.6292591, 9.2987814, -16.9982910, 17.0108414
13: -18.2735844, 4.8716698, -18.2735844, 4.8716698, -21.0322266, 21.0365982
14: -55.3459702, -25.9025135, -55.3459702, -25.9025135, -19.2879753, 19.2655983
15: -24.2777100, -9.2060947, -24.2777100, -9.2060947, -12.9044323, 12.8995171
16: -11.7747812, 12.8350105, -11.7747812, 12.8350105, -21.4264297, 21.4223518
17: -55.9953918, -21.7292957, -55.9953918, -21.7292957, -24.5073242, 24.4965210
18: -21.0252552, 0.8269024, -21.0252552, 0.8269024, -16.6598129, 16.6633873
19: -10.6373339, 1.5452659, -10.6373339, 1.5452659, -12.1826000, 12.1826000
20: -9.6884804, 4.7743168, -9.6884804, 4.7743168, -14.3659668, 14.3662338
21: -15.6772175, 2.7126331, -15.6772175, 2.7126331, -17.2678757, 17.2649536
22: -25.0654488, -5.8718472, -25.0654488, -5.8718472, -19.1936016, 19.1936016
23: -7.8815536, 6.5172176, -7.8815536, 6.5172176, -12.8939934, 12.8947525
24: -13.4431181, 3.7769718, -13.4431181, 3.7769718, -17.0151825, 17.0130920
25: -12.3563976, 3.6828027, -12.3563976, 3.6828027, -15.8257904, 15.8253479
26: -28.2146206, -3.0268388, -28.2146206, -3.0268388, -20.4435463, 20.4538918
27: -13.3887596, 4.7263412, -13.3887596, 4.7263412, -17.5078468, 17.5073738
28: -6.9040775, 9.2477264, -6.9040775, 9.2477264, -14.1433411, 14.1455498
29: -22.1368561, -2.5750332, -22.1368561, -2.5750332, -18.1307945, 18.1374969
30: -11.4021978, 7.9766030, -11.4021978, 7.9766030, -16.3971710, 16.3986206
31: -12.1030540, 2.6077662, -12.1030540, 2.6077662, -14.7108202, 14.7108202
32: -0.5789719, 14.1540375, -0.5789719, 14.1540375, -13.0159798, 13.0191669
33: -14.5690069, 14.1898212, -14.5690069, 14.1898212, -24.1484146, 24.1577988
34: -12.9359703, 8.7492723, -12.9359703, 8.7492723, -16.0857544, 16.0922394
35: -14.2780285, 10.7329445, -14.2780285, 10.7329445, -18.5238152, 18.5389328
36: -13.3598738, 10.9356680, -13.3598738, 10.9356680, -19.2612267, 19.2772713
37: -17.5574036, 7.9555950, -17.5574036, 7.9555950, -20.4431763, 20.4580231
38: -18.3111954, 10.2827892, -18.3111954, 10.2827892, -24.2247925, 24.2282791
39: -21.6987114, 10.0361090, -21.6987114, 10.0361090, -28.2163773, 28.2182159
40: -8.4472446, 14.9514971, -8.4472446, 14.9514971, -19.6589394, 19.6668930
41: 3.1895733, 15.4870577, 3.1895733, 15.4870577, -10.3042068, 10.3122559
42: 2.8709769, 13.6422338, 2.8709769, 13.6422338, -10.7712574, 10.7712574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=218, inp2_unstable=218, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=14, inp2_unstable=14, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=32, inp2_unstable=32, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 757

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1512677, upper bound: 7.1758504
time: 31.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 41, lower bound: -7.1564924, upper bound: 7.1706432
time: 24.59 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 58.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 58.77
Output dim: 41, lower bound: -7.1503301, upper bound: 7.1767874
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 58.77
Output dim: 41, lower bound: -7.1555494, upper bound: 7.1715782
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 58.77
Output dim: 41, lower bound: -7.1503280, upper bound: 7.1767901
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 58.77
Output dim: 41, lower bound: -7.1555471, upper bound: 7.1715809
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 58.77
Output dim: 41, lower bound: -7.1512699, upper bound: 7.1758478
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 58.77
Output dim: 41, lower bound: -7.1564947, upper bound: 7.1706406
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 58.77
Output dim: 41, lower bound: -7.1512677, upper bound: 7.1758504
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 58.77
Output dim: 41, lower bound: -7.1564924, upper bound: 7.1706432
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.77
Output dim: 41, lower bound: -7.1705465, upper bound: 7.1622294
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.77
Output dim: 41, lower bound: -7.1705439, upper bound: 7.1622319
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.77
Output dim: 41, lower bound: -7.1622319, upper bound: 7.1705439
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.77
Output dim: 41, lower bound: -7.1622294, upper bound: 7.1705465
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.77
Output dim: 41, lower bound: -7.1760627, upper bound: 7.1567075
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.77
Output dim: 41, lower bound: -7.1760600, upper bound: 7.1567099
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.77
Output dim: 41, lower bound: -7.1769996, upper bound: 7.1557637
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.77
Output dim: 41, lower bound: -7.1769967, upper bound: 7.1557661

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 57.68 + 1786.58 = 1844.26 seconds
