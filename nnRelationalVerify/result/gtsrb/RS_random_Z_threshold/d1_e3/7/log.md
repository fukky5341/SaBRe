## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 7)
Time budget: 1800 seconds
Split limit: 100
Threshold: 5.9289981984


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7347260, 30.7347336)
1: (-4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6182404, 17.6182404)
2: (1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0191345, 17.0191345)
3: (-2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6993408, 16.6993408)
4: (-2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1902313, 22.1902313)
5: (-0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654)
6: (-41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5760269, 22.5760269)
7: (0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5908661, 16.5908699)
8: (-2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9716110, 25.9716110)
9: (-3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1163177, 17.1163177)
10: (-10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.3270569, 23.3270569)
11: (-11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5197678, 15.5197678)
12: (-33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2626648, 19.2626648)
13: (-20.8587570, 11.3056412, -20.8587570, 11.3056412, -25.0104904, 25.0104904)
14: (-34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3349304, 31.3349304)
15: (-11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380)
16: (-19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9367332, 14.9367332)
17: (-36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2783127, 18.2783089)
18: (-26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8767395, 19.8767395)
19: (-11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3457794, 15.3457832)
20: (-5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4712067, 17.4712067)
21: (-11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2376022, 19.2376022)
22: (-12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1823730, 15.1823730)
23: (-7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8720093, 17.8720093)
24: (-16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8490906, 15.8490906)
25: (-11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.2094345, 16.2094307)
26: (-17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1701508, 24.1701508)
27: (-14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.7317200, 19.7317200)
28: (-8.5007658, 12.0290146, -8.5007658, 12.0290146, -20.0145721, 20.0145721)
29: (-13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.6088295, 14.6088333)
30: (-13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8598442, 18.8598442)
31: (-20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9716568, 20.9716568)
32: (-30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4501801, 21.4501801)
33: (-61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3652267, 27.3652267)
34: (-60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3422852, 19.3422852)
35: (-54.4683228, -24.3051224, -54.4683228, -24.3051224, -23.0022202, 23.0022202)
36: (-45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9730606, 23.9730644)
37: (-74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9424210, 24.9424210)
38: (-55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5782242, 23.5782242)
39: (-60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.2147675, 25.2147675)
40: (-55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.5088806, 15.5088806)
41: (-39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8636475, 25.8636475)
42: (-25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6900330, 17.6900368)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.24 + 57.67 = 59.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -5.9768127, upper bound: 5.9768127

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 985

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 748

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9766750, upper bound: 5.9622926
time: 41.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9622926, upper bound: 5.9766750
time: 77.01 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 118.37 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 118.37
Output dim: 5, lower bound: -5.9766750, upper bound: 5.9622926
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 118.37
Output dim: 5, lower bound: -5.9622926, upper bound: 5.9766750

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7333145, 30.7317886
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6179047, 17.6175842
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0181808, 17.0172043
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6988220, 16.6984100
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1900635, 22.1919556
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5766830, 22.5759583
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5899353, 16.5891151
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9683380, 25.9688568
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1163177, 17.1163101
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.3232727, 23.3217392
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5209045, 15.5196533
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2633362, 19.2608871
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -25.0104599, 25.0104599
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3341751, 31.3333359
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9352684, 14.9341927
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2761726, 18.2738533
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8750153, 19.8747330
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3435974, 15.3447342
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4670181, 17.4691772
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2342834, 19.2356567
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1762466, 15.1798096
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8715744, 17.8719254
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8436508, 15.8455086
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.2034149, 16.2066956
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1663589, 24.1677780
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.7261124, 19.7288361
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -20.0111618, 20.0129395
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.6064873, 14.6067963
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8551140, 18.8551826
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9674072, 20.9696426
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4510345, 21.4501038
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3569641, 27.3610077
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3362350, 19.3387108
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9945068, 22.9971962
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9669571, 23.9690704
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9421921, 24.9423943
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5728683, 23.5750122
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.2093964, 25.2135086
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.5082970, 15.5087738
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8635254, 25.8634758
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6909180, 17.6897354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 568

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9763707, upper bound: 5.9487258
time: 49.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9631133, upper bound: 5.9619899
time: 47.19 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7317886, 30.7333221
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6175842, 17.6179008
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0172043, 17.0181808
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6984024, 16.6988220
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1919556, 22.1900635
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5759583, 22.5766830
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5891113, 16.5899315
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9688568, 25.9683380
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1163101, 17.1163177
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.3217392, 23.3232727
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5196533, 15.5209084
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2608871, 19.2633362
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -25.0104523, 25.0104599
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3333359, 31.3341675
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9341927, 14.9352684
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2738533, 18.2761765
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8747330, 19.8750153
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3447342, 15.3435974
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4691772, 17.4670181
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2356567, 19.2342834
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1798096, 15.1762505
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8719254, 17.8715744
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8455124, 15.8436546
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.2066956, 16.2034187
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1677780, 24.1663589
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.7288361, 19.7261124
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -20.0129395, 20.0111618
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.6068001, 14.6064911
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8551903, 18.8551140
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9696426, 20.9674072
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4501038, 21.4510345
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3610077, 27.3569641
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3387070, 19.3362312
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9971924, 22.9945068
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9690704, 23.9669609
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9423981, 24.9421921
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5750122, 23.5728645
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.2135086, 25.2093964
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.5087738, 15.5082932
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8634720, 25.8635254
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6897354, 17.6909180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 753

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9619038, upper bound: 5.9728653
time: 11.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9584800, upper bound: 5.9762849
time: 38.93 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 51.82 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 51.82
Output dim: 5, lower bound: -5.9763707, upper bound: 5.9487258
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 51.82
Output dim: 5, lower bound: -5.9631133, upper bound: 5.9619899
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 51.82
Output dim: 5, lower bound: -5.9619038, upper bound: 5.9728653
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 51.82
Output dim: 5, lower bound: -5.9584800, upper bound: 5.9762849

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7346191, 30.7343750
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6149292, 17.6140327
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0190964, 17.0181122
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6770020, 16.6723862
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1954269, 22.1987534
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5685883, 22.5654869
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5748367, 16.5710526
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9670639, 25.9674759
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1155472, 17.1102066
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.2992630, 23.2929306
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5297089, 15.5262032
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2844772, 19.2779007
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -24.9621124, 24.9525833
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3170090, 31.3182068
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9352531, 14.9341736
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2409248, 18.2451553
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8018341, 19.8135834
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3260384, 15.3300629
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4654922, 17.4678650
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2338943, 19.2352905
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1510010, 15.1587105
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8603516, 17.8623047
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8084488, 15.8160896
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1805115, 16.1875496
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1284332, 24.1360855
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.6890030, 19.6978226
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -19.9887466, 19.9942093
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.5862465, 14.5898857
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8516235, 18.8522110
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9394836, 20.9463120
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4641037, 21.4554787
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3487930, 27.3513184
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3362885, 19.3387527
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9964867, 22.9989319
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9660645, 23.9672318
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9192734, 24.9204330
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5727196, 23.5746918
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1805496, 25.1789856
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.5008507, 15.5014915
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8579330, 25.8553085
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6898651, 17.6832619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9745960, upper bound: 5.9367600
time: 62.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9644028, upper bound: 5.9469608
time: 25.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7359009, 30.7331009
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6143494, 17.6146164
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0190887, 17.0181198
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6727982, 16.6765900
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1968613, 22.1973190
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5662155, 22.5678596
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5718689, 16.5740204
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9669571, 25.9675827
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1102066, 17.1155434
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.2944641, 23.2977295
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5274582, 15.5284538
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2803574, 19.2820282
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -24.9525909, 24.9621048
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3190384, 31.3161774
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9352493, 14.9341774
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2474785, 18.2386017
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8138657, 19.8015518
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3289223, 15.3271751
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4657135, 17.4676437
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2339172, 19.2352676
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1551514, 15.1545639
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8619461, 17.8607063
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8142319, 15.8102989
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1842728, 16.1837845
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1346664, 24.1298523
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.6950989, 19.6917267
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -19.9924316, 19.9905243
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.5895729, 14.5865555
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8521423, 18.8516960
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9440765, 20.9417267
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4564056, 21.4631767
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3472748, 27.3528290
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3362732, 19.3387718
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9962425, 22.9991760
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9651184, 23.9681702
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9202271, 24.9194717
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5725441, 23.5748672
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1748734, 25.1846619
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.5010147, 15.5013275
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8553543, 25.8578873
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6844482, 17.6886826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 755

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 934

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9574442, upper bound: 5.9618289
time: 9.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9629521, upper bound: 5.9563206
time: 35.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7315445, 30.7335968
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6173325, 17.6176758
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0171738, 17.0180588
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6976776, 16.6976242
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1904602, 22.1912231
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5711441, 22.5708351
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5889053, 16.5889435
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9622803, 25.9585876
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1117401, 17.1116257
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.3097305, 23.3086090
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5189247, 15.5186348
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2605972, 19.2625427
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -25.0062866, 25.0077896
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3274307, 31.3274689
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9224663, 14.9216309
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2684631, 18.2681999
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8645706, 19.8681641
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3439217, 15.3432083
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4659195, 17.4646149
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2334976, 19.2329178
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1704254, 15.1697922
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8713608, 17.8685646
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8393631, 15.8392639
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.2063446, 16.2031517
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1669464, 24.1658859
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.7225571, 19.7210846
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -20.0129166, 20.0104294
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.6062050, 14.6060829
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8549309, 18.8548088
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9656906, 20.9648972
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4451370, 21.4475212
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3499451, 27.3495026
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3311157, 19.3311119
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9875717, 22.9880066
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9570999, 23.9588852
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9396973, 24.9415970
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5546646, 23.5588112
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.2024307, 25.2019234
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.5085373, 15.5075073
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8633881, 25.8632584
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6885910, 17.6864281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1698

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 971

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9600037, upper bound: 5.9727806
time: 53.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9618192, upper bound: 5.9709591
time: 35.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7320633, 30.7330856
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6173630, 17.6176491
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0170822, 17.0181427
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6972122, 16.6980896
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1931152, 22.1885681
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5701065, 22.5718727
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5881271, 16.5897217
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9591064, 25.9617615
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1116180, 17.1117439
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.3070831, 23.3112640
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5173836, 15.5201836
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2600937, 19.2630386
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -25.0077820, 25.0062866
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3266373, 31.3282700
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9205551, 14.9235420
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2658768, 18.2707825
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8678741, 19.8648605
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3443489, 15.3427811
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4667740, 17.4637680
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2342834, 19.2321243
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1733551, 15.1668663
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8689194, 17.8710060
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8411179, 15.8375130
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.2064285, 16.2030716
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1673050, 24.1655273
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.7238083, 19.7198334
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -20.0122070, 20.0111389
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.6063881, 14.6058960
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8548775, 18.8548584
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9671326, 20.9634552
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4465942, 21.4460640
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3535461, 27.3459091
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3335876, 19.3286400
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9906998, 22.9848747
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9609909, 23.9549866
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9418030, 24.9394951
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5609665, 23.5525131
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.2060394, 25.1983185
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.5079842, 15.5080566
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8632050, 25.8634377
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6852493, 17.6897774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 934

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 899

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9516995, upper bound: 5.9762216
time: 35.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9584168, upper bound: 5.9695066
time: 52.02 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 89.39 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 89.39
Output dim: 5, lower bound: -5.9745960, upper bound: 5.9367600
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 89.39
Output dim: 5, lower bound: -5.9644028, upper bound: 5.9469608
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 89.39
Output dim: 5, lower bound: -5.9574442, upper bound: 5.9618289
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 89.39
Output dim: 5, lower bound: -5.9629521, upper bound: 5.9563206
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 89.39
Output dim: 5, lower bound: -5.9600037, upper bound: 5.9727806
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 89.39
Output dim: 5, lower bound: -5.9618192, upper bound: 5.9709591
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 89.39
Output dim: 5, lower bound: -5.9516995, upper bound: 5.9762216
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 89.39
Output dim: 5, lower bound: -5.9584168, upper bound: 5.9695066

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7344971, 30.7349854
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6148911, 17.6139832
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0187607, 17.0175705
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6759415, 16.6686249
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1942215, 22.1958542
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5672836, 22.5646706
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5737610, 16.5686073
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9660034, 25.9648132
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1130905, 17.1037712
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.2982941, 23.2909088
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5282364, 15.5246315
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2813187, 19.2766342
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -24.9620590, 24.9524994
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3154526, 31.3203659
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9341393, 14.9314957
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2337303, 18.2425194
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8001404, 19.8123169
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3251228, 15.3292999
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4660873, 17.4677582
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2341537, 19.2352295
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1487961, 15.1577682
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8583450, 17.8605995
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8032455, 15.8130341
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1762238, 16.1849594
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1286697, 24.1360397
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.6867599, 19.6959839
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -19.9887619, 19.9942017
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.5818214, 14.5881920
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8492584, 18.8505173
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9369507, 20.9441986
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4637909, 21.4553795
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3485107, 27.3512192
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3335190, 19.3374710
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9934959, 22.9977531
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9617615, 23.9656181
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9148331, 24.9184418
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5673637, 23.5730324
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1792297, 25.1784630
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.5001907, 15.5009804
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8572922, 25.8549690
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6898499, 17.6831589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1387

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 567

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9739654, upper bound: 5.9264310
time: 36.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9612849, upper bound: 5.9356499
time: 38.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7352448, 30.7342377
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6148834, 17.6139946
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0185623, 17.0177765
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6732407, 16.6713257
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1925278, 22.1975479
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5677719, 22.5641861
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5723953, 16.5699730
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9644012, 25.9664154
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1091080, 17.1077499
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.2972336, 23.2919617
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5281372, 15.5247307
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2832108, 19.2747421
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -24.9620209, 24.9525375
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3191605, 31.3166504
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9325752, 14.9330635
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2382851, 18.2379608
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8005753, 19.8118820
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3252754, 15.3291473
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4653778, 17.4684601
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2338409, 19.2355423
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1500626, 15.1565056
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8586502, 17.8602905
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8053970, 15.8108902
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1779175, 16.1832657
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1283875, 24.1363220
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.6871643, 19.6955795
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -19.9887390, 19.9942169
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.5845528, 14.5854530
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8499298, 18.8498459
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9373703, 20.9437866
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4640045, 21.4551659
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3486938, 27.3510284
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3350143, 19.3359833
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9953041, 22.9959450
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9644470, 23.9629326
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9172821, 24.9159927
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5710564, 23.5693398
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1800232, 25.1776581
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.5003357, 15.5008316
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8575974, 25.8546638
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6897583, 17.6832542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 823

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9641533, upper bound: 5.9410817
time: 42.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9564727, upper bound: 5.9466899
time: 52.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7404327, 30.7370605
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6128693, 17.6131363
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0193558, 17.0183792
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6673584, 16.6717300
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1972656, 22.1977234
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5683136, 22.5701866
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5611115, 16.5650444
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9639359, 25.9647293
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1078720, 17.1134033
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.2916565, 23.2956009
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5369186, 15.5404205
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2680435, 19.2684250
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -24.9520569, 24.9605026
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3217621, 31.3185577
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9185333, 14.9195175
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2483139, 18.2389183
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8152237, 19.8029556
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3347244, 15.3331070
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4715805, 17.4749146
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2405319, 19.2437057
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1530838, 15.1512794
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8656693, 17.8648376
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8182678, 15.8142433
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1869926, 16.1864929
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1320877, 24.1269608
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.6888351, 19.6855087
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -19.9952011, 19.9936981
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.5907402, 14.5867615
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8564606, 18.8589630
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9495850, 20.9478912
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4528427, 21.4591522
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3399048, 27.3442612
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3377075, 19.3398857
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9923248, 22.9940071
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9676399, 23.9686508
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9066544, 24.9027061
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5741425, 23.5748177
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1668015, 25.1749840
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.4979744, 15.4980431
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8546066, 25.8564720
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6847572, 17.6889877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 756

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1649

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9573887, upper bound: 5.9507306
time: 9.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9463487, upper bound: 5.9617734
time: 33.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7398682, 30.7376328
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6128693, 17.6131363
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0193481, 17.0183868
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6679459, 16.6711502
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1972656, 22.1977234
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5685425, 22.5699654
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5628967, 16.5632629
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9641037, 25.9645767
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1080704, 17.1132088
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.2923355, 23.2949142
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5394211, 15.5379105
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2667542, 19.2697144
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -24.9509888, 24.9615707
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3214264, 31.3189087
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9205933, 14.9174614
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2477951, 18.2394409
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8152695, 19.8029099
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3348618, 15.3329735
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4729767, 17.4735107
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2423477, 19.2418823
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1518631, 15.1524963
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8660812, 17.8644257
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8181763, 15.8143311
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1869774, 16.1865082
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1317749, 24.1272736
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.6888809, 19.6854630
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -19.9956055, 19.9932938
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.5897789, 14.5877190
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8594055, 18.8560181
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9502411, 20.9472351
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4523849, 21.4596100
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3386993, 27.3454590
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3373871, 19.3402100
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9910736, 22.9952583
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9655952, 23.9706955
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9034653, 24.9058990
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5724945, 23.5764656
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1651993, 25.1765938
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.4977264, 15.4982872
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8539429, 25.8571320
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6847572, 17.6889877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 738

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 756

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9627878, upper bound: 5.9489697
time: 53.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9556015, upper bound: 5.9561575
time: 37.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7301636, 30.7321854
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6161575, 17.6167717
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0171204, 17.0178757
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6962967, 16.6964188
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1904068, 22.1911926
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5702438, 22.5678253
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5887756, 16.5887985
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9608154, 25.9568634
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1062126, 17.1047134
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.3096695, 23.3085480
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5178680, 15.5175209
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2533188, 19.2533569
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -25.0071793, 25.0077362
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3276443, 31.3276062
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9223900, 14.9214325
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2710457, 18.2713890
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8688736, 19.8724289
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3425140, 15.3433876
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4639282, 17.4628601
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2307129, 19.2310791
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1731377, 15.1739464
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8699417, 17.8684845
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8230438, 15.8266487
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1918030, 16.1927147
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1574936, 24.1541977
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.7235031, 19.7225113
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -20.0136795, 20.0123215
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.6048355, 14.6054726
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8472900, 18.8490295
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9575577, 20.9607544
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4340820, 21.4340172
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3475266, 27.3466492
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3361435, 19.3359299
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9862137, 22.9862099
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9538422, 23.9549789
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9390106, 24.9392395
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5554123, 23.5593567
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.2026329, 25.2015915
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.4955330, 15.4906616
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8573074, 25.8548851
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6853142, 17.6804924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1462

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1698

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9599030, upper bound: 5.9720320
time: 39.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9592554, upper bound: 5.9726799
time: 44.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7301483, 30.7322083
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6164322, 17.6165009
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0169830, 17.0180054
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6964722, 16.6962433
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1904297, 22.1911774
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5681381, 22.5699310
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5887604, 16.5888138
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9605560, 25.9571304
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1048317, 17.1060982
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.3096695, 23.3085556
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5178223, 15.5175705
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2514114, 19.2552643
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -25.0062332, 25.0086899
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3275833, 31.3276825
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9222679, 14.9215546
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2716560, 18.2707901
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8688354, 19.8724670
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3441010, 15.3418083
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4641647, 17.4626312
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2316666, 19.2301331
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1745872, 15.1725082
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8712769, 17.8671494
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8267441, 15.8229446
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1959076, 16.1886024
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1552582, 24.1564331
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.7239838, 19.7220306
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -20.0148163, 20.0111923
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.6055984, 14.6047096
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8491516, 18.8471680
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9615479, 20.9567566
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4316330, 21.4364662
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3470917, 27.3470764
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3359375, 19.3361359
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9857712, 22.9866524
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9531937, 23.9556274
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9373398, 24.9409103
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5551987, 23.5595627
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.2020988, 25.2021294
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.4916916, 15.4944992
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8550186, 25.8571739
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6826591, 17.6831551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 750

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 752

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9614902, upper bound: 5.9707616
time: 32.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9614925, upper bound: 5.9705212
time: 50.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7243958, 30.7261505
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6150742, 17.6155281
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0143738, 17.0158310
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6964111, 16.6974258
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1828232, 22.1799545
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5603180, 22.5600624
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5847626, 16.5867958
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9575882, 25.9603882
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1038666, 17.1052246
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.3052444, 23.3096237
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5166855, 15.5195007
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2499466, 19.2500610
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -25.0180435, 25.0142975
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3336487, 31.3340988
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9068222, 14.9112320
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2620659, 18.2657623
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8621597, 19.8623352
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3511772, 15.3508911
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4677963, 17.4649239
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2416840, 19.2410355
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1738739, 15.1673698
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8707352, 17.8730583
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8425446, 15.8388405
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.2041779, 16.2006607
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1710663, 24.1697388
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.7233505, 19.7193298
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -20.0157089, 20.0151825
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.6037216, 14.6024361
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8509064, 18.8505859
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9758301, 20.9736023
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4389801, 21.4363823
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3480835, 27.3395081
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3263626, 19.3191605
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9850616, 22.9781342
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9532585, 23.9448318
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9356537, 24.9320068
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5493355, 23.5386200
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1963768, 25.1865845
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.5066376, 15.5064659
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8585129, 25.8574753
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6736221, 17.6757736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 657

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 763

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9512257, upper bound: 5.9739701
time: 39.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9495211, upper bound: 5.9757453
time: 26.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7251282, 30.7254105
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6152420, 17.6153641
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0147705, 17.0154343
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6965485, 16.6972961
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1845093, 22.1782684
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5583038, 22.5620804
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5851974, 16.5863647
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9577408, 25.9602356
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1051025, 17.1039925
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.3054352, 23.3094330
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5166931, 15.5194855
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2471085, 19.2529068
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -25.0157928, 25.0165482
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3324585, 31.3352814
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9082451, 14.9098129
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2608604, 18.2669716
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8653488, 19.8591385
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3524590, 15.3496094
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4679337, 17.4647865
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2431946, 19.2395248
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1738586, 15.1673851
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8709717, 17.8728256
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8424530, 15.8389320
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.2040176, 16.2008247
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1715164, 24.1692963
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.7232971, 19.7193832
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -20.0162430, 20.0146484
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.6029282, 14.6032257
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8506088, 18.8508835
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9772797, 20.9721603
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4369125, 21.4384537
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3471451, 27.3404541
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3241043, 19.3214149
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9839554, 22.9792404
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9508400, 23.9472580
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9343109, 24.9333496
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5470695, 23.5408859
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1943016, 25.1886597
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.5063934, 15.5067139
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8572464, 25.8587418
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6712418, 17.6781616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1727

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9459073, upper bound: 5.9693682
time: 37.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9582793, upper bound: 5.9637173
time: 11.64 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 50.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9739654, upper bound: 5.9264310
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9612849, upper bound: 5.9356499
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9641533, upper bound: 5.9410817
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9564727, upper bound: 5.9466899
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9573887, upper bound: 5.9507306
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9463487, upper bound: 5.9617734
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9627878, upper bound: 5.9489697
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9556015, upper bound: 5.9561575
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9599030, upper bound: 5.9720320
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9592554, upper bound: 5.9726799
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9614902, upper bound: 5.9707616
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9614925, upper bound: 5.9705212
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9512257, upper bound: 5.9739701
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9495211, upper bound: 5.9757453
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9459073, upper bound: 5.9693682
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.94
Output dim: 5, lower bound: -5.9582793, upper bound: 5.9637173

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7282715, 30.7340088
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6138916, 17.6132278
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0186768, 17.0179214
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6738815, 16.6612091
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1922226, 22.1967697
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5658646, 22.5588112
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5731049, 16.5648155
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9655075, 25.9643707
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1188622, 17.1022758
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.2974319, 23.2809448
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5315437, 15.5234375
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2833481, 19.2755203
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -24.9590073, 24.9412460
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3090286, 31.3189774
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9339218, 14.9313278
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2217140, 18.2422371
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.7807922, 19.8074112
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3213387, 15.3281136
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4645157, 17.4672318
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2330322, 19.2350006
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1422043, 15.1554832
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8568268, 17.8600845
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.7944107, 15.8112793
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1698303, 16.1837578
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1237488, 24.1340866
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.6786194, 19.6935501
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -19.9835052, 19.9927216
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.5789833, 14.5874825
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8476753, 18.8501549
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9290543, 20.9426498
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4681931, 21.4508286
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3480301, 27.3492432
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3331528, 19.3362999
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9927788, 22.9985313
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9617004, 23.9651031
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9066849, 24.9146729
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5653343, 23.5676422
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1785583, 25.1764259
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.4987602, 15.4984779
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8575134, 25.8520012
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6896172, 17.6753502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 739

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9732353, upper bound: 5.9261977
time: 10.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9737289, upper bound: 5.9257030
time: 33.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7330780, 30.7287674
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6140594, 17.6129837
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0190735, 17.0174866
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6685257, 16.6665649
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1951370, 22.1938553
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5614243, 22.5632439
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5699692, 16.5679474
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9655685, 25.9643173
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1115990, 17.1095390
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.2883301, 23.2899551
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5270424, 15.5279350
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2802048, 19.2786636
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -24.9507980, 24.9494476
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3139420, 31.3139420
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9339752, 14.9312782
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2334404, 18.2305069
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.7952347, 19.7929688
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3239326, 15.3255196
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4655609, 17.4661636
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2339325, 19.2341003
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1465149, 15.1511765
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8577957, 17.8590851
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8014908, 15.8041992
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1750183, 16.1785583
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1267166, 24.1311264
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.6843185, 19.6878433
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -19.9872818, 19.9889450
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.5811043, 14.5853539
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8488960, 18.8488503
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9354019, 20.9362946
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4592361, 21.4597855
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3463135, 27.3507385
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3323441, 19.3371010
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9942436, 22.9970322
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9612503, 23.9655571
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9102402, 24.9102936
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5619545, 23.5709991
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1771851, 25.1777992
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.4976540, 15.4995499
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8543243, 25.8551903
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6820488, 17.6829185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 950

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9608718, upper bound: 5.9212726
time: 57.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9469168, upper bound: 5.9352362
time: 45.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7333679, 30.7331009
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6146317, 17.6137466
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0184097, 17.0179596
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6707764, 16.6671295
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1907959, 22.1965408
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5676651, 22.5639687
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5722122, 16.5694962
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9637299, 25.9662018
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1106796, 17.1066322
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.2966232, 23.2904510
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5251007, 15.5201035
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2838669, 19.2732391
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -24.9541550, 24.9383163
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3150101, 31.3132019
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9323463, 14.9336433
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2342186, 18.2359695
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.7919235, 19.8069382
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3225441, 15.3275146
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4653625, 17.4684525
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2330093, 19.2349701
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1494827, 15.1561546
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8535767, 17.8570557
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.7970734, 15.8067131
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1743317, 16.1814270
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1240082, 24.1335297
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.6792908, 19.6911316
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -19.9842529, 19.9918060
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.5812111, 14.5837173
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8441315, 18.8466797
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9335785, 20.9418564
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4639435, 21.4516144
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3474045, 27.3479843
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3348312, 19.3353691
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9946747, 22.9950638
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9640923, 23.9608231
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9147034, 24.9128494
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5733109, 23.5687408
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1758347, 25.1697731
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.4997826, 15.4998512
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8575134, 25.8544922
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6890259, 17.6815567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1462

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9598907, upper bound: 5.9407320
time: 9.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9638181, upper bound: 5.9368181
time: 24.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7341003, 30.7323608
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6146317, 17.6137466
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0187378, 17.0176239
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6690521, 16.6688614
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1915207, 22.1958160
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5675507, 22.5640793
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5719147, 16.5697899
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9641724, 25.9657516
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1079941, 17.1093140
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.2957230, 23.2913361
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5235062, 15.5216942
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2817001, 19.2754059
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -24.9477997, 24.9446640
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3156509, 31.3124847
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9331589, 14.9328346
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2364311, 18.2338905
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.7956238, 19.8032379
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3236427, 15.3264122
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4653778, 17.4684448
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2332611, 19.2347183
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1497116, 15.1559219
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8553848, 17.8552170
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8012161, 15.8025742
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1760712, 16.1796761
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1255951, 24.1319351
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.6827087, 19.6877136
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -19.9863205, 19.9897385
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.5828209, 14.5821037
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8467331, 18.8440437
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9354477, 20.9399948
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4604568, 21.4551086
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3456421, 27.3497467
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3344040, 19.3357964
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9944229, 22.9952393
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9623375, 23.9625778
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9140320, 24.9134140
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5704651, 23.5715904
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1721420, 25.1734657
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.4993477, 15.5002823
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8574219, 25.8545799
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6880646, 17.6825180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 752

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 719

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9499760, upper bound: 5.9364926
time: 52.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9462840, upper bound: 5.9401859
time: 50.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7330933, 30.7295761
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6108475, 17.6110115
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0140915, 17.0120697
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6705704, 16.6750336
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1841812, 22.1818924
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5534096, 22.5522652
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5564575, 16.5596275
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9608154, 25.9607697
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1171074, 17.1243019
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.2938156, 23.3000870
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5297203, 15.5340195
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2590256, 19.2594223
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -24.9778290, 24.9909286
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3047028, 31.3065720
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9198914, 14.9210167
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2295151, 18.2246704
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8208847, 19.8053741
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3382912, 15.3366852
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4724884, 17.4757881
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2410507, 19.2442627
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1524734, 15.1507912
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8703690, 17.8697739
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8195801, 15.8156204
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1678352, 16.1704483
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1348801, 24.1296158
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.6986771, 19.6941757
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -19.9927216, 19.9915237
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.5911713, 14.5873680
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8471603, 18.8509560
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9526978, 20.9510498
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4484024, 21.4540787
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3392105, 27.3435440
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3379669, 19.3401108
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9888687, 22.9915695
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9718781, 23.9729500
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9151878, 24.9105606
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5707855, 23.5713272
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1538620, 25.1641960
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.4882927, 15.4864273
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8353500, 25.8330154
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6813126, 17.6852646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1662

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 611

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9432420, upper bound: 5.9495759
time: 48.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9562418, upper bound: 5.9365136
time: 30.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7329407, 30.7297211
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6107407, 17.6111183
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0130386, 17.0131149
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6706619, 16.6749420
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1814270, 22.1846390
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5503960, 22.5552750
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5556946, 16.5603905
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9599915, 25.9616013
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1187706, 17.1226425
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.2961426, 23.2977600
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5305138, 15.5332222
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2590408, 19.2594147
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -24.9824829, 24.9862671
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3097839, 31.3014908
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9200325, 14.9208755
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2340698, 18.2201195
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8176422, 19.8086166
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3382988, 15.3366737
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4724579, 17.4758186
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2410889, 19.2442169
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1525955, 15.1506729
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8706055, 17.8695412
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8196411, 15.8155594
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1709557, 16.1673317
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1347427, 24.1297531
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.6975098, 19.6953430
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -19.9930191, 19.9912262
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.5913391, 14.5871925
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8484573, 18.8496628
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9527435, 20.9510040
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4477692, 21.4547119
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3391953, 27.3435669
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3379364, 19.3401413
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9898834, 22.9905472
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9719391, 23.9728813
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9145164, 24.9112396
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5706482, 23.5714607
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1560135, 25.1620445
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.4863586, 15.4883652
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8311462, 25.8372154
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6810226, 17.6855507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 756

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9420763, upper bound: 5.9614221
time: 44.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9459981, upper bound: 5.9575086
time: 31.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7397766, 30.7369385
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6122208, 17.6119156
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0184479, 17.0177765
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6671982, 16.6696548
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1972504, 22.1977234
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5630417, 22.5662994
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5621796, 16.5621872
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9605408, 25.9604874
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1060257, 17.1091499
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.2915726, 23.2933960
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5392380, 15.5377998
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2666397, 19.2694016
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -24.9499130, 24.9593124
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3202667, 31.3166199
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9188995, 14.9140930
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2466736, 18.2372017
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8133087, 19.8028564
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3340836, 15.3325844
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4714584, 17.4727097
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2418976, 19.2416534
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1483841, 15.1507416
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8659973, 17.8644180
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8146973, 15.8125839
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1857872, 16.1859016
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1317596, 24.1272659
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.6853638, 19.6836929
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -19.9947510, 19.9932175
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.5864449, 14.5860405
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8579254, 18.8552780
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9465714, 20.9453888
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4504623, 21.4568100
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3369522, 27.3445816
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3341293, 19.3385735
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9894485, 22.9944420
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9615898, 23.9686813
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9024277, 24.9053764
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5663910, 23.5734024
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1638184, 25.1758957
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.4977341, 15.4982910
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8518295, 25.8560715
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6852341, 17.6889381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 899

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1661

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9621619, upper bound: 5.9286892
time: 9.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9425098, upper bound: 5.9483468
time: 59.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7391663, 30.7375565
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6116562, 17.6124840
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0187378, 17.0174866
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6664581, 16.6704025
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1972656, 22.1977081
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5648727, 22.5644684
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5618286, 16.5625420
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9600067, 25.9610138
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1040115, 17.1111679
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.2908096, 23.2941513
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5393143, 15.5377274
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2664337, 19.2696075
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -24.9487305, 24.9604950
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3191223, 31.3177567
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9172249, 14.9157677
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2455521, 18.2383156
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8152161, 19.8009415
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3344650, 15.3321991
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4721832, 17.4719849
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2421188, 19.2414322
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1501083, 15.1490021
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8660736, 17.8643456
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8164291, 15.8108482
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.1863823, 16.1853104
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1317673, 24.1272583
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.6871109, 19.6819382
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -19.9955368, 19.9924316
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.5881004, 14.5843811
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8586655, 18.8545380
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9484024, 20.9435577
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4495926, 21.4576836
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3378143, 27.3437119
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3357468, 19.3369560
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -22.9902573, 22.9936371
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9635811, 23.9666901
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9029465, 24.9048653
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5694275, 23.5703697
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.1644974, 25.1752090
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.4977303, 15.4982986
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8528824, 25.8550186
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6847000, 17.6894760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 940

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 764

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9554702, upper bound: 5.9481139
time: 38.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9439372, upper bound: 5.9558057
time: 10.44 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 50.49 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9732353, upper bound: 5.9261977
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9737289, upper bound: 5.9257030
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9608718, upper bound: 5.9212726
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9469168, upper bound: 5.9352362
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9598907, upper bound: 5.9407320
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9638181, upper bound: 5.9368181
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9499760, upper bound: 5.9364926
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9462840, upper bound: 5.9401859
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9432420, upper bound: 5.9495759
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9562418, upper bound: 5.9365136
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9420763, upper bound: 5.9614221
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9459981, upper bound: 5.9575086
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9621619, upper bound: 5.9286892
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9425098, upper bound: 5.9483468
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9554702, upper bound: 5.9481139
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 50.49
Output dim: 5, lower bound: -5.9439372, upper bound: 5.9558057
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.49
Output dim: 5, lower bound: -5.9599030, upper bound: 5.9720320
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.49
Output dim: 5, lower bound: -5.9592554, upper bound: 5.9726799
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.49
Output dim: 5, lower bound: -5.9614902, upper bound: 5.9707616
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.49
Output dim: 5, lower bound: -5.9614925, upper bound: 5.9705212
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.49
Output dim: 5, lower bound: -5.9512257, upper bound: 5.9739701
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.49
Output dim: 5, lower bound: -5.9495211, upper bound: 5.9757453
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.49
Output dim: 5, lower bound: -5.9459073, upper bound: 5.9693682
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.49
Output dim: 5, lower bound: -5.9582793, upper bound: 5.9637173

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 59.91 + 1756.86 = 1816.77 seconds
