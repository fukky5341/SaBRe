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
execution time: IAR + RelationalAnalysis = 2.55 + 57.66 = 60.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -5.9768127, upper bound: 5.9768127

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 934

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 723

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9485673, upper bound: 5.9485592
time: 9.21 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9485592, upper bound: 5.9485673
time: 15.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 25.04 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 25.04
Output dim: 5, lower bound: -5.9485673, upper bound: 5.9485592
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 25.04
Output dim: 5, lower bound: -5.9485592, upper bound: 5.9485673

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7376251, 30.7345276
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6207352, 17.6181107
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0230408, 17.0189438
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6993332, 16.6987991
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1901703, 22.1902237
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5759659, 22.5764923
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5934448, 16.5907555
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9765015, 25.9713745
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1178246, 17.1162376
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.3270645, 23.3262711
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5197296, 15.5197754
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2637024, 19.2626648
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -25.0104828, 25.0089493
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3354492, 31.3348846
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9391060, 14.9366302
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2782974, 18.2782555
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8767319, 19.8768845
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3456955, 15.3475151
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4710693, 17.4740334
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2374496, 19.2409058
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1820450, 15.1890182
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8707962, 17.8720169
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8489609, 15.8518143
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.2091217, 16.2157555
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1690903, 24.1701431
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.7316971, 19.7321014
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -20.0145111, 20.0158920
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.6086464, 14.6124687
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8595695, 18.8653603
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9714355, 20.9759827
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4512558, 21.4501801
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3650208, 27.3689270
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3419724, 19.3484230
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -23.0019531, 23.0078583
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9728851, 23.9765167
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9406433, 24.9424133
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5780106, 23.5826836
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.2146149, 25.2169800
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.5146027, 15.5088654
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8653412, 25.8636360
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6941986, 17.6900253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 934

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 721

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -5.9270560, upper bound: 5.9270478
time: 39.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -5.9270560, upper bound: 5.9270478
time: 40.23 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -31.3629379, 0.4185939, -31.3629379, 0.4185939, -30.7345276, 30.7347336
1: -4.5983768, 14.7148495, -4.5983768, 14.7148495, -17.6181107, 17.6182404
2: 1.7367579, 19.6495686, 1.7367579, 19.6495686, -17.0189438, 17.0191345
3: -2.5960343, 16.3845253, -2.5960343, 16.3845253, -16.6993408, 16.6993332
4: -2.1790190, 20.0988331, -2.1790190, 20.0988331, -22.1902237, 22.1902313
5: -0.7926209, 16.9564438, -0.7926209, 16.9564438, -17.7490654, 17.7490654
6: -41.4600410, -13.2835903, -41.4600410, -13.2835903, -22.5760269, 22.5759659
7: 0.5206641, 20.1329765, 0.5206641, 20.1329765, -16.5907593, 16.5908699
8: -2.9317336, 26.5977516, -2.9317336, 26.5977516, -25.9713745, 25.9716110
9: -3.4009962, 17.6484394, -3.4009962, 17.6484394, -17.1162376, 17.1163177
10: -10.5248709, 17.0772953, -10.5248709, 17.0772953, -23.3270569, 23.3270645
11: -11.6084976, 6.6921549, -11.6084976, 6.6921549, -15.5197754, 15.5197678
12: -33.6966858, -10.0755930, -33.6966858, -10.0755930, -19.2626648, 19.2626648
13: -20.8587570, 11.3056412, -20.8587570, 11.3056412, -25.0104904, 25.0104828
14: -34.9670525, -1.6429014, -34.9670525, -1.6429014, -31.3348846, 31.3349304
15: -11.6983480, 9.2032890, -11.6983480, 9.2032890, -20.9016380, 20.9016380
16: -19.2502575, 0.5712500, -19.2502575, 0.5712500, -14.9366302, 14.9367332
17: -36.2652130, -10.7428312, -36.2652130, -10.7428312, -18.2783127, 18.2783012
18: -26.8024483, -0.4794850, -26.8024483, -0.4794850, -19.8767395, 19.8767319
19: -11.5242262, 5.8098726, -11.5242262, 5.8098726, -15.3457794, 15.3456955
20: -5.7158918, 13.3404474, -5.7158918, 13.3404474, -17.4712067, 17.4710655
21: -11.9724140, 9.2558746, -11.9724140, 9.2558746, -19.2376022, 19.2374496
22: -12.3844757, 6.8067007, -12.3844757, 6.8067007, -15.1823730, 15.1820450
23: -7.1456704, 11.0822029, -7.1456704, 11.0822029, -17.8720169, 17.8720093
24: -16.6659985, 5.3403668, -16.6659985, 5.3403668, -15.8490906, 15.8489609
25: -11.7478676, 7.8849926, -11.7478676, 7.8849926, -16.2094345, 16.2091255
26: -17.4798164, 11.9750071, -17.4798164, 11.9750071, -24.1701431, 24.1701508
27: -14.4218454, 9.9000015, -14.4218454, 9.9000015, -19.7317200, 19.7316971
28: -8.5007658, 12.0290146, -8.5007658, 12.0290146, -20.0145721, 20.0145111
29: -13.2190323, 4.4575744, -13.2190323, 4.4575744, -14.6088295, 14.6086502
30: -13.6661015, 9.7105665, -13.6661015, 9.7105665, -18.8598442, 18.8595657
31: -20.8779793, 4.4925275, -20.8779793, 4.4925275, -20.9716568, 20.9714355
32: -30.3303928, -4.1048207, -30.3303928, -4.1048207, -21.4501801, 21.4501801
33: -61.0354042, -25.3752823, -61.0354042, -25.3752823, -27.3652267, 27.3650208
34: -60.6730919, -34.0663338, -60.6730919, -34.0663338, -19.3422852, 19.3419685
35: -54.4683228, -24.3051224, -54.4683228, -24.3051224, -23.0022202, 23.0019493
36: -45.6614494, -15.1538296, -45.6614494, -15.1538296, -23.9730606, 23.9728851
37: -74.4962540, -40.8712616, -74.4962540, -40.8712616, -24.9424133, 24.9424210
38: -55.1617126, -23.4025707, -55.1617126, -23.4025707, -23.5782242, 23.5780106
39: -60.3135567, -24.9995975, -60.3135567, -24.9995975, -25.2147675, 25.2146187
40: -55.7474899, -33.7856598, -55.7474899, -33.7856598, -15.5088654, 15.5088806
41: -39.8310089, -9.0047054, -39.8310089, -9.0047054, -25.8636398, 25.8636475
42: -25.9453106, -7.4144001, -25.9453106, -7.4144001, -17.6900330, 17.6900368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=67, inp2_unstable=67, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 934

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 721

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -5.9270478, upper bound: 5.9270560
time: 52.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -5.9270478, upper bound: 5.9270560
time: 52.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 107.19 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 107.19
Output dim: 5, lower bound: -5.9270560, upper bound: 5.9270478
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 107.19
Output dim: 5, lower bound: -5.9270560, upper bound: 5.9270478
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 107.19
Output dim: 5, lower bound: -5.9270478, upper bound: 5.9270560
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 107.19
Output dim: 5, lower bound: -5.9270478, upper bound: 5.9270560

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 60.20 + 214.12 = 274.33 seconds
