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
execution time: IAR + RelationalAnalysis = 2.31 + 58.07 = 60.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -5.9768127, upper bound: 5.9768127

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1685

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9751429, upper bound: 5.9681978
time: 30.54 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9751429, upper bound: 5.9751429
time: 17.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 48.33 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 48.33
Output dim: 5, lower bound: -5.9751429, upper bound: 5.9681978
IS_A2, status: Status.UNKNOWN, split count: 1, time: 48.33
Output dim: 5, lower bound: -5.9751429, upper bound: 5.9751429

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -31.3613758, 0.4129887, -31.3621178, 0.4159813, -30.7245331, 30.7167969
1: -4.5980811, 14.7102652, -4.5982242, 14.7126083, -17.6112061, 17.6052475
2: 1.7420990, 19.6489124, 1.7392868, 19.6492271, -17.0115814, 17.0132751
3: -2.5949640, 16.3711433, -2.5954943, 16.3782921, -16.6920242, 16.6864014
4: -2.1710839, 20.0971947, -2.1751704, 20.0980492, -22.1804504, 22.1800079
5: -0.7914858, 16.9399376, -0.7920861, 16.9486084, -17.7400932, 17.7320232
6: -41.4579010, -13.2842960, -41.4590187, -13.2839203, -22.5666656, 22.5686531
7: 0.5212125, 20.1214943, 0.5209427, 20.1273537, -16.5842133, 16.5786438
8: -2.9302840, 26.5965118, -2.9310155, 26.5971737, -25.9668121, 25.9658966
9: -3.4002728, 17.6086121, -3.4006453, 17.6298237, -17.0960236, 17.0763245
10: -10.5236502, 17.0210876, -10.5243130, 17.0509567, -23.2990723, 23.2715454
11: -11.6075535, 6.6722145, -11.6080666, 6.6827922, -15.5073471, 15.5062561
12: -33.6939087, -10.0767355, -33.6953468, -10.0761547, -19.2599716, 19.2596588
13: -20.8580055, 11.3029146, -20.8583946, 11.3043442, -25.0070190, 25.0044098
14: -34.9656601, -1.6553707, -34.9664078, -1.6487970, -31.3247681, 31.3151703
15: -11.6913042, 9.2023125, -11.6950312, 9.2028408, -20.8941460, 20.8973427
16: -19.2500725, 0.5372162, -19.2501640, 0.5553608, -14.9172134, 14.8953896
17: -36.2528000, -10.7428703, -36.2594452, -10.7428532, -18.2679024, 18.2733574
18: -26.7932129, -0.4807606, -26.7981472, -0.4801114, -19.8667984, 19.8700790
19: -11.5176964, 5.8097630, -11.5210609, 5.8098111, -15.3325005, 15.3393288
20: -5.7146072, 13.3375463, -5.7152824, 13.3390951, -17.4610748, 17.4633331
21: -11.9706001, 9.2482347, -11.9715633, 9.2522879, -19.2191696, 19.2236481
22: -12.3628025, 6.8064761, -12.3743315, 6.8066072, -15.1624985, 15.1727524
23: -7.1405530, 11.0815678, -7.1432433, 11.0818939, -17.8629837, 17.8671188
24: -16.6516342, 5.3398118, -16.6593037, 5.3400898, -15.8307877, 15.8400421
25: -11.7436657, 7.8847232, -11.7458363, 7.8848629, -16.1990051, 16.2027245
26: -17.4651566, 11.9748182, -17.4726067, 11.9748878, -24.1577148, 24.1639709
27: -14.4091692, 9.8992481, -14.4159489, 9.8996258, -19.7246780, 19.7273331
28: -8.4917765, 12.0282831, -8.4964218, 12.0286713, -20.0041885, 20.0090790
29: -13.2098017, 4.4568653, -13.2146845, 4.4572349, -14.6006699, 14.6043243
30: -13.6651850, 9.6961594, -13.6656551, 9.7036657, -18.8487396, 18.8466759
31: -20.8702240, 4.4920397, -20.8743305, 4.4922929, -20.9530334, 20.9619293
32: -30.3237915, -4.1055126, -30.3272438, -4.1051273, -21.4440460, 21.4465561
33: -60.9983826, -25.3760815, -61.0176468, -25.3756866, -27.3243866, 27.3439941
34: -60.6578827, -34.0668030, -60.6660004, -34.0665436, -19.3300781, 19.3349228
35: -54.4414978, -24.3057365, -54.4557953, -24.3054314, -22.9744873, 22.9887352
36: -45.6356888, -15.1542206, -45.6494026, -15.1540337, -23.9541054, 23.9629517
37: -74.4510727, -40.8720131, -74.4751129, -40.8716278, -24.8950615, 24.9187050
38: -55.1337280, -23.4028511, -55.1480522, -23.4027290, -23.5444450, 23.5592270
39: -60.2757835, -24.9999771, -60.2956200, -24.9997978, -25.1760101, 25.1955566
40: -55.7286072, -33.7862396, -55.7386932, -33.7859306, -15.4894180, 15.4989777
41: -39.8122215, -9.0054789, -39.8219833, -9.0050697, -25.8495102, 25.8559608
42: -25.9433784, -7.4200916, -25.9444008, -7.4172144, -17.6800842, 17.6822052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9648247, upper bound: 5.9677486
time: 51.82 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9748168, upper bound: 5.9678725
time: 41.01 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -31.3647022, 0.4159021, -31.3626900, 0.4154539, -30.7500763, 30.7257614
1: -4.6012192, 14.7145100, -4.5983119, 14.7135086, -17.6300125, 17.6098824
2: 1.7380658, 19.6532421, 1.7380319, 19.6494446, -17.0149307, 17.0180206
3: -2.6049371, 16.3745613, -2.5959408, 16.3794479, -16.7030640, 16.6892471
4: -2.1738076, 20.1013603, -2.1752911, 20.0987377, -22.1917419, 22.1750183
5: -0.8099990, 16.9532337, -0.7925305, 16.9545898, -17.7645893, 17.7457638
6: -41.4600220, -13.2819996, -41.4596558, -13.2837257, -22.5680695, 22.5786209
7: 0.5075629, 20.1310997, 0.5207118, 20.1319351, -16.6043243, 16.5844879
8: -2.9351172, 26.5965919, -2.9313679, 26.5971756, -25.9753571, 25.9665680
9: -3.4465551, 17.6460896, -3.4008598, 17.6465244, -17.1648636, 17.0996704
10: -10.5889492, 17.0736523, -10.5245886, 17.0745850, -23.3948212, 23.3007431
11: -11.6321220, 6.6908784, -11.6084538, 6.6910195, -15.5100632, 15.5226135
12: -33.6951180, -10.0715532, -33.6957932, -10.0757332, -19.2615814, 19.2627258
13: -20.8583107, 11.3031387, -20.8587036, 11.3032627, -25.0122070, 25.0092926
14: -34.9763184, -1.6501884, -34.9667091, -1.6475687, -31.3498764, 31.3268280
15: -11.6984119, 9.2101917, -11.6975412, 9.2031994, -20.9016113, 20.9077339
16: -19.2907009, 0.5698881, -19.2502480, 0.5672607, -14.9844704, 14.9160271
17: -36.2678642, -10.7292480, -36.2641144, -10.7428322, -18.2766647, 18.2802811
18: -26.7942429, -0.4743085, -26.7982368, -0.4795823, -19.8685532, 19.8733597
19: -11.5173073, 5.8108897, -11.5201397, 5.8098741, -15.3436737, 15.3596764
20: -5.7222109, 13.3399115, -5.7157478, 13.3400669, -17.4600372, 17.4758492
21: -11.9817867, 9.2546024, -11.9722347, 9.2549877, -19.2139664, 19.2440414
22: -12.3840542, 6.8301287, -12.3833132, 6.8066750, -15.1748657, 15.1971931
23: -7.1444631, 11.0848761, -7.1442266, 11.0821400, -17.8692780, 17.8797379
24: -16.6592674, 5.3506923, -16.6618919, 5.3403120, -15.8373413, 15.8642273
25: -11.7495718, 7.8871503, -11.7473927, 7.8849654, -16.2021561, 16.2141228
26: -17.4796047, 11.9850903, -17.4778824, 11.9749832, -24.1660004, 24.1728134
27: -14.4125099, 9.9075451, -14.4170237, 9.8999557, -19.7271805, 19.7219086
28: -8.5011845, 12.0379753, -8.4996758, 12.0289478, -20.0143890, 20.0263138
29: -13.2183161, 4.4682012, -13.2182407, 4.4573178, -14.6055145, 14.6145706
30: -13.6849451, 9.7099495, -13.6660433, 9.7097578, -18.8552361, 18.8580475
31: -20.8739014, 4.4952841, -20.8743725, 4.4924664, -20.9645691, 20.9890289
32: -30.3286781, -4.0979605, -30.3292294, -4.1048722, -21.4476166, 21.4546165
33: -61.0353622, -25.3302555, -61.0327072, -25.3754616, -27.3485718, 27.4150391
34: -60.6689415, -34.0484695, -60.6709900, -34.0663910, -19.3354416, 19.3491631
35: -54.4642906, -24.2745838, -54.4658661, -24.3053055, -22.9890366, 23.0304375
36: -45.6570892, -15.1214075, -45.6586075, -15.1538391, -23.9627075, 23.9903641
37: -74.4924774, -40.8178864, -74.4940872, -40.8713913, -24.9136047, 25.0014763
38: -55.1554031, -23.3728962, -55.1583481, -23.4026031, -23.5562210, 23.6002426
39: -60.3123932, -24.9554710, -60.3116531, -24.9996243, -25.1974564, 25.2626076
40: -55.7459641, -33.7638474, -55.7465591, -33.7857208, -15.4987488, 15.5324020
41: -39.8294754, -8.9782534, -39.8295174, -9.0048008, -25.8563843, 25.8713379
42: -25.9470100, -7.4203167, -25.9451981, -7.4179029, -17.6749001, 17.6905937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=67, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9648247, upper bound: 5.9746942
time: 31.41 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9748168, upper bound: 5.9748168
time: 46.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 80.32 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 80.32
Output dim: 5, lower bound: -5.9648247, upper bound: 5.9677486
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 80.32
Output dim: 5, lower bound: -5.9748168, upper bound: 5.9678725
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 80.32
Output dim: 5, lower bound: -5.9648247, upper bound: 5.9746942
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 80.32
Output dim: 5, lower bound: -5.9748168, upper bound: 5.9748168

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -31.3578300, 0.4099150, -31.3524227, 0.4093943, -30.7117386, 30.6799088
1: -4.5972710, 14.7006168, -4.5858831, 14.6927500, -17.5907059, 17.5835304
2: 1.7432340, 19.6439400, 1.7440184, 19.6390705, -17.0001144, 17.0030212
3: -2.5937450, 16.3527813, -2.5719628, 16.3405781, -16.6530533, 16.6441650
4: -2.1704588, 20.0911674, -2.1686323, 20.0855827, -22.1647949, 22.1627502
5: -0.7908444, 16.9285393, -0.7785232, 16.9252815, -17.7161255, 17.7070618
6: -41.4519386, -13.2853346, -41.4466553, -13.2907381, -22.5550652, 22.5552139
7: 0.5218843, 20.1109886, 0.5308653, 20.1065769, -16.5601196, 16.5536308
8: -2.9293113, 26.5788231, -2.9087915, 26.5608273, -25.9290619, 25.9261017
9: -3.3937464, 17.6085491, -3.3858118, 17.6288185, -17.0960312, 17.0580025
10: -10.5091267, 17.0205212, -10.4936333, 17.0398407, -23.2786865, 23.2417221
11: -11.6073208, 6.6692638, -11.6076431, 6.6756678, -15.4991112, 15.5029221
12: -33.6750641, -10.0780258, -33.6578178, -10.0917749, -19.2165833, 19.2163086
13: -20.8497906, 11.3020000, -20.8407364, 11.2998943, -24.9932098, 24.9843216
14: -34.9642067, -1.6613779, -34.9614143, -1.6610374, -31.3069000, 31.2979431
15: -11.6899624, 9.1986971, -11.6925259, 9.1952496, -20.8852119, 20.8912239
16: -19.2446651, 0.5370216, -19.2391968, 0.5554380, -14.9162979, 14.8853722
17: -36.2517624, -10.7481089, -36.2560272, -10.7536745, -18.2521667, 18.2574234
18: -26.7913895, -0.4836583, -26.7939796, -0.4872499, -19.8473434, 19.8615952
19: -11.5097504, 5.8089237, -11.5049496, 5.8043976, -15.3121758, 15.3179245
20: -5.7142086, 13.3322067, -5.7114244, 13.3276434, -17.4488068, 17.4601593
21: -11.9688110, 9.2453566, -11.9674625, 9.2454367, -19.2082138, 19.2153168
22: -12.3620758, 6.8032031, -12.3726301, 6.7995925, -15.1534004, 15.1674118
23: -7.1400356, 11.0804825, -7.1419630, 11.0789700, -17.8546829, 17.8633194
24: -16.6514053, 5.3355188, -16.6588383, 5.3310847, -15.8197632, 15.8373985
25: -11.7433033, 7.8800755, -11.7446175, 7.8748322, -16.1869507, 16.1993141
26: -17.4643211, 11.9670525, -17.4665737, 11.9586716, -24.1417160, 24.1580887
27: -14.4081450, 9.8838406, -14.4048595, 9.8680325, -19.6915512, 19.7009277
28: -8.4910421, 12.0224266, -8.4930668, 12.0159512, -19.9928513, 20.0020599
29: -13.2091465, 4.4538412, -13.2131824, 4.4501529, -14.5902672, 14.5954208
30: -13.6648903, 9.6798601, -13.6533432, 9.6708069, -18.8186798, 18.8217659
31: -20.8599281, 4.4903603, -20.8533115, 4.4856291, -20.9244537, 20.9329376
32: -30.3092613, -4.1063290, -30.2966080, -4.1204724, -21.4138260, 21.4147415
33: -60.9859505, -25.3780861, -60.9925461, -25.3931427, -27.2938690, 27.3166199
34: -60.6570358, -34.0723686, -60.6632385, -34.0780563, -19.3146439, 19.3222122
35: -54.4403152, -24.3062897, -54.4531479, -24.3073540, -22.9703293, 22.9844017
36: -45.6348267, -15.1554499, -45.6470451, -15.1573639, -23.9486465, 23.9584198
37: -74.4299927, -40.8726807, -74.4320145, -40.8975792, -24.8479919, 24.8754807
38: -55.1310501, -23.4063015, -55.1425133, -23.4101276, -23.5326843, 23.5480042
39: -60.2563019, -25.0006466, -60.2557220, -25.0231094, -25.1396408, 25.1571236
40: -55.7122269, -33.7871361, -55.7051468, -33.8063469, -15.4525986, 15.4649734
41: -39.7985916, -9.0061150, -39.7936249, -9.0193319, -25.8249359, 25.8278656
42: -25.9381638, -7.4205790, -25.9333763, -7.4223270, -17.6711655, 17.6716118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1661

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9445487, upper bound: 5.9671297
time: 38.50 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9642088, upper bound: 5.9671297
time: 18.55 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -31.3612213, 0.4120998, -31.3617325, 0.4140639, -30.7168274, 30.7404480
1: -4.5980444, 14.7097845, -4.5981054, 14.7115765, -17.6061249, 17.6047211
2: 1.7421614, 19.6486244, 1.7394241, 19.6486664, -17.0093765, 17.0129623
3: -2.5949225, 16.3703041, -2.5953593, 16.3763103, -16.6622238, 16.6854095
4: -2.1710336, 20.0963058, -2.1750627, 20.0962315, -22.1790314, 22.1811676
5: -0.7914619, 16.9394112, -0.7919672, 16.9473495, -17.7388115, 17.7313786
6: -41.4571381, -13.2843227, -41.4572678, -13.2840548, -22.5660858, 22.5623779
7: 0.5212598, 20.1208858, 0.5210469, 20.1261024, -16.5687408, 16.5780220
8: -2.9302011, 26.5957088, -2.9308906, 26.5953026, -25.9484253, 25.9649734
9: -3.3995571, 17.6086006, -3.3989801, 17.6297951, -17.0889206, 17.0955009
10: -10.5225277, 17.0210514, -10.5216064, 17.0508747, -23.2966537, 23.2500000
11: -11.6075468, 6.6715708, -11.6080236, 6.6812372, -15.5098648, 15.5032196
12: -33.6928902, -10.0768137, -33.6930466, -10.0762653, -19.2589111, 19.2306900
13: -20.8566914, 11.3028641, -20.8554935, 11.3042345, -25.0032272, 24.9934998
14: -34.9655838, -1.6565886, -34.9661179, -1.6516352, -31.3107834, 31.3145905
15: -11.6912127, 9.2006798, -11.6948624, 9.1990414, -20.8902550, 20.8955421
16: -19.2482681, 0.5372005, -19.2459679, 0.5553017, -14.9120178, 14.9013557
17: -36.2527313, -10.7443953, -36.2592659, -10.7459965, -18.2523880, 18.2730560
18: -26.7928085, -0.4809165, -26.7972031, -0.4804924, -19.8770447, 19.8664246
19: -11.5159788, 5.8097267, -11.5170012, 5.8097272, -15.3318100, 15.3176537
20: -5.7145758, 13.3371830, -5.7151670, 13.3383236, -17.4691925, 17.4582443
21: -11.9701557, 9.2479115, -11.9705219, 9.2514791, -19.2198944, 19.2222977
22: -12.3627367, 6.8056045, -12.3741817, 6.8045506, -15.1599922, 15.1718903
23: -7.1404085, 11.0813551, -7.1429477, 11.0813627, -17.8665771, 17.8643303
24: -16.6516056, 5.3391280, -16.6592369, 5.3385534, -15.8376389, 15.8367462
25: -11.7436333, 7.8841543, -11.7457428, 7.8834915, -16.2061348, 16.1991272
26: -17.4650898, 11.9738054, -17.4724350, 11.9726686, -24.1632919, 24.1574707
27: -14.4091244, 9.8987131, -14.4157505, 9.8984699, -19.6997528, 19.7247696
28: -8.4917307, 12.0275612, -8.4963217, 12.0270109, -20.0016327, 20.0055237
29: -13.2097645, 4.4560943, -13.2145681, 4.4554925, -14.5991478, 14.6057053
30: -13.6651716, 9.6953907, -13.6655827, 9.7019310, -18.8278046, 18.8404732
31: -20.8678169, 4.4919729, -20.8688087, 4.4921179, -20.9522018, 20.9408493
32: -30.3230495, -4.1055145, -30.3255692, -4.1052423, -21.4432602, 21.4224243
33: -60.9978027, -25.3762054, -61.0162811, -25.3759441, -27.3236008, 27.3235931
34: -60.6578598, -34.0685196, -60.6659050, -34.0705566, -19.3158112, 19.3340683
35: -54.4412537, -24.3057861, -54.4552460, -24.3055439, -22.9735489, 22.9865494
36: -45.6355896, -15.1546011, -45.6491470, -15.1549006, -23.9530640, 23.9626465
37: -74.4501114, -40.8720627, -74.4728622, -40.8717575, -24.8945160, 24.8674011
38: -55.1335983, -23.4041290, -55.1477585, -23.4057484, -23.5408287, 23.5579681
39: -60.2748795, -25.0000286, -60.2935562, -24.9998646, -25.1750259, 25.1565742
40: -55.7278824, -33.7862930, -55.7369347, -33.7860451, -15.4886322, 15.4636002
41: -39.8116417, -9.0054970, -39.8205414, -9.0051479, -25.8471527, 25.8312225
42: -25.9430542, -7.4201260, -25.9436512, -7.4172835, -17.6796570, 17.6754875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1387

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1661

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9741892, upper bound: 5.9475974
time: 32.52 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9741892, upper bound: 5.9672463
time: 9.36 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -31.3611813, 0.4128275, -31.3530159, 0.4088984, -30.7372742, 30.6889038
1: -4.6004229, 14.7048397, -4.5859804, 14.6936359, -17.6095123, 17.5881424
2: 1.7391741, 19.6483021, 1.7427275, 19.6392879, -17.0034332, 17.0077591
3: -2.6037354, 16.3562202, -2.5724144, 16.3417168, -16.6640625, 16.6469803
4: -2.1732011, 20.0953178, -2.1687624, 20.0862541, -22.1761093, 22.1577682
5: -0.8093526, 16.9418373, -0.7789414, 16.9312611, -17.7406139, 17.7207794
6: -41.4540367, -13.2830429, -41.4472389, -13.2905197, -22.5564728, 22.5652161
7: 0.5082097, 20.1206322, 0.5306532, 20.1111507, -16.5802002, 16.5594559
8: -2.9341817, 26.5789165, -2.9091620, 26.5608292, -25.9376068, 25.9267731
9: -3.4400463, 17.6460094, -3.3860154, 17.6455212, -17.1648636, 17.0813446
10: -10.5744133, 17.0731125, -10.4939079, 17.0634899, -23.3744431, 23.2708664
11: -11.6318493, 6.6879578, -11.6080074, 6.6839199, -15.5018158, 15.5192833
12: -33.6762886, -10.0728903, -33.6582642, -10.0912991, -19.2181702, 19.2193527
13: -20.8500843, 11.3022385, -20.8410301, 11.2987499, -24.9984055, 24.9891739
14: -34.9747772, -1.6562576, -34.9617157, -1.6598377, -31.3320312, 31.3095856
15: -11.6970415, 9.2065678, -11.6950417, 9.1956072, -20.8926487, 20.9016094
16: -19.2852707, 0.5697122, -19.2392616, 0.5673618, -14.9835587, 14.9060173
17: -36.2668495, -10.7344999, -36.2607384, -10.7536564, -18.2609253, 18.2643166
18: -26.7924099, -0.4771700, -26.7940712, -0.4867072, -19.8490982, 19.8648682
19: -11.5093241, 5.8100467, -11.5040131, 5.8044300, -15.3233566, 15.3382988
20: -5.7218475, 13.3345165, -5.7118812, 13.3286381, -17.4477692, 17.4726791
21: -11.9800053, 9.2517595, -11.9681301, 9.2481594, -19.2030411, 19.2357254
22: -12.3833046, 6.8268390, -12.3815975, 6.7996750, -15.1657753, 15.1918373
23: -7.1439390, 11.0838003, -7.1429620, 11.0792313, -17.8609467, 17.8759346
24: -16.6590290, 5.3464050, -16.6614571, 5.3312984, -15.8262825, 15.8616180
25: -11.7492352, 7.8825002, -11.7462034, 7.8749251, -16.1900864, 16.2106438
26: -17.4788132, 11.9772882, -17.4718781, 11.9587822, -24.1500015, 24.1669464
27: -14.4114742, 9.8921356, -14.4059200, 9.8683453, -19.6940613, 19.6954346
28: -8.5004549, 12.0321493, -8.4962921, 12.0162334, -20.0030518, 20.0192719
29: -13.2176762, 4.4651589, -13.2167778, 4.4502196, -14.5951462, 14.6056442
30: -13.6846657, 9.6936502, -13.6536741, 9.6769171, -18.8251762, 18.8331337
31: -20.8636589, 4.4936080, -20.8533382, 4.4858298, -20.9360275, 20.9600143
32: -30.3141270, -4.0987458, -30.2986164, -4.1201758, -21.4174042, 21.4227982
33: -61.0229149, -25.3322029, -61.0075417, -25.3929234, -27.3180618, 27.3876724
34: -60.6680756, -34.0540695, -60.6682091, -34.0778618, -19.3199768, 19.3364677
35: -54.4631195, -24.2751732, -54.4632263, -24.3072472, -22.9848709, 23.0261269
36: -45.6561813, -15.1226711, -45.6562653, -15.1571865, -23.9572754, 23.9858170
37: -74.4713593, -40.8186035, -74.4509888, -40.8973579, -24.8665695, 24.9582787
38: -55.1527557, -23.3763180, -55.1527939, -23.4100037, -23.5444336, 23.5890007
39: -60.2929459, -24.9561424, -60.2717514, -25.0230122, -25.1611023, 25.2241783
40: -55.7296028, -33.7646904, -55.7130127, -33.8061333, -15.4618950, 15.4984016
41: -39.8158951, -8.9788742, -39.8011742, -9.0190697, -25.8318253, 25.8432465
42: -25.9417648, -7.4208021, -25.9341660, -7.4230289, -17.6660080, 17.6799889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1661

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9445487, upper bound: 5.9740716
time: 11.83 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9445487, upper bound: 5.9740716
time: 44.94 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -31.3645840, 0.4150205, -31.3623047, 0.4135466, -30.7423553, 30.7494354
1: -4.6011724, 14.7140589, -4.5982161, 14.7124767, -17.6248856, 17.6093445
2: 1.7381208, 19.6529617, 1.7381459, 19.6488686, -17.0127487, 17.0176697
3: -2.6048806, 16.3737221, -2.5957997, 16.3774662, -16.6732941, 16.6882553
4: -2.1737511, 20.1004848, -2.1751897, 20.0969353, -22.1903381, 22.1761932
5: -0.8099604, 16.9527092, -0.7924285, 16.9533768, -17.7633362, 17.7451382
6: -41.4592590, -13.2820473, -41.4578743, -13.2838993, -22.5674896, 22.5723228
7: 0.5075798, 20.1305370, 0.5208238, 20.1306629, -16.5888443, 16.5838928
8: -2.9350944, 26.5958042, -2.9312406, 26.5952873, -25.9569550, 25.9656372
9: -3.4458332, 17.6460762, -3.3991728, 17.6464958, -17.1577492, 17.1188393
10: -10.5877972, 17.0736523, -10.5219021, 17.0745068, -23.3924103, 23.2791672
11: -11.6320820, 6.6902294, -11.6084099, 6.6894422, -15.5125542, 15.5195541
12: -33.6941299, -10.0716467, -33.6934967, -10.0758762, -19.2605057, 19.2337646
13: -20.8569622, 11.3030891, -20.8557281, 11.3031235, -25.0084152, 24.9983521
14: -34.9761658, -1.6513958, -34.9664116, -1.6504526, -31.3359375, 31.3262177
15: -11.6982899, 9.2085714, -11.6974020, 9.1994076, -20.8976974, 20.9059734
16: -19.2888947, 0.5698857, -19.2460442, 0.5671930, -14.9792709, 14.9219971
17: -36.2677956, -10.7307577, -36.2639542, -10.7459793, -18.2611389, 18.2799683
18: -26.7938557, -0.4744654, -26.7972775, -0.4799476, -19.8787918, 19.8697052
19: -11.5155697, 5.8108606, -11.5160856, 5.8097677, -15.3430061, 15.3380280
20: -5.7221694, 13.3395557, -5.7156391, 13.3393421, -17.4681473, 17.4707947
21: -11.9813290, 9.2542706, -11.9711990, 9.2541866, -19.2147217, 19.2426758
22: -12.3839817, 6.8292575, -12.3831615, 6.8046641, -15.1723862, 15.1963387
23: -7.1443224, 11.0846338, -7.1439152, 11.0816097, -17.8728561, 17.8769341
24: -16.6592312, 5.3500257, -16.6618271, 5.3388000, -15.8441620, 15.8609314
25: -11.7495422, 7.8865519, -11.7472935, 7.8836098, -16.2092819, 16.2105064
26: -17.4795074, 11.9840755, -17.4777412, 11.9727631, -24.1715698, 24.1663208
27: -14.4124470, 9.9070168, -14.4168415, 9.8987923, -19.7022324, 19.7192993
28: -8.5011349, 12.0372601, -8.4995594, 12.0273018, -20.0118179, 20.0227737
29: -13.2182646, 4.4674168, -13.2181377, 4.4555683, -14.6039925, 14.6159515
30: -13.6849394, 9.7092113, -13.6659632, 9.7080345, -18.8342972, 18.8518524
31: -20.8715630, 4.4952312, -20.8687973, 4.4923000, -20.9637451, 20.9679489
32: -30.3279686, -4.0979967, -30.3275566, -4.1049414, -21.4468689, 21.4304924
33: -61.0347672, -25.3303699, -61.0313797, -25.3757229, -27.3477783, 27.3946152
34: -60.6688614, -34.0501862, -60.6708755, -34.0703773, -19.3211632, 19.3483276
35: -54.4640694, -24.2746639, -54.4653091, -24.3053741, -22.9880829, 23.0282745
36: -45.6569824, -15.1218033, -45.6583481, -15.1547508, -23.9616928, 23.9900322
37: -74.4914856, -40.8179474, -74.4918213, -40.8714600, -24.9130478, 24.9501648
38: -55.1552734, -23.3742218, -55.1580429, -23.4056358, -23.5525513, 23.5990143
39: -60.3115196, -24.9555206, -60.3095779, -24.9997215, -25.1964951, 25.2236404
40: -55.7451859, -33.7638969, -55.7448235, -33.7858276, -15.4979439, 15.4970436
41: -39.8288422, -8.9782324, -39.8280907, -9.0048790, -25.8540344, 25.8466034
42: -25.9466896, -7.4203415, -25.9444866, -7.4179802, -17.6744690, 17.6838875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1387

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1661

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9741892, upper bound: 5.9545385
time: 26.42 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9741892, upper bound: 5.9741892
time: 23.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 52.36 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 52.36
Output dim: 5, lower bound: -5.9445487, upper bound: 5.9671297
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 52.36
Output dim: 5, lower bound: -5.9642088, upper bound: 5.9671297
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 52.36
Output dim: 5, lower bound: -5.9741892, upper bound: 5.9475974
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 52.36
Output dim: 5, lower bound: -5.9741892, upper bound: 5.9672463
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 52.36
Output dim: 5, lower bound: -5.9445487, upper bound: 5.9740716
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 52.36
Output dim: 5, lower bound: -5.9445487, upper bound: 5.9740716
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 52.36
Output dim: 5, lower bound: -5.9741892, upper bound: 5.9545385
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 52.36
Output dim: 5, lower bound: -5.9741892, upper bound: 5.9741892

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -31.3560047, 0.3962560, -31.3269730, 0.3788171, -30.6781921, 30.6385193
1: -4.5967765, 14.6870432, -4.5644288, 14.6636410, -17.5614319, 17.5483055
2: 1.7437509, 19.6320744, 1.7625368, 19.6135845, -16.9740219, 16.9722214
3: -2.5928657, 16.3353386, -2.5447707, 16.3039989, -16.6156616, 16.5993805
4: -2.1694391, 20.0750179, -2.1419222, 20.0521965, -22.1302795, 22.1194839
5: -0.7898462, 16.9144249, -0.7558837, 16.8951015, -17.6849480, 17.6703091
6: -41.4453773, -13.2861471, -41.4323502, -13.3007355, -22.5356369, 22.5384178
7: 0.5226923, 20.0966663, 0.5540345, 20.0766487, -16.5295410, 16.5159378
8: -2.9278340, 26.5608521, -2.8769946, 26.5233459, -25.8899231, 25.8760300
9: -3.3928614, 17.5894470, -3.3513036, 17.5897274, -17.0561485, 17.0046577
10: -10.5076313, 17.0126801, -10.4749966, 17.0221424, -23.2593536, 23.2145233
11: -11.5956631, 6.6684198, -11.5824671, 6.6559954, -15.4679108, 15.4778519
12: -33.6639442, -10.0793247, -33.6324005, -10.1131124, -19.1830215, 19.1883316
13: -20.8491974, 11.2877522, -20.8208923, 11.2675610, -24.9611282, 24.9500351
14: -34.9579086, -1.6617250, -34.9426651, -1.6637821, -31.2942200, 31.2760620
15: -11.6887321, 9.1872969, -11.6727629, 9.1701975, -20.8589287, 20.8600597
16: -19.2442074, 0.5248485, -19.2173576, 0.5276194, -14.8874016, 14.8503914
17: -36.2384567, -10.7482986, -36.2235985, -10.7726297, -18.2205582, 18.2244987
18: -26.7760658, -0.4845853, -26.7612324, -0.5140629, -19.8048096, 19.8288040
19: -11.4950275, 5.8085437, -11.4743786, 5.7826195, -15.2747498, 15.2875519
20: -5.7020216, 13.3319798, -5.6824861, 13.3090048, -17.4162521, 17.4318924
21: -11.9553785, 9.2449207, -11.9373779, 9.2283773, -19.1760559, 19.1859055
22: -12.3474464, 6.8029866, -12.3406849, 6.7785454, -15.1174927, 15.1355324
23: -7.1238055, 11.0797291, -7.1076970, 11.0540562, -17.8132782, 17.8287735
24: -16.6308994, 5.3348799, -16.6155300, 5.2984176, -15.7664452, 15.7941818
25: -11.7219400, 7.8796358, -11.6987619, 7.8425007, -16.1323357, 16.1537361
26: -17.4475098, 11.9665976, -17.4277878, 11.9333935, -24.0979614, 24.1190872
27: -14.3951511, 9.8832321, -14.3768826, 9.8505993, -19.6606827, 19.6727448
28: -8.4754639, 12.0216312, -8.4588680, 11.9914246, -19.9514694, 19.9678802
29: -13.1966276, 4.4532146, -13.1871033, 4.4285946, -14.5561676, 14.5684166
30: -13.6501007, 9.6790543, -13.6204967, 9.6497440, -18.7826614, 18.7890434
31: -20.8380222, 4.4896679, -20.8073883, 4.4547687, -20.8709412, 20.8870544
32: -30.3048019, -4.1070037, -30.2849274, -4.1282506, -21.3912125, 21.3982391
33: -60.9807472, -25.3793163, -60.9788246, -25.4051170, -27.2724762, 27.2997284
34: -60.6452217, -34.0728111, -60.6374893, -34.1009445, -19.2791672, 19.2954941
35: -54.4314232, -24.3074608, -54.4343414, -24.3255577, -22.9459534, 22.9662132
36: -45.6240845, -15.1561346, -45.6247864, -15.1777220, -23.9205933, 23.9377937
37: -74.4181442, -40.8737602, -74.4071274, -40.9186096, -24.8150101, 24.8495522
38: -55.1139603, -23.4070282, -55.1048355, -23.4370232, -23.4889297, 23.5092735
39: -60.2508774, -25.0011616, -60.2423935, -25.0291309, -25.1294594, 25.1446648
40: -55.7116737, -33.7882805, -55.7033882, -33.8101006, -15.4452324, 15.4613228
41: -39.7948990, -9.0071802, -39.7850990, -9.0292482, -25.8098450, 25.8176003
42: -25.9335690, -7.4214573, -25.9222946, -7.4304790, -17.6505356, 17.6586456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9242202, upper bound: 5.9667443
time: 28.68 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9441833, upper bound: 5.9667657
time: 53.62 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -31.3576279, 0.4089174, -31.3520584, 0.4074373, -30.7089386, 30.6791382
1: -4.5971417, 14.6996737, -4.5855961, 14.6908770, -17.5837784, 17.5823212
2: 1.7434205, 19.6431179, 1.7443815, 19.6374302, -16.9923325, 17.0017395
3: -2.5932927, 16.3516293, -2.5710425, 16.3382473, -16.6319885, 16.6419678
4: -2.1701832, 20.0900993, -2.1681216, 20.0833893, -22.1509933, 22.1609650
5: -0.7904253, 16.9275990, -0.7776554, 16.9233818, -17.7138062, 17.7052536
6: -41.4507370, -13.2854404, -41.4444084, -13.2908821, -22.5540733, 22.5470772
7: 0.5222099, 20.1100197, 0.5315193, 20.1046143, -16.5457153, 16.5519943
8: -2.9288006, 26.5775700, -2.9078240, 26.5583382, -25.9109650, 25.9238205
9: -3.3935790, 17.6072273, -3.3854594, 17.6261425, -17.0653458, 17.0563431
10: -10.5089760, 17.0200939, -10.4933701, 17.0389843, -23.2725296, 23.2407608
11: -11.6065664, 6.6690035, -11.6060801, 6.6751642, -15.4977875, 15.4900742
12: -33.6742554, -10.0782337, -33.6561928, -10.0921497, -19.2151566, 19.2073593
13: -20.8497581, 11.3010435, -20.8405495, 11.2980080, -24.9880447, 24.9829712
14: -34.9628563, -1.6614885, -34.9587784, -1.6611996, -31.2986145, 31.2935410
15: -11.6897612, 9.1979351, -11.6922092, 9.1936712, -20.8834324, 20.8901443
16: -19.2445946, 0.5361366, -19.2390480, 0.5536213, -14.9022789, 14.8842659
17: -36.2507248, -10.7481461, -36.2540016, -10.7537012, -18.2510071, 18.2335548
18: -26.7903519, -0.4840450, -26.7919273, -0.4879694, -19.8456116, 19.8471909
19: -11.5087223, 5.8086414, -11.5029411, 5.8038101, -15.3105927, 15.3054848
20: -5.7133865, 13.3321552, -5.7097254, 13.3275270, -17.4479523, 17.4581566
21: -11.9678421, 9.2451744, -11.9655056, 9.2450838, -19.2068100, 19.2116852
22: -12.3610020, 6.8030157, -12.3705273, 6.7992134, -15.1518784, 15.1519852
23: -7.1389847, 11.0801239, -7.1398439, 11.0782719, -17.8528442, 17.8524323
24: -16.6500206, 5.3351836, -16.6560898, 5.3304057, -15.8177338, 15.8049660
25: -11.7418537, 7.8798714, -11.7417336, 7.8743963, -16.1850471, 16.1728897
26: -17.4631405, 11.9667873, -17.4642448, 11.9581985, -24.1400833, 24.1544724
27: -14.4071960, 9.8836422, -14.4030161, 9.8675880, -19.6901169, 19.6831894
28: -8.4899960, 12.0221405, -8.4909601, 12.0153646, -19.9911880, 19.9964828
29: -13.2082548, 4.4535761, -13.2113953, 4.4496059, -14.5886765, 14.5719070
30: -13.6638784, 9.6797371, -13.6512947, 9.6705647, -18.8173409, 18.8050919
31: -20.8584461, 4.4900522, -20.8503666, 4.4850321, -20.9223633, 20.9107742
32: -30.3082962, -4.1064043, -30.2947655, -4.1206532, -21.4167862, 21.4112930
33: -60.9847603, -25.3782101, -60.9903793, -25.3933868, -27.2929535, 27.3143692
34: -60.6561737, -34.0724640, -60.6614532, -34.0782013, -19.3134079, 19.3014641
35: -54.4395866, -24.3064804, -54.4518013, -24.3077660, -22.9691582, 22.9672585
36: -45.6339226, -15.1556768, -45.6452637, -15.1578026, -23.9473305, 23.9381981
37: -74.4291534, -40.8730469, -74.4303970, -40.8983078, -24.8464355, 24.8537750
38: -55.1298561, -23.4065075, -55.1400146, -23.4104500, -23.5310135, 23.5172882
39: -60.2548981, -25.0007668, -60.2530594, -25.0233364, -25.1375732, 25.1489372
40: -55.7120552, -33.7881088, -55.7048340, -33.8083954, -15.4549599, 15.4608727
41: -39.7979813, -9.0063171, -39.7925110, -9.0197134, -25.8236847, 25.8239441
42: -25.9377117, -7.4207044, -25.9324684, -7.4225302, -17.6743851, 17.6678467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1387

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9638222, upper bound: 5.9468101
time: 33.89 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9441625, upper bound: 5.9667655
time: 64.89 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -31.3357201, 0.3815384, -31.3598938, 0.4003563, -30.6754074, 30.7069092
1: -4.5765672, 14.6807051, -4.5976009, 14.6980162, -17.5709229, 17.5754395
2: 1.7606790, 19.6231213, 1.7399316, 19.6368256, -16.9785919, 16.9868164
3: -2.5676720, 16.3337383, -2.5944781, 16.3588123, -16.6174622, 16.6480408
4: -2.1442950, 20.0629482, -2.1740553, 20.0800743, -22.1357422, 22.1466446
5: -0.7688611, 16.9091988, -0.7909737, 16.9332657, -17.7021275, 17.7001724
6: -41.4428902, -13.2944412, -41.4507103, -13.2848177, -22.5492935, 22.5429153
7: 0.5444448, 20.0909309, 0.5218502, 20.1118183, -16.5310745, 16.5474663
8: -2.8984251, 26.5582008, -2.9294033, 26.5772667, -25.8983688, 25.9258194
9: -3.3650403, 17.5695152, -3.3980927, 17.6107044, -17.0355644, 17.0556526
10: -10.5039148, 17.0033665, -10.5200605, 17.0430355, -23.2694321, 23.2306519
11: -11.5823946, 6.6519036, -11.5963545, 6.6804118, -15.4847832, 15.4720001
12: -33.6674500, -10.0981035, -33.6818771, -10.0776081, -19.2309418, 19.1970978
13: -20.8368893, 11.2705040, -20.8548393, 11.2900200, -24.9689484, 24.9614182
14: -34.9468613, -1.6593056, -34.9598770, -1.6519709, -31.2888947, 31.3018799
15: -11.6714420, 9.1756611, -11.6936264, 9.1876450, -20.8590870, 20.8692875
16: -19.2264137, 0.5094080, -19.2455196, 0.5431399, -14.8770180, 14.8724937
17: -36.2202606, -10.7633982, -36.2459412, -10.7461863, -18.2194519, 18.2414398
18: -26.7600536, -0.5077326, -26.7818184, -0.4814069, -19.8442230, 19.8239136
19: -11.4854193, 5.7879524, -11.5023003, 5.8093252, -15.3014679, 15.2802353
20: -5.6856413, 13.3185549, -5.7029667, 13.3381500, -17.4409561, 17.4256668
21: -11.9400873, 9.2307882, -11.9570541, 9.2510338, -19.1904755, 19.1901627
22: -12.3307486, 6.7845616, -12.3595610, 6.8043337, -15.1280823, 15.1360054
23: -7.1061540, 11.0564413, -7.1267271, 11.0806007, -17.8320389, 17.8229370
24: -16.6083202, 5.3064642, -16.6387062, 5.3379397, -15.7944031, 15.7834206
25: -11.6977482, 7.8518085, -11.7243681, 7.8830519, -16.1606102, 16.1445084
26: -17.4263000, 11.9485025, -17.4556351, 11.9722242, -24.1242905, 24.1137085
27: -14.3811283, 9.8812599, -14.4027891, 9.8978739, -19.6715698, 19.6939011
28: -8.4575233, 12.0030289, -8.4807510, 12.0261927, -19.9674454, 19.9641647
29: -13.1836834, 4.4345474, -13.2020531, 4.4548187, -14.5720978, 14.5715942
30: -13.6323509, 9.6743059, -13.6507683, 9.7011309, -18.7951050, 18.8044548
31: -20.8219109, 4.4611111, -20.8469296, 4.4913754, -20.9063339, 20.8873138
32: -30.3113728, -4.1133256, -30.3211479, -4.1059589, -21.4267654, 21.3998032
33: -60.9841843, -25.3882027, -61.0111237, -25.3771839, -27.3066635, 27.3022079
34: -60.6320992, -34.0914383, -60.6540680, -34.0709839, -19.2890472, 19.2985764
35: -54.4225006, -24.3239460, -54.4463425, -24.3067150, -22.9553452, 22.9622116
36: -45.6132660, -15.1749668, -45.6384163, -15.1555882, -23.9323959, 23.9345779
37: -74.4252014, -40.8930511, -74.4610367, -40.8727608, -24.8685722, 24.8344231
38: -55.0959396, -23.4311180, -55.1306381, -23.4064274, -23.5020981, 23.5142365
39: -60.2615356, -25.0060673, -60.2881012, -25.0003948, -25.1625519, 25.1463966
40: -55.7261124, -33.7900543, -55.7363358, -33.7872543, -15.4849701, 15.4562378
41: -39.8031082, -9.0154285, -39.8168030, -9.0062180, -25.8369217, 25.8160973
42: -25.9319706, -7.4283104, -25.9391098, -7.4181585, -17.6666489, 17.6548386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1387

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9638223, upper bound: 5.9272767
time: 13.43 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9638223, upper bound: 5.9272767
time: 51.58 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -31.3608418, 0.4101582, -31.3615208, 0.4130669, -30.7160492, 30.7376709
1: -4.5977402, 14.7079420, -4.5979729, 14.7106571, -17.6049194, 17.5978012
2: 1.7425326, 19.6469727, 1.7396388, 19.6478462, -17.0081558, 17.0051727
3: -2.5939715, 16.3679581, -2.5948894, 16.3751278, -16.6600647, 16.6643600
4: -2.1704831, 20.0941429, -2.1748154, 20.0951347, -22.1772461, 22.1673508
5: -0.7906249, 16.9375114, -0.7915673, 16.9464035, -17.7370281, 17.7290783
6: -41.4549446, -13.2845211, -41.4560585, -13.2841263, -22.5579147, 22.5613899
7: 0.5219380, 20.1189251, 0.5213461, 20.1251335, -16.5670929, 16.5636215
8: -2.9292593, 26.5932198, -2.9304008, 26.5940323, -25.9461517, 25.9468536
9: -3.3992348, 17.6059227, -3.3988070, 17.6284657, -17.0872726, 17.0648270
10: -10.5222206, 17.0202122, -10.5214424, 17.0504208, -23.2957077, 23.2438049
11: -11.6059952, 6.6710386, -11.6072521, 6.6809554, -15.4970169, 15.5018768
12: -33.6913109, -10.0772209, -33.6922379, -10.0765171, -19.2499695, 19.2292633
13: -20.8565197, 11.3009739, -20.8554039, 11.3032665, -25.0018692, 24.9883423
14: -34.9629555, -1.6567802, -34.9647903, -1.6517048, -31.3064194, 31.3062592
15: -11.6909027, 9.1991386, -11.6946802, 9.1982803, -20.8891830, 20.8938179
16: -19.2480965, 0.5354052, -19.2458820, 0.5544190, -14.9109077, 14.8873444
17: -36.2506638, -10.7444105, -36.2582321, -10.7460165, -18.2285271, 18.2718849
18: -26.7907658, -0.4816384, -26.7961655, -0.4807847, -19.8626099, 19.8647461
19: -11.5140133, 5.8091092, -11.5160122, 5.8094382, -15.3193893, 15.3160973
20: -5.7128887, 13.3370743, -5.7143373, 13.3382730, -17.4672241, 17.4573898
21: -11.9681950, 9.2475643, -11.9695015, 9.2513065, -19.2163239, 19.2208939
22: -12.3606071, 6.8052106, -12.3731489, 6.8043709, -15.1445694, 15.1703987
23: -7.1382866, 11.0806274, -7.1418877, 11.0809975, -17.8556900, 17.8624954
24: -16.6488724, 5.3384275, -16.6578617, 5.3382120, -15.8051987, 15.8347130
25: -11.7407236, 7.8837109, -11.7442722, 7.8832941, -16.1797218, 16.1972313
26: -17.4627609, 11.9732723, -17.4712982, 11.9724350, -24.1596603, 24.1558533
27: -14.4072533, 9.8982925, -14.4147835, 9.8982811, -19.6820068, 19.7233124
28: -8.4896269, 12.0269756, -8.4952765, 12.0267048, -19.9960709, 20.0038910
29: -13.2079582, 4.4555788, -13.2136259, 4.4552059, -14.5756149, 14.6041069
30: -13.6630993, 9.6951637, -13.6645489, 9.7018433, -18.8111305, 18.8391380
31: -20.8648701, 4.4913831, -20.8673153, 4.4918294, -20.9300919, 20.9387741
32: -30.3212509, -4.1056523, -30.3246536, -4.1052847, -21.4398575, 21.4253883
33: -60.9956818, -25.3764591, -61.0151367, -25.3760452, -27.3213425, 27.3226700
34: -60.6561127, -34.0686836, -60.6650467, -34.0706520, -19.2950134, 19.3328476
35: -54.4398613, -24.3062057, -54.4545441, -24.3057423, -22.9563904, 22.9854050
36: -45.6337929, -15.1550570, -45.6482468, -15.1551609, -23.9328423, 23.9613152
37: -74.4484634, -40.8727608, -74.4720459, -40.8721085, -24.8727951, 24.8658218
38: -55.1311646, -23.4045582, -55.1465416, -23.4059372, -23.5101357, 23.5563049
39: -60.2722015, -25.0002403, -60.2921181, -24.9999809, -25.1668243, 25.1545143
40: -55.7275848, -33.7883301, -55.7368011, -33.7870483, -15.4845200, 15.4659729
41: -39.8105240, -9.0059090, -39.8199615, -9.0053654, -25.8432770, 25.8300095
42: -25.9421730, -7.4203129, -25.9432144, -7.4173832, -17.6759071, 17.6787033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1387

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9738051, upper bound: 5.9469383
time: 14.73 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9738265, upper bound: 5.9668827
time: 53.94 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -31.3593502, 0.3991117, -31.3275471, 0.3783288, -30.7037048, 30.6474762
1: -4.5998893, 14.6912537, -4.5645142, 14.6645393, -17.5802155, 17.5529404
2: 1.7396998, 19.6364403, 1.7612437, 19.6137772, -16.9773254, 16.9769516
3: -2.6028254, 16.3387642, -2.5451665, 16.3051834, -16.6267242, 16.6021919
4: -2.1721556, 20.0791893, -2.1420445, 20.0528755, -22.1415863, 22.1144791
5: -0.8083622, 16.9277210, -0.7563529, 16.9010696, -17.7094326, 17.6840744
6: -41.4475021, -13.2838554, -41.4329681, -13.3005581, -22.5370293, 22.5483742
7: 0.5090441, 20.1063118, 0.5538189, 20.0812321, -16.5496445, 16.5218239
8: -2.9327264, 26.5609303, -2.8773389, 26.5233440, -25.8984222, 25.8767242
9: -3.4391398, 17.6269341, -3.3515029, 17.6064167, -17.1250114, 17.0279846
10: -10.5728989, 17.0652580, -10.4753122, 17.0457573, -23.3551025, 23.2436447
11: -11.6201954, 6.6871214, -11.5828991, 6.6642356, -15.4706383, 15.4941864
12: -33.6651688, -10.0742092, -33.6327782, -10.1126709, -19.1846161, 19.1913605
13: -20.8494606, 11.2880154, -20.8211842, 11.2664595, -24.9662476, 24.9549026
14: -34.9685440, -1.6565781, -34.9429893, -1.6625776, -31.3193817, 31.2876892
15: -11.6958122, 9.1951790, -11.6752548, 9.1705608, -20.8663731, 20.8704338
16: -19.2848263, 0.5575304, -19.2174473, 0.5395403, -14.9546928, 14.8710518
17: -36.2535553, -10.7346754, -36.2283020, -10.7726402, -18.2293167, 18.2313881
18: -26.7770519, -0.4781013, -26.7613430, -0.5134616, -19.8065567, 19.8320694
19: -11.4946098, 5.8096871, -11.4734516, 5.7826428, -15.2859421, 15.3079262
20: -5.7096314, 13.3343277, -5.6829453, 13.3099823, -17.4151764, 17.4444199
21: -11.9665604, 9.2513065, -11.9380493, 9.2310276, -19.1708984, 19.2062988
22: -12.3686867, 6.8266521, -12.3496265, 6.7786655, -15.1298523, 15.1599884
23: -7.1277232, 11.0830469, -7.1086636, 11.0543079, -17.8195801, 17.8413582
24: -16.6385193, 5.3457985, -16.6181469, 5.2986317, -15.7729874, 15.8183784
25: -11.7278652, 7.8820753, -11.7003345, 7.8426099, -16.1355057, 16.1651001
26: -17.4620152, 11.9769087, -17.4330444, 11.9334717, -24.1062241, 24.1279068
27: -14.3985119, 9.8915081, -14.3779545, 9.8508797, -19.6631775, 19.6672745
28: -8.4848623, 12.0313253, -8.4620848, 11.9916878, -19.9617081, 19.9850845
29: -13.2051487, 4.4645195, -13.1906929, 4.4286823, -14.5610123, 14.5786362
30: -13.6698542, 9.6928825, -13.6208668, 9.6558123, -18.7891655, 18.8004074
31: -20.8417358, 4.4929256, -20.8074093, 4.4549751, -20.8824692, 20.9141464
32: -30.3097553, -4.0994873, -30.2869358, -4.1279745, -21.3947754, 21.4062958
33: -61.0177002, -25.3334465, -60.9938469, -25.4049301, -27.2966995, 27.3707581
34: -60.6562614, -34.0544777, -60.6424713, -34.1007347, -19.2845039, 19.3097267
35: -54.4542313, -24.2763748, -54.4444504, -24.3253899, -22.9604874, 23.0079193
36: -45.6454887, -15.1233606, -45.6339798, -15.1775780, -23.9292145, 23.9652023
37: -74.4595184, -40.8196182, -74.4260635, -40.9183578, -24.8335457, 24.9322968
38: -55.1356430, -23.3770409, -55.1150742, -23.4368973, -23.5006714, 23.5502853
39: -60.2875748, -24.9566841, -60.2584686, -25.0290680, -25.1509285, 25.2117310
40: -55.7290115, -33.7658920, -55.7112961, -33.8098755, -15.4545212, 15.4947433
41: -39.8121567, -8.9799347, -39.7926407, -9.0289726, -25.8166656, 25.8329773
42: -25.9372082, -7.4216704, -25.9231052, -7.4311528, -17.6453781, 17.6670303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9242202, upper bound: 5.9736861
time: 36.85 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9242202, upper bound: 5.9737076
time: 13.82 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -31.3609886, 0.4118557, -31.3526077, 0.4069195, -30.7344666, 30.6881256
1: -4.6002798, 14.7039213, -4.5857086, 14.6917553, -17.6025772, 17.5869446
2: 1.7393818, 19.6474628, 1.7430835, 19.6376400, -16.9956741, 17.0064774
3: -2.6032703, 16.3550682, -2.5714474, 16.3394012, -16.6430359, 16.6448135
4: -2.1729052, 20.0942535, -2.1681905, 20.0840778, -22.1623001, 22.1559906
5: -0.8089421, 16.9408951, -0.7781115, 16.9293613, -17.7383041, 17.7190056
6: -41.4528198, -13.2831259, -41.4450150, -13.2906866, -22.5555115, 22.5570374
7: 0.5085564, 20.1196690, 0.5312994, 20.1091671, -16.5657806, 16.5578423
8: -2.9336886, 26.5776672, -2.9081936, 26.5583706, -25.9195099, 25.9244919
9: -3.4398656, 17.6446648, -3.3856936, 17.6428585, -17.1341934, 17.0796814
10: -10.5742788, 17.0726166, -10.4936600, 17.0626335, -23.3682480, 23.2699509
11: -11.6310883, 6.6876845, -11.6064825, 6.6834087, -15.5004845, 15.5064278
12: -33.6755104, -10.0730801, -33.6566544, -10.0917511, -19.2167358, 19.2104187
13: -20.8499775, 11.3012924, -20.8408432, 11.2968807, -24.9932556, 24.9878082
14: -34.9734840, -1.6562977, -34.9591103, -1.6600161, -31.3237457, 31.3051910
15: -11.6968718, 9.2058058, -11.6947174, 9.1940451, -20.8909168, 20.9005241
16: -19.2852116, 0.5688195, -19.2391052, 0.5655541, -14.9695549, 14.9049110
17: -36.2658081, -10.7345219, -36.2586861, -10.7536812, -18.2597313, 18.2404594
18: -26.7913666, -0.4775400, -26.7920227, -0.4874268, -19.8473663, 19.8504486
19: -11.5083218, 5.8097649, -11.5020351, 5.8038654, -15.3218002, 15.3258553
20: -5.7210002, 13.3344746, -5.7101831, 13.3285065, -17.4469070, 17.4707146
21: -11.9789925, 9.2515650, -11.9661684, 9.2477827, -19.2016602, 19.2321320
22: -12.3822641, 6.8266587, -12.3794804, 6.7992954, -15.1642838, 15.1764183
23: -7.1428928, 11.0834503, -7.1408358, 11.0785179, -17.8591232, 17.8650398
24: -16.6576691, 5.3460636, -16.6587200, 5.3306284, -15.8242569, 15.8291779
25: -11.7477970, 7.8822951, -11.7432909, 7.8744979, -16.1882172, 16.1842537
26: -17.4776535, 11.9770403, -17.4695511, 11.9582777, -24.1483154, 24.1633377
27: -14.4105339, 9.8919230, -14.4040604, 9.8679323, -19.6926193, 19.6777267
28: -8.4994144, 12.0318422, -8.4942245, 12.0156479, -20.0013504, 20.0137177
29: -13.2167664, 4.4649034, -13.2149687, 4.4496884, -14.5935440, 14.5821571
30: -13.6836376, 9.6935501, -13.6516209, 9.6766357, -18.8238258, 18.8164673
31: -20.8621445, 4.4933167, -20.8503799, 4.4852552, -20.9338989, 20.9378891
32: -30.3132019, -4.0988045, -30.2967758, -4.1203256, -21.4203720, 21.4193611
33: -61.0217133, -25.3323669, -61.0054398, -25.3931961, -27.3171387, 27.3854370
34: -60.6672325, -34.0541458, -60.6664505, -34.0780373, -19.3187599, 19.3156853
35: -54.4624023, -24.2753334, -54.4618492, -24.3076706, -22.9836807, 23.0089760
36: -45.6553116, -15.1229162, -45.6545105, -15.1576576, -23.9559326, 23.9655838
37: -74.4705276, -40.8189545, -74.4493103, -40.8980598, -24.8649712, 24.9365501
38: -55.1515503, -23.3765259, -55.1503372, -23.4103413, -23.5427628, 23.5583115
39: -60.2915573, -24.9563026, -60.2690735, -25.0231972, -25.1590652, 25.2159843
40: -55.7294388, -33.7657089, -55.7127151, -33.8081665, -15.4642792, 15.4942741
41: -39.8153000, -8.9790936, -39.8000488, -9.0194788, -25.8305664, 25.8393326
42: -25.9413300, -7.4209027, -25.9332733, -7.4232240, -17.6692276, 17.6762581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1387

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9441625, upper bound: 5.9537557
time: 44.94 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9638443, upper bound: 5.9737075
time: 46.70 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -31.3391018, 0.3844194, -31.3604889, 0.3998446, -30.7009125, 30.7158737
1: -4.5796995, 14.6849537, -4.5977173, 14.6988859, -17.5897217, 17.5800438
2: 1.7566271, 19.6274796, 1.7386711, 19.6370125, -16.9819336, 16.9915161
3: -2.5776732, 16.3371811, -2.5948946, 16.3599663, -16.6284943, 16.6508713
4: -2.1470304, 20.0670948, -2.1741850, 20.0807400, -22.1470718, 22.1416550
5: -0.7873478, 16.9224911, -0.7914093, 16.9392204, -17.7265682, 17.7138996
6: -41.4449539, -13.2921085, -41.4513550, -13.2846317, -22.5506821, 22.5528946
7: 0.5307627, 20.1005955, 0.5216359, 20.1163616, -16.5511627, 16.5533295
8: -2.9032784, 26.5583076, -2.9297738, 26.5772552, -25.9069138, 25.9265060
9: -3.4113469, 17.6070156, -3.3983068, 17.6274033, -17.1044159, 17.0789795
10: -10.5691652, 17.0559196, -10.5203848, 17.0666466, -23.3651962, 23.2598343
11: -11.6069126, 6.6705751, -11.5967569, 6.6886234, -15.4874573, 15.4883461
12: -33.6686554, -10.0929737, -33.6823578, -10.0771732, -19.2325439, 19.2001724
13: -20.8371544, 11.2707586, -20.8551617, 11.2889042, -24.9741287, 24.9662704
14: -34.9574585, -1.6541357, -34.9601860, -1.6507912, -31.3140488, 31.3135071
15: -11.6785336, 9.1835127, -11.6961517, 9.1879797, -20.8665123, 20.8796654
16: -19.2670326, 0.5421357, -19.2455940, 0.5550451, -14.9442787, 14.8931236
17: -36.2353325, -10.7497587, -36.2506561, -10.7461510, -18.2282143, 18.2483292
18: -26.7610588, -0.5012629, -26.7819290, -0.4808307, -19.8459930, 19.8271942
19: -11.4850101, 5.7890511, -11.5013809, 5.8093967, -15.3126373, 15.3005981
20: -5.6932335, 13.3208656, -5.7034235, 13.3391142, -17.4398956, 17.4382095
21: -11.9512329, 9.2371397, -11.9577370, 9.2537003, -19.1852798, 19.2105484
22: -12.3520031, 6.8082047, -12.3685017, 6.8044090, -15.1404724, 15.1604385
23: -7.1100521, 11.0597458, -7.1277080, 11.0808325, -17.8382950, 17.8355141
24: -16.6159134, 5.3173451, -16.6412945, 5.3381615, -15.8009491, 15.8076134
25: -11.7036886, 7.8542795, -11.7259207, 7.8831530, -16.1637573, 16.1558685
26: -17.4407501, 11.9588308, -17.4609070, 11.9723368, -24.1325378, 24.1225662
27: -14.3844719, 9.8895550, -14.4038916, 9.8981581, -19.6740875, 19.6884232
28: -8.4669418, 12.0127401, -8.4839725, 12.0264530, -19.9776382, 19.9813843
29: -13.1921864, 4.4458795, -13.2056141, 4.4549241, -14.5769882, 14.5818176
30: -13.6521015, 9.6881123, -13.6511383, 9.7072239, -18.8015633, 18.8158417
31: -20.8255806, 4.4643946, -20.8469143, 4.4915886, -20.9179001, 20.9144287
32: -30.3162746, -4.1057391, -30.3231659, -4.1056428, -21.4303436, 21.4078560
33: -61.0211029, -25.3423977, -61.0261765, -25.3769646, -27.3308792, 27.3732452
34: -60.6431885, -34.0731392, -60.6590500, -34.0708199, -19.2943916, 19.3128357
35: -54.4453011, -24.2928085, -54.4564285, -24.3065605, -22.9698563, 23.0038910
36: -45.6346893, -15.1422052, -45.6476364, -15.1554089, -23.9410210, 23.9619827
37: -74.4665680, -40.8389435, -74.4799652, -40.8725586, -24.8870850, 24.9171600
38: -55.1176147, -23.4011288, -55.1409340, -23.4062901, -23.5138397, 23.5552330
39: -60.2981949, -24.9615593, -60.3041229, -25.0002060, -25.1840134, 25.2134552
40: -55.7434883, -33.7676582, -55.7442398, -33.7869797, -15.4942741, 15.4896469
41: -39.8203659, -8.9881306, -39.8243217, -9.0059595, -25.8437881, 25.8314667
42: -25.9356194, -7.4284945, -25.9399281, -7.4188576, -17.6615067, 17.6632462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1387

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9638223, upper bound: 5.9342213
time: 47.58 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9738265, upper bound: 5.9541743
time: 22.67 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -31.3641968, 0.4130592, -31.3621235, 0.4125566, -30.7415924, 30.7466049
1: -4.6009221, 14.7121887, -4.5980897, 14.7115364, -17.6237183, 17.6024055
2: 1.7384844, 19.6513138, 1.7383227, 19.6480637, -17.0114746, 17.0098877
3: -2.6039596, 16.3713875, -2.5953517, 16.3763161, -16.6711121, 16.6671524
4: -2.1732008, 20.0983429, -2.1749029, 20.0958614, -22.1885605, 22.1623688
5: -0.8091080, 16.9507942, -0.7919972, 16.9524155, -17.7615242, 17.7427921
6: -41.4570084, -13.2822104, -41.4566650, -13.2839489, -22.5593567, 22.5713730
7: 0.5082579, 20.1285725, 0.5211384, 20.1296997, -16.5871964, 16.5694885
8: -2.9341087, 26.5933132, -2.9307485, 26.5940514, -25.9546967, 25.9475555
9: -3.4454947, 17.6433849, -3.3990078, 17.6451588, -17.1561317, 17.0881577
10: -10.5874891, 17.0727730, -10.5217514, 17.0740337, -23.3914490, 23.2729797
11: -11.6305265, 6.6897197, -11.6076174, 6.6892071, -15.4997139, 15.5182304
12: -33.6925507, -10.0720539, -33.6927032, -10.0760508, -19.2515640, 19.2323151
13: -20.8568077, 11.3011875, -20.8556671, 11.3021994, -25.0070343, 24.9932022
14: -34.9735641, -1.6516037, -34.9651299, -1.6505547, -31.3315430, 31.3179398
15: -11.6980028, 9.2069979, -11.6972256, 9.1986227, -20.8966255, 20.9042244
16: -19.2887096, 0.5680876, -19.2459412, 0.5663185, -14.9781494, 14.9079781
17: -36.2657356, -10.7308178, -36.2629013, -10.7459984, -18.2372780, 18.2787971
18: -26.7917805, -0.4751570, -26.7962513, -0.4802840, -19.8643417, 19.8679886
19: -11.5135984, 5.8102584, -11.5150814, 5.8094864, -15.3305931, 15.3364716
20: -5.7204995, 13.3394089, -5.7147808, 13.3392687, -17.4661636, 17.4699211
21: -11.9793673, 9.2539091, -11.9701939, 9.2540064, -19.2111435, 19.2413101
22: -12.3818617, 6.8288703, -12.3820839, 6.8044691, -15.1569252, 15.1948395
23: -7.1421967, 11.0839520, -7.1428638, 11.0812531, -17.8619537, 17.8750877
24: -16.6564922, 5.3493462, -16.6604519, 5.3384504, -15.8117294, 15.8589211
25: -11.7466564, 7.8861551, -11.7458525, 7.8833737, -16.1828728, 16.2085953
26: -17.4772034, 11.9836168, -17.4766083, 11.9724989, -24.1679306, 24.1646957
27: -14.4106283, 9.9066029, -14.4158726, 9.8985806, -19.6845322, 19.7178497
28: -8.4990501, 12.0366831, -8.4985161, 12.0270042, -20.0062790, 20.0210953
29: -13.2164736, 4.4668698, -13.2172346, 4.4553051, -14.5804825, 14.6143723
30: -13.6828918, 9.7089624, -13.6649055, 9.7078991, -18.8176079, 18.8505173
31: -20.8685532, 4.4946485, -20.8673248, 4.4920177, -20.9416275, 20.9658661
32: -30.3261776, -4.0981317, -30.3266392, -4.1050448, -21.4434280, 21.4334183
33: -61.0326118, -25.3306160, -61.0301895, -25.3758411, -27.3455429, 27.3937378
34: -60.6671371, -34.0504265, -60.6699982, -34.0704727, -19.3003616, 19.3470688
35: -54.4627037, -24.2750664, -54.4645996, -24.3055916, -22.9709244, 23.0271072
36: -45.6552048, -15.1222744, -45.6574707, -15.1550131, -23.9414406, 23.9887199
37: -74.4898453, -40.8186493, -74.4909821, -40.8718262, -24.8913460, 24.9485931
38: -55.1528435, -23.3745708, -55.1568527, -23.4058132, -23.5218582, 23.5973396
39: -60.3088684, -24.9557343, -60.3081474, -24.9998093, -25.1883011, 25.2215805
40: -55.7448997, -33.7659569, -55.7446747, -33.7868500, -15.4938393, 15.4993935
41: -39.8277740, -8.9786301, -39.8274994, -9.0051346, -25.8501511, 25.8454056
42: -25.9458046, -7.4205070, -25.9440269, -7.4180727, -17.6707420, 17.6870995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1387

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9538797, upper bound: 5.9738051
time: 50.15 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9738263, upper bound: 5.9738264
time: 25.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 78.15 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9242202, upper bound: 5.9667443
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9441833, upper bound: 5.9667657
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9638222, upper bound: 5.9468101
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9441625, upper bound: 5.9667655
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9638223, upper bound: 5.9272767
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9638223, upper bound: 5.9272767
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9738051, upper bound: 5.9469383
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9738265, upper bound: 5.9668827
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9242202, upper bound: 5.9736861
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9242202, upper bound: 5.9737076
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9441625, upper bound: 5.9537557
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9638443, upper bound: 5.9737075
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9638223, upper bound: 5.9342213
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9738265, upper bound: 5.9541743
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9538797, upper bound: 5.9738051
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 78.15
Output dim: 5, lower bound: -5.9738263, upper bound: 5.9738264

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -31.3523445, 0.3578281, -31.2757893, 0.3028040, -30.6007996, 30.5522385
1: -4.5958920, 14.6700325, -4.5444727, 14.6297455, -17.5265503, 17.5112534
2: 1.7444952, 19.6154079, 1.7831299, 19.5800056, -16.9397964, 16.9349976
3: -2.5919135, 16.3281288, -2.5344431, 16.2897701, -16.5999374, 16.5788040
4: -2.1677060, 20.0643921, -2.1257329, 20.0307198, -22.1036758, 22.0771408
5: -0.7884932, 16.8965359, -0.7324855, 16.8597679, -17.6482620, 17.6290207
6: -41.4303780, -13.2877312, -41.4018784, -13.3209505, -22.4992294, 22.5060577
7: 0.5238883, 20.0819893, 0.5741160, 20.0476837, -16.4999008, 16.4801826
8: -2.9255986, 26.5397263, -2.8467493, 26.4816437, -25.8446808, 25.8220978
9: -3.3914695, 17.5746288, -3.3313932, 17.5602322, -17.0256348, 16.9695854
10: -10.5058203, 16.9896679, -10.4438543, 16.9764118, -23.2137756, 23.1611099
11: -11.5917397, 6.6670313, -11.5733232, 6.6497602, -15.4445457, 15.4639740
12: -33.6535110, -10.0817308, -33.6121750, -10.1278038, -19.1551514, 19.1645050
13: -20.8440914, 11.2854500, -20.8099861, 11.2689924, -24.9617386, 24.9387054
14: -34.9550095, -1.6932402, -34.9002495, -1.7262735, -31.2296295, 31.2028198
15: -11.6862144, 9.1809006, -11.6594296, 9.1566458, -20.8428612, 20.8403301
16: -19.2439575, 0.5057902, -19.1958656, 0.4890385, -14.8492737, 14.8085976
17: -36.2360535, -10.7662697, -36.1969452, -10.8078642, -18.1824684, 18.1793098
18: -26.7735424, -0.4866228, -26.7579803, -0.5199566, -19.7966690, 19.8240738
19: -11.4880333, 5.8081245, -11.4599047, 5.7743030, -15.2590485, 15.2725449
20: -5.6886978, 13.3314419, -5.6549554, 13.2936230, -17.3877487, 17.4042664
21: -11.9459400, 9.2440395, -11.9166021, 9.2203465, -19.1577682, 19.1634674
22: -12.3357048, 6.8027706, -12.3167553, 6.7667780, -15.0926437, 15.1116219
23: -7.1219735, 11.0783892, -7.1083946, 11.0492649, -17.7958908, 17.8254166
24: -16.6296158, 5.3336301, -16.6130848, 5.2934823, -15.7567825, 15.7900734
25: -11.7150106, 7.8790240, -11.6838198, 7.8328695, -16.1122971, 16.1383591
26: -17.4342575, 11.9661303, -17.3994675, 11.9221001, -24.0719604, 24.0897598
27: -14.3847151, 9.8820782, -14.3547039, 9.8399391, -19.6390991, 19.6484756
28: -8.4611216, 12.0203114, -8.4295235, 11.9724417, -19.9178314, 19.9377747
29: -13.1920433, 4.4523616, -13.1771107, 4.4231772, -14.5442810, 14.5574455
30: -13.6449022, 9.6772127, -13.6080570, 9.6470146, -18.7672729, 18.7715836
31: -20.8299389, 4.4885569, -20.7904072, 4.4464488, -20.8522034, 20.8692017
32: -30.2876568, -4.1084900, -30.2500687, -4.1509204, -21.3510666, 21.3623276
33: -60.9557571, -25.3805199, -60.9293823, -25.4397545, -27.2126312, 27.2502365
34: -60.6242027, -34.0735054, -60.5956268, -34.1269226, -19.2313919, 19.2531471
35: -54.4080391, -24.3082504, -54.3887634, -24.3564301, -22.8912277, 22.9205589
36: -45.5952148, -15.1568356, -45.5674438, -15.2140465, -23.8541412, 23.8804855
37: -74.4087372, -40.8750496, -74.3884583, -40.9319229, -24.7892265, 24.8293610
38: -55.0800705, -23.4076233, -55.0373764, -23.4712124, -23.4191513, 23.4415283
39: -60.2280960, -25.0016251, -60.1981506, -25.0582066, -25.0763626, 25.0994530
40: -55.6993408, -33.7894592, -55.6785622, -33.8263702, -15.4182014, 15.4354019
41: -39.7820396, -9.0086479, -39.7588844, -9.0475368, -25.7750854, 25.7900124
42: -25.9266529, -7.4230695, -25.9078312, -7.4413404, -17.6258011, 17.6421890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=64, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 567

## Relational analysis of IS_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9235878, upper bound: 5.9537870
time: 50.21 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9235878, upper bound: 5.9661105
time: 41.15 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -31.3555450, 0.3950500, -31.3260193, 0.3762169, -30.6515503, 30.6364594
1: -4.5965362, 14.6865101, -4.5639019, 14.6625204, -17.5578156, 17.5472679
2: 1.7439317, 19.6315823, 1.7629514, 19.6125031, -16.9639130, 16.9712906
3: -2.5927107, 16.3350544, -2.5444560, 16.3034515, -16.6144638, 16.5994377
4: -2.1690922, 20.0747108, -2.1411126, 20.0515118, -22.1252136, 22.1317368
5: -0.7896824, 16.9138374, -0.7555456, 16.8938332, -17.6835155, 17.6693840
6: -41.4449615, -13.2866020, -41.4313889, -13.3018017, -22.5340652, 22.5313873
7: 0.5228981, 20.0961971, 0.5544168, 20.0756283, -16.5236664, 16.5150223
8: -2.9276304, 26.5602226, -2.8765116, 26.5218964, -25.8852234, 25.8745346
9: -3.3926115, 17.5890350, -3.3507481, 17.5887947, -17.0500336, 17.0036469
10: -10.5073881, 17.0120277, -10.4745646, 17.0206852, -23.2372208, 23.2131042
11: -11.5954990, 6.6680846, -11.5821915, 6.6552372, -15.4781265, 15.4736595
12: -33.6634216, -10.0794506, -33.6313095, -10.1134167, -19.1836929, 19.1864929
13: -20.8469143, 11.2876511, -20.8162842, 11.2672281, -24.9570999, 24.9564285
14: -34.9572220, -1.6626482, -34.9411621, -1.6657495, -31.2779617, 31.2736664
15: -11.6884804, 9.1870365, -11.6722097, 9.1696510, -20.8581314, 20.8592453
16: -19.2441578, 0.5242171, -19.2172165, 0.5262966, -14.8625603, 14.8495522
17: -36.2379494, -10.7487946, -36.2223854, -10.7737341, -18.2005920, 18.2227898
18: -26.7758446, -0.4854350, -26.7608433, -0.5158608, -19.8069077, 19.8275604
19: -11.4948177, 5.8083391, -11.4739399, 5.7822127, -15.2741089, 15.2854767
20: -5.7015500, 13.3317060, -5.6814585, 13.3084030, -17.4151917, 17.4162560
21: -11.9550762, 9.2447357, -11.9366894, 9.2279472, -19.1752777, 19.1812439
22: -12.3470507, 6.8029652, -12.3397655, 6.7784758, -15.1167984, 15.1154785
23: -7.1237178, 11.0776854, -7.1074877, 11.0496416, -17.8263474, 17.8241997
24: -16.6307831, 5.3346758, -16.6152954, 5.2979574, -15.7647285, 15.7905846
25: -11.7216969, 7.8795662, -11.6982069, 7.8423462, -16.1316032, 16.1411057
26: -17.4470367, 11.9663935, -17.4267731, 11.9329720, -24.0969696, 24.1096954
27: -14.3947783, 9.8830547, -14.3761339, 9.8502026, -19.6599426, 19.6622238
28: -8.4750061, 12.0213776, -8.4579029, 11.9908695, -19.9504318, 19.9609528
29: -13.1965017, 4.4529710, -13.1867971, 4.4280686, -14.5554428, 14.5676537
30: -13.6494932, 9.6788979, -13.6193628, 9.6494236, -18.7819405, 18.7871399
31: -20.8377762, 4.4894667, -20.8067646, 4.4543085, -20.8701401, 20.8811569
32: -30.3041878, -4.1073241, -30.2835045, -4.1288896, -21.3899231, 21.3875694
33: -60.9800339, -25.3794861, -60.9772339, -25.4054451, -27.2712021, 27.2680435
34: -60.6445236, -34.0729523, -60.6360397, -34.1012726, -19.2781563, 19.2706604
35: -54.4307594, -24.3076305, -54.4328880, -24.3258438, -22.9448395, 22.9388733
36: -45.6232605, -15.1562376, -45.6229630, -15.1779156, -23.9194336, 23.9068375
37: -74.4178543, -40.8739586, -74.4065247, -40.9189911, -24.8156624, 24.8479538
38: -55.1128464, -23.4070816, -55.1024094, -23.4371758, -23.4876671, 23.4679260
39: -60.2501488, -25.0012283, -60.2408600, -25.0292606, -25.1285934, 25.1064262
40: -55.7112846, -33.7886620, -55.7025452, -33.8108749, -15.4440002, 15.4581070
41: -39.7944984, -9.0075817, -39.7842827, -9.0301304, -25.8103027, 25.8153305
42: -25.9333363, -7.4220285, -25.9217815, -7.4317341, -17.6562195, 17.6552925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=64, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 567

## Relational analysis of IS_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9312241, upper bound: 5.9661320
time: 35.71 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9435515, upper bound: 5.9661320
time: 30.97 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -31.3064537, 0.3329329, -31.3484116, 0.3690143, -30.6226349, 30.6017990
1: -4.5772038, 14.6657553, -4.5847497, 14.6739159, -17.5467072, 17.5474319
2: 1.7640150, 19.6095390, 1.7451316, 19.6207485, -16.9551392, 16.9675674
3: -2.5829856, 16.3373966, -2.5700898, 16.3310757, -16.6113968, 16.6262436
4: -2.1539643, 20.0686378, -2.1663156, 20.0727882, -22.1087341, 22.1343613
5: -0.7670472, 16.8923149, -0.7762892, 16.9055176, -17.6725655, 17.6686039
6: -41.4202423, -13.3056221, -41.4293671, -13.2924423, -22.5217133, 22.5106583
7: 0.5422990, 20.0810509, 0.5327212, 20.0899467, -16.5099487, 16.5223389
8: -2.8986588, 26.5359573, -2.9056273, 26.5371990, -25.8570709, 25.8785629
9: -3.3736567, 17.5777035, -3.3840885, 17.6113358, -17.0302773, 17.0258636
10: -10.4777832, 16.9743462, -10.4915094, 17.0159359, -23.2191315, 23.1952057
11: -11.5973434, 6.6627483, -11.6021652, 6.6737628, -15.4839325, 15.4667206
12: -33.6540833, -10.0929108, -33.6458511, -10.0946217, -19.1913223, 19.1795044
13: -20.8387451, 11.3025284, -20.8354225, 11.2956295, -24.9767151, 24.9835892
14: -34.9204102, -1.7240353, -34.9558868, -1.6927567, -31.2253723, 31.2289734
15: -11.6765184, 9.1843414, -11.6896858, 9.1873264, -20.8638458, 20.8740273
16: -19.2230797, 0.4975829, -19.2387505, 0.5345445, -14.8605156, 14.8461227
17: -36.2241096, -10.7833509, -36.2515144, -10.7716866, -18.2058029, 18.1954498
18: -26.7871037, -0.4899499, -26.7894402, -0.4900012, -19.8408890, 19.8390045
19: -11.4942741, 5.8003244, -11.4959497, 5.8033934, -15.2955475, 15.2898064
20: -5.6858654, 13.3167534, -5.6964293, 13.3269844, -17.4203873, 17.4296646
21: -11.9470806, 9.2372074, -11.9560547, 9.2442293, -19.1843567, 19.1934128
22: -12.3370972, 6.7912598, -12.3587675, 6.7989683, -15.1279831, 15.1271210
23: -7.1396637, 11.0753269, -7.1380091, 11.0769444, -17.8495331, 17.8350525
24: -16.6476173, 5.3302789, -16.6548100, 5.3291645, -15.8136139, 15.7953224
25: -11.7269554, 7.8702140, -11.7347908, 7.8737731, -16.1696663, 16.1528511
26: -17.4348717, 11.9554691, -17.4509945, 11.9577351, -24.1107712, 24.1284561
27: -14.3849783, 9.8729429, -14.3925724, 9.8664589, -19.6658249, 19.6615906
28: -8.4606647, 12.0031891, -8.4766350, 12.0140257, -19.9610748, 19.9628525
29: -13.1982584, 4.4481688, -13.2067699, 4.4487786, -14.5777206, 14.5600510
30: -13.6514416, 9.6769590, -13.6460896, 9.6686935, -18.7998810, 18.7897224
31: -20.8414726, 4.4817591, -20.8422260, 4.4839315, -20.9045334, 20.8920822
32: -30.2734451, -4.1290407, -30.2776413, -4.1220646, -21.3808670, 21.3711853
33: -60.9353333, -25.4127960, -60.9653931, -25.3945923, -27.2434845, 27.2545471
34: -60.6142387, -34.0984192, -60.6404877, -34.0789413, -19.2710800, 19.2536926
35: -54.3939590, -24.3373909, -54.4284210, -24.3085728, -22.9235229, 22.9125252
36: -45.5766373, -15.1920223, -45.6163635, -15.1585140, -23.8900375, 23.8717346
37: -74.4104919, -40.8863831, -74.4210205, -40.8995743, -24.8262787, 24.8280258
38: -55.0624352, -23.4407272, -55.1061249, -23.4110451, -23.4632721, 23.4475327
39: -60.2106514, -25.0297985, -60.2303123, -25.0238152, -25.0923538, 25.0958633
40: -55.6872787, -33.8043823, -55.6925049, -33.8096008, -15.4290543, 15.4338303
41: -39.7718315, -9.0246325, -39.7796898, -9.0212078, -25.7960739, 25.7891922
42: -25.9232616, -7.4315395, -25.9255238, -7.4241323, -17.6579514, 17.6431427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1387

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 567

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9508628, upper bound: 5.9461766
time: 46.37 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9631911, upper bound: 5.9461766
time: 46.23 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -31.3567162, 0.4063253, -31.3516312, 0.4062352, -30.7068939, 30.6525955
1: -4.5966187, 14.6985617, -4.5853643, 14.6903458, -17.5827408, 17.5787354
2: 1.7438213, 19.6420059, 1.7445687, 19.6369324, -16.9914093, 16.9916534
3: -2.5930061, 16.3510704, -2.5708976, 16.3379555, -16.6320190, 16.6407623
4: -2.1693702, 20.0894279, -2.1676924, 20.0830841, -22.1632690, 22.1559067
5: -0.7900846, 16.9263325, -0.7774947, 16.9228039, -17.7128887, 17.7038269
6: -41.4497528, -13.2864866, -41.4439545, -13.2913513, -22.5470619, 22.5455017
7: 0.5226252, 20.1089554, 0.5317169, 20.1041698, -16.5447540, 16.5461311
8: -2.9283295, 26.5762138, -2.9076056, 26.5576630, -25.9095230, 25.9190903
9: -3.3930378, 17.6062832, -3.3851953, 17.6257095, -17.0643463, 17.0502510
10: -10.5085001, 17.0186501, -10.4931269, 17.0383186, -23.2711334, 23.2186050
11: -11.6062326, 6.6682158, -11.6059313, 6.6748171, -15.4935989, 15.5003014
12: -33.6732674, -10.0785265, -33.6557770, -10.0923357, -19.2133255, 19.2080688
13: -20.8450966, 11.3007412, -20.8382206, 11.2978325, -24.9944229, 24.9789963
14: -34.9613266, -1.6634760, -34.9580536, -1.6621075, -31.2962112, 31.2772293
15: -11.6892662, 9.1973934, -11.6919661, 9.1934566, -20.8827229, 20.8893585
16: -19.2444363, 0.5347948, -19.2389755, 0.5530062, -14.9014282, 14.8594170
17: -36.2495575, -10.7492189, -36.2534523, -10.7541990, -18.2493057, 18.2135925
18: -26.7899704, -0.4858656, -26.7917137, -0.4887862, -19.8443680, 19.8492508
19: -11.5083065, 5.8082333, -11.5027714, 5.8036141, -15.3085327, 15.3048401
20: -5.7123661, 13.3315315, -5.7092695, 13.3272390, -17.4323425, 17.4571152
21: -11.9671173, 9.2447586, -11.9651976, 9.2448997, -19.2021561, 19.2109375
22: -12.3601246, 6.8029299, -12.3701029, 6.7991614, -15.1318588, 15.1512566
23: -7.1387935, 11.0757027, -7.1397657, 11.0762205, -17.8482971, 17.8654785
24: -16.6498051, 5.3346953, -16.6560040, 5.3301806, -15.8141556, 15.8032799
25: -11.7413044, 7.8797145, -11.7414742, 7.8743429, -16.1724091, 16.1721268
26: -17.4621658, 11.9663382, -17.4638042, 11.9580412, -24.1306686, 24.1535034
27: -14.4064198, 9.8832664, -14.4026594, 9.8674412, -19.6796188, 19.6824646
28: -8.4890709, 12.0215855, -8.4905367, 12.0151043, -19.9842224, 19.9954529
29: -13.2079420, 4.4530621, -13.2112417, 4.4493780, -14.5879517, 14.5711632
30: -13.6627388, 9.6794310, -13.6506929, 9.6704483, -18.8154526, 18.8043518
31: -20.8578148, 4.4896283, -20.8500862, 4.4848323, -20.9164963, 20.9099808
32: -30.3069305, -4.1070457, -30.2941628, -4.1208987, -21.4061050, 21.4100151
33: -60.9831619, -25.3785343, -60.9896965, -25.3935432, -27.2613068, 27.3131180
34: -60.6547012, -34.0727921, -60.6608047, -34.0783768, -19.2885818, 19.3004303
35: -54.4381218, -24.3068180, -54.4511185, -24.3079338, -22.9418335, 22.9661560
36: -45.6321793, -15.1558599, -45.6444397, -15.1579075, -23.9163780, 23.9370575
37: -74.4285736, -40.8734589, -74.4301300, -40.8984909, -24.8448792, 24.8544197
38: -55.1274719, -23.4066448, -55.1389427, -23.4105644, -23.4896584, 23.5160675
39: -60.2533531, -25.0008621, -60.2523346, -25.0233765, -25.0993271, 25.1480713
40: -55.7112770, -33.7888985, -55.7044601, -33.8087769, -15.4517632, 15.4596443
41: -39.7971992, -9.0072393, -39.7921295, -9.0201569, -25.8214111, 25.8244514
42: -25.9371643, -7.4219279, -25.9322243, -7.4231014, -17.6710739, 17.6735458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1387

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 567

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9508833, upper bound: 5.9661319
time: 60.35 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9632129, upper bound: 5.9661319
time: 13.56 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -31.2845554, 0.3055201, -31.3562279, 0.3619728, -30.5890656, 30.6295395
1: -4.5566168, 14.6467743, -4.5967593, 14.6810322, -17.5338364, 17.5405731
2: 1.7812754, 19.5895195, 1.7406900, 19.6201210, -16.9414215, 16.9526367
3: -2.5573816, 16.3194981, -2.5935280, 16.3516159, -16.5968933, 16.6323090
4: -2.1280987, 20.0415230, -2.1723130, 20.0694904, -22.0934677, 22.1200333
5: -0.7454560, 16.8739147, -0.7896161, 16.9153633, -17.6608200, 17.6635303
6: -41.4123383, -13.3145504, -41.4356651, -13.2864132, -22.5169067, 22.5065002
7: 0.5645236, 20.0620060, 0.5230280, 20.0971203, -16.4953003, 16.5178108
8: -2.8681898, 26.5165749, -2.9271979, 26.5561752, -25.8444443, 25.8805542
9: -3.3451467, 17.5400066, -3.3967094, 17.5958843, -17.0004959, 17.0251312
10: -10.4726982, 16.9576416, -10.5182562, 17.0199776, -23.2159805, 23.1851120
11: -11.5731802, 6.6456146, -11.5924263, 6.6789923, -15.4709244, 15.4486313
12: -33.6472244, -10.1128788, -33.6715393, -10.0800381, -19.2071381, 19.1692810
13: -20.8259144, 11.2719126, -20.8497467, 11.2876730, -24.9576263, 24.9620361
14: -34.9044724, -1.7218542, -34.9569283, -1.6834679, -31.2156754, 31.2373123
15: -11.6581717, 9.1620932, -11.6911163, 9.1812544, -20.8394260, 20.8532104
16: -19.2049103, 0.4708381, -19.2452583, 0.5240564, -14.8352547, 14.8343620
17: -36.1936493, -10.7985888, -36.2434807, -10.7641411, -18.1742859, 18.2033463
18: -26.7567749, -0.5136204, -26.7793694, -0.4834123, -19.8395233, 19.8157349
19: -11.4709663, 5.7796459, -11.4952869, 5.8089151, -15.2864418, 15.2645226
20: -5.6581001, 13.3031549, -5.6896796, 13.3376102, -17.4133606, 17.3971786
21: -11.9192934, 9.2227831, -11.9476213, 9.2501802, -19.1680527, 19.1718292
22: -12.3068638, 6.7728071, -12.3478270, 6.8041315, -15.1041794, 15.1111374
23: -7.1068282, 11.0516052, -7.1248856, 11.0792780, -17.8287048, 17.8055229
24: -16.6058502, 5.3015337, -16.6374416, 5.3366756, -15.7902985, 15.7737656
25: -11.6828423, 7.8421803, -11.7173977, 7.8824205, -16.1452293, 16.1244850
26: -17.3979607, 11.9372034, -17.4423828, 11.9717550, -24.0949631, 24.0877151
27: -14.3589306, 9.8706074, -14.3923540, 9.8966961, -19.6472626, 19.6722946
28: -8.4281940, 11.9840965, -8.4663982, 12.0248880, -19.9373398, 19.9305267
29: -13.1736889, 4.4291396, -13.1974163, 4.4540081, -14.5611267, 14.5597038
30: -13.6198864, 9.6715298, -13.6455784, 9.6992731, -18.7776260, 18.7890816
31: -20.8049202, 4.4527783, -20.8387756, 4.4902992, -20.8885193, 20.8685989
32: -30.2765274, -4.1359215, -30.3040390, -4.1073503, -21.3908691, 21.3596954
33: -60.9347420, -25.4228325, -60.9861221, -25.3783913, -27.2571945, 27.2423477
34: -60.5902176, -34.1173859, -60.6330795, -34.0717125, -19.2467232, 19.2508278
35: -54.3768959, -24.3548279, -54.4229965, -24.3074703, -22.9097023, 22.9074593
36: -45.5559845, -15.2112713, -45.6095352, -15.1562481, -23.8751373, 23.8681030
37: -74.4065628, -40.9063644, -74.4516907, -40.8740463, -24.8483963, 24.8086319
38: -55.0285034, -23.4652596, -55.0967369, -23.4070435, -23.4343643, 23.4444542
39: -60.2172852, -25.0350552, -60.2653275, -25.0008698, -25.1173477, 25.0933189
40: -55.7013016, -33.8063202, -55.7240715, -33.7884636, -15.4590530, 15.4292145
41: -39.7768707, -9.0336933, -39.8039742, -9.0077000, -25.8093262, 25.7813492
42: -25.9175358, -7.4391303, -25.9321594, -7.4197645, -17.6502228, 17.6301155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=64, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1387

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 567

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9508629, upper bound: 5.9266432
time: 29.17 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9731743, upper bound: 5.9266432
time: 44.33 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -31.3348045, 0.3789330, -31.3594589, 0.3991785, -30.6734009, 30.6803131
1: -4.5760241, 14.6795645, -4.5973759, 14.6974945, -17.5698471, 17.5718384
2: 1.7610941, 19.6220474, 1.7401202, 19.6363354, -16.9776764, 16.9767609
3: -2.5674481, 16.3331623, -2.5943542, 16.3585453, -16.6174774, 16.6468124
4: -2.1435227, 20.0623016, -2.1736619, 20.0797691, -22.1480408, 22.1415787
5: -0.7684917, 16.9079399, -0.7908249, 16.9326897, -17.7011814, 17.6987648
6: -41.4419403, -13.2954454, -41.4502945, -13.2853222, -22.5422249, 22.5413513
7: 0.5448041, 20.0899506, 0.5220411, 20.1113358, -16.5301132, 16.5415840
8: -2.8979120, 26.5568428, -2.9291987, 26.5766678, -25.8968887, 25.9211197
9: -3.3644934, 17.5685997, -3.3978348, 17.6102905, -17.0345650, 17.0495338
10: -10.5034389, 17.0019016, -10.5198498, 17.0423737, -23.2680511, 23.2085419
11: -11.5820589, 6.6511388, -11.5962324, 6.6800623, -15.4805756, 15.4822235
12: -33.6664238, -10.0984850, -33.6814346, -10.0777464, -19.2291260, 19.1978149
13: -20.8322411, 11.2701912, -20.8525734, 11.2898769, -24.9753342, 24.9574127
14: -34.9453430, -1.6612864, -34.9591522, -1.6528606, -31.2864990, 31.2855988
15: -11.6709108, 9.1751328, -11.6933899, 9.1873760, -20.8582878, 20.8685226
16: -19.2262859, 0.5080891, -19.2454453, 0.5425382, -14.8761711, 14.8476639
17: -36.2190819, -10.7644577, -36.2453766, -10.7466688, -18.2177887, 18.2214737
18: -26.7596397, -0.5095689, -26.7816734, -0.4822514, -19.8429794, 19.8260269
19: -11.4849977, 5.7875218, -11.5021114, 5.8091526, -15.2993431, 15.2796021
20: -5.6845894, 13.3179388, -5.7025175, 13.3378611, -17.4253387, 17.4246368
21: -11.9393826, 9.2303724, -11.9567766, 9.2508421, -19.1858444, 19.1893539
22: -12.3298607, 6.7844915, -12.3591614, 6.8043051, -15.1080742, 15.1352654
23: -7.1059427, 11.0519962, -7.1266232, 11.0785732, -17.8274918, 17.8359528
24: -16.6080456, 5.3060055, -16.6385937, 5.3377285, -15.7908211, 15.7817383
25: -11.6971970, 7.8516688, -11.7241011, 7.8829732, -16.1479492, 16.1437759
26: -17.4252338, 11.9480829, -17.4551678, 11.9720364, -24.1148911, 24.1127090
27: -14.3803949, 9.8808928, -14.4024220, 9.8976889, -19.6610641, 19.6931763
28: -8.4565601, 12.0024681, -8.4802914, 12.0259457, -19.9605179, 19.9630890
29: -13.1833525, 4.4340477, -13.2018986, 4.4545975, -14.5714035, 14.5708656
30: -13.6312199, 9.6739788, -13.6501789, 9.7010059, -18.7932129, 18.8037338
31: -20.8212967, 4.4606509, -20.8466377, 4.4911680, -20.9004745, 20.8865204
32: -30.3100281, -4.1139336, -30.3205414, -4.1062031, -21.4160843, 21.3985596
33: -60.9826431, -25.3884964, -61.0103645, -25.3773136, -27.2750092, 27.3009338
34: -60.6306534, -34.0917320, -60.6534119, -34.0711441, -19.2642288, 19.2975883
35: -54.4210281, -24.3242416, -54.4456711, -24.3068542, -22.9279938, 22.9611092
36: -45.6114845, -15.1751575, -45.6376038, -15.1556759, -23.9014778, 23.9334412
37: -74.4246063, -40.8934479, -74.4607697, -40.8729744, -24.8669930, 24.8350449
38: -55.0935440, -23.4312382, -55.1295395, -23.4064960, -23.4607506, 23.5129700
39: -60.2600021, -25.0061646, -60.2873955, -25.0004578, -25.1243248, 25.1455307
40: -55.7253113, -33.7908096, -55.7359543, -33.7875900, -15.4817734, 15.4550056
41: -39.8023071, -9.0163021, -39.8164406, -9.0066395, -25.8346481, 25.8166008
42: -25.9314346, -7.4294949, -25.9388657, -7.4187088, -17.6633453, 17.6605301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=64, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1387

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 567

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9608661, upper bound: 5.9466007
time: 35.70 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9731955, upper bound: 5.9466007
time: 54.34 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -31.3096867, 0.3341227, -31.3579330, 0.3746500, -30.6297607, 30.6602783
1: -4.5778279, 14.6740170, -4.5971432, 14.6936340, -17.5678711, 17.5629272
2: 1.7631663, 19.6133728, 1.7403485, 19.6311665, -16.9709320, 16.9709625
3: -2.5836911, 16.3537178, -2.5939524, 16.3679638, -16.6394501, 16.6486053
4: -2.1542749, 20.0727043, -2.1730111, 20.0845585, -22.1349869, 22.1407547
5: -0.7672102, 16.9022064, -0.7901986, 16.9285793, -17.6957893, 17.6924057
6: -41.4244118, -13.3046856, -41.4410095, -13.2857237, -22.5255585, 22.5249672
7: 0.5419872, 20.0899429, 0.5225487, 20.1104507, -16.5313568, 16.5339508
8: -2.8990464, 26.5516243, -2.9281740, 26.5729256, -25.8922577, 25.9016266
9: -3.3793035, 17.5764217, -3.3974304, 17.6136589, -17.0522003, 17.0343323
10: -10.4910240, 16.9744835, -10.5196133, 17.0273762, -23.2422791, 23.1982651
11: -11.5968113, 6.6647778, -11.6033478, 6.6795464, -15.4831619, 15.4785271
12: -33.6710968, -10.0919056, -33.6818810, -10.0789108, -19.2261734, 19.2014084
13: -20.8455296, 11.3024035, -20.8502502, 11.3008890, -24.9905319, 24.9889603
14: -34.9205322, -1.7192965, -34.9619255, -1.6832447, -31.2332077, 31.2416916
15: -11.6776133, 9.1855536, -11.6921835, 9.1918888, -20.8695030, 20.8777370
16: -19.2265892, 0.4968286, -19.2455921, 0.5353713, -14.8691177, 14.8492279
17: -36.2241135, -10.7796297, -36.2557983, -10.7639866, -18.1833496, 18.2337952
18: -26.7874985, -0.4875669, -26.7936707, -0.4828997, -19.8579025, 19.8565598
19: -11.4995365, 5.8008595, -11.5090237, 5.8090448, -15.3043365, 15.3003883
20: -5.6853533, 13.3216724, -5.7010336, 13.3377647, -17.4396362, 17.4289093
21: -11.9474344, 9.2395401, -11.9600754, 9.2504635, -19.1938629, 19.2025909
22: -12.3367443, 6.7934561, -12.3613853, 6.8041387, -15.1206665, 15.1455383
23: -7.1389990, 11.0757904, -7.1400542, 11.0796871, -17.8523636, 17.8451004
24: -16.6464386, 5.3335381, -16.6565800, 5.3369656, -15.8010826, 15.8250618
25: -11.7258186, 7.8740892, -11.7373371, 7.8826494, -16.1643524, 16.1772041
26: -17.4344635, 11.9619703, -17.4580193, 11.9719343, -24.1303558, 24.1297989
27: -14.3850307, 9.8876438, -14.4043770, 9.8971281, -19.6577148, 19.7017288
28: -8.4603062, 12.0080414, -8.4809341, 12.0254059, -19.9659805, 19.9702301
29: -13.1979494, 4.4501557, -13.2090292, 4.4543729, -14.5646515, 14.5922394
30: -13.6506701, 9.6924009, -13.6593418, 9.6999254, -18.7936745, 18.8237724
31: -20.8478737, 4.4830580, -20.8591881, 4.4907165, -20.9122162, 20.9200287
32: -30.2864170, -4.1283278, -30.3074875, -4.1067615, -21.4039459, 21.3852844
33: -60.9462776, -25.4110928, -60.9901848, -25.3772621, -27.2718735, 27.2628403
34: -60.6141853, -34.0946579, -60.6439819, -34.0713577, -19.2526627, 19.2850761
35: -54.3942947, -24.3370819, -54.4311600, -24.3064919, -22.9107475, 22.9306793
36: -45.5765114, -15.1913710, -45.6193466, -15.1558590, -23.8755341, 23.8948288
37: -74.4298096, -40.8860703, -74.4626770, -40.8733444, -24.8526154, 24.8400612
38: -55.0637207, -23.4387169, -55.1126251, -23.4065323, -23.4423676, 23.4865761
39: -60.2278938, -25.0292339, -60.2693710, -25.0004406, -25.1216049, 25.1014404
40: -55.7027473, -33.8046265, -55.7244606, -33.7882767, -15.4585991, 15.4389610
41: -39.7843208, -9.0241947, -39.8071480, -9.0068531, -25.8156509, 25.7952499
42: -25.9277096, -7.4311485, -25.9362507, -7.4190016, -17.6594505, 17.6539764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=64, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1387

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 567

## Relational analysis of IS_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9731745, upper bound: 5.9339753
time: 24.85 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9731745, upper bound: 5.9463042
time: 43.44 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -31.3599052, 0.4075351, -31.3611107, 0.4118795, -30.7140427, 30.7111053
1: -4.5972261, 14.7068129, -4.5977335, 14.7101402, -17.6038666, 17.5941887
2: 1.7429640, 19.6458912, 1.7398188, 19.6473484, -17.0072174, 16.9950485
3: -2.5936990, 16.3673935, -2.5947902, 16.3748760, -16.6600876, 16.6631393
4: -2.1696727, 20.0934753, -2.1744239, 20.0948658, -22.1895218, 22.1623001
5: -0.7902834, 16.9362545, -0.7913978, 16.9458580, -17.7361412, 17.7276516
6: -41.4539795, -13.2855349, -41.4556198, -13.2846241, -22.5508728, 22.5598030
7: 0.5222853, 20.1178989, 0.5215433, 20.1246643, -16.5661697, 16.5577545
8: -2.9288182, 26.5918846, -2.9301834, 26.5934200, -25.9447021, 25.9421768
9: -3.3986859, 17.6050224, -3.3985519, 17.6280327, -17.0862656, 17.0587120
10: -10.5217781, 17.0187340, -10.5212021, 17.0497379, -23.2943039, 23.2216873
11: -11.6056824, 6.6702814, -11.6071043, 6.6806226, -15.4928284, 15.5120850
12: -33.6902618, -10.0775604, -33.6917496, -10.0766239, -19.2481461, 19.2299576
13: -20.8519230, 11.3006153, -20.8531399, 11.3031206, -25.0082550, 24.9843750
14: -34.9614334, -1.6587524, -34.9640808, -1.6526117, -31.3040009, 31.2899933
15: -11.6903925, 9.1985731, -11.6944304, 9.1980209, -20.8884125, 20.8930035
16: -19.2479610, 0.5340757, -19.2458267, 0.5538177, -14.9100494, 14.8625069
17: -36.2495117, -10.7454987, -36.2576752, -10.7465172, -18.2268486, 18.2519073
18: -26.7903595, -0.4834394, -26.7959690, -0.4816723, -19.8613586, 19.8667908
19: -11.5135632, 5.8087463, -11.5157967, 5.8092489, -15.3172684, 15.3154564
20: -5.7118678, 13.3364582, -5.7138600, 13.3380079, -17.4516144, 17.4563446
21: -11.9675121, 9.2471037, -11.9691725, 9.2511082, -19.2116547, 19.2201233
22: -12.3597355, 6.8051405, -12.3727083, 6.8043423, -15.1245346, 15.1696815
23: -7.1381083, 11.0761948, -7.1418018, 11.0789642, -17.8511505, 17.8755417
24: -16.6486397, 5.3379774, -16.6577301, 5.3379951, -15.8016129, 15.8330193
25: -11.7401886, 7.8835630, -11.7440224, 7.8832178, -16.1670876, 16.1964645
26: -17.4617577, 11.9728518, -17.4708290, 11.9722767, -24.1502380, 24.1548615
27: -14.4064999, 9.8978853, -14.4144440, 9.8981333, -19.6715088, 19.7226257
28: -8.4886751, 12.0264263, -8.4948215, 12.0264597, -19.9891739, 20.0028229
29: -13.2076578, 4.4550509, -13.2135258, 4.4549856, -14.5748825, 14.6033974
30: -13.6619511, 9.6948252, -13.6639566, 9.7017002, -18.8092270, 18.8384094
31: -20.8642368, 4.4909506, -20.8670559, 4.4916067, -20.9242096, 20.9379501
32: -30.3199005, -4.1063075, -30.3239956, -4.1056204, -21.4291763, 21.4240875
33: -60.9941711, -25.3768158, -61.0143890, -25.3761864, -27.2896652, 27.3214035
34: -60.6546135, -34.0690117, -60.6643219, -34.0708008, -19.2702065, 19.3318481
35: -54.4384422, -24.3065147, -54.4538727, -24.3058815, -22.9290848, 22.9843025
36: -45.6319923, -15.1552353, -45.6474457, -15.1552334, -23.9018707, 23.9601631
37: -74.4478607, -40.8731842, -74.4717712, -40.8722610, -24.8712234, 24.8664436
38: -55.1287727, -23.4046936, -55.1454201, -23.4060059, -23.4687424, 23.5550652
39: -60.2706528, -25.0003567, -60.2914085, -25.0000153, -25.1285782, 25.1536674
40: -55.7267380, -33.7891312, -55.7364311, -33.7874451, -15.4813194, 15.4647636
41: -39.8096924, -9.0068302, -39.8195648, -9.0057774, -25.8409729, 25.8304901
42: -25.9415989, -7.4215431, -25.9429531, -7.4179626, -17.6725616, 17.6843872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=64, inp2_unstable=66, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1387

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 567

## Relational analysis of IS_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9731958, upper bound: 5.9539276
time: 45.38 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9631914, upper bound: 5.9463042
time: 13.09 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -31.3556614, 0.3607168, -31.2763786, 0.3022943, -30.6263275, 30.5611725
1: -4.5990610, 14.6742887, -4.5445714, 14.6306200, -17.5453568, 17.5158653
2: 1.7404298, 19.6197529, 1.7818341, 19.5801678, -16.9431305, 16.9397278
3: -2.6018713, 16.3315620, -2.5348446, 16.2909164, -16.6109543, 16.5816269
4: -2.1704199, 20.0685692, -2.1258302, 20.0314140, -22.1149521, 22.0721741
5: -0.8069839, 16.9098434, -0.7329154, 16.8657475, -17.6727314, 17.6427593
6: -41.4324417, -13.2854328, -41.4025040, -13.3207521, -22.5006332, 22.5160675
7: 0.5102159, 20.0916214, 0.5738937, 20.0522442, -16.5199814, 16.4860687
8: -2.9304352, 26.5398216, -2.8471060, 26.4816399, -25.8532104, 25.8227692
9: -3.4377623, 17.6121273, -3.3316073, 17.5769062, -17.0945129, 16.9929199
10: -10.5710907, 17.0421715, -10.4441185, 17.0000401, -23.3095322, 23.1902618
11: -11.6162853, 6.6856823, -11.5737076, 6.6580005, -15.4472656, 15.4803391
12: -33.6547203, -10.0766153, -33.6125832, -10.1273594, -19.1567535, 19.1675644
13: -20.8443604, 11.2856827, -20.8102760, 11.2678928, -24.9668884, 24.9435883
14: -34.9656067, -1.6880865, -34.9005890, -1.7251072, -31.2547607, 31.2144928
15: -11.6932936, 9.1888180, -11.6619930, 9.1569920, -20.8502846, 20.8508110
16: -19.2845669, 0.5384536, -19.1958942, 0.5009732, -14.9165611, 14.8292542
17: -36.2511024, -10.7526312, -36.2016220, -10.8078279, -18.1912117, 18.1862373
18: -26.7745609, -0.4801474, -26.7580662, -0.5194042, -19.7984238, 19.8273544
19: -11.4876289, 5.8092637, -11.4589767, 5.7743464, -15.2702408, 15.2929115
20: -5.6963587, 13.3337936, -5.6554222, 13.2946053, -17.3867035, 17.4168091
21: -11.9570923, 9.2504244, -11.9172812, 9.2230387, -19.1526260, 19.1838684
22: -12.3569355, 6.8263879, -12.3257170, 6.7668667, -15.1050262, 15.1360703
23: -7.1258645, 11.0817165, -7.1093435, 11.0494843, -17.8021774, 17.8380203
24: -16.6372375, 5.3445530, -16.6156998, 5.2937102, -15.7633286, 15.8142662
25: -11.7209215, 7.8814383, -11.6854219, 7.8329606, -16.1154480, 16.1497345
26: -17.4487114, 11.9764385, -17.4047642, 11.9221907, -24.0802536, 24.0986023
27: -14.3880978, 9.8903751, -14.3557682, 9.8402367, -19.6415710, 19.6429901
28: -8.4705172, 12.0299978, -8.4327545, 11.9727173, -19.9280167, 19.9550018
29: -13.2005320, 4.4636955, -13.1806965, 4.4232950, -14.5491333, 14.5676651
30: -13.6646862, 9.6909504, -13.6084242, 9.6530657, -18.7737885, 18.7829590
31: -20.8336048, 4.4918156, -20.7904015, 4.4466472, -20.8637772, 20.8963242
32: -30.2926140, -4.1009116, -30.2520638, -4.1506209, -21.3546906, 21.3703651
33: -60.9927216, -25.3346806, -60.9444122, -25.4395561, -27.2368164, 27.3212814
34: -60.6352501, -34.0552063, -60.6006126, -34.1267471, -19.2367363, 19.2673798
35: -54.4308662, -24.2771378, -54.3988113, -24.3563232, -22.9057770, 22.9622536
36: -45.6165771, -15.1240225, -45.5766830, -15.2139082, -23.8627472, 23.9078827
37: -74.4501801, -40.8209229, -74.4074097, -40.9316788, -24.8077774, 24.9121437
38: -55.1017227, -23.3776703, -55.0476913, -23.4711056, -23.4309120, 23.4825554
39: -60.2647247, -24.9571342, -60.2142029, -25.0580635, -25.0978279, 25.1665154
40: -55.7167320, -33.7670898, -55.6864853, -33.8261414, -15.4274902, 15.4688339
41: -39.7993050, -8.9813948, -39.7664108, -9.0472889, -25.7819366, 25.8054047
42: -25.9302864, -7.4233012, -25.9086323, -7.4420261, -17.6206322, 17.6505394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=66, inp2_unstable=64, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=254, inp2_unstable=254, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 567

## Relational analysis of IS_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9112562, upper bound: 5.9730555
time: 58.80 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -5.9235875, upper bound: 5.9730555
time: 57.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 118.63 seconds
IS_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9235878, upper bound: 5.9537870
IS_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9235878, upper bound: 5.9661105
IS_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9312241, upper bound: 5.9661320
IS_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9435515, upper bound: 5.9661320
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9508628, upper bound: 5.9461766
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9631911, upper bound: 5.9461766
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9508833, upper bound: 5.9661319
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9632129, upper bound: 5.9661319
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9508629, upper bound: 5.9266432
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9731743, upper bound: 5.9266432
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9608661, upper bound: 5.9466007
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9731955, upper bound: 5.9466007
IS_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9731745, upper bound: 5.9339753
IS_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9731745, upper bound: 5.9463042
IS_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9731958, upper bound: 5.9539276
IS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9631914, upper bound: 5.9463042
IS_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9112562, upper bound: 5.9730555
IS_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 118.63
Output dim: 5, lower bound: -5.9235875, upper bound: 5.9730555
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 118.63
Output dim: 5, lower bound: -5.9242202, upper bound: 5.9737076
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 118.63
Output dim: 5, lower bound: -5.9441625, upper bound: 5.9537557
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 118.63
Output dim: 5, lower bound: -5.9638443, upper bound: 5.9737075
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 118.63
Output dim: 5, lower bound: -5.9638223, upper bound: 5.9342213
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 118.63
Output dim: 5, lower bound: -5.9738265, upper bound: 5.9541743
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 118.63
Output dim: 5, lower bound: -5.9538797, upper bound: 5.9738051
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 118.63
Output dim: 5, lower bound: -5.9738263, upper bound: 5.9738264

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 60.38 + 1808.68 = 1869.06 seconds
