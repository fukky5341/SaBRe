## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.3532293525
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.8783393, 3.8783391)
1: (-12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.5584154, 3.5584154)
2: (-13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.2301512, 3.2301512)
3: (-9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981)
4: (-4.5608406, -2.3997998, -4.5608406, -2.3997998, -2.1610408, 2.1610408)
5: (-11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.7072897, 3.7072897)
6: (-17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.9770737, 3.9770737)
7: (-6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.8377733, 2.8377733)
8: (-2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.2236829, 2.2236829)
9: (2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.7430749, 2.7430749)

## BASE Result
execution time: IAR + LP analysis = 14.52 + 37.04 = 51.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -2.1375822, upper bound: 2.1375790


# Binary Search by BASE starts (time budget: 3548.44 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.467238664627075
rel_dist={9: [-1.6640502761084588, 1.6640497405138106]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.2999587059020996
rel_dist={9: [-1.360028225390102, 1.3600276268876046]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.188438653945923
rel_dist={9: [-1.0985895441039362, 1.0985873760165363]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.244198799133301
rel_dist={9: [-1.2428127078072388, 1.2428113778761434]}

## Binary Search Result
Binary search time: 224.90 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3323.54 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7532678, upper bound: 1.7318902
time: 4.23 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7532678, upper bound: 1.7532676
time: 9.14 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.60 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.60
Output dim: 9, lower bound: -1.7532678, upper bound: 1.7318902
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.60
Output dim: 9, lower bound: -1.7532678, upper bound: 1.7532676

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -14.2953281, -10.2914829, -14.2989922, -10.2875051, -3.1263227, 3.1261048
1: -12.4783459, -8.9430885, -12.4903154, -8.9369202, -3.0240822, 3.0299563
2: -13.4055109, -10.1823139, -13.4090919, -10.1799526, -3.0180569, 3.0204811
3: -9.8860092, -6.9041572, -9.8893652, -6.9027524, -2.9832568, 2.9852080
4: -4.5463209, -2.4078240, -4.5570164, -2.4006119, -1.8632779, 1.8671007
5: -11.0696669, -7.3708925, -11.0728312, -7.3670454, -3.0798531, 3.0806441
6: -17.5701237, -13.6092243, -17.5777149, -13.6039028, -3.3656297, 3.3685260
7: -6.4289093, -3.6039596, -6.4327278, -3.5974965, -2.5065417, 2.5050695
8: -2.0346689, 0.1787167, -2.0392828, 0.1824327, -2.0329952, 2.0338736
9: 2.4283915, 5.1467175, 2.4185758, 5.1565900, -2.5081263, 2.5080056

Time for backsubstitution: 12.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7532654, upper bound: 1.7294444
time: 4.82 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7532654, upper bound: 1.7318883
time: 4.14 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -14.3000641, -10.2871685, -14.3000660, -10.2871666, -3.1313267, 3.1309881
1: -12.4945726, -8.9361610, -12.4945774, -8.9361620, -3.0360279, 3.0408382
2: -13.4097595, -10.1796112, -13.4097614, -10.1796103, -3.0254469, 3.0240293
3: -9.8902349, -6.9025426, -9.8902388, -6.9025407, -2.9876943, 2.9876962
4: -4.5608354, -2.3998003, -4.5608406, -2.3997998, -1.8683372, 1.8783069
5: -11.0733891, -7.3661041, -11.0733919, -7.3661022, -3.0880980, 3.0859742
6: -17.5802155, -13.6031475, -17.5802174, -13.6031437, -3.3737144, 3.3771005
7: -6.4332161, -3.5954437, -6.4332151, -3.5954418, -2.5140724, 2.5076902
8: -2.0399027, 0.1837783, -2.0399036, 0.1837792, -2.0398755, 2.0371957
9: 2.4171572, 5.1602278, 2.4171557, 5.1602306, -2.5229955, 2.5181053

Time for backsubstitution: 12.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318886, upper bound: 1.7532695
time: 4.01 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318886, upper bound: 1.7532695
time: 4.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.06 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.06
Output dim: 9, lower bound: -1.7532654, upper bound: 1.7294444
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.06
Output dim: 9, lower bound: -1.7532654, upper bound: 1.7318883
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.06
Output dim: 9, lower bound: -1.7318886, upper bound: 1.7532695
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.06
Output dim: 9, lower bound: -1.7318886, upper bound: 1.7532695

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -14.2938175, -10.2915964, -14.2882805, -10.2883196, -3.1224313, 3.1147022
1: -12.4776812, -8.9436512, -12.4855299, -8.9410133, -3.0203028, 3.0238209
2: -13.4038467, -10.1832657, -13.3970776, -10.1873808, -3.0083914, 3.0017023
3: -9.8855505, -6.9075627, -9.8862467, -6.9265270, -2.9590235, 2.9786839
4: -4.5461292, -2.4082541, -4.5555034, -2.4037504, -1.8555055, 1.8599370
5: -11.0689840, -7.3733053, -11.0675154, -7.3843231, -3.0573845, 3.0720129
6: -17.5685043, -13.6093941, -17.5658302, -13.6050835, -3.3604650, 3.3545744
7: -6.4270821, -3.6046615, -6.4195733, -3.6027145, -2.4992085, 2.4944258
8: -2.0341759, 0.1777983, -2.0357962, 0.1758242, -2.0218253, 2.0283096
9: 2.4291964, 5.1466031, 2.4243631, 5.1557317, -2.5053358, 2.5012736

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7508214, upper bound: 1.7294443
time: 4.27 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7508236, upper bound: 1.7294446
time: 5.03 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -14.2953186, -10.2914848, -14.3099689, -10.2690487, -3.1530094, 3.1410499
1: -12.4783421, -8.9430904, -12.4927549, -8.9039383, -3.0664883, 3.0409718
2: -13.4055090, -10.1823177, -13.4110565, -10.1159296, -3.0709171, 3.0342321
3: -9.8860073, -6.9041681, -9.9426765, -6.8911157, -2.9948916, 3.0385084
4: -4.5463200, -2.4078250, -4.5645423, -2.3906200, -1.8692265, 1.8834631
5: -11.0696669, -7.3709030, -11.1356850, -7.3622942, -3.0892458, 3.1448445
6: -17.5701218, -13.6092291, -17.6137867, -13.5976763, -3.3984141, 3.4268370
7: -6.4289017, -3.6039600, -6.4419069, -3.5584178, -2.5541747, 2.5208015
8: -2.0346699, 0.1787114, -2.0630922, 0.1844096, -2.0483665, 2.0649195
9: 2.4283938, 5.1467171, 2.4107432, 5.1618052, -2.5138657, 2.5172913

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7508214, upper bound: 1.7318863
time: 4.53 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7508214, upper bound: 1.7318865
time: 4.31 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -14.3000641, -10.2871685, -14.2953281, -10.2914829, -3.1266804, 3.1266580
1: -12.4945726, -8.9361610, -12.4783459, -8.9430885, -3.0341330, 3.0246925
2: -13.4097595, -10.1796112, -13.4055109, -10.1823139, -3.0197401, 3.0174980
3: -9.8902349, -6.9025426, -9.8860092, -6.9041572, -2.9860778, 2.9834666
4: -4.5608354, -2.3998003, -4.5463209, -2.4078240, -1.8706565, 1.8639405
5: -11.0733891, -7.3661041, -11.0696669, -7.3708925, -3.0807362, 3.0813837
6: -17.5802155, -13.6031475, -17.5701237, -13.6092243, -3.3709393, 3.3663752
7: -6.4332161, -3.5954437, -6.4289093, -3.6039596, -2.5052791, 2.5094080
8: -2.0399027, 0.1837783, -2.0346689, 0.1787167, -2.0345864, 2.0343935
9: 2.4171572, 5.1602278, 2.4283915, 5.1467175, -2.5094323, 2.5117793

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7294443, upper bound: 1.7532647
time: 3.98 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318861, upper bound: 1.7532648
time: 3.88 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -14.3000641, -10.2871685, -14.3000641, -10.2871685, -3.1311698, 3.1311698
1: -12.4945726, -8.9361610, -12.4945726, -8.9361610, -3.0360260, 3.0360265
2: -13.4097595, -10.1796112, -13.4097595, -10.1796112, -3.0244389, 3.0244384
3: -9.8902349, -6.9025426, -9.8902349, -6.9025426, -2.9876924, 2.9876924
4: -4.5608354, -2.3998003, -4.5608354, -2.3998003, -1.8683367, 1.8683367
5: -11.0733891, -7.3661041, -11.0733891, -7.3661041, -3.0880961, 3.0880957
6: -17.5802155, -13.6031475, -17.5802155, -13.6031475, -3.3737135, 3.3737140
7: -6.4332161, -3.5954437, -6.4332161, -3.5954437, -2.5073519, 2.5073521
8: -2.0399027, 0.1837783, -2.0399027, 0.1837783, -2.0371943, 2.0371940
9: 2.4171572, 5.1602278, 2.4171572, 5.1602278, -2.5181029, 2.5181034

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7294443, upper bound: 1.7532654
time: 4.01 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318861, upper bound: 1.7532651
time: 4.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.66 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 9, lower bound: -1.7508214, upper bound: 1.7294443
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 9, lower bound: -1.7508236, upper bound: 1.7294446
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 9, lower bound: -1.7508214, upper bound: 1.7318863
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 9, lower bound: -1.7508214, upper bound: 1.7318865
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 9, lower bound: -1.7294443, upper bound: 1.7532647
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 9, lower bound: -1.7318861, upper bound: 1.7532648
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 9, lower bound: -1.7294443, upper bound: 1.7532654
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 9, lower bound: -1.7318861, upper bound: 1.7532651

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14.2846308, -10.2922974, -14.2882805, -10.2883196, -3.1128602, 3.1126432
1: -12.4735651, -8.9471931, -12.4855299, -8.9410133, -3.0153360, 3.0212097
2: -13.3935213, -10.1897430, -13.3970776, -10.1873808, -2.9930353, 2.9954610
3: -9.8828888, -6.9279327, -9.8862467, -6.9265270, -2.9563618, 2.9583139
4: -4.5448055, -2.4109602, -4.5555034, -2.4037504, -1.8501582, 1.8539796
5: -11.0643578, -7.3881693, -11.0675154, -7.3843231, -3.0525074, 3.0532990
6: -17.5582485, -13.6104107, -17.5658302, -13.6050835, -3.3488674, 3.3517425
7: -6.4157553, -3.6091747, -6.4195733, -3.6027145, -2.4906940, 2.4892125
8: -2.0311818, 0.1721077, -2.0357962, 0.1758242, -2.0182686, 2.0191460
9: 2.4341812, 5.1458588, 2.4243631, 5.1557317, -2.4997683, 2.4996474

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7424352, upper bound: 1.7294447
time: 4.81 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7424352, upper bound: 1.7294465
time: 4.74 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -14.3062782, -10.2730274, -14.2882805, -10.2883196, -3.1339936, 3.1417012
1: -12.4807825, -8.9101238, -12.4855299, -8.9410133, -3.0253372, 3.0666347
2: -13.4075050, -10.1182690, -13.3970776, -10.1873808, -3.0079594, 3.0554631
3: -9.9393272, -6.8925219, -9.8862467, -6.9265270, -3.0128002, 2.9937248
4: -4.5538464, -2.3978233, -4.5555034, -2.4037504, -1.8599682, 1.8668587
5: -11.1325302, -7.3661418, -11.0675154, -7.3843231, -3.1223240, 3.0750856
6: -17.6061974, -13.6030016, -17.5658302, -13.6050835, -3.4146280, 3.3877912
7: -6.4380989, -3.5648832, -6.4195733, -3.6027145, -2.5137968, 2.5436485
8: -2.0584745, 0.1806898, -2.0357962, 0.1758242, -2.0534325, 2.0288048
9: 2.4205651, 5.1519318, 2.4243631, 5.1557317, -2.5135207, 2.5072863

Time for backsubstitution: 12.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7424352, upper bound: 1.7294440
time: 5.29 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7424330, upper bound: 1.7294465
time: 4.66 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14.2846308, -10.2922974, -14.3099689, -10.2690487, -3.1419182, 3.1337833
1: -12.4735651, -8.9471931, -12.4927549, -8.9039383, -3.0607557, 3.0312109
2: -13.3935213, -10.1897430, -13.4110565, -10.1159296, -3.0530472, 3.0103860
3: -9.8828888, -6.9279327, -9.9426765, -6.8911157, -2.9917731, 3.0147438
4: -4.5448055, -2.4109602, -4.5645423, -2.3906200, -1.8630362, 1.8637893
5: -11.0643578, -7.3881693, -11.1356850, -7.3622942, -3.0742960, 3.1231151
6: -17.5582485, -13.6104107, -17.6137867, -13.5976763, -3.3849158, 3.4175212
7: -6.4157553, -3.6091747, -6.4419069, -3.5584178, -2.5442202, 2.5123076
8: -2.0311818, 0.1721077, -2.0630922, 0.1844096, -2.0279303, 2.0543087
9: 2.4341812, 5.1458588, 2.4107432, 5.1618052, -2.5074065, 2.5133998

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7424316, upper bound: 1.7318858
time: 3.95 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7424337, upper bound: 1.7318883
time: 5.13 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -14.3062782, -10.2730274, -14.3099689, -10.2690487, -3.1583896, 3.1581697
1: -12.4807825, -8.9101238, -12.4927549, -8.9039383, -3.0728817, 3.0787592
2: -13.4075050, -10.1182690, -13.4110565, -10.1159296, -3.0719557, 3.0743723
3: -9.9393272, -6.8925219, -9.9426765, -6.8911157, -3.0417738, 3.0402451
4: -4.5538464, -2.3978233, -4.5645423, -2.3906200, -1.8849010, 1.8887224
5: -11.1325302, -7.3661418, -11.1356850, -7.3622942, -3.1182528, 3.1190467
6: -17.6061974, -13.6030016, -17.6137867, -13.5976763, -3.4480157, 3.4486744
7: -6.4380989, -3.5648832, -6.4419069, -3.5584178, -2.5647569, 2.5632632
8: -2.0584745, 0.1806898, -2.0630922, 0.1844096, -2.0652099, 2.0660853
9: 2.4205651, 5.1519318, 2.4107432, 5.1618052, -2.5185878, 2.5184636

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7424316, upper bound: 1.7294453
time: 4.56 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7424316, upper bound: 1.7294465
time: 4.93 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.2893553, -10.2879810, -14.2938175, -10.2915964, -3.1152649, 3.1227674
1: -12.4897871, -8.9402523, -12.4776812, -8.9436512, -3.0279961, 3.0209150
2: -13.3977480, -10.1870403, -13.4038467, -10.1832657, -3.0009594, 3.0078316
3: -9.8871183, -6.9263182, -9.8855505, -6.9075627, -2.9795556, 2.9592323
4: -4.5593214, -2.4029384, -4.5461292, -2.4082541, -1.8634932, 1.8561678
5: -11.0680809, -7.3833776, -11.0689840, -7.3733053, -3.0721064, 3.0589156
6: -17.5683289, -13.6043262, -17.5685043, -13.6093941, -3.3569908, 3.3612154
7: -6.4200597, -3.6006608, -6.4270821, -3.6046615, -2.4946346, 2.5020742
8: -2.0364180, 0.1771660, -2.0341759, 0.1777983, -2.0290217, 2.0232239
9: 2.4229450, 5.1593699, 2.4291964, 5.1466031, -2.5027018, 2.5089874

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7294443, upper bound: 1.7508208
time: 4.44 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7294443, upper bound: 1.7532648
time: 4.06 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.3110447, -10.2687092, -14.2953186, -10.2914848, -3.1416426, 3.1533470
1: -12.4970131, -8.9031763, -12.4783421, -8.9430904, -3.0451484, 3.0670962
2: -13.4117231, -10.1156006, -13.4055090, -10.1823177, -3.0334888, 3.0699532
3: -9.9435472, -6.8909054, -9.8860073, -6.9041681, -3.0393791, 2.9951019
4: -4.5683608, -2.3898096, -4.5463200, -2.4078250, -1.8870182, 1.8698885
5: -11.1362419, -7.3613496, -11.0696669, -7.3709030, -3.1449366, 3.0907760
6: -17.6162872, -13.5969229, -17.5701218, -13.6092291, -3.4292512, 3.3991625
7: -6.4423943, -3.5563631, -6.4289017, -3.6039600, -2.5210133, 2.5550218
8: -2.0637145, 0.1857529, -2.0346699, 0.1787114, -2.0656304, 2.0497644
9: 2.4093242, 5.1654425, 2.4283938, 5.1467171, -2.5187168, 2.5175183

Time for backsubstitution: 12.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318884, upper bound: 1.7508208
time: 4.80 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318862, upper bound: 1.7532647
time: 4.75 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.2893553, -10.2879810, -14.2985554, -10.2872820, -3.1197686, 3.1272812
1: -12.4897871, -8.9402523, -12.4939041, -8.9367256, -3.0298891, 3.0322504
2: -13.3977480, -10.1870403, -13.4080915, -10.1805630, -3.0056601, 3.0147710
3: -9.8871183, -6.9263182, -9.8897781, -6.9059443, -2.9811740, 2.9634600
4: -4.5593214, -2.4029384, -4.5606451, -2.4002309, -1.8611722, 1.8605630
5: -11.0680809, -7.3833776, -11.0727072, -7.3685164, -3.0794663, 3.0656281
6: -17.5683289, -13.6043262, -17.5785904, -13.6033125, -3.3597608, 3.3685541
7: -6.4200597, -3.6006608, -6.4313865, -3.5961466, -2.4967070, 2.5000174
8: -2.0364180, 0.1771660, -2.0394135, 0.1828589, -2.0316305, 2.0260234
9: 2.4229450, 5.1593699, 2.4179621, 5.1601138, -2.5113735, 2.5153131

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7294443, upper bound: 1.7508212
time: 4.61 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7294443, upper bound: 1.7532652
time: 4.14 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.3110447, -10.2687092, -14.3000631, -10.2871695, -3.1461143, 3.1578608
1: -12.4970131, -8.9031763, -12.4945669, -8.9361649, -3.0470419, 3.0784316
2: -13.4117231, -10.1156006, -13.4097576, -10.1796141, -3.0381885, 3.0801165
3: -9.9435472, -6.8909054, -9.8902359, -6.9025526, -3.0409946, 2.9993305
4: -4.5683608, -2.3898096, -4.5608368, -2.3998022, -1.8846974, 1.8742843
5: -11.1362419, -7.3613496, -11.0733852, -7.3661122, -3.1522951, 3.0974884
6: -17.6162872, -13.5969229, -17.5802116, -13.6031504, -3.4320259, 3.4065008
7: -6.4423943, -3.5563631, -6.4332037, -3.5954466, -2.5230861, 2.5600674
8: -2.0637145, 0.1857529, -2.0399008, 0.1837730, -2.0682383, 2.0525653
9: 2.4093242, 5.1654425, 2.4171600, 5.1602273, -2.5273876, 2.5238433

Time for backsubstitution: 12.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318884, upper bound: 1.7508214
time: 4.65 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318862, upper bound: 1.7532652
time: 3.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.63 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7424352, upper bound: 1.7294447
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7424352, upper bound: 1.7294465
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7424352, upper bound: 1.7294440
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7424330, upper bound: 1.7294465
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7424316, upper bound: 1.7318858
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7424337, upper bound: 1.7318883
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7424316, upper bound: 1.7294453
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7424316, upper bound: 1.7294465
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7294443, upper bound: 1.7508208
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7294443, upper bound: 1.7532648
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7318884, upper bound: 1.7508208
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7318862, upper bound: 1.7532647
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7294443, upper bound: 1.7508212
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7294443, upper bound: 1.7532652
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7318884, upper bound: 1.7508214
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.63
Output dim: 9, lower bound: -1.7318862, upper bound: 1.7532652

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14.2846308, -10.2922974, -14.2846308, -10.2922974, -3.1090174, 3.1090183
1: -12.4735651, -8.9471931, -12.4735651, -8.9471931, -3.0092473, 3.0092468
2: -13.3935213, -10.1897430, -13.3935213, -10.1897430, -2.9895992, 2.9895992
3: -9.8828888, -6.9279327, -9.8828888, -6.9279327, -2.9549561, 2.9549561
4: -4.5448055, -2.4109602, -4.5448055, -2.4109602, -1.8431740, 1.8431740
5: -11.0643578, -7.3881693, -11.0643578, -7.3881693, -3.0492468, 3.0492468
6: -17.5582485, -13.6104107, -17.5582485, -13.6104107, -3.3434534, 3.3434539
7: -6.4157553, -3.6091747, -6.4157553, -3.6091747, -2.4847636, 2.4847641
8: -2.0311818, 0.1721077, -2.0311818, 0.1721077, -2.0143814, 2.0143816
9: 2.4341812, 5.1458588, 2.4341812, 5.1458588, -2.4898627, 2.4898627

Time for backsubstitution: 12.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7393669, upper bound: 1.7285489
time: 4.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7424283, upper bound: 1.7294411
time: 4.25 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.2846308, -10.2922974, -14.2893553, -10.2879810, -3.1131959, 3.1132054
1: -12.4735651, -8.9471931, -12.4897871, -8.9402523, -3.0159483, 3.0253849
2: -13.3935213, -10.1897430, -13.3977480, -10.1870403, -2.9924755, 2.9947181
3: -9.8828888, -6.9279327, -9.8871183, -6.9263182, -2.9565706, 2.9591856
4: -4.5448055, -2.4109602, -4.5593214, -2.4029384, -1.8508205, 1.8575358
5: -11.0643578, -7.3881693, -11.0680809, -7.3833776, -3.0540380, 3.0533919
6: -17.5582485, -13.6104107, -17.5683289, -13.6043262, -3.3496180, 3.3541577
7: -6.4157553, -3.6091747, -6.4200597, -3.6006608, -2.4935594, 2.4894216
8: -2.0311818, 0.1721077, -2.0364180, 0.1771660, -2.0196676, 2.0198581
9: 2.4341812, 5.1458588, 2.4229450, 5.1593699, -2.5034199, 2.5010755

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7393671, upper bound: 1.7285511
time: 4.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7424283, upper bound: 1.7294411
time: 5.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.3062782, -10.2730274, -14.2846308, -10.2922974, -3.1301527, 3.1380763
1: -12.4807825, -8.9101238, -12.4735651, -8.9471931, -3.0192475, 3.0546718
2: -13.4075050, -10.1182690, -13.3935213, -10.1897430, -3.0045233, 3.0496087
3: -9.9393272, -6.8925219, -9.8828888, -6.9279327, -3.0113945, 2.9903669
4: -4.5538464, -2.3978233, -4.5448055, -2.4109602, -1.8529835, 1.8560526
5: -11.1325302, -7.3661418, -11.0643578, -7.3881693, -3.1190634, 3.0710340
6: -17.6061974, -13.6030016, -17.5582485, -13.6104107, -3.4092140, 3.3795025
7: -6.4380989, -3.5648832, -6.4157553, -3.6091747, -2.5078659, 2.5391836
8: -2.0584745, 0.1806898, -2.0311818, 0.1721077, -2.0495453, 2.0240407
9: 2.4205651, 5.1519318, 2.4341812, 5.1458588, -2.5036156, 2.4975016

Time for backsubstitution: 12.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7418075, upper bound: 1.7285499
time: 3.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7448688, upper bound: 1.7294395
time: 4.14 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.3062782, -10.2730274, -14.2893553, -10.2879810, -3.1343312, 3.1422634
1: -12.4807825, -8.9101238, -12.4897871, -8.9402523, -3.0259495, 3.0708094
2: -13.4075050, -10.1182690, -13.3977480, -10.1870403, -3.0073996, 3.0544693
3: -9.9393272, -6.8925219, -9.8871183, -6.9263182, -3.0130091, 2.9945965
4: -4.5538464, -2.3978233, -4.5593214, -2.4029384, -1.8606305, 1.8704147
5: -11.1325302, -7.3661418, -11.0680809, -7.3833776, -3.1238546, 3.0751786
6: -17.6061974, -13.6030016, -17.5683289, -13.6043262, -3.4153767, 3.3902066
7: -6.4380989, -3.5648832, -6.4200597, -3.6006608, -2.5166616, 2.5420532
8: -2.0584745, 0.1806898, -2.0364180, 0.1771660, -2.0546193, 2.0295172
9: 2.4205651, 5.1519318, 2.4229450, 5.1593699, -2.5171728, 2.5087144

Time for backsubstitution: 12.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7418075, upper bound: 1.7285499
time: 4.31 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7448688, upper bound: 1.7294396
time: 5.16 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.2846308, -10.2922974, -14.3062782, -10.2730274, -3.1380758, 3.1301527
1: -12.4735651, -8.9471931, -12.4807825, -8.9101238, -3.0546713, 3.0192480
2: -13.3935213, -10.1897430, -13.4075050, -10.1182690, -3.0496082, 3.0045233
3: -9.8828888, -6.9279327, -9.9393272, -6.8925219, -2.9903669, 3.0113945
4: -4.5448055, -2.4109602, -4.5538464, -2.3978233, -1.8560524, 1.8529835
5: -11.0643578, -7.3881693, -11.1325302, -7.3661418, -3.0710335, 3.1190634
6: -17.5582485, -13.6104107, -17.6061974, -13.6030016, -3.3795023, 3.4092133
7: -6.4157553, -3.6091747, -6.4380989, -3.5648832, -2.5391836, 2.5078661
8: -2.0311818, 0.1721077, -2.0584745, 0.1806898, -2.0240412, 2.0495448
9: 2.4341812, 5.1458588, 2.4205651, 5.1519318, -2.4975016, 2.5036151

Time for backsubstitution: 12.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7393678, upper bound: 1.7309890
time: 4.98 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7424268, upper bound: 1.7318812
time: 4.54 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.2846308, -10.2922974, -14.3110447, -10.2687092, -3.1422553, 3.1343799
1: -12.4735651, -8.9471931, -12.4970131, -8.9031763, -3.0613642, 3.0353861
2: -13.3935213, -10.1897430, -13.4117231, -10.1156006, -3.0520835, 3.0096416
3: -9.8828888, -6.9279327, -9.9435472, -6.8909054, -2.9919834, 3.0156145
4: -4.5448055, -2.4109602, -4.5683608, -2.3898096, -1.8636980, 1.8673446
5: -11.0643578, -7.3881693, -11.1362419, -7.3613496, -3.0758247, 3.1232076
6: -17.5582485, -13.6104107, -17.6162872, -13.5969229, -3.3856649, 3.4199357
7: -6.4157553, -3.6091747, -6.4423943, -3.5563631, -2.5450668, 2.5125165
8: -2.0311818, 0.1721077, -2.0637145, 0.1857529, -2.0293303, 2.0550194
9: 2.4341812, 5.1458588, 2.4093242, 5.1654425, -2.5110595, 2.5148275

Time for backsubstitution: 12.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7393657, upper bound: 1.7309915
time: 4.27 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7424268, upper bound: 1.7318835
time: 4.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.3062782, -10.2730274, -14.3062782, -10.2730274, -3.1545467, 3.1545467
1: -12.4807825, -8.9101238, -12.4807825, -8.9101238, -3.0667982, 3.0667982
2: -13.4075050, -10.1182690, -13.4075050, -10.1182690, -3.0685177, 3.0685179
3: -9.9393272, -6.8925219, -9.9393272, -6.8925219, -3.0379353, 3.0379348
4: -4.5538464, -2.3978233, -4.5538464, -2.3978233, -1.8779173, 1.8779171
5: -11.1325302, -7.3661418, -11.1325302, -7.3661418, -3.1149964, 3.1149960
6: -17.6061974, -13.6030016, -17.6061974, -13.6030016, -3.4425635, 3.4425633
7: -6.4380989, -3.5648832, -6.4380989, -3.5648832, -2.5588164, 2.5588164
8: -2.0584745, 0.1806898, -2.0584745, 0.1806898, -2.0613213, 2.0613217
9: 2.4205651, 5.1519318, 2.4205651, 5.1519318, -2.5086813, 2.5086815

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7418077, upper bound: 1.7285471
time: 7.50 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7448689, upper bound: 1.7294395
time: 4.25 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.3062782, -10.2730274, -14.3110447, -10.2687092, -3.1587267, 3.1587625
1: -12.4807825, -8.9101238, -12.4970131, -8.9031763, -3.0734911, 3.0829358
2: -13.4075050, -10.1182690, -13.4117231, -10.1156006, -3.0709920, 3.0733781
3: -9.9393272, -6.8925219, -9.9435472, -6.8909054, -3.0406952, 3.0423899
4: -4.5538464, -2.3978233, -4.5683608, -2.3898096, -1.8855629, 1.8922780
5: -11.1325302, -7.3661418, -11.1362419, -7.3613496, -3.1197815, 3.1191401
6: -17.6061974, -13.6030016, -17.6162872, -13.5969229, -3.4466448, 3.4494698
7: -6.4380989, -3.5648832, -6.4423943, -3.5563631, -2.5676246, 2.5634747
8: -2.0584745, 0.1806898, -2.0637145, 0.1857529, -2.0666080, 2.0667958
9: 2.4205651, 5.1519318, 2.4093242, 5.1654425, -2.5222402, 2.5198891

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7418077, upper bound: 1.7285472
time: 5.07 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7448689, upper bound: 1.7294418
time: 5.16 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14.2893553, -10.2879810, -14.2846308, -10.2922974, -3.1132059, 3.1131959
1: -12.4897871, -8.9402523, -12.4735651, -8.9471931, -3.0253854, 3.0159488
2: -13.3977480, -10.1870403, -13.3935213, -10.1897430, -2.9947176, 2.9924755
3: -9.8871183, -6.9263182, -9.8828888, -6.9279327, -2.9591856, 2.9565706
4: -4.5593214, -2.4029384, -4.5448055, -2.4109602, -1.8575358, 1.8508205
5: -11.0680809, -7.3833776, -11.0643578, -7.3881693, -3.0533915, 3.0540380
6: -17.5683289, -13.6043262, -17.5582485, -13.6104107, -3.3541574, 3.3496180
7: -6.4200597, -3.6006608, -6.4157553, -3.6091747, -2.4894218, 2.4935594
8: -2.0364180, 0.1771660, -2.0311818, 0.1721077, -2.0198579, 2.0196676
9: 2.4229450, 5.1593699, 2.4341812, 5.1458588, -2.5010755, 2.5034204

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7263752, upper bound: 1.7499129
time: 4.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7294396, upper bound: 1.7508171
time: 4.96 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.2893553, -10.2879810, -14.3062782, -10.2730274, -3.1422634, 3.1343307
1: -12.4897871, -8.9402523, -12.4807825, -8.9101238, -3.0708094, 3.0259495
2: -13.3977480, -10.1870403, -13.4075050, -10.1182690, -3.0544691, 3.0074000
3: -9.8871183, -6.9263182, -9.9393272, -6.8925219, -2.9945965, 3.0130091
4: -4.5593214, -2.4029384, -4.5538464, -2.3978233, -1.8704147, 1.8606305
5: -11.0680809, -7.3833776, -11.1325302, -7.3661418, -3.0751781, 3.1238546
6: -17.5683289, -13.6043262, -17.6061974, -13.6030016, -3.3902063, 3.4153771
7: -6.4200597, -3.6006608, -6.4380989, -3.5648832, -2.5420532, 2.5166619
8: -2.0364180, 0.1771660, -2.0584745, 0.1806898, -2.0295172, 2.0546193
9: 2.4229450, 5.1593699, 2.4205651, 5.1519318, -2.5087144, 2.5171731

Time for backsubstitution: 12.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7263774, upper bound: 1.7523506
time: 4.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7294396, upper bound: 1.7532597
time: 4.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.3110447, -10.2687092, -14.2846308, -10.2922974, -3.1343794, 3.1422558
1: -12.4970131, -8.9031763, -12.4735651, -8.9471931, -3.0353856, 3.0613642
2: -13.4117231, -10.1156006, -13.3935213, -10.1897430, -3.0096416, 3.0520835
3: -9.9435472, -6.8909054, -9.8828888, -6.9279327, -3.0156145, 2.9919834
4: -4.5683608, -2.3898096, -4.5448055, -2.4109602, -1.8673444, 1.8636985
5: -11.1362419, -7.3613496, -11.0643578, -7.3881693, -3.1232071, 3.0758243
6: -17.6162872, -13.5969229, -17.5582485, -13.6104107, -3.4199352, 3.3856652
7: -6.4423943, -3.5563631, -6.4157553, -3.6091747, -2.5125160, 2.5450671
8: -2.0637145, 0.1857529, -2.0311818, 0.1721077, -2.0550194, 2.0293298
9: 2.4093242, 5.1654425, 2.4341812, 5.1458588, -2.5148275, 2.5110595

Time for backsubstitution: 12.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7288171, upper bound: 1.7499110
time: 4.16 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318813, upper bound: 1.7508163
time: 4.27 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.3110447, -10.2687092, -14.3062782, -10.2730274, -3.1587620, 3.1587267
1: -12.4970131, -8.9031763, -12.4807825, -8.9101238, -3.0829363, 3.0734916
2: -13.4117231, -10.1156006, -13.4075050, -10.1182690, -3.0733776, 3.0709927
3: -9.9435472, -6.8909054, -9.9393272, -6.8925219, -3.0423899, 3.0406957
4: -4.5683608, -2.3898096, -4.5538464, -2.3978233, -1.8922782, 1.8855627
5: -11.1362419, -7.3613496, -11.1325302, -7.3661418, -3.1191401, 3.1197820
6: -17.6162872, -13.5969229, -17.6061974, -13.6030016, -3.4494700, 3.4466455
7: -6.4423943, -3.5563631, -6.4380989, -3.5648832, -2.5634751, 2.5676241
8: -2.0637145, 0.1857529, -2.0584745, 0.1806898, -2.0667958, 2.0666082
9: 2.4093242, 5.1654425, 2.4205651, 5.1519318, -2.5198889, 2.5222404

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7288172, upper bound: 1.7499117
time: 4.28 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318814, upper bound: 1.7508162
time: 4.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.2893553, -10.2879810, -14.2893553, -10.2879810, -3.1177106, 3.1177111
1: -12.4897871, -8.9402523, -12.4897871, -8.9402523, -3.0272813, 3.0272813
2: -13.3977480, -10.1870403, -13.3977480, -10.1870403, -2.9994121, 2.9994125
3: -9.8871183, -6.9263182, -9.8871183, -6.9263182, -2.9608002, 2.9608002
4: -4.5593214, -2.4029384, -4.5593214, -2.4029384, -1.8552160, 1.8552163
5: -11.0680809, -7.3833776, -11.0680809, -7.3833776, -3.0607548, 3.0607548
6: -17.5683289, -13.6043262, -17.5683289, -13.6043262, -3.3569393, 3.3569388
7: -6.4200597, -3.6006608, -6.4200597, -3.6006608, -2.4915018, 2.4915018
8: -2.0364180, 0.1771660, -2.0364180, 0.1771660, -2.0224652, 2.0224650
9: 2.4229450, 5.1593699, 2.4229450, 5.1593699, -2.5097454, 2.5097451

Time for backsubstitution: 12.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7263752, upper bound: 1.7499112
time: 4.97 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7294396, upper bound: 1.7508203
time: 4.02 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.2893553, -10.2879810, -14.3110447, -10.2687092, -3.1467705, 3.1388521
1: -12.4897871, -8.9402523, -12.4970131, -8.9031763, -3.0726976, 3.0372815
2: -13.3977480, -10.1870403, -13.4117231, -10.1156006, -3.0622449, 3.0143361
3: -9.8871183, -6.9263182, -9.9435472, -6.8909054, -2.9962130, 3.0172291
4: -4.5593214, -2.4029384, -4.5683608, -2.3898096, -1.8680940, 1.8650258
5: -11.0680809, -7.3833776, -11.1362419, -7.3613496, -3.0825415, 3.1305695
6: -17.5683289, -13.6043262, -17.6162872, -13.5969229, -3.3929853, 3.4227152
7: -6.4200597, -3.6006608, -6.4423943, -3.5563631, -2.5501451, 2.5145960
8: -2.0364180, 0.1771660, -2.0637145, 0.1857529, -2.0321269, 2.0576270
9: 2.4229450, 5.1593699, 2.4093242, 5.1654425, -2.5173843, 2.5234966

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7263752, upper bound: 1.7523483
time: 4.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7294396, upper bound: 1.7532601
time: 5.20 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.3110447, -10.2687092, -14.2893553, -10.2879810, -3.1388516, 3.1467705
1: -12.4970131, -8.9031763, -12.4897871, -8.9402523, -3.0372815, 3.0726976
2: -13.4117231, -10.1156006, -13.3977480, -10.1870403, -3.0143361, 3.0622451
3: -9.9435472, -6.8909054, -9.8871183, -6.9263182, -3.0172291, 2.9962130
4: -4.5683608, -2.3898096, -4.5593214, -2.4029384, -1.8650255, 1.8680940
5: -11.1362419, -7.3613496, -11.0680809, -7.3833776, -3.1305695, 3.0825410
6: -17.6162872, -13.5969229, -17.5683289, -13.6043262, -3.4227142, 3.3929853
7: -6.4423943, -3.5563631, -6.4200597, -3.6006608, -2.5145955, 2.5501451
8: -2.0637145, 0.1857529, -2.0364180, 0.1771660, -2.0576272, 2.0321269
9: 2.4093242, 5.1654425, 2.4229450, 5.1593699, -2.5234964, 2.5173845

Time for backsubstitution: 12.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7288171, upper bound: 1.7499121
time: 4.57 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318813, upper bound: 1.7508189
time: 4.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.3110447, -10.2687092, -14.3110447, -10.2687092, -3.1632366, 3.1632361
1: -12.4970131, -8.9031763, -12.4970131, -8.9031763, -3.0848236, 3.0848231
2: -13.4117231, -10.1156006, -13.4117231, -10.1156006, -3.0807528, 3.0807528
3: -9.9435472, -6.8909054, -9.9435472, -6.8909054, -3.0526419, 3.0526419
4: -4.5683608, -2.3898096, -4.5683608, -2.3898096, -1.8899565, 1.8899570
5: -11.1362419, -7.3613496, -11.1362419, -7.3613496, -3.1264992, 3.1264987
6: -17.6162872, -13.5969229, -17.6162872, -13.5969229, -3.4550028, 3.4550028
7: -6.4423943, -3.5563631, -6.4423943, -3.5563631, -2.5655670, 2.5655670
8: -2.0637145, 0.1857529, -2.0637145, 0.1857529, -2.0694036, 2.0694039
9: 2.4093242, 5.1654425, 2.4093242, 5.1654425, -2.5285602, 2.5285604

Time for backsubstitution: 12.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7288172, upper bound: 1.7499122
time: 4.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318814, upper bound: 1.7508189
time: 4.13 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 21.98 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7393669, upper bound: 1.7285489
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7424283, upper bound: 1.7294411
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7393671, upper bound: 1.7285511
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7424283, upper bound: 1.7294411
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7418075, upper bound: 1.7285499
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7448688, upper bound: 1.7294395
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7418075, upper bound: 1.7285499
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7448688, upper bound: 1.7294396
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7393678, upper bound: 1.7309890
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7424268, upper bound: 1.7318812
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7393657, upper bound: 1.7309915
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7424268, upper bound: 1.7318835
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7418077, upper bound: 1.7285471
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7448689, upper bound: 1.7294395
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7418077, upper bound: 1.7285472
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7448689, upper bound: 1.7294418
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7263752, upper bound: 1.7499129
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7294396, upper bound: 1.7508171
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7263774, upper bound: 1.7523506
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7294396, upper bound: 1.7532597
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7288171, upper bound: 1.7499110
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7318813, upper bound: 1.7508163
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7288172, upper bound: 1.7499117
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7318814, upper bound: 1.7508162
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7263752, upper bound: 1.7499112
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7294396, upper bound: 1.7508203
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7263752, upper bound: 1.7523483
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7294396, upper bound: 1.7532601
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7288171, upper bound: 1.7499121
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7318813, upper bound: 1.7508189
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7288172, upper bound: 1.7499122
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.98
Output dim: 9, lower bound: -1.7318814, upper bound: 1.7508189

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14.2752590, -10.3249445, -14.2825356, -10.2996864, -3.0940456, 3.0740814
1: -12.4692907, -8.9608955, -12.4726162, -8.9503136, -2.9981823, 2.9924369
2: -13.3810949, -10.1954117, -13.3906803, -10.1910105, -2.9726796, 2.9769740
3: -9.8727856, -6.9312177, -9.8805962, -6.9286652, -2.9441204, 2.9493785
4: -4.5442271, -2.4259353, -4.5446749, -2.4143639, -1.8333232, 1.8242185
5: -11.0555973, -7.3901548, -11.0623646, -7.3886137, -3.0340099, 3.0376983
6: -17.5522270, -13.6359434, -17.5568924, -13.6161842, -3.3264246, 3.3128631
7: -6.4127755, -3.6144905, -6.4150710, -3.6103859, -2.4772263, 2.4760354
8: -2.0253696, 0.1595144, -2.0298038, 0.1692462, -2.0042577, 1.9990828
9: 2.4437881, 5.1446662, 2.4363651, 5.1455894, -2.4780052, 2.4823813

Time for backsubstitution: 13.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7393669, upper bound: 1.7393693
time: 4.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7393669, upper bound: 1.7415397
time: 4.91 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -14.3285809, -10.2895708, -14.2846260, -10.2923069, -3.1532106, 3.1050558
1: -12.4841938, -8.9443293, -12.4735622, -8.9471989, -3.0171075, 3.0077219
2: -13.3973303, -10.1770477, -13.3935146, -10.1897469, -2.9908762, 3.0001702
3: -9.8874311, -6.9168262, -9.8828859, -6.9279327, -2.9594984, 2.9660597
4: -4.5567980, -2.4067566, -4.5448050, -2.4109697, -1.8499637, 1.8457816
5: -11.0691080, -7.3824968, -11.0643520, -7.3881693, -3.0569820, 3.0478706
6: -17.5995598, -13.6090937, -17.5582428, -13.6104164, -3.3801794, 3.3377841
7: -6.4207535, -3.6054564, -6.4157534, -3.6091771, -2.4897938, 2.4883671
8: -2.0506783, 0.1753235, -2.0311799, 0.1721053, -2.0321765, 2.0167124
9: 2.4303093, 5.1495018, 2.4341860, 5.1458588, -2.4927130, 2.4894881

Time for backsubstitution: 13.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7415373, upper bound: 1.7393666
time: 5.13 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7415375, upper bound: 1.7424280
time: 4.84 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14.2752590, -10.3249445, -14.2872429, -10.2953663, -3.0982237, 3.0782495
1: -12.4692907, -8.9608955, -12.4888363, -8.9433813, -3.0048885, 3.0085759
2: -13.3810949, -10.1954117, -13.3949299, -10.1883087, -2.9755559, 2.9820933
3: -9.8727856, -6.9312177, -9.8848228, -6.9270487, -2.9457369, 2.9536052
4: -4.5442271, -2.4259353, -4.5591922, -2.4063401, -1.8409729, 1.8385806
5: -11.0555973, -7.3901548, -11.0660858, -7.3838234, -3.0388002, 3.0418429
6: -17.5522270, -13.6359434, -17.5669708, -13.6101017, -3.3325891, 3.3235700
7: -6.4127755, -3.6144905, -6.4193773, -3.6018791, -2.4860153, 2.4806929
8: -2.0253696, 0.1595144, -2.0350413, 0.1743064, -2.0095444, 2.0045638
9: 2.4437881, 5.1446662, 2.4251337, 5.1590996, -2.4915633, 2.4936013

Time for backsubstitution: 12.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7477308, upper bound: 1.7263767
time: 5.16 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7477308, upper bound: 1.7285490
time: 4.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -14.3285809, -10.2895708, -14.2893505, -10.2879896, -3.1573868, 3.1092443
1: -12.4841938, -8.9443293, -12.4897842, -8.9402552, -3.0238094, 3.0238605
2: -13.3973303, -10.1770477, -13.3977423, -10.1870441, -2.9937530, 3.0052896
3: -9.8874311, -6.9168262, -9.8871126, -6.9263177, -2.9611135, 2.9702864
4: -4.5567980, -2.4067566, -4.5593219, -2.4029472, -1.8576109, 1.8601441
5: -11.0691080, -7.3824968, -11.0680752, -7.3833799, -3.0617738, 3.0520163
6: -17.5995598, -13.6090937, -17.5683250, -13.6043339, -3.3863430, 3.3484879
7: -6.4207535, -3.6054564, -6.4200592, -3.6006641, -2.4985890, 2.4930253
8: -2.0506783, 0.1753235, -2.0364170, 0.1771641, -2.0374632, 2.0221882
9: 2.4303093, 5.1495018, 2.4229479, 5.1593695, -2.5062709, 2.5007014

Time for backsubstitution: 12.84 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.522998809814453
rel_dist={9: [-1.7535540543698414, 1.753553758870325]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678720, upper bound: 1.4559398
time: 10.37 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678720, upper bound: 1.4678717
time: 4.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.22 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.22
Output dim: 9, lower bound: -1.4678720, upper bound: 1.4559398
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.22
Output dim: 9, lower bound: -1.4678720, upper bound: 1.4678717

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -14.2953281, -10.2914829, -14.2982788, -10.2877312, -2.6776752, 2.6773205
1: -12.4783459, -8.9430885, -12.4874964, -8.9374256, -2.6653252, 2.6688428
2: -13.4055109, -10.1823139, -13.4086456, -10.1801786, -2.6593485, 2.6616287
3: -9.8860092, -6.9041572, -9.8887882, -6.9028931, -2.7788019, 2.7760210
4: -4.5463209, -2.4078240, -4.5544891, -2.4011540, -1.6482279, 1.6501391
5: -11.0696669, -7.3708925, -11.0724583, -7.3676720, -2.6852822, 2.6867270
6: -17.5701237, -13.6092243, -17.5760574, -13.6044035, -2.9811521, 2.9829478
7: -6.4289093, -3.6039596, -6.4324040, -3.5988584, -2.2626123, 2.2626784
8: -2.0346689, 0.1787167, -2.0388684, 0.1815438, -1.8322592, 1.8335910
9: 2.4283915, 5.1467175, 2.4195199, 5.1541786, -2.3384261, 2.3397760

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678705, upper bound: 1.4543305
time: 4.95 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678705, upper bound: 1.4559388
time: 9.66 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -14.3000641, -10.2871685, -14.3000660, -10.2871675, -2.6826277, 2.6825433
1: -12.4945726, -8.9361610, -12.4945774, -8.9361639, -2.6768312, 2.6824899
2: -13.4097595, -10.1796112, -13.4097595, -10.1796122, -2.6660438, 2.6653209
3: -9.8902349, -6.9025426, -9.8902378, -6.9025407, -2.7834044, 2.7892389
4: -4.5608354, -2.3998003, -4.5608397, -2.3997991, -1.6519654, 1.6636972
5: -11.0733891, -7.3661041, -11.0733919, -7.3661022, -2.6942892, 2.6923213
6: -17.5802155, -13.6031475, -17.5802193, -13.6031466, -2.9891357, 2.9931178
7: -6.4332161, -3.5954437, -6.4332175, -3.5954418, -2.2720456, 2.2644141
8: -2.0399027, 0.1837783, -2.0399036, 0.1837788, -1.8400674, 1.8369143
9: 2.4171572, 5.1602278, 2.4171553, 5.1602316, -2.3557143, 2.3499622

Time for backsubstitution: 13.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678705, upper bound: 1.4662647
time: 4.75 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678705, upper bound: 1.4678703
time: 4.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.87 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.87
Output dim: 9, lower bound: -1.4678705, upper bound: 1.4543305
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.87
Output dim: 9, lower bound: -1.4678705, upper bound: 1.4559388
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.87
Output dim: 9, lower bound: -1.4678705, upper bound: 1.4662647
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.87
Output dim: 9, lower bound: -1.4678705, upper bound: 1.4678703

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -14.2906685, -10.2918320, -14.2875690, -10.2885494, -2.6705580, 2.6652484
1: -12.4762840, -8.9448605, -12.4827147, -8.9415207, -2.6599045, 2.6618457
2: -13.4003410, -10.1853466, -13.3966331, -10.1876078, -2.6444902, 2.6407614
3: -9.8846092, -6.9146233, -9.8856688, -6.9266691, -2.7575350, 2.7641573
4: -4.5457067, -2.4091644, -4.5529747, -2.4042923, -1.6386590, 1.6409421
5: -11.0674953, -7.3783722, -11.0671473, -7.3849468, -2.6612263, 2.6717315
6: -17.5651512, -13.6097383, -17.5641708, -13.6055861, -2.9720306, 2.9680643
7: -6.4232359, -3.6061616, -6.4192486, -3.6040716, -2.2524438, 2.2503121
8: -2.0331459, 0.1758676, -2.0353804, 0.1749334, -1.8199091, 1.8249512
9: 2.4308910, 5.1463590, 2.4253068, 5.1533203, -2.3337488, 2.3324845

Time for backsubstitution: 12.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662636, upper bound: 1.4543329
time: 5.62 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662636, upper bound: 1.4543302
time: 6.08 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -14.2953186, -10.2914820, -14.3092556, -10.2692728, -2.7043586, 2.6903758
1: -12.4783392, -8.9430885, -12.4899330, -8.9044456, -2.7077312, 2.6789465
2: -13.4055080, -10.1823168, -13.4106092, -10.1161480, -2.7078142, 2.6698723
3: -9.8860073, -6.9041753, -9.9421005, -6.8912573, -2.7937355, 2.8305786
4: -4.5463200, -2.4078269, -4.5620131, -2.3911600, -1.6541743, 1.6647770
5: -11.0696659, -7.3709078, -11.1353140, -7.3629169, -2.6877289, 2.7506137
6: -17.5701199, -13.6092300, -17.6121292, -13.5981789, -3.0139275, 3.0404396
7: -6.4288931, -3.6039619, -6.4415879, -3.5597763, -2.3070707, 2.2763920
8: -2.0346670, 0.1787086, -2.0626774, 0.1835179, -1.8448296, 1.8625224
9: 2.4283948, 5.1467171, 2.4116874, 5.1593947, -2.3441634, 2.3479378

Time for backsubstitution: 12.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4653074, upper bound: 1.4547419
time: 4.69 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678660, upper bound: 1.4559343
time: 5.08 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -14.2954063, -10.2875156, -14.2893572, -10.2879810, -2.6755161, 2.6704688
1: -12.4925060, -8.9379320, -12.4897900, -8.9402504, -2.6714096, 2.6754913
2: -13.4045820, -10.1826420, -13.3977509, -10.1870413, -2.6511817, 2.6444535
3: -9.8888359, -6.9130068, -9.8871193, -6.9263158, -2.7621374, 2.7773728
4: -4.5602241, -2.4011421, -4.5593252, -2.4029372, -1.6423965, 1.6545010
5: -11.0712166, -7.3735800, -11.0680790, -7.3833771, -2.6702371, 2.6773300
6: -17.5752354, -13.6036568, -17.5683327, -13.6043272, -2.9800344, 2.9782367
7: -6.4275389, -3.5976477, -6.4200573, -3.6006565, -2.2618756, 2.2520452
8: -2.0383835, 0.1809287, -2.0364184, 0.1771684, -1.8277159, 1.8282719
9: 2.4196558, 5.1598692, 2.4229431, 5.1593723, -2.3510404, 2.3426719

Time for backsubstitution: 12.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662636, upper bound: 1.4662647
time: 4.49 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662636, upper bound: 1.4662627
time: 6.17 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -14.3000574, -10.2871666, -14.3110456, -10.2687082, -2.7093163, 2.6956177
1: -12.4945621, -8.9361649, -12.4970169, -8.9031744, -2.7192345, 2.6925931
2: -13.4097557, -10.1796160, -13.4117241, -10.1155987, -2.7165270, 2.6735630
3: -9.8902349, -6.9025583, -9.9435482, -6.8909039, -2.7983360, 2.8384817
4: -4.5608358, -2.3998032, -4.5683637, -2.3898091, -1.6579096, 1.6783345
5: -11.0733881, -7.3661184, -11.1362429, -7.3613510, -2.6967373, 2.7558565
6: -17.5802116, -13.6031466, -17.6162891, -13.5969210, -3.0219193, 3.0506101
7: -6.4331946, -3.5954480, -6.4423938, -3.5563607, -2.3149500, 2.2781343
8: -2.0399008, 0.1837697, -2.0637140, 0.1857538, -1.8526368, 1.8657334
9: 2.4171619, 5.1602268, 2.4093227, 5.1654458, -2.3614528, 2.3581195

Time for backsubstitution: 12.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4653074, upper bound: 1.4666764
time: 5.56 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678660, upper bound: 1.4678657
time: 4.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.46 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.46
Output dim: 9, lower bound: -1.4662636, upper bound: 1.4543329
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.46
Output dim: 9, lower bound: -1.4662636, upper bound: 1.4543302
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.46
Output dim: 9, lower bound: -1.4653074, upper bound: 1.4547419
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.46
Output dim: 9, lower bound: -1.4678660, upper bound: 1.4559343
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.46
Output dim: 9, lower bound: -1.4662636, upper bound: 1.4662647
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.46
Output dim: 9, lower bound: -1.4662636, upper bound: 1.4662627
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.46
Output dim: 9, lower bound: -1.4653074, upper bound: 1.4666764
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.46
Output dim: 9, lower bound: -1.4678660, upper bound: 1.4678657

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14.2846308, -10.2922974, -14.2875690, -10.2885494, -2.6642122, 2.6638579
1: -12.4735651, -8.9471931, -12.4827147, -8.9415207, -2.6565809, 2.6600952
2: -13.3935213, -10.1897430, -13.3966331, -10.1876078, -2.6343279, 2.6366076
3: -9.8828888, -6.9279327, -9.8856688, -6.9266691, -2.7559118, 2.7531319
4: -4.5448055, -2.4109602, -4.5529747, -2.4042923, -1.6351080, 1.6370182
5: -11.0643578, -7.3881693, -11.0671473, -7.3849468, -2.6579351, 2.6593790
6: -17.5582485, -13.6104107, -17.5641708, -13.6055861, -2.9643869, 2.9661613
7: -6.4157553, -3.6091747, -6.4192486, -3.6040716, -2.2467642, 2.2468224
8: -2.0311818, 0.1721077, -2.0353804, 0.1749334, -1.8175325, 1.8188648
9: 2.4341812, 5.1458588, 2.4253068, 5.1533203, -2.3300672, 2.3314178

Time for backsubstitution: 13.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4639275, upper bound: 1.4543329
time: 5.75 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4639275, upper bound: 1.4543297
time: 5.30 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -14.3062782, -10.2730274, -14.2875690, -10.2885494, -2.6853466, 2.6929159
1: -12.4807825, -8.9101238, -12.4827147, -8.9415207, -2.6665831, 2.7055202
2: -13.4075050, -10.1182690, -13.3966331, -10.1876078, -2.6492510, 2.6922235
3: -9.9393272, -6.8925219, -9.8856688, -6.9266691, -2.8095727, 2.7906008
4: -4.5538464, -2.3978233, -4.5529747, -2.4042923, -1.6449180, 1.6498973
5: -11.1325302, -7.3661418, -11.0671473, -7.3849468, -2.7274609, 2.6811657
6: -17.6061974, -13.6030016, -17.5641708, -13.6055861, -3.0301456, 3.0022101
7: -6.4380989, -3.5648832, -6.4192486, -3.6040716, -2.2698669, 2.2974842
8: -2.0584745, 0.1806898, -2.0353804, 0.1749334, -1.8492746, 1.8285236
9: 2.4205651, 5.1519318, 2.4253068, 5.1533203, -2.3438196, 2.3390567

Time for backsubstitution: 12.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4639275, upper bound: 1.4543302
time: 5.33 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4639275, upper bound: 1.4543328
time: 5.34 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14.2859592, -10.3241224, -14.3047705, -10.2849884, -2.6806550, 2.6534185
1: -12.4740620, -8.9566889, -12.4879131, -8.9112110, -2.6928134, 2.6598620
2: -13.3930998, -10.1882935, -13.4046383, -10.1189804, -2.6882420, 2.6536431
3: -9.8759270, -6.9074430, -9.9372025, -6.8926458, -2.7808409, 2.8204439
4: -4.5456862, -2.4228191, -4.5617056, -2.3984652, -1.6397719, 1.6443243
5: -11.0608978, -7.3728886, -11.1309471, -7.3638654, -2.6700497, 2.7348826
6: -17.5639420, -13.6347551, -17.6090279, -13.6104612, -2.9902267, 3.0069368
7: -6.4259181, -3.6092825, -6.4401655, -3.5624409, -2.2976365, 2.2661841
8: -2.0288186, 0.1661243, -2.0596266, 0.1774387, -1.8312254, 1.8453531
9: 2.4379845, 5.1455183, 2.4163356, 5.1587849, -2.3309727, 2.3377614

Time for backsubstitution: 12.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4629705, upper bound: 1.4547425
time: 8.13 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4629705, upper bound: 1.4547448
time: 4.98 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -14.3393240, -10.2887726, -14.3092508, -10.2692871, -2.7252469, 2.6806388
1: -12.4890003, -8.9402266, -12.4899311, -8.9044571, -2.7156959, 2.6743269
2: -13.4093208, -10.1695557, -13.4106026, -10.1161499, -2.7070413, 2.6810908
3: -9.8906212, -6.8927360, -9.9420929, -6.8912587, -2.7975564, 2.8342535
4: -4.5583162, -2.4036865, -4.5620141, -2.3911743, -1.6610467, 1.6629137
5: -11.0742025, -7.3651772, -11.1353092, -7.3629179, -2.6934004, 2.7468395
6: -17.6116161, -13.6079102, -17.6121254, -13.5981874, -3.0408235, 3.0286160
7: -6.4338684, -3.6003213, -6.4415865, -3.5597796, -2.3121533, 2.2798126
8: -2.0542879, 0.1819248, -2.0626726, 0.1835117, -1.8562474, 1.8635244
9: 2.4245210, 5.1503634, 2.4116936, 5.1593933, -2.3456285, 2.3476083

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4655335, upper bound: 1.4559341
time: 5.73 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4655310, upper bound: 1.4559335
time: 5.15 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.2893553, -10.2879810, -14.2893572, -10.2879810, -2.6691704, 2.6690784
1: -12.4897871, -8.9402523, -12.4897900, -8.9402504, -2.6680851, 2.6737428
2: -13.3977480, -10.1870403, -13.3977509, -10.1870413, -2.6410174, 2.6402946
3: -9.8871183, -6.9263182, -9.8871193, -6.9263158, -2.7605124, 2.7663455
4: -4.5593214, -2.4029384, -4.5593252, -2.4029372, -1.6388445, 1.6505771
5: -11.0680809, -7.3833776, -11.0680790, -7.3833771, -2.6669488, 2.6649785
6: -17.5683289, -13.6043262, -17.5683327, -13.6043272, -2.9723601, 2.9763422
7: -6.4200597, -3.6006608, -6.4200573, -3.6006565, -2.2561946, 2.2485631
8: -2.0364180, 0.1771660, -2.0364184, 0.1771684, -1.8253379, 1.8221850
9: 2.4229450, 5.1593699, 2.4229431, 5.1593723, -2.3473563, 2.3416035

Time for backsubstitution: 12.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4543337, upper bound: 1.4662631
time: 6.52 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4543337, upper bound: 1.4662634
time: 6.80 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.3110447, -10.2687092, -14.2893572, -10.2879810, -2.6903114, 2.6981378
1: -12.4970131, -8.9031763, -12.4897900, -8.9402504, -2.6780853, 2.7191582
2: -13.4117231, -10.1156006, -13.3977509, -10.1870413, -2.6559405, 2.6953566
3: -9.9435472, -6.8909054, -9.8871193, -6.9263158, -2.8143864, 2.8038192
4: -4.5683608, -2.3898096, -4.5593252, -2.4029372, -1.6486540, 1.6634545
5: -11.1362419, -7.3613496, -11.0680790, -7.3833771, -2.7347727, 2.6867642
6: -17.6162872, -13.5969229, -17.5683327, -13.6043272, -3.0381360, 3.0123897
7: -6.4423943, -3.5563631, -6.4200573, -3.6006565, -2.2792892, 2.3031530
8: -2.0637145, 0.1857529, -2.0364184, 0.1771684, -1.8571434, 1.8318472
9: 2.4093242, 5.1654425, 2.4229431, 5.1593723, -2.3611083, 2.3492429

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4543313, upper bound: 1.4662625
time: 5.03 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4543313, upper bound: 1.4662634
time: 7.31 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.2906132, -10.3198061, -14.3065634, -10.2844181, -2.6856260, 2.6586380
1: -12.4902821, -8.9498062, -12.4949932, -8.9099426, -2.7043180, 2.6735244
2: -13.3974438, -10.1855965, -13.4057522, -10.1184320, -2.6969452, 2.6573339
3: -9.8801517, -6.9058242, -9.9386520, -6.8922915, -2.7854404, 2.8283479
4: -4.5602007, -2.4147861, -4.5680532, -2.3971131, -1.6435118, 1.6578951
5: -11.0646248, -7.3680992, -11.1318779, -7.3622975, -2.6790624, 2.7401259
6: -17.5740223, -13.6286736, -17.6131897, -13.6092014, -2.9982381, 3.0171130
7: -6.4302254, -3.6008024, -6.4409723, -3.5590291, -2.3055010, 2.2678792
8: -2.0340734, 0.1711884, -2.0606651, 0.1796770, -1.8390503, 1.8485701
9: 2.4267769, 5.1590261, 2.4139743, 5.1648369, -2.3482919, 2.3479526

Time for backsubstitution: 12.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4533753, upper bound: 1.4666748
time: 4.61 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4533753, upper bound: 1.4666758
time: 5.37 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.3440609, -10.2844667, -14.3110399, -10.2687225, -2.7303543, 2.6858706
1: -12.5052395, -8.9332819, -12.4970140, -8.9031868, -2.7272134, 2.6880364
2: -13.4136257, -10.1668453, -13.4117165, -10.1156025, -2.7157717, 2.6847830
3: -9.8948441, -6.8911180, -9.9435406, -6.8909054, -2.8021574, 2.8421586
4: -4.5728321, -2.3956451, -4.5683641, -2.3898242, -1.6647851, 1.6764867
5: -11.0779285, -7.3603935, -11.1362391, -7.3613505, -2.7023954, 2.7520795
6: -17.6217308, -13.6018181, -17.6162872, -13.5969296, -3.0485320, 3.0387952
7: -6.4381742, -3.5918007, -6.4423909, -3.5563645, -2.3200336, 2.2815681
8: -2.0595136, 0.1869884, -2.0637121, 0.1857476, -1.8641310, 1.8667352
9: 2.4132409, 5.1638784, 2.4093285, 5.1654449, -2.3629017, 2.3578036

Time for backsubstitution: 12.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4559367, upper bound: 1.4678661
time: 5.39 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4559342, upper bound: 1.4678654
time: 5.13 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.63 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4639275, upper bound: 1.4543329
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4639275, upper bound: 1.4543297
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4639275, upper bound: 1.4543302
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4639275, upper bound: 1.4543328
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4629705, upper bound: 1.4547425
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4629705, upper bound: 1.4547448
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4655335, upper bound: 1.4559341
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4655310, upper bound: 1.4559335
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4543337, upper bound: 1.4662631
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4543337, upper bound: 1.4662634
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4543313, upper bound: 1.4662625
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4543313, upper bound: 1.4662634
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4533753, upper bound: 1.4666748
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4533753, upper bound: 1.4666758
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4559367, upper bound: 1.4678661
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.63
Output dim: 9, lower bound: -1.4559342, upper bound: 1.4678654

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14.2846308, -10.2922974, -14.2846308, -10.2922974, -2.6607008, 2.6607018
1: -12.4735651, -8.9471931, -12.4735651, -8.9471931, -2.6508985, 2.6508989
2: -13.3935213, -10.1897430, -13.3935213, -10.1897430, -2.6311903, 2.6311903
3: -9.8828888, -6.9279327, -9.8828888, -6.9279327, -2.7522402, 2.7522402
4: -4.5448055, -2.4109602, -4.5448055, -2.4109602, -1.6285644, 1.6285646
5: -11.0643578, -7.3881693, -11.0643578, -7.3881693, -2.6556883, 2.6556888
6: -17.5582485, -13.6104107, -17.5582485, -13.6104107, -2.9594717, 2.9594712
7: -6.4157553, -3.6091747, -6.4157553, -3.6091747, -2.2427373, 2.2427375
8: -2.0311818, 0.1721077, -2.0311818, 0.1721077, -1.8145733, 1.8145735
9: 2.4341812, 5.1458588, 2.4341812, 5.1458588, -2.3225825, 2.3225830

Time for backsubstitution: 12.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4613541, upper bound: 1.4531328
time: 5.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4639230, upper bound: 1.4543270
time: 4.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.2846308, -10.2922974, -14.2893553, -10.2879810, -2.6648793, 2.6648889
1: -12.4735651, -8.9471931, -12.4897871, -8.9402523, -2.6576004, 2.6670370
2: -13.3935213, -10.1897430, -13.3977480, -10.1870403, -2.6340656, 2.6363091
3: -9.8828888, -6.9279327, -9.8871183, -6.9263182, -2.7550020, 2.7567015
4: -4.5448055, -2.4109602, -4.5593214, -2.4029384, -1.6362114, 1.6429265
5: -11.0643578, -7.3881693, -11.0680809, -7.3833776, -2.6604795, 2.6598339
6: -17.5582485, -13.6104107, -17.5683289, -13.6043262, -2.9656353, 2.9701753
7: -6.4157553, -3.6091747, -6.4200597, -3.6006608, -2.2515326, 2.2473950
8: -2.0311818, 0.1721077, -2.0364180, 0.1771660, -1.8198586, 1.8200498
9: 2.4341812, 5.1458588, 2.4229450, 5.1593699, -2.3361402, 2.3337958

Time for backsubstitution: 12.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4613542, upper bound: 1.4531328
time: 5.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4639230, upper bound: 1.4543267
time: 4.96 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.3062782, -10.2730274, -14.2846308, -10.2922974, -2.6818361, 2.6897597
1: -12.4807825, -8.9101238, -12.4735651, -8.9471931, -2.6608996, 2.6963239
2: -13.4075050, -10.1182690, -13.3935213, -10.1897430, -2.6461143, 2.6868024
3: -9.9393272, -6.8925219, -9.8828888, -6.9279327, -2.8058920, 2.7897091
4: -4.5538464, -2.3978233, -4.5448055, -2.4109602, -1.6383743, 1.6414435
5: -11.1325302, -7.3661418, -11.0643578, -7.3881693, -2.7251716, 2.6774759
6: -17.6061974, -13.6030016, -17.5582485, -13.6104107, -3.0252304, 2.9955201
7: -6.4380989, -3.5648832, -6.4157553, -3.6091747, -2.2658396, 2.2933815
8: -2.0584745, 0.1806898, -2.0311818, 0.1721077, -1.8476954, 1.8242323
9: 2.4205651, 5.1519318, 2.4341812, 5.1458588, -2.3363354, 2.3302219

Time for backsubstitution: 12.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4629727, upper bound: 1.4531324
time: 5.14 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4655333, upper bound: 1.4543264
time: 5.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.3062782, -10.2730274, -14.2893553, -10.2879810, -2.6860147, 2.6939468
1: -12.4807825, -8.9101238, -12.4897871, -8.9402523, -2.6676025, 2.7124615
2: -13.4075050, -10.1182690, -13.3977480, -10.1870403, -2.6489906, 2.6916633
3: -9.9393272, -6.8925219, -9.8871183, -6.9263182, -2.8082771, 2.7941699
4: -4.5538464, -2.3978233, -4.5593214, -2.4029384, -1.6460214, 1.6558056
5: -11.1325302, -7.3661418, -11.0680809, -7.3833776, -2.7284031, 2.6816206
6: -17.6061974, -13.6030016, -17.5683289, -13.6043262, -3.0313950, 3.0062242
7: -6.4380989, -3.5648832, -6.4200597, -3.6006608, -2.2746353, 2.2962508
8: -2.0584745, 0.1806898, -2.0364180, 0.1771660, -1.8497367, 1.8297091
9: 2.4205651, 5.1519318, 2.4229450, 5.1593699, -2.3498931, 2.3414342

Time for backsubstitution: 12.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4629704, upper bound: 1.4531348
time: 4.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4655333, upper bound: 1.4543259
time: 6.21 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.2859592, -10.3241224, -14.3018370, -10.2887402, -2.6771426, 2.6502595
1: -12.4740620, -8.9566889, -12.4787626, -8.9168768, -2.6871314, 2.6506658
2: -13.3930998, -10.1882935, -13.4014864, -10.1210995, -2.6851001, 2.6482234
3: -9.8759270, -6.9074430, -9.9344311, -6.8939118, -2.7771711, 2.8159289
4: -4.5456862, -2.4228191, -4.5535364, -2.4051352, -1.6332254, 1.6358705
5: -11.0608978, -7.3728886, -11.1281624, -7.3670878, -2.6678066, 2.7311749
6: -17.5639420, -13.6347551, -17.6030998, -13.6152840, -2.9853125, 3.0002232
7: -6.4259181, -3.6092825, -6.4366765, -3.5675378, -2.2939072, 2.2621043
8: -2.0288186, 0.1661243, -2.0554209, 0.1746101, -1.8282633, 1.8410475
9: 2.4379845, 5.1455183, 2.4252024, 5.1513238, -2.3234882, 2.3289189

Time for backsubstitution: 12.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4613533, upper bound: 1.4547422
time: 4.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4613533, upper bound: 1.4531348
time: 5.04 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.2859592, -10.3241224, -14.3065586, -10.2844219, -2.6813231, 2.6544309
1: -12.4740620, -8.9566889, -12.4949884, -8.9099426, -2.6938319, 2.6668043
2: -13.3930998, -10.1882935, -13.4057484, -10.1184349, -2.6875744, 2.6533427
3: -9.8759270, -6.9074430, -9.9386501, -6.8922920, -2.7799311, 2.8207750
4: -4.5456862, -2.4228191, -4.5680523, -2.3971157, -1.6408772, 1.6462220
5: -11.0608978, -7.3728886, -11.1318808, -7.3622990, -2.6725941, 2.7341166
6: -17.5639420, -13.6347551, -17.6131859, -13.6092024, -2.9914742, 3.0086997
7: -6.4259181, -3.6092825, -6.4409719, -3.5590324, -2.2990403, 2.2667625
8: -2.0288186, 0.1661243, -2.0606656, 0.1796756, -1.8335514, 1.8452110
9: 2.4379845, 5.1455183, 2.4139752, 5.1648335, -2.3370457, 2.3401408

Time for backsubstitution: 13.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4613533, upper bound: 1.4547422
time: 5.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4613557, upper bound: 1.4531324
time: 6.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.3393240, -10.2887726, -14.3062763, -10.2730389, -2.7217340, 2.6774850
1: -12.4890003, -8.9402266, -12.4807806, -8.9101324, -2.7100191, 2.6651297
2: -13.4093208, -10.1695557, -13.4074984, -10.1182709, -2.7039013, 2.6756716
3: -9.8906212, -6.8927360, -9.9393196, -6.8925242, -2.7938876, 2.8297412
4: -4.5583162, -2.4036865, -4.5538464, -2.3978391, -1.6545036, 1.6552932
5: -11.0742025, -7.3651772, -11.1325235, -7.3661423, -2.6911569, 2.7431333
6: -17.6116161, -13.6079102, -17.6061954, -13.6030130, -3.0358887, 3.0219045
7: -6.4338684, -3.6003213, -6.4380965, -3.5648885, -2.3084176, 2.2757339
8: -2.0542879, 0.1819248, -2.0584712, 0.1806841, -1.8546672, 1.8592250
9: 2.4245210, 5.1503634, 2.4205723, 5.1519308, -2.3381443, 2.3387752

Time for backsubstitution: 12.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4639222, upper bound: 1.4559340
time: 5.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4639246, upper bound: 1.4543257
time: 5.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.3393240, -10.2887726, -14.3110390, -10.2687225, -2.7254391, 2.6816998
1: -12.4890003, -8.9402266, -12.4970112, -8.9031868, -2.7167120, 2.6812668
2: -13.4093208, -10.1695557, -13.4117146, -10.1156044, -2.7063751, 2.6807914
3: -9.8906212, -6.8927360, -9.9435387, -6.8909063, -2.7966480, 2.8345842
4: -4.5583162, -2.4036865, -4.5683608, -2.3898239, -1.6621497, 1.6638494
5: -11.0742025, -7.3651772, -11.1362362, -7.3613544, -2.6959438, 2.7460754
6: -17.6116161, -13.6079102, -17.6162834, -13.5969315, -3.0399685, 3.0305555
7: -6.4338684, -3.6003213, -6.4423909, -3.5563660, -2.3135581, 2.2803905
8: -2.0542879, 0.1819248, -2.0637112, 0.1857471, -1.8567080, 1.8633809
9: 2.4245210, 5.1503634, 2.4093299, 5.1654415, -2.3517015, 2.3499818

Time for backsubstitution: 12.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4639222, upper bound: 1.4559336
time: 4.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4639222, upper bound: 1.4543257
time: 5.19 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14.2893553, -10.2879810, -14.2846308, -10.2922974, -2.6648893, 2.6648793
1: -12.4897871, -8.9402523, -12.4735651, -8.9471931, -2.6670365, 2.6576009
2: -13.3977480, -10.1870403, -13.3935213, -10.1897430, -2.6363087, 2.6340661
3: -9.8871183, -6.9263182, -9.8828888, -6.9279327, -2.7567015, 2.7550015
4: -4.5593214, -2.4029384, -4.5448055, -2.4109602, -1.6429267, 1.6362114
5: -11.0680809, -7.3833776, -11.0643578, -7.3881693, -2.6598339, 2.6604800
6: -17.5683289, -13.6043262, -17.5582485, -13.6104107, -2.9701757, 2.9656358
7: -6.4200597, -3.6006608, -6.4157553, -3.6091747, -2.2473955, 2.2515328
8: -2.0364180, 0.1771660, -2.0311818, 0.1721077, -1.8200498, 1.8198593
9: 2.4229450, 5.1593699, 2.4341812, 5.1458588, -2.3337953, 2.3361402

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4517606, upper bound: 1.4650669
time: 4.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4543269, upper bound: 1.4662589
time: 6.07 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.2893553, -10.2879810, -14.2893553, -10.2879810, -2.6690459, 2.6690459
1: -12.4897871, -8.9402523, -12.4897871, -8.9402523, -2.6680846, 2.6680841
2: -13.3977480, -10.1870403, -13.3977480, -10.1870403, -2.6402240, 2.6402240
3: -9.8871183, -6.9263182, -9.8871183, -6.9263182, -2.7663441, 2.7663441
4: -4.5593214, -2.4029384, -4.5593214, -2.4029384, -1.6388435, 1.6388440
5: -11.0680809, -7.3833776, -11.0680809, -7.3833776, -2.6669474, 2.6669474
6: -17.5683289, -13.6043262, -17.5683289, -13.6043262, -2.9723597, 2.9723597
7: -6.4200597, -3.6006608, -6.4200597, -3.6006608, -2.2482967, 2.2482972
8: -2.0364180, 0.1771660, -2.0364180, 0.1771660, -1.8221841, 1.8221841
9: 2.4229450, 5.1593699, 2.4229450, 5.1593699, -2.3416021, 2.3416018

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4517630, upper bound: 1.4650659
time: 5.94 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4543269, upper bound: 1.4662597
time: 7.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.3110447, -10.2687092, -14.2846308, -10.2922974, -2.6860628, 2.6939392
1: -12.4970131, -8.9031763, -12.4735651, -8.9471931, -2.6770377, 2.7030163
2: -13.4117231, -10.1156006, -13.3935213, -10.1897430, -2.6512327, 2.6892772
3: -9.9435472, -6.8909054, -9.8828888, -6.9279327, -2.8107352, 2.7924752
4: -4.5683608, -2.3898096, -4.5448055, -2.4109602, -1.6527352, 1.6490891
5: -11.1362419, -7.3613496, -11.0643578, -7.3881693, -2.7281132, 2.6822662
6: -17.6162872, -13.5969229, -17.5582485, -13.6104107, -3.0359535, 3.0016828
7: -6.4423943, -3.5563631, -6.4157553, -3.6091747, -2.2704897, 2.2985222
8: -2.0637145, 0.1857529, -2.0311818, 0.1721077, -1.8518505, 1.8295214
9: 2.4093242, 5.1654425, 2.4341812, 5.1458588, -2.3475473, 2.3437793

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4533751, upper bound: 1.4650646
time: 4.67 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4559341, upper bound: 1.4662583
time: 4.69 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.3110447, -10.2687092, -14.2893553, -10.2879810, -2.6901870, 2.6981053
1: -12.4970131, -8.9031763, -12.4897871, -8.9402523, -2.6780849, 2.7135005
2: -13.4117231, -10.1156006, -13.3977480, -10.1870403, -2.6551471, 2.6986544
3: -9.9435472, -6.8909054, -9.8871183, -6.9263182, -2.8183055, 2.8038177
4: -4.5683608, -2.3898096, -4.5593214, -2.4029384, -1.6486530, 1.6517217
5: -11.1362419, -7.3613496, -11.0680809, -7.3833776, -2.7347722, 2.6887341
6: -17.6162872, -13.5969229, -17.5683289, -13.6043262, -3.0381355, 3.0084062
7: -6.4423943, -3.5563631, -6.4200597, -3.6006608, -2.2713914, 2.3031526
8: -2.0637145, 0.1857529, -2.0364180, 0.1771660, -1.8548412, 1.8318460
9: 2.4093242, 5.1654425, 2.4229450, 5.1593699, -2.3553531, 2.3492413

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4533775, upper bound: 1.4650659
time: 5.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4559341, upper bound: 1.4662586
time: 5.09 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.2906132, -10.3198061, -14.3018370, -10.2887402, -2.6812491, 2.6544371
1: -12.4902821, -8.9498062, -12.4787626, -8.9168768, -2.7032723, 2.6573815
2: -13.3974438, -10.1855965, -13.4014864, -10.1210995, -2.6899524, 2.6511049
3: -9.8801517, -6.9058242, -9.9344311, -6.8939118, -2.7816324, 2.8183186
4: -4.5602007, -2.4147861, -4.5535364, -2.4051352, -1.6475883, 1.6435292
5: -11.0646248, -7.3680992, -11.1281624, -7.3670878, -2.6719522, 2.7344136
6: -17.5740223, -13.6286736, -17.6030998, -13.6152840, -2.9960527, 3.0063851
7: -6.4302254, -3.6008024, -6.4366765, -3.5675378, -2.2967763, 2.2708468
8: -2.0340734, 0.1711884, -2.0554209, 0.1746101, -1.8337607, 1.8430903
9: 2.4267769, 5.1590261, 2.4252024, 5.1513238, -2.3347311, 2.3424807

Time for backsubstitution: 12.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4517598, upper bound: 1.4666748
time: 5.03 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4517598, upper bound: 1.4650649
time: 5.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.2906132, -10.3198061, -14.3065586, -10.2844219, -2.6855021, 2.6586061
1: -12.4902821, -8.9498062, -12.4949884, -8.9099426, -2.7043171, 2.6678658
2: -13.3974438, -10.1855965, -13.4057484, -10.1184349, -2.6969433, 2.6572628
3: -9.8801517, -6.9058242, -9.9386501, -6.8922920, -2.7912731, 2.8283470
4: -4.5602007, -2.4147861, -4.5680523, -2.3971157, -1.6435113, 1.6461613
5: -11.0646248, -7.3680992, -11.1318808, -7.3622990, -2.6790609, 2.7407751
6: -17.5740223, -13.6286736, -17.6131859, -13.6092024, -2.9982352, 3.0131297
7: -6.4302254, -3.6008024, -6.4409719, -3.5590324, -2.3036737, 2.2676113
8: -2.0340734, 0.1711884, -2.0606656, 0.1796756, -1.8358965, 1.8482046
9: 2.4267769, 5.1590261, 2.4139752, 5.1648335, -2.3425376, 2.3479507

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4517598, upper bound: 1.4666750
time: 7.50 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4517598, upper bound: 1.4650678
time: 4.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.3440609, -10.2844667, -14.3062763, -10.2730389, -2.7257719, 2.6816540
1: -12.5052395, -8.9332819, -12.4807806, -8.9101324, -2.7251301, 2.6718950
2: -13.4136257, -10.1668453, -13.4074984, -10.1182709, -2.7087803, 2.6785545
3: -9.8948441, -6.8911180, -9.9393196, -6.8925242, -2.7983470, 2.8321290
4: -4.5728321, -2.3956451, -4.5538464, -2.3978391, -1.6641879, 1.6620911
5: -11.0779285, -7.3603935, -11.1325235, -7.3661423, -2.6952844, 2.7463765
6: -17.6217308, -13.6018181, -17.6061954, -13.6030130, -3.0418224, 3.0280712
7: -6.4381742, -3.5918007, -6.4380965, -3.5648885, -2.3112919, 2.2845366
8: -2.0595136, 0.1869884, -2.0584712, 0.1806841, -1.8588362, 1.8612604
9: 2.4132409, 5.1638784, 2.4205723, 5.1519308, -2.3493404, 2.3523452

Time for backsubstitution: 12.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4543261, upper bound: 1.4678657
time: 5.17 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4543261, upper bound: 1.4662604
time: 4.96 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.3440609, -10.2844667, -14.3110390, -10.2687225, -2.7303543, 2.6858144
1: -12.5052395, -8.9332819, -12.4970112, -8.9031868, -2.7272124, 2.6823764
2: -13.4136257, -10.1668453, -13.4117146, -10.1156044, -2.7157707, 2.6847129
3: -9.8948441, -6.8911180, -9.9435387, -6.8909063, -2.8079910, 2.8421574
4: -4.5728321, -2.3956451, -4.5683608, -2.3898239, -1.6647847, 1.6655648
5: -11.0779285, -7.3603935, -11.1362362, -7.3613544, -2.7023931, 2.7527299
6: -17.6217308, -13.6018181, -17.6162834, -13.5969315, -3.0477371, 3.0348125
7: -6.4381742, -3.5918007, -6.4423909, -3.5563660, -2.3181968, 2.2812998
8: -2.0595136, 0.1869884, -2.0637112, 0.1857471, -1.8618283, 1.8663721
9: 2.4132409, 5.1638784, 2.4093299, 5.1654415, -2.3571472, 2.3578012

Time for backsubstitution: 12.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4543261, upper bound: 1.4678666
time: 5.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4517598, upper bound: 1.4662586
time: 6.61 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.67 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4613541, upper bound: 1.4531328
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4639230, upper bound: 1.4543270
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4613542, upper bound: 1.4531328
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4639230, upper bound: 1.4543267
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4629727, upper bound: 1.4531324
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4655333, upper bound: 1.4543264
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4629704, upper bound: 1.4531348
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4655333, upper bound: 1.4543259
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4613533, upper bound: 1.4547422
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4613533, upper bound: 1.4531348
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4613533, upper bound: 1.4547422
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4613557, upper bound: 1.4531324
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4639222, upper bound: 1.4559340
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4639246, upper bound: 1.4543257
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4639222, upper bound: 1.4559336
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4639222, upper bound: 1.4543257
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4517606, upper bound: 1.4650669
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4543269, upper bound: 1.4662589
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4517630, upper bound: 1.4650659
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4543269, upper bound: 1.4662597
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4533751, upper bound: 1.4650646
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4559341, upper bound: 1.4662583
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4533775, upper bound: 1.4650659
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4559341, upper bound: 1.4662586
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4517598, upper bound: 1.4666748
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4517598, upper bound: 1.4650649
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4517598, upper bound: 1.4666750
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4517598, upper bound: 1.4650678
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4543261, upper bound: 1.4678657
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4543261, upper bound: 1.4662604
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4543261, upper bound: 1.4678666
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.67
Output dim: 9, lower bound: -1.4517598, upper bound: 1.4662586

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14.2752590, -10.3249445, -14.2801571, -10.3080082, -2.6372447, 2.6238780
1: -12.4692907, -8.9608955, -12.4715319, -8.9538136, -2.6360002, 2.6321263
2: -13.3810949, -10.1954117, -13.3874969, -10.1924477, -2.6118999, 2.6147799
3: -9.8727856, -6.9312177, -9.8780127, -6.9294958, -2.7393932, 2.7427397
4: -4.5442271, -2.4259353, -4.5445294, -2.4181895, -1.6142220, 1.6081123
5: -11.0555973, -7.3901548, -11.0601215, -7.3891191, -2.6382694, 2.6407523
6: -17.5522270, -13.6359434, -17.5553589, -13.6226931, -2.9352651, 2.9261346
7: -6.4127755, -3.6144905, -6.4143095, -3.6117439, -2.2333083, 2.2325110
8: -2.0253696, 0.1595144, -2.0283542, 0.1660280, -1.8009696, 1.7974792
9: 2.4437881, 5.1446662, 2.4388199, 5.1452837, -2.3094263, 2.3123779

Time for backsubstitution: 12.86 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.3557186126708984
rel_dist={9: [-1.46816030629895, 1.4681598898989336]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597462, upper bound: 1.3509751
time: 5.44 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597462, upper bound: 1.3597476
time: 4.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.33 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.33
Output dim: 9, lower bound: -1.3597462, upper bound: 1.3509751
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.33
Output dim: 9, lower bound: -1.3597462, upper bound: 1.3597476

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -14.2953281, -10.2914829, -14.2979240, -10.2878437, -2.5280714, 2.5276504
1: -12.4783459, -8.9430885, -12.4861088, -8.9376755, -2.5456743, 2.5480328
2: -13.4055109, -10.1823139, -13.4084244, -10.1802902, -2.5397305, 2.5419393
3: -9.8860092, -6.9041572, -9.8885021, -6.9029617, -2.7091703, 2.7057734
4: -4.5463209, -2.4078240, -4.5532451, -2.4014206, -1.5764737, 1.5774472
5: -11.0696669, -7.3708925, -11.0722733, -7.3679771, -2.5535975, 2.5553641
6: -17.5701237, -13.6092243, -17.5752392, -13.6046543, -2.8529124, 2.8541684
7: -6.4289093, -3.6039596, -6.4322467, -3.5995255, -2.1809988, 2.1818244
8: -2.0346689, 0.1787167, -2.0386639, 0.1811042, -1.7652006, 1.7667544
9: 2.4283915, 5.1467175, 2.4199858, 5.1529927, -2.2814744, 2.2835472

Time for backsubstitution: 12.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597447, upper bound: 1.3495662
time: 5.67 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597447, upper bound: 1.3509737
time: 5.15 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -14.3000641, -10.2871685, -14.3000660, -10.2871685, -2.5330462, 2.5331044
1: -12.4945726, -8.9361610, -12.4945736, -8.9361629, -2.5570984, 2.5630393
2: -13.4097595, -10.1796112, -13.4097605, -10.1796150, -2.5461426, 2.5456128
3: -9.8902349, -6.9025426, -9.8902369, -6.9025412, -2.7136326, 2.7194467
4: -4.5608354, -2.3998003, -4.5608387, -2.3997991, -1.5798416, 1.5921590
5: -11.0733891, -7.3661041, -11.0733900, -7.3661013, -2.5630207, 2.5610590
6: -17.5802155, -13.6031475, -17.5802174, -13.6031475, -2.8609428, 2.8651233
7: -6.4332161, -3.5954437, -6.4332156, -3.5954423, -2.1913686, 2.1832888
8: -2.0399027, 0.1837783, -2.0399032, 0.1837788, -1.7734632, 1.7701540
9: 2.4171572, 5.1602278, 2.4171562, 5.1602306, -2.2999542, 2.2939141

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597447, upper bound: 1.3583398
time: 4.64 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597447, upper bound: 1.3597460
time: 4.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.37 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.37
Output dim: 9, lower bound: -1.3597447, upper bound: 1.3495662
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.37
Output dim: 9, lower bound: -1.3597447, upper bound: 1.3509737
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.37
Output dim: 9, lower bound: -1.3597447, upper bound: 1.3583398
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.37
Output dim: 9, lower bound: -1.3597447, upper bound: 1.3597460

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -14.2893953, -10.2919321, -14.2872181, -10.2886581, -2.5196366, 2.5152988
1: -12.4757156, -8.9453630, -12.4813271, -8.9417696, -2.5395727, 2.5406790
2: -13.3989172, -10.1862259, -13.3964128, -10.1877174, -2.5227575, 2.5201926
3: -9.8842344, -6.9174528, -9.8853836, -6.9267378, -2.6875553, 2.6915274
4: -4.5455284, -2.4095387, -4.5517302, -2.4045596, -1.5661697, 1.5674250
5: -11.0668669, -7.3804235, -11.0669594, -7.3852549, -2.5288754, 2.5377860
6: -17.5637512, -13.6098804, -17.5633545, -13.6058350, -2.8422256, 2.8388958
7: -6.4216723, -3.6067820, -6.4190903, -3.6047416, -2.1696630, 2.1687367
8: -2.0327234, 0.1750832, -2.0351763, 0.1744943, -1.7523632, 1.7568526
9: 2.4315786, 5.1462564, 2.4257746, 5.1521349, -2.2760301, 2.2760296

Time for backsubstitution: 12.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3574718, upper bound: 1.3482328
time: 5.54 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597406, upper bound: 1.3495621
time: 5.55 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -14.2953196, -10.2914867, -14.3089066, -10.2693863, -2.5547543, 2.5400777
1: -12.4783382, -8.9430914, -12.4885464, -8.9046974, -2.5880814, 2.5578341
2: -13.4055052, -10.1823177, -13.4103899, -10.1162577, -2.5867310, 2.5483470
3: -9.8860073, -6.9041772, -9.9418182, -6.8913245, -2.7228031, 2.7581522
4: -4.5463204, -2.4078269, -4.5607700, -2.3914261, -1.5824194, 1.5915098
5: -11.0696640, -7.3709116, -11.1351318, -7.3632221, -2.5537319, 2.6165998
6: -17.5701180, -13.6092281, -17.6113129, -13.5984268, -2.8856850, 2.9095533
7: -6.4288898, -3.6039629, -6.4414282, -3.5604458, -2.2246950, 2.1948650
8: -2.0346680, 0.1787071, -2.0624733, 0.1830788, -1.7768373, 1.7942147
9: 2.4283948, 5.1467161, 2.4121542, 5.1582088, -2.2872126, 2.2913346

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3574718, upper bound: 1.3496336
time: 5.47 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597406, upper bound: 1.3509697
time: 4.59 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -14.2941303, -10.2876167, -14.2893553, -10.2879810, -2.5246148, 2.5207400
1: -12.4919367, -8.9384365, -12.4897919, -8.9402494, -2.5509963, 2.5556846
2: -13.4031525, -10.1835251, -13.3977489, -10.1870403, -2.5291653, 2.5238652
3: -9.8884621, -6.9158382, -9.8871183, -6.9263158, -2.6920161, 2.7051978
4: -4.5600443, -2.4015167, -4.5593247, -2.4029377, -1.5695369, 1.5821397
5: -11.0705910, -7.3756332, -11.0680790, -7.3833771, -2.5383024, 2.5434852
6: -17.5738373, -13.6037989, -17.5683327, -13.6043291, -2.8502707, 2.8498507
7: -6.4259748, -3.5982671, -6.4200592, -3.6006579, -2.1800308, 2.1701996
8: -2.0379601, 0.1801429, -2.0364180, 0.1771674, -1.7606235, 1.7602496
9: 2.4203444, 5.1597667, 2.4229431, 5.1593709, -2.2945111, 2.2863970

Time for backsubstitution: 12.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3574741, upper bound: 1.3570042
time: 4.93 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597406, upper bound: 1.3583341
time: 5.21 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -14.3000584, -10.2871685, -14.3110476, -10.2687073, -2.5597348, 2.5455480
1: -12.4945641, -8.9361649, -12.4970160, -8.9031754, -2.5995007, 2.5728393
2: -13.4097538, -10.1796169, -13.4117241, -10.1155977, -2.5953283, 2.5520177
3: -9.8902330, -6.9025602, -9.9435463, -6.8909044, -2.7272601, 2.7661374
4: -4.5608363, -2.3998034, -4.5683641, -2.3898098, -1.5857852, 1.6062231
5: -11.0733852, -7.3661213, -11.1362438, -7.3613501, -2.5631533, 2.6217098
6: -17.5802078, -13.6031475, -17.6162910, -13.5969219, -2.8937254, 2.9202716
7: -6.4331932, -3.5954480, -6.4423928, -3.5563607, -2.2330155, 2.1963365
8: -2.0399008, 0.1837683, -2.0637150, 0.1857538, -1.7851000, 1.7974129
9: 2.4171610, 5.1602263, 2.4093232, 5.1654449, -2.3056910, 2.3016975

Time for backsubstitution: 12.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3574718, upper bound: 1.3584070
time: 4.74 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597406, upper bound: 1.3597420
time: 4.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.36 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 9, lower bound: -1.3574718, upper bound: 1.3482328
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 9, lower bound: -1.3597406, upper bound: 1.3495621
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 9, lower bound: -1.3574718, upper bound: 1.3496336
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 9, lower bound: -1.3597406, upper bound: 1.3509697
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 9, lower bound: -1.3574741, upper bound: 1.3570042
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 9, lower bound: -1.3597406, upper bound: 1.3583341
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 9, lower bound: -1.3574718, upper bound: 1.3584070
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 9, lower bound: -1.3597406, upper bound: 1.3597420

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14.2800303, -10.3245773, -14.2817593, -10.3078213, -2.4925919, 2.4776878
1: -12.4714375, -8.9589462, -12.4788408, -8.9498615, -2.5230870, 2.5209770
2: -13.3864994, -10.1920338, -13.3891401, -10.1910238, -2.5024843, 2.5024471
3: -9.8741417, -6.9207373, -9.8794422, -6.9286475, -2.6742640, 2.6808896
4: -4.5449224, -2.4245210, -4.5513921, -2.4133654, -1.5499654, 1.5463622
5: -11.0581112, -7.3824081, -11.0618057, -7.3864141, -2.5104408, 2.5214467
6: -17.5575600, -13.6354132, -17.5598240, -13.6208191, -2.8153019, 2.8044786
7: -6.4186940, -3.6120954, -6.4173303, -3.6078835, -2.1594348, 2.1579471
8: -2.0269094, 0.1624942, -2.0317454, 0.1670866, -1.7373400, 1.7390437
9: 2.4411802, 5.1450605, 2.4314313, 5.1514344, -2.2623320, 2.2647171

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3557161, upper bound: 1.3482328
time: 5.34 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3557161, upper bound: 1.3482331
time: 5.61 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -14.3333702, -10.2892113, -14.2872143, -10.2886791, -2.5582929, 2.5037236
1: -12.4863567, -8.9424257, -12.4813242, -8.9417820, -2.5474796, 2.5352983
2: -13.4027300, -10.1735010, -13.3964005, -10.1877213, -2.5214686, 2.5311913
3: -9.8888168, -6.9061785, -9.8853741, -6.9267378, -2.6913137, 2.7017183
4: -4.5575213, -2.4053650, -4.5517287, -2.4045753, -1.5729928, 1.5651383
5: -11.0715122, -7.3747225, -11.0669537, -7.3852549, -2.5337496, 2.5364361
6: -17.6050587, -13.6085634, -17.5633507, -13.6058483, -2.8795810, 2.8250699
7: -6.4266596, -3.6030993, -6.4190893, -3.6047449, -2.1746473, 2.1722016
8: -2.0522833, 0.1782990, -2.0351710, 0.1744871, -1.7702684, 1.7575078
9: 2.4277067, 5.1499019, 2.4257817, 5.1521349, -2.2770290, 2.2756753

Time for backsubstitution: 13.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3579839, upper bound: 1.3495647
time: 6.19 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3579839, upper bound: 1.3495624
time: 5.10 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14.2859545, -10.3241243, -14.3034477, -10.2885532, -2.5275059, 2.5023012
1: -12.4740582, -8.9566898, -12.4860764, -8.9129429, -2.5715647, 2.5379248
2: -13.3930969, -10.1882935, -13.4031200, -10.1197166, -2.5661926, 2.5305467
3: -9.8759251, -6.9074459, -9.9358501, -6.8930225, -2.7094402, 2.7469435
4: -4.5456853, -2.4228199, -4.5603924, -2.4003334, -1.5661385, 1.5704300
5: -11.0608969, -7.3728876, -11.1298094, -7.3643799, -2.5351467, 2.5996099
6: -17.5639400, -13.6347542, -17.6075306, -13.6134090, -2.8589964, 2.8748693
7: -6.4259157, -3.6092830, -6.4396992, -3.5636902, -2.2144954, 2.1840281
8: -2.0288205, 0.1661224, -2.0587997, 0.1756721, -1.7617846, 1.7762630
9: 2.4379845, 5.1455183, 2.4178171, 5.1574664, -2.2734823, 2.2800241

Time for backsubstitution: 12.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3557161, upper bound: 1.3496337
time: 5.05 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3557161, upper bound: 1.3496335
time: 5.72 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -14.3393211, -10.2887726, -14.3088942, -10.2694025, -2.5731401, 2.5284481
1: -12.4890003, -8.9402256, -12.4885454, -8.9047089, -2.5960436, 2.5522890
2: -13.4093227, -10.1695557, -13.4103785, -10.1162596, -2.5853186, 2.5595651
3: -9.8906221, -6.8927383, -9.9418087, -6.8913279, -2.7265787, 2.7618272
4: -4.5583167, -2.4036875, -4.5607691, -2.3914425, -1.5892887, 1.5876670
5: -11.0742025, -7.3651810, -11.1351242, -7.3632245, -2.5586677, 2.6128230
6: -17.6116161, -13.6079130, -17.6113110, -13.5984373, -2.9099798, 2.8957496
7: -6.4338665, -3.6003227, -6.4414272, -3.5604496, -2.2297764, 2.1982558
8: -2.0542879, 0.1819239, -2.0624690, 0.1830730, -1.7869239, 1.7948070
9: 2.4245205, 5.1503639, 2.4121614, 5.1582079, -2.2882140, 2.2910042

Time for backsubstitution: 13.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3579839, upper bound: 1.3509718
time: 5.26 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3579839, upper bound: 1.3509694
time: 5.58 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.2846737, -10.3202562, -14.2838421, -10.3071423, -2.4975863, 2.4830713
1: -12.4876566, -8.9520683, -12.4873037, -8.9483461, -2.5345144, 2.5359993
2: -13.3908339, -10.1893387, -13.3904715, -10.1903477, -2.5088921, 2.5061159
3: -9.8783684, -6.9191160, -9.8811779, -6.9282260, -2.6787233, 2.6945620
4: -4.5594406, -2.4164891, -4.5589876, -2.4117429, -1.5533383, 1.5610898
5: -11.0618382, -7.3776197, -11.0629234, -7.3845363, -2.5198746, 2.5271468
6: -17.5676384, -13.6293325, -17.5647926, -13.6193142, -2.8233490, 2.8154378
7: -6.4230013, -3.6036153, -6.4183011, -3.6038098, -2.1698046, 2.1593719
8: -2.0321631, 0.1675549, -2.0329928, 0.1697617, -1.7456183, 1.7424481
9: 2.4299717, 5.1585703, 2.4286098, 5.1586723, -2.2808402, 2.2750909

Time for backsubstitution: 12.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3487030, upper bound: 1.3570041
time: 6.69 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3487030, upper bound: 1.3570045
time: 4.76 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.3380976, -10.2849026, -14.2893486, -10.2880001, -2.5634451, 2.5091572
1: -12.5025940, -8.9354858, -12.4897890, -8.9402628, -2.5589166, 2.5503697
2: -13.4070225, -10.1707916, -13.3977394, -10.1870441, -2.5278955, 2.5348644
3: -9.8930416, -6.9045639, -9.8871088, -6.9263186, -2.6957726, 2.7153873
4: -4.5720391, -2.3973236, -4.5593243, -2.4029539, -1.5763626, 1.5798466
5: -11.0752392, -7.3699408, -11.0680742, -7.3833790, -2.5431662, 2.5421324
6: -17.6151657, -13.6024723, -17.5683250, -13.6043396, -2.8876657, 2.8360367
7: -6.4309645, -3.5945778, -6.4200583, -3.6006622, -2.1850195, 2.1736760
8: -2.0575070, 0.1833615, -2.0364141, 0.1771603, -1.7785177, 1.7609048
9: 2.4164228, 5.1634150, 2.4229517, 5.1593719, -2.2954931, 2.2860525

Time for backsubstitution: 12.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3509702, upper bound: 1.3583338
time: 4.76 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3509702, upper bound: 1.3583345
time: 5.38 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.2906113, -10.3198071, -14.3055649, -10.2878733, -2.5325012, 2.5077133
1: -12.4902802, -8.9498081, -12.4945431, -8.9114246, -2.5829878, 2.5529456
2: -13.3974419, -10.1855965, -13.4044533, -10.1190605, -2.5747819, 2.5342178
3: -9.8801517, -6.9058270, -9.9375820, -6.8926010, -2.7139001, 2.7549288
4: -4.5602002, -2.4147866, -4.5679865, -2.3987143, -1.5695100, 1.5850538
5: -11.0646229, -7.3681030, -11.1309261, -7.3625040, -2.5445738, 2.6047220
6: -17.5740185, -13.6286755, -17.6125069, -13.6119013, -2.8670545, 2.8855915
7: -6.4302220, -3.6008034, -6.4406662, -3.5596151, -2.2227912, 2.1854529
8: -2.0340743, 0.1711869, -2.0600462, 0.1783495, -1.7700648, 1.7794697
9: 2.4267783, 5.1590266, 2.4149933, 5.1647034, -2.2919912, 2.2903981

Time for backsubstitution: 12.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3487030, upper bound: 1.3584038
time: 5.46 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3487030, upper bound: 1.3584043
time: 7.20 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.3440571, -10.2844677, -14.3110418, -10.2687216, -2.5782938, 2.5339103
1: -12.5052395, -8.9332809, -12.4970121, -8.9031887, -2.6074781, 2.5673580
2: -13.4136257, -10.1668453, -13.4117136, -10.1156044, -2.5939364, 2.5632391
3: -9.8948421, -6.8911214, -9.9435406, -6.8909054, -2.7310357, 2.7698143
4: -4.5728326, -2.3956447, -4.5683632, -2.3898265, -1.5926566, 1.6020174
5: -11.0779295, -7.3603973, -11.1362371, -7.3613534, -2.5680771, 2.6179311
6: -17.6217308, -13.6018190, -17.6162872, -13.5969334, -2.9175434, 2.9064767
7: -6.4381728, -3.5917993, -6.4423914, -3.5563650, -2.2380996, 2.1997406
8: -2.0595136, 0.1869874, -2.0637112, 0.1857476, -1.7951226, 1.7980046
9: 2.4132423, 5.1638784, 2.4093289, 5.1654439, -2.3066764, 2.3013785

Time for backsubstitution: 12.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3509702, upper bound: 1.3597397
time: 4.77 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3509702, upper bound: 1.3597402
time: 4.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.80 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3557161, upper bound: 1.3482328
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3557161, upper bound: 1.3482331
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3579839, upper bound: 1.3495647
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3579839, upper bound: 1.3495624
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3557161, upper bound: 1.3496337
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3557161, upper bound: 1.3496335
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3579839, upper bound: 1.3509718
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3579839, upper bound: 1.3509694
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3487030, upper bound: 1.3570041
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3487030, upper bound: 1.3570045
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3509702, upper bound: 1.3583338
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3509702, upper bound: 1.3583345
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3487030, upper bound: 1.3584038
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3487030, upper bound: 1.3584043
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3509702, upper bound: 1.3597397
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 9, lower bound: -1.3509702, upper bound: 1.3597402

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14.2800303, -10.3245773, -14.2791662, -10.3114643, -2.4892454, 2.4747577
1: -12.4714375, -8.9589462, -12.4710789, -8.9552631, -2.5175991, 2.5131397
2: -13.3864994, -10.1920338, -13.3861866, -10.1930466, -2.4994955, 2.4972467
3: -9.8741417, -6.9207373, -9.8769474, -6.9298434, -2.6706753, 2.6806984
4: -4.5449224, -2.4245210, -4.5444684, -2.4197717, -1.5436366, 1.5390649
5: -11.0581112, -7.3824081, -11.0591993, -7.3893290, -2.5086937, 2.5179310
6: -17.5575600, -13.6354132, -17.5547237, -13.6253967, -2.8106337, 2.7985649
7: -6.4186940, -3.6120954, -6.4139957, -3.6123056, -2.1563640, 2.1540406
8: -2.0269094, 0.1624942, -2.0277491, 0.1647010, -1.7348351, 1.7349823
9: 2.4411802, 5.1450605, 2.4398379, 5.1451597, -2.2560375, 2.2563398

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3543330, upper bound: 1.3482334
time: 5.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3543329, upper bound: 1.3482321
time: 6.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.2800303, -10.3245773, -14.2838402, -10.3071442, -2.4934244, 2.4788904
1: -12.4714375, -8.9589462, -12.4872990, -8.9483471, -2.5243120, 2.5292788
2: -13.3864994, -10.1920338, -13.3904724, -10.1903496, -2.5023732, 2.5023651
3: -9.8741417, -6.9207373, -9.8811760, -6.9282269, -2.6734390, 2.6851583
4: -4.5449224, -2.4245210, -4.5589843, -2.4117444, -1.5512910, 1.5534272
5: -11.0581112, -7.3824081, -11.0629225, -7.3845387, -2.5134850, 2.5220776
6: -17.5575600, -13.6354132, -17.5647926, -13.6193161, -2.8167996, 2.8092780
7: -6.4186940, -3.6120954, -6.4183006, -3.6038122, -2.1651416, 2.1586990
8: -2.0269094, 0.1624942, -2.0329928, 0.1697612, -1.7401237, 1.7404671
9: 2.4411802, 5.1450605, 2.4286113, 5.1586676, -2.2695956, 2.2675705

Time for backsubstitution: 12.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3543329, upper bound: 1.3482331
time: 4.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3543329, upper bound: 1.3482327
time: 5.13 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.3333702, -10.2892113, -14.2846241, -10.2923164, -2.5549459, 2.5007973
1: -12.4863567, -8.9424257, -12.4735651, -8.9472065, -2.5419979, 2.5274630
2: -13.4027300, -10.1735010, -13.3935099, -10.1897478, -2.5184784, 2.5259919
3: -9.8888168, -6.9061785, -9.8828802, -6.9279342, -2.6877251, 2.7015257
4: -4.5575213, -2.4053650, -4.5448055, -2.4109769, -1.5666671, 1.5578413
5: -11.0715122, -7.3747225, -11.0643501, -7.3881721, -2.5320015, 2.5329232
6: -17.6050587, -13.6085634, -17.5582428, -13.6104183, -2.8749142, 2.8191648
7: -6.4266596, -3.6030993, -6.4157529, -3.6091809, -2.1715565, 2.1682961
8: -2.0522833, 0.1782990, -2.0311785, 0.1721010, -1.7677641, 1.7534509
9: 2.4277067, 5.1499019, 2.4341893, 5.1458588, -2.2707357, 2.2673073

Time for backsubstitution: 12.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3565773, upper bound: 1.3495648
time: 5.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3565773, upper bound: 1.3495626
time: 4.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.3333702, -10.2892113, -14.2893486, -10.2879982, -2.5586462, 2.5049863
1: -12.4863567, -8.9424257, -12.4897842, -8.9402618, -2.5486994, 2.5436010
2: -13.4027300, -10.1735010, -13.3977375, -10.1870441, -2.5213552, 2.5311122
3: -9.8888168, -6.9061785, -9.8871117, -6.9263191, -2.6904893, 2.7059870
4: -4.5575213, -2.4053650, -4.5593214, -2.4029565, -1.5743141, 1.5722029
5: -11.0715122, -7.3747225, -11.0680733, -7.3833799, -2.5367942, 2.5370674
6: -17.6050587, -13.6085634, -17.5683250, -13.6043415, -2.8810787, 2.8298683
7: -6.4266596, -3.6030993, -6.4200583, -3.6006651, -2.1803532, 2.1729541
8: -2.0522833, 0.1782990, -2.0364146, 0.1771598, -1.7730503, 1.7589271
9: 2.4277067, 5.1499019, 2.4229527, 5.1593685, -2.2842929, 2.2785227

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3565773, upper bound: 1.3495625
time: 5.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3565773, upper bound: 1.3495620
time: 5.49 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.2859545, -10.3241243, -14.3008461, -10.2921944, -2.5241604, 2.4993739
1: -12.4740582, -8.9566898, -12.4783096, -8.9183540, -2.5660830, 2.5300889
2: -13.3930969, -10.1882935, -13.4001789, -10.1217270, -2.5631990, 2.5253458
3: -9.8759251, -6.9074459, -9.9333611, -6.8942208, -2.7058544, 2.7427580
4: -4.5456853, -2.4228199, -4.5534692, -2.4067359, -1.5598104, 1.5631325
5: -11.0608969, -7.3728876, -11.1272058, -7.3672981, -2.5334005, 2.5960779
6: -17.5639400, -13.6347542, -17.6024189, -13.6179876, -2.8543272, 2.8707790
7: -6.4259157, -3.6092830, -6.4363699, -3.5681176, -2.2111998, 2.1801288
8: -2.0288205, 0.1661224, -2.0547986, 0.1732826, -1.7592788, 1.7721937
9: 2.4379845, 5.1455183, 2.4262195, 5.1511922, -2.2671885, 2.2716503

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3543347, upper bound: 1.3496334
time: 8.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3543323, upper bound: 1.3482361
time: 4.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.2859545, -10.3241243, -14.3055630, -10.2878733, -2.5283403, 2.5035334
1: -12.4740582, -8.9566898, -12.4945354, -8.9114246, -2.5727844, 2.5462260
2: -13.3930969, -10.1882935, -13.4044533, -10.1190634, -2.5656743, 2.5304656
3: -9.8759251, -6.9074459, -9.9375801, -6.8926020, -2.7086143, 2.7476037
4: -4.5456853, -2.4228199, -4.5679846, -2.3987148, -1.5674629, 1.5718356
5: -11.0608969, -7.3728876, -11.1309242, -7.3625069, -2.5381880, 2.5990198
6: -17.5639400, -13.6347542, -17.6125050, -13.6119041, -2.8604913, 2.8764546
7: -6.4259157, -3.6092830, -6.4406633, -3.5596166, -2.2160857, 2.1847866
8: -2.0288205, 0.1661224, -2.0600452, 0.1783485, -1.7640862, 1.7763588
9: 2.4379845, 5.1455183, 2.4149952, 5.1647005, -2.2807460, 2.2828760

Time for backsubstitution: 12.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3543323, upper bound: 1.3496354
time: 4.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3543323, upper bound: 1.3482332
time: 4.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.3393211, -10.2887726, -14.3062744, -10.2730436, -2.5697913, 2.5255237
1: -12.4890003, -8.9402256, -12.4807816, -8.9101334, -2.5905666, 2.5444522
2: -13.4093227, -10.1695557, -13.4074945, -10.1182709, -2.5823269, 2.5543647
3: -9.8906221, -6.8927383, -9.9393187, -6.8925228, -2.7229910, 2.7576425
4: -4.5583167, -2.4036875, -4.5538473, -2.3978424, -1.5829635, 1.5819519
5: -11.0742025, -7.3651810, -11.1325226, -7.3661413, -2.5569229, 2.6092925
6: -17.6116161, -13.6079130, -17.6061954, -13.6030169, -2.9052992, 2.8916006
7: -6.4338665, -3.6003227, -6.4380975, -3.5648890, -2.2264831, 2.1943567
8: -2.0542879, 0.1819239, -2.0584707, 0.1806827, -1.7856584, 1.7907441
9: 2.4245205, 5.1503639, 2.4205728, 5.1519327, -2.2819209, 2.2826395

Time for backsubstitution: 12.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3565767, upper bound: 1.3509717
time: 4.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3565767, upper bound: 1.3495626
time: 5.50 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.3393211, -10.2887726, -14.3110390, -10.2687263, -2.5734963, 2.5297408
1: -12.4890003, -8.9402256, -12.4970093, -8.9031878, -2.5972595, 2.5605912
2: -13.4093227, -10.1695557, -13.4117136, -10.1156063, -2.5847998, 2.5594845
3: -9.8906221, -6.8927383, -9.9435387, -6.8909073, -2.7257519, 2.7624869
4: -4.5583167, -2.4036875, -4.5683608, -2.3898268, -1.5897512, 1.5887854
5: -11.0742025, -7.3651810, -11.1362362, -7.3613505, -2.5617104, 2.6122341
6: -17.6116161, -13.6079130, -17.6162834, -13.5969334, -2.9093800, 2.8973281
7: -6.4338665, -3.6003227, -6.4423919, -3.5563669, -2.2313759, 2.1990144
8: -2.0542879, 0.1819239, -2.0637102, 0.1857448, -1.7874756, 1.7948995
9: 2.4245205, 5.1503639, 2.4093308, 5.1654420, -2.2954779, 2.2938457

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3565791, upper bound: 1.3509699
time: 5.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3565791, upper bound: 1.3495631
time: 5.11 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14.2846737, -10.3202562, -14.2791662, -10.3114643, -2.4933453, 2.4789376
1: -12.4876566, -8.9520683, -12.4710789, -8.9552631, -2.5337391, 2.5198574
2: -13.3908339, -10.1893387, -13.3861866, -10.1930466, -2.5046129, 2.5001273
3: -9.8783684, -6.9191160, -9.8769474, -6.9298434, -2.6751356, 2.6834636
4: -4.5594406, -2.4164891, -4.5444684, -2.4197717, -1.5580003, 1.5467241
5: -11.0618382, -7.3776197, -11.0591993, -7.3893290, -2.5128422, 2.5227213
6: -17.5676384, -13.6293325, -17.5547237, -13.6253967, -2.8213615, 2.8047228
7: -6.4230013, -3.6036153, -6.4139957, -3.6123056, -2.1610231, 2.1627891
8: -2.0321631, 0.1675549, -2.0277491, 0.1647010, -1.7403297, 1.7402723
9: 2.4299717, 5.1585703, 2.4398379, 5.1451597, -2.2672803, 2.2698982

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3473195, upper bound: 1.3570065
time: 4.89 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3473195, upper bound: 1.3570033
time: 5.15 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.2846737, -10.3202562, -14.2838402, -10.3071442, -2.4974890, 2.4829969
1: -12.4876566, -8.9520683, -12.4872990, -8.9483471, -2.5345140, 2.5300579
2: -13.3908339, -10.1893387, -13.3904724, -10.1903496, -2.5082703, 2.5060248
3: -9.8783684, -6.9191160, -9.8811760, -6.9282269, -2.6845360, 2.6945596
4: -4.5594406, -2.4164891, -4.5589843, -2.4117444, -1.5533378, 1.5487695
5: -11.0618382, -7.3776197, -11.0629225, -7.3845387, -2.5198717, 2.5291085
6: -17.5676384, -13.6293325, -17.5647926, -13.6193161, -2.8233490, 2.8112574
7: -6.4230013, -3.6036153, -6.4183006, -3.6038122, -2.1615148, 2.1591625
8: -2.0321631, 0.1675549, -2.0329928, 0.1697612, -1.7423091, 1.7424469
9: 2.4299717, 5.1585703, 2.4286113, 5.1586676, -2.2747993, 2.2750897

Time for backsubstitution: 12.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3473195, upper bound: 1.3570069
time: 9.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3473218, upper bound: 1.3570039
time: 7.24 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.3380976, -10.2849026, -14.2846241, -10.2923164, -2.5589514, 2.5049667
1: -12.5025940, -8.9354858, -12.4735651, -8.9472065, -2.5581532, 2.5342293
2: -13.4070225, -10.1707916, -13.3935099, -10.1897478, -2.5236187, 2.5288744
3: -9.8930416, -6.9045639, -9.8828802, -6.9279342, -2.6921854, 2.7042894
4: -4.5720391, -2.3973236, -4.5448055, -2.4109769, -1.5777311, 1.5654821
5: -11.0752392, -7.3699408, -11.0643501, -7.3881721, -2.5361328, 2.5377097
6: -17.6151657, -13.6024723, -17.5582428, -13.6104183, -2.8856812, 2.8253303
7: -6.4309645, -3.5945778, -6.4157529, -3.6091809, -2.1762190, 2.1770940
8: -2.0575070, 0.1833615, -2.0311785, 0.1721010, -1.7732306, 1.7587368
9: 2.4164228, 5.1634150, 2.4341893, 5.1458588, -2.2819333, 2.2808766

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3495660, upper bound: 1.3583346
time: 5.15 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3495636, upper bound: 1.3583332
time: 5.35 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.3380976, -10.2849026, -14.2893486, -10.2879982, -2.5634446, 2.5090184
1: -12.5025940, -8.9354858, -12.4897842, -8.9402618, -2.5589166, 2.5444288
2: -13.4070225, -10.1707916, -13.3977375, -10.1870441, -2.5272741, 2.5347729
3: -9.8930416, -6.9045639, -9.8871117, -6.9263191, -2.7015858, 2.7153859
4: -4.5720391, -2.3973236, -4.5593214, -2.4029565, -1.5763619, 1.5675271
5: -11.0752392, -7.3699408, -11.0680733, -7.3833799, -2.5431657, 2.5440955
6: -17.6151657, -13.6024723, -17.5683250, -13.6043415, -2.8876648, 2.8318548
7: -6.4309645, -3.5945778, -6.4200583, -3.6006651, -2.1767297, 2.1734672
8: -2.0575070, 0.1833615, -2.0364146, 0.1771598, -1.7752075, 1.7609048
9: 2.4164228, 5.1634150, 2.4229527, 5.1593685, -2.2894523, 2.2860525

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3495636, upper bound: 1.3583371
time: 4.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3495636, upper bound: 1.3583340
time: 5.27 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.2906113, -10.3198071, -14.3008461, -10.2921944, -2.5282650, 2.5035515
1: -12.4902802, -8.9498081, -12.4783096, -8.9183540, -2.5822239, 2.5368047
2: -13.3974419, -10.1855965, -13.4001789, -10.1217270, -2.5680523, 2.5282278
3: -9.8801517, -6.9058270, -9.9333611, -6.8942208, -2.7103167, 2.7451470
4: -4.5602002, -2.4147866, -4.5534692, -2.4067359, -1.5741732, 1.5706578
5: -11.0646229, -7.3681030, -11.1272058, -7.3672981, -2.5375462, 2.5992112
6: -17.5740185, -13.6286755, -17.6024189, -13.6179876, -2.8650694, 2.8749170
7: -6.4302220, -3.6008034, -6.4363699, -3.5681176, -2.2140698, 2.1888704
8: -2.0340743, 0.1711869, -2.0547986, 0.1732826, -1.7647767, 1.7740138
9: 2.4267783, 5.1590266, 2.4262195, 5.1511922, -2.2784314, 2.2852101

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3473188, upper bound: 1.3584038
time: 5.53 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3473188, upper bound: 1.3570032
time: 4.86 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.2906113, -10.3198071, -14.3055630, -10.2878733, -2.5324030, 2.5076060
1: -12.4902802, -8.9498081, -12.4945354, -8.9114246, -2.5829868, 2.5470042
2: -13.3974419, -10.1855965, -13.4044533, -10.1190634, -2.5747824, 2.5341249
3: -9.8801517, -6.9058270, -9.9375801, -6.8926020, -2.7197132, 2.7549276
4: -4.5602002, -2.4147866, -4.5679846, -2.3987148, -1.5695090, 1.5728374
5: -11.0646229, -7.3681030, -11.1309242, -7.3625069, -2.5445738, 2.6055949
6: -17.5740185, -13.6286755, -17.6125050, -13.6119041, -2.8670540, 2.8824830
7: -6.4302220, -3.6008034, -6.4406633, -3.5596166, -2.2205703, 2.1852422
8: -2.0340743, 0.1711869, -2.0600452, 0.1783485, -1.7667551, 1.7791955
9: 2.4267783, 5.1590266, 2.4149952, 5.1647005, -2.2859497, 2.2903972

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3473188, upper bound: 1.3584046
time: 4.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3473188, upper bound: 1.3570046
time: 4.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.3440571, -10.2844677, -14.3062744, -10.2730436, -2.5737967, 2.5296926
1: -12.5052395, -8.9332809, -12.4807816, -8.9101334, -2.6020155, 2.5512166
2: -13.4136257, -10.1668453, -13.4074945, -10.1182709, -2.5872059, 2.5572476
3: -9.8948421, -6.8911214, -9.9393187, -6.8925228, -2.7274523, 2.7600315
4: -4.5728326, -2.3956447, -4.5538473, -2.3978424, -1.5909319, 1.5876219
5: -11.0779295, -7.3603973, -11.1325226, -7.3661413, -2.5610495, 2.6124310
6: -17.6217308, -13.6018190, -17.6061954, -13.6030169, -2.9109068, 2.8958101
7: -6.4381728, -3.5917993, -6.4380975, -3.5648890, -2.2293570, 2.2031591
8: -2.0595136, 0.1869874, -2.0584707, 0.1806827, -1.7898278, 1.7925553
9: 2.4132423, 5.1638784, 2.4205728, 5.1519327, -2.2931170, 2.2962093

Time for backsubstitution: 12.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3495630, upper bound: 1.3597395
time: 4.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3473188, upper bound: 1.3583341
time: 4.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.3440571, -10.2844677, -14.3110390, -10.2687263, -2.5782933, 2.5337391
1: -12.5052395, -8.9332809, -12.4970093, -8.9031878, -2.6074777, 2.5614157
2: -13.4136257, -10.1668453, -13.4117136, -10.1156063, -2.5939341, 2.5631461
3: -9.8948421, -6.8911214, -9.9435387, -6.8909073, -2.7368493, 2.7698140
4: -4.5728326, -2.3956447, -4.5683608, -2.3898268, -1.5926561, 1.5916352
5: -11.0779295, -7.3603973, -11.1362362, -7.3613505, -2.5680761, 2.6188054
6: -17.6217308, -13.6018190, -17.6162834, -13.5969334, -2.9169483, 2.9033685
7: -6.4381728, -3.5917993, -6.4423919, -3.5563669, -2.2358658, 2.1995301
8: -2.0595136, 0.1869874, -2.0637102, 0.1857448, -1.7926631, 1.7977321
9: 2.4132423, 5.1638784, 2.4093308, 5.1654420, -2.3006344, 2.3013766

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3495630, upper bound: 1.3597400
time: 4.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3495630, upper bound: 1.3583340
time: 5.50 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.98 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3543330, upper bound: 1.3482334
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3543329, upper bound: 1.3482321
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3543329, upper bound: 1.3482331
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3543329, upper bound: 1.3482327
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3565773, upper bound: 1.3495648
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3565773, upper bound: 1.3495626
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3565773, upper bound: 1.3495625
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3565773, upper bound: 1.3495620
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3543347, upper bound: 1.3496334
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3543323, upper bound: 1.3482361
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3543323, upper bound: 1.3496354
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3543323, upper bound: 1.3482332
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3565767, upper bound: 1.3509717
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3565767, upper bound: 1.3495626
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3565791, upper bound: 1.3509699
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3565791, upper bound: 1.3495631
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3473195, upper bound: 1.3570065
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3473195, upper bound: 1.3570033
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3473195, upper bound: 1.3570069
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3473218, upper bound: 1.3570039
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3495660, upper bound: 1.3583346
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3495636, upper bound: 1.3583332
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3495636, upper bound: 1.3583371
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3495636, upper bound: 1.3583340
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3473188, upper bound: 1.3584038
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3473188, upper bound: 1.3570032
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3473188, upper bound: 1.3584046
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3473188, upper bound: 1.3570046
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3495630, upper bound: 1.3597395
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3473188, upper bound: 1.3583341
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3495630, upper bound: 1.3597400
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.98
Output dim: 9, lower bound: -1.3495630, upper bound: 1.3583340

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14.2752590, -10.3249445, -14.2791662, -10.3114643, -2.4842854, 2.4736509
1: -12.4692907, -8.9608955, -12.4710789, -8.9552631, -2.5149574, 2.5118661
2: -13.3810949, -10.1954117, -13.3861866, -10.1930466, -2.4914484, 2.4937458
3: -9.8727856, -6.9312177, -9.8769474, -6.9298434, -2.6693926, 2.6720533
4: -4.5442271, -2.4259353, -4.5444684, -2.4197717, -1.5408213, 1.5359554
5: -11.0555973, -7.3901548, -11.0591993, -7.3893290, -2.5061798, 2.5081596
6: -17.5522270, -13.6359434, -17.5547237, -13.6253967, -2.8042898, 2.7970133
7: -6.4127755, -3.6144905, -6.4139957, -3.6123056, -2.1518497, 2.1512187
8: -2.0253696, 0.1595144, -2.0277491, 0.1647010, -1.7329264, 1.7301533
9: 2.4437881, 5.1446662, 2.4398379, 5.1451597, -2.2531326, 2.2554853

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5747

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3468243, upper bound: 1.3553397
time: 5.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3544259, upper bound: 1.3553427
time: 7.97 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -14.2966518, -10.3057356, -14.2791662, -10.3114643, -2.5048771, 2.5017796
1: -12.4765005, -8.9260149, -12.4710789, -8.9552631, -2.5240297, 2.5545883
2: -13.3950758, -10.1266994, -13.3861866, -10.1930466, -2.5062275, 2.5440331
3: -9.9285297, -6.8970156, -9.8769474, -6.9298434, -2.7200642, 2.7076640
4: -4.5526423, -2.4130335, -4.5444684, -2.4197717, -1.5499461, 1.5486641
5: -11.1208363, -7.3684053, -11.0591993, -7.3893290, -2.5707254, 2.5297470
6: -17.5973969, -13.6285667, -17.5547237, -13.6253967, -2.8655577, 2.8307920
7: -6.4351382, -3.5717525, -6.4139957, -3.6123056, -2.1742940, 2.1985250
8: -2.0517807, 0.1680789, -2.0277491, 0.1647010, -1.7619476, 1.7394004
9: 2.4302959, 5.1505985, 2.4398379, 5.1451597, -2.2667634, 2.2629352

Time for backsubstitution: 12.58 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.2999587059020996
rel_dist={9: [-1.360028225390102, 1.3600276268876046]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2432.41 seconds
