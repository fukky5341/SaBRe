## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.3120148312
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.5105915, -10.3324680, -12.5105915, -10.3324680, -2.1781235, 2.1781235)
1: (3.1649604, 4.4064875, 3.1649604, 4.4064875, -1.1255052, 1.1255053)
2: (-4.9406466, -3.7702384, -4.9406466, -3.7702384, -1.1704082, 1.1704082)
3: (-12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.7759883, 1.7759886)
4: (-2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.4544666, 1.4544665)
5: (-10.0812330, -8.6300726, -10.0812330, -8.6300726, -1.3241396, 1.3241397)
6: (-8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.7115004, 1.7115003)
7: (-2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.8193477, 0.8193477)
8: (-3.7933545, -2.3891320, -3.7933545, -2.3891320, -1.4042225, 1.4042225)
9: (-12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.5232468, 1.5232465)

## BASE Result
execution time: IAR + LP analysis = 13.30 + 31.94 = 45.24 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3554.76 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=0.8663967847824097
rel_dist={1: [-0.5771277606023322, 0.5771254352696031]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=0.7368426322937012
rel_dist={1: [-0.38431871261449, 0.38431867244677376]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=0.6504731178283691
rel_dist={1: [-0.2375077651971229, 0.23750601401867089]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=0.6936578750610352
rel_dist={1: [-0.3128648053580161, 0.31286157369120016]}

## Binary Search Result
Binary search time: 187.73 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Individual Split (IS_dual_ind) starts
Time budget: 3367.03 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start
Binary search (step 0): status=Status.ADV_EXAMPLE, k_low=2, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=None

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5815
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5815

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4501825, upper bound: 0.4439257
time: 3.42 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509254, upper bound: 0.4509245
time: 3.38 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.96 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 6.96
Output dim: 1, lower bound: -0.4501825, upper bound: 0.4439257
IS_A2, status: Status.UNKNOWN, split count: 1, time: 6.96
Output dim: 1, lower bound: -0.4509254, upper bound: 0.4509245

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.4172287, -10.4103403, -12.5045977, -10.3702469, -1.5056851, 1.5585427
1: 3.2003183, 4.3859591, 3.1689262, 4.3971572, -0.7352107, 0.7530195
2: -4.9007716, -3.8007307, -4.9384508, -3.7855086, -0.8675041, 0.8892967
3: -12.6646280, -10.9396000, -12.7049980, -10.8597231, -1.1990869, 1.1506343
4: -2.4103954, -0.9145901, -2.4237161, -0.9086497, -1.1086111, 1.1111283
5: -10.0592670, -8.6618805, -10.0707741, -8.6334591, -0.8511039, 0.8370783
6: -8.0244331, -6.4216833, -8.0865364, -6.3963509, -1.1513684, 1.1848718
7: -2.7749660, -1.9314392, -2.7793074, -1.9270303, -0.5802813, 0.5804886
8: -3.7502260, -2.4320984, -3.7903309, -2.4096999, -0.9258177, 0.9469641
9: -12.4648705, -10.9785194, -12.4826956, -10.9589033, -1.0685978, 1.0670943

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6193

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4441267, upper bound: 0.4433113
time: 3.28 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4501783, upper bound: 0.4439185
time: 5.20 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -12.5105877, -10.3325186, -12.5105896, -10.3324928, -1.5798602, 1.5929289
1: 3.1649652, 4.4064684, 3.1649628, 4.4064789, -0.7698855, 0.7726337
2: -4.9406443, -3.7702651, -4.9406452, -3.7702515, -0.9010482, 0.8920180
3: -12.7447414, -10.8541403, -12.7447805, -10.8541355, -1.2133043, 1.2334883
4: -2.4278302, -0.9076324, -2.4278333, -0.9076309, -1.1251149, 1.1248498
5: -10.0812159, -8.6300774, -10.0812254, -8.6300764, -0.8692052, 0.8717290
6: -8.0952511, -6.3753991, -8.0952568, -6.3753858, -1.2286313, 1.2398872
7: -2.7799153, -1.9236879, -2.7799168, -1.9236858, -0.5883070, 0.5891856
8: -3.7933502, -2.3891625, -3.7933521, -2.3891459, -0.9599640, 0.9696902
9: -12.4935284, -10.9539337, -12.4935379, -10.9539318, -1.1034970, 1.1041341

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4503063, upper bound: 0.4459495
time: 3.60 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509137, upper bound: 0.4509125
time: 3.38 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.62 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 19.62
Output dim: 1, lower bound: -0.4441267, upper bound: 0.4433113
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 19.62
Output dim: 1, lower bound: -0.4501783, upper bound: 0.4439185
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 19.62
Output dim: 1, lower bound: -0.4503063, upper bound: 0.4459495
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 19.62
Output dim: 1, lower bound: -0.4509137, upper bound: 0.4509125

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.4052658, -10.4579678, -12.4015903, -10.4672308, -1.4019954, 1.4069045
1: 3.2050037, 4.3754635, 3.1995130, 4.3744383, -0.7044251, 0.7114228
2: -4.8966675, -3.8231747, -4.8861713, -3.8309939, -0.8170803, 0.8142613
3: -12.6408625, -10.9480534, -12.6569166, -10.9156046, -1.1140056, 1.0979540
4: -2.4046144, -0.9262419, -2.3875136, -0.9337628, -1.0737481, 1.0515951
5: -10.0531349, -8.6652908, -10.0568886, -8.6524973, -0.8233578, 0.8171680
6: -8.0150032, -6.4401851, -8.0406046, -6.4369307, -1.0902770, 1.1136376
7: -2.7737124, -1.9426157, -2.7618773, -1.9513602, -0.5526702, 0.5488476
8: -3.7450933, -2.4409533, -3.7677050, -2.4318438, -0.8942246, 0.9079881
9: -12.4591532, -10.9965363, -12.4415874, -10.9949608, -1.0260448, 1.0082901

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4391514, upper bound: 0.4426995
time: 3.67 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4441145, upper bound: 0.4433018
time: 3.41 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.4172249, -10.4103899, -12.5045919, -10.3703413, -1.4545460, 1.5024610
1: 3.2003217, 4.3859539, 3.1689301, 4.3971467, -0.7231891, 0.7466559
2: -4.9007702, -3.8007345, -4.9384465, -3.7855189, -0.8345222, 0.8651407
3: -12.6646156, -10.9396048, -12.7049770, -10.8597307, -1.1814289, 1.1291339
4: -2.4103930, -0.9145963, -2.4237103, -0.9086688, -1.1058407, 1.1112971
5: -10.0592651, -8.6618814, -10.0707684, -8.6334620, -0.8558623, 0.8346260
6: -8.0244284, -6.4216962, -8.0865269, -6.3963780, -1.1350033, 1.1848524
7: -2.7749655, -1.9314451, -2.7793064, -1.9270425, -0.5593408, 0.5758440
8: -3.7502246, -2.4321089, -3.7903271, -2.4097128, -0.9017751, 0.9482405
9: -12.4648676, -10.9785357, -12.4826908, -10.9589405, -1.0561323, 1.0670784

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4452161, upper bound: 0.4433299
time: 3.71 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4501662, upper bound: 0.4439089
time: 3.75 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -12.5081367, -10.3369265, -12.5054903, -10.3410797, -1.5657322, 1.5803933
1: 3.1748104, 4.4050856, 3.1846952, 4.4036274, -0.7509964, 0.7490295
2: -4.9380322, -3.7749121, -4.9352903, -3.7795892, -0.8887174, 0.8821609
3: -12.7432175, -10.8698168, -12.7416134, -10.8856010, -1.1806976, 1.2118890
4: -2.4253201, -0.9209971, -2.4227090, -0.9344289, -1.0953007, 1.1065910
5: -10.0520849, -8.6327705, -10.0226994, -8.6355534, -0.8274700, 0.8109888
6: -8.0722036, -6.3763790, -8.0490074, -6.3774133, -1.1913185, 1.1887062
7: -2.7792554, -1.9304576, -2.7785959, -1.9371865, -0.5687466, 0.5748297
8: -3.7908139, -2.4107113, -3.7881212, -2.4324794, -0.9138958, 0.9359152
9: -12.4789629, -10.9564981, -12.4645252, -10.9591579, -1.0659087, 1.0592937

Time for backsubstitution: 12.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4498220, upper bound: 0.4400255
time: 3.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4503028, upper bound: 0.4459454
time: 3.58 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -12.5105839, -10.3325262, -12.5141687, -10.3310862, -1.5788984, 1.5951467
1: 3.1649837, 4.4064665, 3.1629658, 4.4210672, -0.7698467, 0.7655727
2: -4.9406414, -3.7702746, -4.9461985, -3.7676482, -0.9024370, 0.8968241
3: -12.7447414, -10.8541622, -12.7674809, -10.8529816, -1.2051222, 1.2363205
4: -2.4278288, -0.9076490, -2.4501038, -0.9069717, -1.1154411, 1.1380827
5: -10.0811968, -8.6300793, -10.0845356, -8.5760784, -0.8711109, 0.8417651
6: -8.0952349, -6.3754010, -8.0979605, -6.3389149, -1.2272320, 1.2198975
7: -2.7799144, -1.9236972, -2.7885220, -1.9220746, -0.5841197, 0.5927994
8: -3.7933493, -2.3891854, -3.8365726, -2.3891320, -0.9399549, 0.9744600
9: -12.4935179, -10.9539375, -12.4970760, -10.9363194, -1.1039488, 1.0976181

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4504294, upper bound: 0.4449930
time: 3.84 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509101, upper bound: 0.4509085
time: 3.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.80 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.80
Output dim: 1, lower bound: -0.4391514, upper bound: 0.4426995
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.80
Output dim: 1, lower bound: -0.4441145, upper bound: 0.4433018
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.80
Output dim: 1, lower bound: -0.4452161, upper bound: 0.4433299
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.80
Output dim: 1, lower bound: -0.4501662, upper bound: 0.4439089
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.80
Output dim: 1, lower bound: -0.4498220, upper bound: 0.4400255
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.80
Output dim: 1, lower bound: -0.4503028, upper bound: 0.4459454
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.80
Output dim: 1, lower bound: -0.4504294, upper bound: 0.4449930
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.80
Output dim: 1, lower bound: -0.4509101, upper bound: 0.4509085

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.4001541, -10.4665909, -12.3991356, -10.4716244, -1.3894365, 1.3927445
1: 3.2243690, 4.3726153, 3.2092628, 4.3730574, -0.6814982, 0.6944263
2: -4.8913755, -3.8324921, -4.8835912, -3.8356271, -0.8072536, 0.8018880
3: -12.6376934, -10.9794521, -12.6553926, -10.9312582, -1.0936174, 1.0653028
4: -2.3996539, -0.9530401, -2.3850129, -0.9471283, -1.0556567, 1.0217912
5: -9.9946146, -8.6708345, -10.0277605, -8.6552277, -0.7626365, 0.7786994
6: -7.9688454, -6.4422388, -8.0175152, -6.4379168, -1.0396641, 1.0827141
7: -2.7724230, -1.9560430, -2.7612336, -1.9580936, -0.5383111, 0.5293367
8: -3.7399006, -2.4842443, -3.7651839, -2.4533796, -0.8621042, 0.8620837
9: -12.4302130, -11.0016785, -12.4269524, -10.9974928, -0.9815187, 0.9708720

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4426996
time: 3.61 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4426977
time: 3.49 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.4088507, -10.4565907, -12.4015875, -10.4672403, -1.4041939, 1.4059031
1: 3.2030110, 4.3900509, 3.1995311, 4.3744369, -0.6973425, 0.7132756
2: -4.9022379, -3.8205991, -4.8861685, -3.8310027, -0.8237495, 0.8156211
3: -12.6635647, -10.9469366, -12.6569138, -10.9156275, -1.1180564, 1.0897690
4: -2.4270182, -0.9255822, -2.3875122, -0.9337800, -1.0886524, 1.0419214
5: -10.0564690, -8.6112070, -10.0568695, -8.6525002, -0.7933800, 0.8223348
6: -8.0175724, -6.4037118, -8.0405874, -6.4369321, -1.0701261, 1.1242776
7: -2.7823269, -1.9410284, -2.7618766, -1.9513695, -0.5565058, 0.5446891
8: -3.7883234, -2.4409490, -3.7677035, -2.4318652, -0.9005853, 0.8879374
9: -12.4625769, -10.9788885, -12.4415798, -10.9949636, -1.0197339, 1.0089791

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4433018
time: 3.49 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4432986
time: 4.46 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.4121151, -10.4190197, -12.5021362, -10.3747559, -1.4419823, 1.4883077
1: 3.2197318, 4.3831053, 3.1787519, 4.3957644, -0.6997705, 0.7277260
2: -4.8954554, -3.8100655, -4.9358449, -3.7901669, -0.8235693, 0.8527775
3: -12.6614485, -10.9710131, -12.7034521, -10.8754015, -1.1598225, 1.0964880
4: -2.4054356, -0.9413953, -2.4212294, -0.9220338, -1.0836489, 1.0808759
5: -10.0007410, -8.6674004, -10.0416384, -8.6361599, -0.7950663, 0.7919930
6: -7.9782352, -6.4237514, -8.0634499, -6.3973589, -1.0844288, 1.1477737
7: -2.7736650, -1.9448988, -2.7786531, -1.9338059, -0.5449865, 0.5554556
8: -3.7450213, -2.4754071, -3.7877960, -2.4312592, -0.8680364, 0.9021833
9: -12.4359560, -10.9836941, -12.4681292, -10.9614944, -1.0116296, 1.0278773

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4396814, upper bound: 0.4433299
time: 3.49 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4396814, upper bound: 0.4433298
time: 3.42 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.4208069, -10.4090118, -12.5045910, -10.3703499, -1.4567513, 1.5014679
1: 3.1983204, 4.4005413, 3.1689472, 4.3971453, -0.7161201, 0.7466142
2: -4.9063373, -3.7981586, -4.9384446, -3.7855279, -0.8373890, 0.8665069
3: -12.6873178, -10.9384804, -12.7049751, -10.8597527, -1.1842585, 1.1209445
4: -2.4327846, -0.9139373, -2.4237084, -0.9086850, -1.1118069, 1.1017039
5: -10.0625944, -8.6078176, -10.0707493, -8.6334648, -0.8258988, 0.8356610
6: -8.0270529, -6.3852243, -8.0865088, -6.3963780, -1.1149065, 1.1837654
7: -2.7835782, -1.9298546, -2.7793055, -1.9270523, -0.5624893, 0.5691859
8: -3.7934542, -2.4320989, -3.7903266, -2.4097352, -0.9065378, 0.9282216
9: -12.4683266, -10.9608946, -12.4826841, -10.9589443, -1.0497792, 1.0574570

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4446190, upper bound: 0.4439089
time: 3.70 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4446188, upper bound: 0.4439071
time: 3.66 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.4070721, -10.4338837, -12.4955006, -10.3886347, -1.4058666, 1.4759707
1: 3.2046809, 4.3821793, 3.1886320, 4.3929510, -0.7049367, 0.7177218
2: -4.8862963, -3.8205018, -4.9317226, -3.8020897, -0.8056448, 0.8319938
3: -12.6951389, -10.9245367, -12.7178259, -10.8928499, -1.1275996, 1.1204113
4: -2.3891435, -0.9463186, -2.4171808, -0.9459062, -1.0362704, 1.0714741
5: -10.0379705, -8.6514721, -10.0163879, -8.6385651, -0.8071474, 0.7825969
6: -8.0273218, -6.4173522, -8.0409193, -6.3961959, -1.1127772, 1.1268809
7: -2.7619021, -1.9548469, -2.7774134, -1.9485695, -0.5368265, 0.5475707
8: -3.7691383, -2.4328589, -3.7839727, -2.4413195, -0.8731971, 0.9038687
9: -12.4378777, -10.9928665, -12.4576035, -10.9764662, -1.0071404, 1.0146198

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4426975, upper bound: 0.4391508
time: 3.73 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4426975, upper bound: 0.4400269
time: 3.49 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.5081310, -10.3370180, -12.5054903, -10.3411245, -1.5071964, 1.5317464
1: 3.1748142, 4.4050779, 3.1846981, 4.4036226, -0.7400284, 0.7357104
2: -4.9380288, -3.7749228, -4.9352880, -3.7795932, -0.8566763, 0.8465884
3: -12.7431965, -10.8698225, -12.7416039, -10.8856030, -1.1603338, 1.1894759
4: -2.4253149, -0.9210155, -2.4227078, -0.9344392, -1.0910463, 1.0982928
5: -10.0520811, -8.6327744, -10.0226965, -8.6355543, -0.8212339, 0.8089026
6: -8.0721941, -6.3764019, -8.0490026, -6.3774271, -1.1765945, 1.1687024
7: -2.7792544, -1.9304686, -2.7785954, -1.9371924, -0.5629458, 0.5540950
8: -3.7908096, -2.4107275, -3.7881203, -2.4324861, -0.9096458, 0.9115719
9: -12.4789581, -10.9565382, -12.4645224, -10.9591761, -1.0605536, 1.0468290

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4433278, upper bound: 0.4452154
time: 5.23 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4433278, upper bound: 0.4459460
time: 3.43 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.4095249, -10.4295082, -12.5041800, -10.3786507, -1.4190335, 1.4907167
1: 3.1949234, 4.3835592, 3.1669426, 4.4103870, -0.7237570, 0.7343503
2: -4.8888845, -3.8158786, -4.9426146, -3.7901604, -0.8193207, 0.8461248
3: -12.6966600, -10.9089012, -12.7436924, -10.8602438, -1.1520355, 1.1448455
4: -2.3916748, -0.9329700, -2.4445984, -0.9184477, -1.0564294, 1.1021941
5: -10.0670767, -8.6487513, -10.0782280, -8.5790510, -0.8508006, 0.8134074
6: -8.0504179, -6.4163713, -8.0898533, -6.3576980, -1.1487365, 1.1580565
7: -2.7625523, -1.9481163, -2.7873328, -1.9334831, -0.5521969, 0.5651577
8: -3.7716608, -2.4113412, -3.8324156, -2.3979826, -0.8992141, 0.9423914
9: -12.4525042, -10.9903221, -12.4901495, -10.9536371, -1.0404859, 1.0530088

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4432998, upper bound: 0.4441160
time: 3.71 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4432997, upper bound: 0.4449917
time: 3.62 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.5105791, -10.3326206, -12.5141678, -10.3311348, -1.5203657, 1.5465045
1: 3.1649866, 4.4064598, 3.1629677, 4.4210625, -0.7588534, 0.7522125
2: -4.9406376, -3.7702842, -4.9461966, -3.7676528, -0.8703791, 0.8603718
3: -12.7447205, -10.8541679, -12.7674704, -10.8529825, -1.1847634, 1.2139095
4: -2.4278240, -0.9076672, -2.4501009, -0.9069817, -1.1118460, 1.1265903
5: -10.0811920, -8.6300850, -10.0845346, -8.5760803, -0.8648876, 0.8396925
6: -8.0952244, -6.3754244, -8.0979548, -6.3389282, -1.2125654, 1.1999495
7: -2.7799144, -1.9237089, -2.7885220, -1.9220800, -0.5767089, 0.5720068
8: -3.7933455, -2.3892002, -3.8365707, -2.3891401, -0.9357064, 0.9501125
9: -12.4935131, -10.9539747, -12.4970760, -10.9363365, -1.0901222, 1.0823103

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4439068, upper bound: 0.4501658
time: 4.18 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4439068, upper bound: 0.4501683
time: 3.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.09 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4426996
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4426977
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4433018
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4432986
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4396814, upper bound: 0.4433299
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4396814, upper bound: 0.4433298
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4446190, upper bound: 0.4439089
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4446188, upper bound: 0.4439071
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4426975, upper bound: 0.4391508
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4426975, upper bound: 0.4400269
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4433278, upper bound: 0.4452154
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4433278, upper bound: 0.4459460
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4432998, upper bound: 0.4441160
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4432997, upper bound: 0.4449917
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4439068, upper bound: 0.4501658
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.09
Output dim: 1, lower bound: -0.4439068, upper bound: 0.4501683

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.4001541, -10.4665909, -12.3080730, -10.5118294, -1.3585167, 1.3060195
1: 3.2243690, 4.3726153, 3.2414255, 4.3617992, -0.6691600, 0.6640862
2: -4.8913755, -3.8324921, -4.8454027, -3.8507476, -0.7918055, 0.7643180
3: -12.6376934, -10.9794521, -12.6149521, -11.0131340, -1.0123775, 1.0326312
4: -2.3996539, -0.9530401, -2.3722448, -0.9532638, -1.0445056, 1.0091850
5: -9.9946146, -8.6708345, -10.0161638, -8.6836395, -0.7360682, 0.7688032
6: -7.9688454, -6.4422388, -7.9526148, -6.4629040, -1.0111058, 1.0184586
7: -2.7724230, -1.9560430, -2.7569575, -1.9621696, -0.5341558, 0.5244956
8: -3.7399006, -2.4842443, -3.7233586, -2.4761200, -0.8468338, 0.8202108
9: -12.4302130, -11.0016785, -12.4103851, -11.0180035, -0.9610584, 0.9539179

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4372677
time: 3.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4426996
time: 3.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.4001541, -10.4665909, -12.4070721, -10.4341850, -1.3911510, 1.3315752
1: 3.2243690, 4.3726153, 3.2048960, 4.3821797, -0.6852870, 0.6803421
2: -4.8913755, -3.8324921, -4.8862953, -3.8213732, -0.8119218, 0.7807962
3: -12.6376934, -10.9794521, -12.6951389, -10.9249401, -1.0296361, 1.0657551
4: -2.3996539, -0.9530401, -2.3888383, -0.9463186, -1.0539494, 1.0261492
5: -9.9946146, -8.6708345, -10.0379686, -8.6516371, -0.7555329, 0.7816433
6: -7.9688454, -6.4422388, -8.0272121, -6.4173536, -1.0536389, 1.0583751
7: -2.7724230, -1.9560430, -2.7619014, -1.9551692, -0.5418305, 0.5303419
8: -3.7399006, -2.4842443, -3.7691360, -2.4334097, -0.8642015, 0.8361549
9: -12.4302130, -11.0016785, -12.4378490, -10.9928665, -0.9861960, 0.9818790

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4372657
time: 3.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4336174, upper bound: 0.4426969
time: 3.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.4088507, -10.4565907, -12.3105316, -10.5074463, -1.3752518, 1.3191867
1: 3.2030110, 4.3900509, 3.2318425, 4.3631773, -0.6850041, 0.6828002
2: -4.9022379, -3.8205991, -4.8479671, -3.8461289, -0.8083012, 0.7780225
3: -12.6635647, -10.9469366, -12.6164761, -10.9975300, -1.0368316, 1.0575516
4: -2.4270182, -0.9255822, -2.3746860, -0.9399152, -1.0773492, 1.0292543
5: -10.0564690, -8.6112070, -10.0452728, -8.6808720, -0.7667618, 0.8124388
6: -8.0175724, -6.4037118, -7.9756036, -6.4619083, -1.0415540, 1.0599754
7: -2.7823269, -1.9410284, -2.7575955, -1.9554651, -0.5516403, 0.5398257
8: -3.7883234, -2.4409490, -3.7258668, -2.4546194, -0.8852986, 0.8460380
9: -12.4625769, -10.9788885, -12.4249916, -11.0154896, -0.9992495, 0.9940066

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4378738
time: 3.51 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4433018
time: 3.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.4088507, -10.4565907, -12.4095249, -10.4298096, -1.4059107, 1.3447320
1: 3.2030110, 4.3900509, 3.1951351, 4.3835583, -0.7016215, 0.6991301
2: -4.9022379, -3.8205991, -4.8888836, -3.8167491, -0.8257217, 0.7944978
3: -12.6635647, -10.9469366, -12.6966600, -10.9093037, -1.0540724, 1.0902224
4: -2.4270182, -0.9255822, -2.3913698, -0.9329700, -1.0853891, 1.0463083
5: -10.0564690, -8.6112070, -10.0670776, -8.6489162, -0.7863780, 0.8252789
6: -8.0175724, -6.4037118, -8.0503082, -6.4163713, -1.0847378, 1.0943501
7: -2.7823269, -1.9410284, -2.7625513, -1.9484382, -0.5590263, 0.5457034
8: -3.7883234, -2.4409490, -3.7716603, -2.4118910, -0.9026883, 0.8621188
9: -12.4625769, -10.9788885, -12.4524755, -10.9903221, -1.0244009, 1.0145090

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4378708
time: 4.21 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4432991
time: 4.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.4121151, -10.4190197, -12.4147615, -10.4148540, -1.4148226, 1.4019177
1: 3.2197318, 4.3831053, 3.2100120, 4.3845677, -0.6899897, 0.6969053
2: -4.8954554, -3.8100655, -4.8981762, -3.8053846, -0.8124917, 0.8150803
3: -12.6614485, -10.9710131, -12.6630802, -10.9552574, -1.0785551, 1.0702150
4: -2.4054356, -0.9413953, -2.4079621, -0.9279680, -1.0728614, 1.0655043
5: -10.0007410, -8.6674004, -10.0301323, -8.6645956, -0.7680341, 0.7821054
6: -7.9782352, -6.4237514, -8.0013990, -6.4227037, -1.0555582, 1.0843973
7: -2.7736650, -1.9448988, -2.7743151, -1.9381986, -0.5403805, 0.5503976
8: -3.7450213, -2.4754071, -3.7477002, -2.4536519, -0.8528454, 0.8599526
9: -12.4359560, -10.9836941, -12.4503250, -10.9810839, -0.9917638, 1.0113909

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4372642
time: 5.29 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4372664
time: 3.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.4121151, -10.4190197, -12.5081320, -10.3373222, -1.4437296, 1.4329832
1: 3.2197318, 4.3831053, 3.1750283, 4.4050779, -0.7021118, 0.7152026
2: -4.8954554, -3.8100655, -4.9380274, -3.7757928, -0.8253262, 0.8317294
3: -12.6614485, -10.9710131, -12.7431965, -10.8702278, -1.0989206, 1.0970144
4: -2.4054356, -0.9413953, -2.4250095, -0.9210155, -1.0810070, 1.0820615
5: -10.0007410, -8.6674004, -10.0520821, -8.6329393, -0.7817029, 0.7952547
6: -7.9782352, -6.4237514, -8.0720539, -6.3764033, -1.0929120, 1.1217693
7: -2.7736650, -1.9448988, -2.7792544, -1.9307911, -0.5485433, 0.5562826
8: -3.7450213, -2.4754071, -3.7908101, -2.4112768, -0.8702908, 0.8726149
9: -12.4359560, -10.9836941, -12.4789314, -10.9565382, -1.0169215, 1.0340806

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4336174, upper bound: 0.4372664
time: 3.45 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4336153, upper bound: 0.4433299
time: 3.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.4208069, -10.4090118, -12.4172173, -10.4104462, -1.4295850, 1.4150858
1: 3.1983204, 4.4005413, 3.2003403, 4.3859491, -0.7058789, 0.7156664
2: -4.9063373, -3.7981586, -4.9007654, -3.8007488, -0.8263063, 0.8288175
3: -12.6873178, -10.9384804, -12.6646042, -10.9396305, -1.1029972, 1.0946726
4: -2.4327846, -0.9139373, -2.4103875, -0.9146199, -1.1010187, 1.0862453
5: -10.0625944, -8.6078176, -10.0592432, -8.6618862, -0.7988455, 0.8257737
6: -8.0270529, -6.3852243, -8.0244055, -6.4217124, -1.0860248, 1.1203423
7: -2.7835782, -1.9298546, -2.7749648, -1.9314604, -0.5579017, 0.5641100
8: -3.7934542, -2.4320989, -3.7502208, -2.4321384, -0.8913337, 0.8859493
9: -12.4683266, -10.9608946, -12.4648561, -10.9785528, -1.0298882, 1.0414200

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4378702
time: 3.28 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4378699
time: 5.52 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.4208069, -10.4090118, -12.5105801, -10.3329220, -1.4585011, 1.4461415
1: 3.1983204, 4.4005413, 3.1652002, 4.4064598, -0.7184615, 0.7340226
2: -4.9063373, -3.7981586, -4.9406366, -3.7711551, -0.8391440, 0.8454384
3: -12.6873178, -10.9384804, -12.7447205, -10.8545723, -1.1233424, 1.1214712
4: -2.4327846, -0.9139373, -2.4275184, -0.9076672, -1.1091642, 1.1028612
5: -10.0625944, -8.6078176, -10.0811920, -8.6302490, -0.8125250, 0.8389223
6: -8.0270529, -6.3852243, -8.0950842, -6.3754244, -1.1240902, 1.1577555
7: -2.7835782, -1.9298546, -2.7799134, -1.9240305, -0.5656184, 0.5699983
8: -3.7934542, -2.4320989, -3.7933464, -2.3897476, -0.9087961, 0.8986379
9: -12.4683266, -10.9608946, -12.4934855, -10.9539738, -1.0519631, 1.0636243

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4378725
time: 3.43 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4385804, upper bound: 0.4378725
time: 3.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.4070721, -10.4338837, -12.4001541, -10.4665909, -1.3315752, 1.3874927
1: 3.2046809, 4.3821793, 3.2243690, 4.3726153, -0.6815238, 0.6852871
2: -4.8862963, -3.8205018, -4.8913755, -3.8324921, -0.7807970, 0.8066040
3: -12.6951389, -10.9245367, -12.6376934, -10.9794521, -1.0657555, 1.0403583
4: -2.3891435, -0.9463186, -2.3996539, -0.9530401, -1.0258317, 1.0539494
5: -10.0379705, -8.6514721, -9.9946146, -8.6708345, -0.7816434, 0.7582061
6: -8.0273218, -6.4173522, -7.9688454, -6.4422388, -1.0602267, 1.0536389
7: -2.7619021, -1.9548469, -2.7724230, -1.9560430, -0.5303423, 0.5404080
8: -3.7691383, -2.4328589, -3.7399006, -2.4842443, -0.8361557, 0.8590715
9: -12.4378777, -10.9928665, -12.4302130, -11.0016785, -0.9821782, 0.9861963

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4383436, upper bound: 0.4391483
time: 5.31 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4383436, upper bound: 0.4391488
time: 3.70 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.4070721, -10.4338837, -12.4954987, -10.3886604, -1.4000108, 1.4560912
1: 3.2046809, 4.3821793, 3.1886330, 4.3929400, -0.7030665, 0.7128972
2: -4.8862963, -3.8205018, -4.9317207, -3.8021038, -0.7950740, 0.8286285
3: -12.6951389, -10.9245367, -12.7177887, -10.8928518, -1.1178365, 1.0908952
4: -2.3891435, -0.9463186, -2.4171791, -0.9459064, -1.0360045, 1.0714712
5: -10.0379705, -8.6514721, -10.0163803, -8.6385689, -0.8034331, 0.7775013
6: -8.0273218, -6.4173522, -8.0409145, -6.3962088, -1.1118674, 1.1140223
7: -2.7619021, -1.9548469, -2.7774129, -1.9485710, -0.5379418, 0.5475692
8: -3.7691383, -2.4328589, -3.7839699, -2.4413333, -0.8731952, 0.8937280
9: -12.4378777, -10.9928665, -12.4575930, -10.9764709, -1.0071375, 1.0139811

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4383415, upper bound: 0.4400259
time: 4.10 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4383415, upper bound: 0.4400282
time: 3.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.5081310, -10.3370180, -12.4121151, -10.4190197, -1.4329832, 1.4400716
1: 3.1748142, 4.4050779, 3.2197318, 4.3831053, -0.7163844, 0.7021117
2: -4.9380288, -3.7749228, -4.8954554, -3.8100655, -0.8317305, 0.8199568
3: -12.7431965, -10.8698225, -12.6614485, -10.9710131, -1.0970142, 1.1094054
4: -2.4253149, -0.9210155, -2.4054356, -0.9413953, -1.0818217, 1.0810070
5: -10.0520811, -8.6327744, -10.0007410, -8.6674004, -0.7952545, 0.7842977
6: -8.0721941, -6.3764019, -7.9782352, -6.4237514, -1.1236207, 1.0929122
7: -2.7792544, -1.9304686, -2.7736650, -1.9448988, -0.5562834, 0.5471526
8: -3.7908096, -2.4107275, -3.7450213, -2.4754071, -0.8726156, 0.8651614
9: -12.4789581, -10.9565382, -12.4359560, -10.9836941, -1.0344174, 1.0169220

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4389718, upper bound: 0.4452153
time: 3.29 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4389718, upper bound: 0.4452152
time: 5.20 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.5081310, -10.3370180, -12.5054865, -10.3411503, -1.5013409, 1.5128169
1: 3.1748142, 4.4050779, 3.1847000, 4.4036131, -0.7381582, 0.7311527
2: -4.9380288, -3.7749228, -4.9352875, -3.7796082, -0.8461059, 0.8431352
3: -12.7431965, -10.8698225, -12.7415657, -10.8856068, -1.1510007, 1.1599602
4: -2.4253149, -0.9210155, -2.4227047, -0.9344392, -1.0905607, 1.0976403
5: -10.0520811, -8.6327744, -10.0226879, -8.6355572, -0.8176973, 0.8038071
6: -8.0721941, -6.3764019, -8.0489979, -6.3774395, -1.1756842, 1.1565342
7: -2.7792544, -1.9304686, -2.7785947, -1.9371943, -0.5643303, 0.5540932
8: -3.7908096, -2.4107275, -3.7881184, -2.4325018, -0.9096435, 0.9018546
9: -12.4789581, -10.9565382, -12.4645100, -10.9591789, -1.0572553, 1.0461888

Time for backsubstitution: 12.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4389718, upper bound: 0.4452181
time: 3.88 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4389718, upper bound: 0.4452182
time: 3.33 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.4095249, -10.4295082, -12.4088507, -10.4565907, -1.3447320, 1.4022524
1: 3.1949234, 4.3835592, 3.2030110, 4.3900509, -0.7003400, 0.7016218
2: -4.8888845, -3.8158786, -4.9022379, -3.8205991, -0.7944988, 0.8203835
3: -12.6966600, -10.9089012, -12.6635647, -10.9469366, -1.0902224, 1.0647942
4: -2.3916748, -0.9329700, -2.4270182, -0.9255822, -1.0459900, 1.0853891
5: -10.0670767, -8.6487513, -10.0564690, -8.6112070, -0.8252791, 0.7890135
6: -8.0504179, -6.4163713, -8.0175724, -6.4037118, -1.0962012, 1.0847380
7: -2.7625523, -1.9481163, -2.7823269, -1.9410284, -0.5457041, 0.5575857
8: -3.7716608, -2.4113412, -3.7883234, -2.4409490, -0.8621193, 0.8975576
9: -12.4525042, -10.9903221, -12.4625769, -10.9788885, -1.0149281, 1.0244019

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4383413, upper bound: 0.4435063
time: 3.33 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4383392, upper bound: 0.4391508
time: 3.48 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.4095249, -10.4295082, -12.5041761, -10.3786755, -1.4131789, 1.4708402
1: 3.1949234, 4.3835592, 3.1669445, 4.4103780, -0.7218869, 0.7294755
2: -4.8888845, -3.8158786, -4.9426136, -3.7901745, -0.8087498, 0.8423679
3: -12.6966600, -10.9089012, -12.7436562, -10.8602467, -1.1422729, 1.1153294
4: -2.3916748, -0.9329700, -2.4445951, -0.9184484, -1.0561633, 1.1013076
5: -10.0670767, -8.6487513, -10.0782204, -8.5790529, -0.8470910, 0.8083122
6: -8.0504179, -6.4163713, -8.0898476, -6.3577118, -1.1478262, 1.1452174
7: -2.7625523, -1.9481163, -2.7873318, -1.9334850, -0.5532343, 0.5651568
8: -3.7716608, -2.4113412, -3.8324118, -2.3979979, -0.8992126, 0.9322398
9: -12.4525042, -10.9903221, -12.4901400, -10.9536390, -1.0369725, 1.0523691

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4383392, upper bound: 0.4443861
time: 3.48 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4383392, upper bound: 0.4400282
time: 3.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.5105791, -10.3326206, -12.4208069, -10.4090118, -1.4461417, 1.4548423
1: 3.1649866, 4.4064598, 3.1983204, 4.4005413, -0.7352049, 0.7184616
2: -4.9406376, -3.7702842, -4.9063373, -3.7981586, -0.8454394, 0.8337750
3: -12.7447205, -10.8541679, -12.6873178, -10.9384804, -1.1214712, 1.1338397
4: -2.4278240, -0.9076672, -2.4327846, -0.9139373, -1.1026211, 1.1091645
5: -10.0811920, -8.6300850, -10.0625944, -8.6078176, -0.8389224, 0.8150846
6: -8.0952244, -6.3754244, -8.0270529, -6.3852243, -1.1596076, 1.1240902
7: -2.7799144, -1.9237089, -2.7835782, -1.9298546, -0.5699990, 0.5642037
8: -3.7933455, -2.3892002, -3.7934542, -2.4320989, -0.8986382, 0.9036667
9: -12.4935131, -10.9539747, -12.4683266, -10.9608946, -1.0639613, 1.0519637

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4389695, upper bound: 0.4495733
time: 3.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4389695, upper bound: 0.4452176
time: 3.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.5105791, -10.3326206, -12.5141659, -10.3311586, -1.5145099, 1.5275784
1: 3.1649866, 4.4064598, 3.1629696, 4.4210515, -0.7569834, 0.7476051
2: -4.9406376, -3.7702842, -4.9461956, -3.7676671, -0.8598088, 0.8568989
3: -12.7447205, -10.8541679, -12.7674332, -10.8529873, -1.1754304, 1.1843934
4: -2.4278240, -0.9076672, -2.4500985, -0.9069810, -1.1113601, 1.1259191
5: -10.0811920, -8.6300850, -10.0845251, -8.5760822, -0.8613551, 0.8345969
6: -8.0952244, -6.3754244, -8.0979509, -6.3389411, -1.2116556, 1.1878014
7: -2.7799144, -1.9237089, -2.7885220, -1.9220824, -0.5780579, 0.5720059
8: -3.7933455, -2.3892002, -3.8365698, -2.3891549, -0.9357046, 0.9403846
9: -12.4935131, -10.9539747, -12.4970636, -10.9363403, -1.0868111, 1.0820378

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4389695, upper bound: 0.4503027
time: 3.39 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4389695, upper bound: 0.4459460
time: 3.51 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 19.59 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4372677
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4426996
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4372657
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4336174, upper bound: 0.4426969
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4378738
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4433018
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4378708
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4432991
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4372642
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4372664
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4336174, upper bound: 0.4372664
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4336153, upper bound: 0.4433299
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4378702
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4378699
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4385805, upper bound: 0.4378725
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4385804, upper bound: 0.4378725
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4383436, upper bound: 0.4391483
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4383436, upper bound: 0.4391488
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4383415, upper bound: 0.4400259
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4383415, upper bound: 0.4400282
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4389718, upper bound: 0.4452153
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4389718, upper bound: 0.4452152
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4389718, upper bound: 0.4452181
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4389718, upper bound: 0.4452182
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4383413, upper bound: 0.4435063
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4383392, upper bound: 0.4391508
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4383392, upper bound: 0.4443861
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4383392, upper bound: 0.4400282
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4389695, upper bound: 0.4495733
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4389695, upper bound: 0.4452176
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4389695, upper bound: 0.4503027
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.59
Output dim: 1, lower bound: -0.4389695, upper bound: 0.4459460

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.3054190, -10.5160379, -12.3080730, -10.5118294, -1.2700868, 1.2675183
1: 3.2510633, 4.3603334, 3.2414255, 4.3617992, -0.6443294, 0.6511829
2: -4.8427029, -3.8554149, -4.8454027, -3.8507476, -0.7432377, 0.7406790
3: -12.6133089, -11.0288696, -12.6149521, -11.0131340, -0.9943414, 0.9792178
4: -2.3697009, -0.9666967, -2.3722448, -0.9532638, -1.0074456, 0.9958190
5: -9.9867754, -8.6865025, -10.0161638, -8.6836395, -0.7247899, 0.7498327
6: -7.9294634, -6.4639740, -7.9526148, -6.4629040, -0.9712534, 0.9910868
7: -2.7563167, -1.9688401, -2.7569575, -1.9621696, -0.5162294, 0.5111196
8: -3.7206903, -2.4978733, -3.7233586, -2.4761200, -0.8117565, 0.7930537
9: -12.3959808, -11.0206079, -12.4103851, -11.0180035, -0.9272017, 0.9344636

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4336208
time: 3.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4336154, upper bound: 0.4379768
time: 3.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.4121122, -10.4190674, -12.3080730, -10.5118294, -1.3051939, 1.3096733
1: 3.2197342, 4.3830996, 3.2414255, 4.3617992, -0.6649973, 0.6680653
2: -4.8954535, -3.8100710, -4.8454027, -3.8507476, -0.7698182, 0.7684783
3: -12.6614389, -10.9710159, -12.6149521, -11.0131340, -1.0130706, 1.0099900
4: -2.4054337, -0.9414015, -2.3722448, -0.9532638, -1.0365310, 1.0187483
5: -10.0007391, -8.6674023, -10.0161638, -8.6836395, -0.7418034, 0.7613277
6: -7.9782286, -6.4237652, -7.9526148, -6.4629040, -1.0184305, 1.0338745
7: -2.7736647, -1.9449046, -2.7569575, -1.9621696, -0.5294254, 0.5310442
8: -3.7450180, -2.4754152, -3.7233586, -2.4761200, -0.8218468, 0.8139040
9: -12.4359531, -10.9837093, -12.4103851, -11.0180035, -0.9660759, 0.9704937

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4336174, upper bound: 0.4390540
time: 3.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4336174, upper bound: 0.4434100
time: 3.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.3054190, -10.5160379, -12.4070721, -10.4341850, -1.3026552, 1.2976897
1: 3.2510633, 4.3603334, 3.2048960, 4.3821797, -0.6603875, 0.6695255
2: -4.8427029, -3.8554149, -4.8862953, -3.8213732, -0.7633119, 0.7644356
3: -12.6133089, -11.0288696, -12.6951389, -10.9249401, -1.0138614, 1.0122567
4: -2.3697009, -0.9666967, -2.3888383, -0.9463186, -1.0168896, 1.0127832
5: -9.9867754, -8.6865025, -10.0379686, -8.6516371, -0.7442418, 0.7626729
6: -7.9294634, -6.4639740, -8.0272121, -6.4173536, -1.0138872, 1.0357890
7: -2.7563167, -1.9688401, -2.7619014, -1.9551692, -0.5239041, 0.5169659
8: -3.7206903, -2.4978733, -3.7691360, -2.4334097, -0.8291242, 0.8088156
9: -12.3959808, -11.0206079, -12.4378490, -10.9928665, -0.9523392, 0.9624250

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4391493, upper bound: 0.4329096
time: 3.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4391493, upper bound: 0.4372656
time: 3.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.4121122, -10.4190674, -12.4070721, -10.4341850, -1.3341136, 1.3352287
1: 3.2197342, 4.3830996, 3.2048960, 4.3821797, -0.6766829, 0.6843212
2: -4.8954535, -3.8100710, -4.8862953, -3.8213732, -0.7823734, 0.7836103
3: -12.6614389, -10.9710159, -12.6951389, -10.9249401, -1.0303290, 1.0367074
4: -2.4054337, -0.9414015, -2.3888383, -0.9463186, -1.0445709, 1.0339236
5: -10.0007391, -8.6674023, -10.0379686, -8.6516371, -0.7566641, 0.7741678
6: -7.9782286, -6.4237652, -8.0272121, -6.4173536, -1.0448427, 1.0682068
7: -2.7736647, -1.9449046, -2.7619014, -1.9551692, -0.5368101, 0.5367730
8: -3.7450180, -2.4754152, -3.7691360, -2.4334097, -0.8392143, 0.8232497
9: -12.4359531, -10.9837093, -12.4378490, -10.9928665, -0.9836588, 0.9908421

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4391493, upper bound: 0.4383415
time: 3.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4391493, upper bound: 0.4426974
time: 3.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.3141193, -10.5060482, -12.3105316, -10.5074463, -1.2868233, 1.2803931
1: 3.2298574, 4.3777637, 3.2318425, 4.3631773, -0.6600574, 0.6719800
2: -4.8535471, -3.8435414, -4.8479671, -3.8461289, -0.7596865, 0.7543690
3: -12.6391783, -10.9964085, -12.6164761, -10.9975300, -1.0210569, 1.0041616
4: -2.3970945, -0.9392393, -2.3746860, -0.9399152, -1.0403008, 1.0158883
5: -10.0486298, -8.6267776, -10.0452728, -8.6808720, -0.7554855, 0.7934594
6: -7.9781294, -6.4254303, -7.9756036, -6.4619083, -1.0017657, 1.0373845
7: -2.7662158, -1.9538660, -2.7575955, -1.9554651, -0.5336093, 0.5264413
8: -3.7691050, -2.4545960, -3.7258668, -2.4546194, -0.8501922, 0.8188547
9: -12.4283772, -10.9978180, -12.4249916, -11.0154896, -0.9655190, 0.9754012

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4379733, upper bound: 0.4336164
time: 3.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4379737, upper bound: 0.4345355
time: 3.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.4208031, -10.4090614, -12.3105316, -10.5074463, -1.3199537, 1.3228493
1: 3.1983213, 4.4005365, 3.2318425, 4.3631773, -0.6813211, 0.6867814
2: -4.9063358, -3.7981637, -4.8479671, -3.8461289, -0.7836061, 0.7821691
3: -12.6873055, -10.9384842, -12.6164761, -10.9975300, -1.0375240, 1.0344509
4: -2.4327810, -0.9139440, -2.3746860, -0.9399152, -1.0647025, 1.0394711
5: -10.0625906, -8.6078205, -10.0452728, -8.6808720, -0.7725005, 0.8050050
6: -8.0270462, -6.3852391, -7.9756036, -6.4619083, -1.0496418, 1.0698190
7: -2.7835789, -1.9298601, -2.7575955, -1.9554651, -0.5413876, 0.5447288
8: -3.7934523, -2.4321089, -3.7258668, -2.4546194, -0.8603047, 0.8398685
9: -12.4683256, -10.9609098, -12.4249916, -11.0154896, -1.0012090, 0.9999167

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4379733, upper bound: 0.4390493
time: 3.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4379737, upper bound: 0.4390516
time: 3.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.3141193, -10.5060482, -12.4095249, -10.4298096, -1.3174150, 1.3108423
1: 3.2298574, 4.3777637, 3.1951351, 4.3835583, -0.6766047, 0.6883099
2: -4.8535471, -3.8435414, -4.8888836, -3.8167491, -0.7770514, 0.7781268
3: -12.6391783, -10.9964085, -12.6966600, -10.9093037, -1.0382977, 1.0367637
4: -2.3970945, -0.9392393, -2.3913698, -0.9329700, -1.0483413, 1.0329423
5: -10.0486298, -8.6267776, -10.0670776, -8.6489162, -0.7750895, 0.8062994
6: -7.9781294, -6.4254303, -8.0503082, -6.4163713, -1.0450501, 1.0717592
7: -2.7662158, -1.9538660, -2.7625513, -1.9484382, -0.5409952, 0.5323191
8: -3.7691050, -2.4545960, -3.7716603, -2.4118910, -0.8675818, 0.8347539
9: -12.4283772, -10.9978180, -12.4524755, -10.9903221, -0.9906704, 0.9992254

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4435066, upper bound: 0.4329075
time: 3.50 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4435071, upper bound: 0.4338268
time: 3.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.4208031, -10.4090614, -12.4095249, -10.4298096, -1.3488810, 1.3483949
1: 3.1983213, 4.4005365, 3.1951351, 4.3835583, -0.6930072, 0.7031114
2: -4.9063358, -3.7981637, -4.8888836, -3.8167491, -0.7961662, 0.7973080
3: -12.6873055, -10.9384842, -12.6966600, -10.9093037, -1.0547647, 1.0611676
4: -2.4327810, -0.9139440, -2.3913698, -0.9329700, -1.0727425, 1.0546865
5: -10.0625906, -8.6078205, -10.0670776, -8.6489162, -0.7875049, 0.8178451
6: -8.0270462, -6.3852391, -8.0503082, -6.4163713, -1.0760584, 1.1041937
7: -2.7835789, -1.9298601, -2.7625513, -1.9484382, -0.5487735, 0.5504628
8: -3.7934523, -2.4321089, -3.7716603, -2.4118910, -0.8776944, 0.8492320
9: -12.4683256, -10.9609098, -12.4524755, -10.9903221, -1.0159340, 1.0204189

Time for backsubstitution: 12.56 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=2, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=0.7800273895263672
rel_dist={1: [-0.4509328816559992, 0.4509326440462238]}

## Binary search (step 2) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5815
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5815

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3120336, upper bound: 0.3087195
time: 3.55 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3128601, upper bound: 0.3128588
time: 3.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.56 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.56
Output dim: 1, lower bound: -0.3120336, upper bound: 0.3087195
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.56
Output dim: 1, lower bound: -0.3128601, upper bound: 0.3128588

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.4172287, -10.4103403, -12.5019283, -10.3858471, -1.3372064, 1.4017956
1: 3.2003183, 4.3859591, 3.1705866, 4.3933010, -0.6445777, 0.6660265
2: -4.9007716, -3.8007307, -4.9374738, -3.7914991, -0.7931604, 0.8201355
3: -12.6646280, -10.9396000, -12.6884060, -10.8620052, -1.0772204, 1.0171981
4: -2.4103954, -0.9145901, -2.4220924, -0.9091063, -1.0260217, 1.0276417
5: -10.0592670, -8.6618805, -10.0664368, -8.6348610, -0.7389917, 0.7213717
6: -8.0244331, -6.4216833, -8.0828323, -6.4050708, -1.0252571, 1.0652679
7: -2.7749660, -1.9314392, -2.7790475, -1.9283173, -0.5214313, 0.5227751
8: -3.7502260, -2.4320984, -3.7889771, -2.4180245, -0.8080220, 0.8375584
9: -12.4648705, -10.9785194, -12.4782801, -10.9611387, -0.9614844, 0.9575908

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6193

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3071317, upper bound: 0.3065976
time: 3.19 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3120286, upper bound: 0.3087161
time: 3.44 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -12.5105877, -10.3325186, -12.5105896, -10.3325024, -1.4232864, 1.4255333
1: 3.1649652, 4.4064684, 3.1649628, 4.4064741, -0.6822556, 0.6822606
2: -4.9406443, -3.7702651, -4.9406447, -3.7702572, -0.8290207, 0.8177297
3: -12.7447414, -10.8541403, -12.7447643, -10.8541393, -1.0804958, 1.1122093
4: -2.4278302, -0.9076324, -2.4278324, -0.9076314, -1.0426164, 1.0424240
5: -10.0812159, -8.6300774, -10.0812206, -8.6300774, -0.7535467, 0.7580578
6: -8.0952511, -6.3753991, -8.0952549, -6.3753924, -1.1099367, 1.1149006
7: -2.7799153, -1.9236879, -2.7799163, -1.9236860, -0.5308601, 0.5314697
8: -3.7933502, -2.3891625, -3.7933526, -2.3891535, -0.8495405, 0.8541151
9: -12.4935284, -10.9539337, -12.4935312, -10.9539337, -0.9986632, 0.9993553

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3125454, upper bound: 0.3104957
time: 4.94 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3128567, upper bound: 0.3128536
time: 3.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.34 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 21.34
Output dim: 1, lower bound: -0.3071317, upper bound: 0.3065976
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.34
Output dim: 1, lower bound: -0.3120286, upper bound: 0.3087161
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.34
Output dim: 1, lower bound: -0.3125454, upper bound: 0.3104957
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.34
Output dim: 1, lower bound: -0.3128567, upper bound: 0.3128536

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.4172239, -10.4104118, -12.5019207, -10.3859444, -1.2747755, 1.3432772
1: 3.2003222, 4.3859529, 3.1705914, 4.3932896, -0.6300151, 0.6580334
2: -4.9007683, -3.8007371, -4.9374695, -3.7915106, -0.7527177, 0.7928396
3: -12.6646118, -10.9396057, -12.6883860, -10.8620119, -1.0574951, 0.9896073
4: -2.4103918, -0.9145994, -2.4220877, -0.9091244, -1.0171068, 1.0239275
5: -10.0592642, -8.6618834, -10.0664339, -8.6348639, -0.7397305, 0.7151564
6: -8.0244255, -6.4217043, -8.0828209, -6.4050989, -1.0072641, 1.0619125
7: -2.7749648, -1.9314477, -2.7790465, -1.9283285, -0.4982119, 0.5169877
8: -3.7502236, -2.4321117, -3.7889733, -2.4180408, -0.7804191, 0.8338106
9: -12.4648647, -10.9785433, -12.4782763, -10.9611750, -0.9477777, 0.9564567

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3095775, upper bound: 0.3083941
time: 3.44 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3120224, upper bound: 0.3087107
time: 3.58 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -12.5070210, -10.3387394, -12.5054913, -10.3410902, -1.4079843, 1.4109859
1: 3.1790667, 4.4044633, 3.1846967, 4.4036236, -0.6597781, 0.6574405
2: -4.9368687, -3.7769294, -4.9352889, -3.7795949, -0.8156416, 0.8049821
3: -12.7425251, -10.8766108, -12.7415962, -10.8855991, -1.0474761, 1.0848986
4: -2.4242048, -0.9267862, -2.4227085, -0.9344296, -1.0118234, 1.0182002
5: -10.0394335, -8.6339607, -10.0226955, -8.6355543, -0.7006944, 0.6961663
6: -8.0622149, -6.3768220, -8.0490074, -6.3774195, -1.0643470, 1.0621902
7: -2.7789714, -1.9333646, -2.7785952, -1.9371877, -0.5099328, 0.5135657
8: -3.7896729, -2.4200711, -3.7881212, -2.4324856, -0.8023636, 0.8129233
9: -12.4727192, -10.9576368, -12.4645195, -10.9591599, -0.9535749, 0.9501151

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3112331, upper bound: 0.3064871
time: 3.65 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3125436, upper bound: 0.3104945
time: 4.20 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -12.5105867, -10.3325310, -12.5141697, -10.3310986, -1.4223242, 1.4274435
1: 3.1649895, 4.4064665, 3.1629653, 4.4210625, -0.6822159, 0.6728219
2: -4.9406414, -3.7702775, -4.9461985, -3.7676549, -0.8301790, 0.8206028
3: -12.7447405, -10.8541737, -12.7674646, -10.8529835, -1.0696838, 1.1150396
4: -2.4278276, -0.9076548, -2.4501023, -0.9069726, -1.0303068, 1.0525618
5: -10.0811882, -8.6300812, -10.0845337, -8.5760784, -0.7544892, 0.7193882
6: -8.0952263, -6.3754015, -8.0979586, -6.3389215, -1.1085353, 1.0888411
7: -2.7799149, -1.9237018, -2.7885225, -1.9220757, -0.5245017, 0.5339575
8: -3.7933483, -2.3891907, -3.8365717, -2.3891382, -0.8243659, 0.8588847
9: -12.4935150, -10.9539385, -12.4970722, -10.9363194, -0.9932981, 0.9852670

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3116831, upper bound: 0.3089858
time: 3.01 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3128529, upper bound: 0.3128508
time: 4.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.02 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 20.02
Output dim: 1, lower bound: -0.3095775, upper bound: 0.3083941
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.02
Output dim: 1, lower bound: -0.3120224, upper bound: 0.3087107
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 20.02
Output dim: 1, lower bound: -0.3112331, upper bound: 0.3064871
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.02
Output dim: 1, lower bound: -0.3125436, upper bound: 0.3104945
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 20.02
Output dim: 1, lower bound: -0.3116831, upper bound: 0.3089858
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.02
Output dim: 1, lower bound: -0.3128529, upper bound: 0.3128508

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.4208050, -10.4090338, -12.5019188, -10.3859558, -1.2766740, 1.3422835
1: 3.1983223, 4.4005399, 3.1706157, 4.3932886, -0.6205602, 0.6579909
2: -4.9063358, -3.7981601, -4.9374661, -3.7915213, -0.7555838, 0.7939758
3: -12.6873131, -10.9384823, -12.6883850, -10.8620443, -1.0603230, 0.9787881
4: -2.4327826, -0.9139397, -2.4220843, -0.9091485, -1.0230680, 1.0116830
5: -10.0625944, -8.6078176, -10.0664062, -8.6348667, -0.7010611, 0.7160542
6: -8.0270481, -6.3852301, -8.0827961, -6.4051008, -0.9811072, 1.0605130
7: -2.7835777, -1.9298573, -2.7790461, -1.9283428, -0.5006413, 0.5081494
8: -3.7934546, -2.4321041, -3.7889709, -2.4180722, -0.7851819, 0.8086259
9: -12.4683266, -10.9609013, -12.4782648, -10.9611788, -0.9364395, 0.9441329

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3094221, upper bound: 0.3086872
time: 4.07 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3120203, upper bound: 0.3087094
time: 3.57 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.5070133, -10.3388329, -12.5054874, -10.3411551, -1.3494473, 1.3517535
1: 3.1790700, 4.4044552, 3.1846991, 4.4036174, -0.6488205, 0.6421430
2: -4.9368639, -3.7769389, -4.9352865, -3.7796016, -0.7836074, 0.7627594
3: -12.7425041, -10.8766165, -12.7415819, -10.8856058, -1.0213434, 1.0613623
4: -2.4242005, -0.9268045, -2.4227047, -0.9344430, -1.0036767, 1.0045424
5: -10.0394306, -8.6339626, -10.0226927, -8.6355562, -0.6926925, 0.6931670
6: -8.0622063, -6.3768463, -8.0489979, -6.3774371, -1.0495949, 1.0395128
7: -2.7789710, -1.9333758, -2.7785959, -1.9371958, -0.5032406, 0.4905486
8: -3.7896690, -2.4200864, -3.7881198, -2.4324965, -0.7946982, 0.7841828
9: -12.4727135, -10.9576731, -12.4645166, -10.9591856, -0.9438047, 0.9345647

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3099179, upper bound: 0.3104599
time: 4.45 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3125418, upper bound: 0.3104959
time: 3.93 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.5105782, -10.3326244, -12.5141640, -10.3311634, -1.3637896, 1.3682172
1: 3.1649938, 4.4064584, 3.1629696, 4.4210558, -0.6712215, 0.6574715
2: -4.9406376, -3.7702866, -4.9461956, -3.7676620, -0.7981207, 0.7783537
3: -12.7447186, -10.8541775, -12.7674484, -10.8529854, -1.0435579, 1.0915065
4: -2.4278235, -0.9076729, -2.4500992, -0.9069853, -1.0228832, 1.0378134
5: -10.0811863, -8.6300840, -10.0845308, -8.5760803, -0.7465029, 0.7164087
6: -8.0952168, -6.3754253, -8.0979509, -6.3389397, -1.0938666, 1.0662360
7: -2.7799137, -1.9237127, -2.7885220, -1.9220834, -0.5159568, 0.5108813
8: -3.7933450, -2.3892093, -3.8365684, -2.3891492, -0.8166997, 0.8301404
9: -12.4935112, -10.9539757, -12.4970675, -10.9363461, -0.9794703, 0.9658873

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3103139, upper bound: 0.3128280
time: 4.97 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3128512, upper bound: 0.3128498
time: 4.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.88 seconds
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 21.88
Output dim: 1, lower bound: -0.3094221, upper bound: 0.3086872
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 1, lower bound: -0.3120203, upper bound: 0.3087094
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 21.88
Output dim: 1, lower bound: -0.3099179, upper bound: 0.3104599
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 1, lower bound: -0.3125418, upper bound: 0.3104959
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 1, lower bound: -0.3103139, upper bound: 0.3128280
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 1, lower bound: -0.3128512, upper bound: 0.3128498

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.4207859, -10.4090347, -12.5031719, -10.3462687, -1.2786689, 1.3433068
1: 3.1983218, 4.4005356, 3.1301184, 4.3953304, -0.6211424, 0.6658818
2: -4.9063358, -3.7981756, -4.9821882, -3.7890112, -0.7562335, 0.7985915
3: -12.6873093, -10.9384823, -12.6897926, -10.8523388, -1.0644712, 0.9807730
4: -2.4327753, -0.9139559, -2.5218813, -0.9073517, -1.0226679, 1.0173063
5: -10.0625868, -8.6078205, -10.0683041, -8.6216736, -0.7075361, 0.7178898
6: -8.0270472, -6.3852444, -8.1100006, -6.4017506, -0.9856627, 1.0694733
7: -2.7835546, -1.9298575, -2.7843089, -1.8809941, -0.5008184, 0.5125139
8: -3.7934537, -2.4321089, -3.8227277, -2.4169660, -0.7853882, 0.8121557
9: -12.4683208, -10.9609022, -12.4868851, -10.9275608, -0.9359035, 0.9534622

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3115532, upper bound: 0.3071210
time: 3.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3120148, upper bound: 0.3087045
time: 3.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.5069933, -10.3388338, -12.5067501, -10.3014441, -1.3514099, 1.3527942
1: 3.1790709, 4.4044495, 3.1441731, 4.4056640, -0.6494045, 0.6499189
2: -4.9368639, -3.7769547, -4.9800777, -3.7771070, -0.7842934, 0.7673310
3: -12.7425022, -10.8766174, -12.7429895, -10.8759432, -1.0254695, 1.0633447
4: -2.4241929, -0.9268200, -2.5224824, -0.9326479, -1.0032778, 1.0100598
5: -10.0394220, -8.6339645, -10.0246105, -8.6223373, -0.6990820, 0.6950060
6: -8.0622034, -6.3768587, -8.0763464, -6.3740849, -1.0519574, 1.0484607
7: -2.7789474, -1.9333763, -2.7839832, -1.8898498, -0.5034752, 0.5000690
8: -3.7896676, -2.4200916, -3.8219099, -2.4313936, -0.7949151, 0.7875533
9: -12.4727058, -10.9576778, -12.4731598, -10.9255466, -0.9419475, 0.9440262

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3121081, upper bound: 0.3089683
time: 3.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3125363, upper bound: 0.3104878
time: 4.21 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.5092106, -10.3327122, -12.5086327, -10.3314705, -1.3427348, 1.3623121
1: 3.1650791, 4.4049001, 3.1633039, 4.4154930, -0.6663990, 0.6539173
2: -4.9406185, -3.7723596, -4.9461188, -3.7749095, -0.7626512, 0.7724394
3: -12.7444887, -10.8542719, -12.7667465, -10.8533316, -1.0396469, 1.0943573
4: -2.4273856, -0.9109023, -2.4485519, -0.9200480, -1.0090408, 1.0305855
5: -10.0805216, -8.6301451, -10.0822115, -8.5763092, -0.7452831, 0.7138981
6: -8.0949488, -6.3762507, -8.0969458, -6.3419886, -1.0908952, 1.0634010
7: -2.7784598, -1.9237525, -2.7827334, -1.9222305, -0.5126764, 0.4966399
8: -3.7931480, -2.3901620, -3.8359647, -2.3930354, -0.8125718, 0.8376676
9: -12.4915829, -10.9541502, -12.4906502, -10.9369717, -0.9713957, 0.9631021

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3098472, upper bound: 0.3112477
time: 4.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3103084, upper bound: 0.3128247
time: 4.07 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.5105553, -10.3326244, -12.5154238, -10.2914820, -1.3657372, 1.3692696
1: 3.1649947, 4.4064536, 3.1223583, 4.4231062, -0.6718110, 0.6655335
2: -4.9406366, -3.7703032, -4.9909697, -3.7651339, -0.7987924, 0.7829385
3: -12.7447166, -10.8541803, -12.7688589, -10.8432083, -1.0476629, 1.0934891
4: -2.4278162, -0.9076886, -2.5498197, -0.9051898, -1.0224838, 1.0433183
5: -10.0811777, -8.6300850, -10.0864840, -8.5628834, -0.7528539, 0.7182863
6: -8.0952139, -6.3754382, -8.1253605, -6.3356099, -1.0962126, 1.0751796
7: -2.7798905, -1.9237137, -2.7938375, -1.8747358, -0.5161818, 0.5150067
8: -3.7933445, -2.3892131, -3.8703465, -2.3880386, -0.8169341, 0.8335190
9: -12.4935017, -10.9539785, -12.5057182, -10.9027252, -0.9776273, 0.9752591

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3123821, upper bound: 0.3112664
time: 4.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3128456, upper bound: 0.3128444
time: 4.74 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 21.73 seconds
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 21.73
Output dim: 1, lower bound: -0.3115532, upper bound: 0.3071210
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 21.73
Output dim: 1, lower bound: -0.3120148, upper bound: 0.3087045
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 1, lower bound: -0.3121081, upper bound: 0.3089683
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 1, lower bound: -0.3125363, upper bound: 0.3104878
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 21.73
Output dim: 1, lower bound: -0.3098472, upper bound: 0.3112477
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 1, lower bound: -0.3103084, upper bound: 0.3128247
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 1, lower bound: -0.3123821, upper bound: 0.3112664
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 1, lower bound: -0.3128456, upper bound: 0.3128444

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.4997244, -10.3436108, -12.5054159, -10.3036613, -1.3409538, 1.3402700
1: 3.1911001, 4.3932009, 3.1471491, 4.3988848, -0.6285639, 0.6370646
2: -4.9095936, -3.7987034, -4.9615870, -3.7796316, -0.7539757, 0.7081885
3: -12.7369881, -10.8844185, -12.7418518, -10.8808165, -1.0126631, 1.0405790
4: -2.4017622, -0.9424851, -2.5085416, -0.9347389, -0.9769301, 0.9768200
5: -10.0296173, -8.6446724, -10.0211554, -8.6297255, -0.6773615, 0.6803843
6: -8.0382385, -6.3936262, -8.0701828, -6.3856473, -1.0094874, 1.0228703
7: -2.7762082, -1.9398701, -2.7828956, -1.8922093, -0.4911969, 0.4826973
8: -3.7830667, -2.4229498, -3.8199530, -2.4328775, -0.7850686, 0.7775105
9: -12.4555130, -10.9677277, -12.4626427, -10.9276142, -0.9232023, 0.9203304

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3076537, upper bound: 0.3083113
time: 3.53 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3121060, upper bound: 0.3089696
time: 4.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.5069885, -10.3388376, -12.5067463, -10.3014469, -1.3479426, 1.3483856
1: 3.1790771, 4.4044266, 3.1441760, 4.4056482, -0.6423869, 0.6411090
2: -4.9368100, -3.7769599, -4.9800396, -3.7771108, -0.7488244, 0.7499396
3: -12.7424994, -10.8766251, -12.7429876, -10.8759499, -1.0220762, 1.0586381
4: -2.4241500, -0.9268246, -2.5224514, -0.9326496, -0.9877009, 0.9978526
5: -10.0394135, -8.6339722, -10.0246048, -8.6223440, -0.6933012, 0.6868291
6: -8.0621948, -6.3768768, -8.0763388, -6.3740983, -1.0392213, 1.0236034
7: -2.7789454, -1.9333804, -2.7839820, -1.8898525, -0.4981645, 0.4878750
8: -3.7896657, -2.4200931, -3.8219061, -2.4313946, -0.7850479, 0.7823832
9: -12.4726877, -10.9576817, -12.4731455, -10.9255514, -0.9359505, 0.9387432

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3081146, upper bound: 0.3098407
time: 3.19 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3125343, upper bound: 0.3104884
time: 3.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.5092068, -10.3327141, -12.5086317, -10.3314734, -1.3391559, 1.3578520
1: 3.1650848, 4.4048777, 3.1633086, 4.4154768, -0.6593928, 0.6450282
2: -4.9405656, -3.7723651, -4.9460812, -3.7749128, -0.7271904, 0.7550576
3: -12.7444868, -10.8542805, -12.7667437, -10.8533363, -1.0362537, 1.0895798
4: -2.4273419, -0.9109061, -2.4485226, -0.9200501, -0.9934807, 1.0183737
5: -10.0805140, -8.6301537, -10.0822058, -8.5763168, -0.7395028, 0.7057464
6: -8.0949392, -6.3762679, -8.0969381, -6.3420010, -1.0782440, 1.0385988
7: -2.7784586, -1.9237559, -2.7827315, -1.9222331, -0.5073274, 0.4844208
8: -3.7931447, -2.3901629, -3.8359642, -2.3930359, -0.8026986, 0.8323539
9: -12.4915619, -10.9541550, -12.4906340, -10.9369745, -0.9654021, 0.9578284

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3058313, upper bound: 0.3122069
time: 3.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3103065, upper bound: 0.3128207
time: 3.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.5032139, -10.3373985, -12.5139742, -10.2937040, -1.3550773, 1.3567190
1: 3.1770515, 4.3952312, 3.1253238, 4.4163094, -0.6509943, 0.6525021
2: -4.9134793, -3.7920234, -4.9724541, -3.7676449, -0.7685142, 0.7238753
3: -12.7391319, -10.8620491, -12.7676611, -10.8481455, -1.0345360, 1.0719019
4: -2.4054427, -0.9233379, -2.5358469, -0.9072580, -0.9961743, 1.0101173
5: -10.0714102, -8.6408186, -10.0830517, -8.5702353, -0.7309566, 0.7036830
6: -8.0712452, -6.3921375, -8.1191044, -6.3471947, -1.0538368, 1.0495489
7: -2.7771821, -1.9302769, -2.7927816, -1.8770912, -0.5039353, 0.4976056
8: -3.7865272, -2.3920979, -3.8682737, -2.3895507, -0.8068097, 0.8234804
9: -12.4762907, -10.9640007, -12.4952707, -10.9047737, -0.9586720, 0.9516429

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3079038, upper bound: 0.3106574
time: 3.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3123802, upper bound: 0.3112648
time: 3.58 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.5105534, -10.3326283, -12.5154238, -10.2914867, -1.3621769, 1.3648152
1: 3.1649990, 4.4064302, 3.1223621, 4.4230900, -0.6648033, 0.6567485
2: -4.9405837, -3.7703078, -4.9909325, -3.7651381, -0.7633550, 0.7655580
3: -12.7447157, -10.8541870, -12.7688589, -10.8432121, -1.0442693, 1.0887115
4: -2.4277723, -0.9076920, -2.5497885, -0.9051929, -1.0069234, 1.0311060
5: -10.0811672, -8.6300945, -10.0864792, -8.5628881, -0.7470733, 0.7101250
6: -8.0952053, -6.3754559, -8.1253548, -6.3356237, -1.0835583, 1.0505271
7: -2.7798889, -1.9237165, -2.7938359, -1.8747392, -0.5108795, 0.5028340
8: -3.7933416, -2.3892169, -3.8703446, -2.3880410, -0.8070796, 0.8282549
9: -12.4934826, -10.9539833, -12.5057020, -10.9027290, -0.9716535, 0.9699863

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3083693, upper bound: 0.3122329
time: 3.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3128437, upper bound: 0.3128421
time: 3.73 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 19.93 seconds
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 19.93
Output dim: 1, lower bound: -0.3076537, upper bound: 0.3083113
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 19.93
Output dim: 1, lower bound: -0.3121060, upper bound: 0.3089696
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 19.93
Output dim: 1, lower bound: -0.3081146, upper bound: 0.3098407
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 19.93
Output dim: 1, lower bound: -0.3125343, upper bound: 0.3104884
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 19.93
Output dim: 1, lower bound: -0.3058313, upper bound: 0.3122069
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 19.93
Output dim: 1, lower bound: -0.3103065, upper bound: 0.3128207
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 19.93
Output dim: 1, lower bound: -0.3079038, upper bound: 0.3106574
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 19.93
Output dim: 1, lower bound: -0.3123802, upper bound: 0.3112648
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 19.93
Output dim: 1, lower bound: -0.3083693, upper bound: 0.3122329
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 19.93
Output dim: 1, lower bound: -0.3128437, upper bound: 0.3128421

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.4997168, -10.3436651, -12.5563545, -10.3026581, -1.3369863, 1.3474505
1: 3.1911039, 4.3931932, 3.1336346, 4.3993807, -0.6288536, 0.6430092
2: -4.9095917, -3.7987220, -4.9825735, -3.7793100, -0.7514968, 0.7130353
3: -12.7369614, -10.8844233, -12.7423058, -10.8472738, -1.0177419, 1.0353416
4: -2.3993578, -0.9424863, -2.5083351, -0.9288580, -0.9775949, 0.9780006
5: -10.0296078, -8.6446753, -10.0216236, -8.6142883, -0.6803663, 0.6789424
6: -8.0382290, -6.3936586, -8.0990200, -6.3837919, -1.0103166, 1.0290847
7: -2.7762079, -1.9398756, -2.7873237, -1.8916698, -0.4906135, 0.4842060
8: -3.7830648, -2.4229860, -3.8432407, -2.4319463, -0.7849648, 0.7816796
9: -12.4554949, -10.9677324, -12.4655790, -10.9191666, -0.9247072, 0.9227043

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3100596, upper bound: 0.3089696
time: 4.27 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3100596, upper bound: 0.3089664
time: 3.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.5069828, -10.3388939, -12.5576839, -10.3004436, -1.3439746, 1.3555694
1: 3.1790800, 4.4044218, 3.1306610, 4.4061456, -0.6426768, 0.6471581
2: -4.9368086, -3.7769792, -5.0010500, -3.7767844, -0.7463496, 0.7547719
3: -12.7424755, -10.8766298, -12.7434435, -10.8423920, -1.0271823, 1.0534006
4: -2.4217443, -0.9268248, -2.5222423, -0.9267690, -0.9883654, 0.9990337
5: -10.0394020, -8.6339760, -10.0250711, -8.6069069, -0.6962638, 0.6853878
6: -8.0621872, -6.3769088, -8.1051559, -6.3722386, -1.0400562, 1.0298488
7: -2.7789450, -1.9333854, -2.7884068, -1.8893039, -0.4975859, 0.4893138
8: -3.7896638, -2.4201279, -3.8451939, -2.4304581, -0.7849492, 0.7865590
9: -12.4726677, -10.9576855, -12.4761047, -10.9171057, -0.9374537, 0.9411407

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3104877, upper bound: 0.3104849
time: 6.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3104877, upper bound: 0.3104851
time: 4.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.5074186, -10.3477964, -12.5058937, -10.3543921, -1.3134313, 1.3372877
1: 3.1661196, 4.4021029, 3.1648870, 4.4112635, -0.6530458, 0.6396295
2: -4.9400001, -3.7778695, -4.9452143, -3.7832747, -0.7177830, 0.7475441
3: -12.7344809, -10.8557043, -12.7515402, -10.8555136, -1.0229065, 1.0728495
4: -2.4264362, -0.9110522, -2.4471505, -0.9202743, -0.9890666, 1.0130911
5: -10.0765467, -8.6308994, -10.0761833, -8.5774479, -0.7332787, 0.6981018
6: -8.0926399, -6.3844748, -8.0934744, -6.3544621, -1.0610700, 1.0240061
7: -2.7782879, -1.9254324, -2.7824738, -1.9247766, -0.5038910, 0.4815935
8: -3.7922010, -2.3972368, -3.8345165, -2.4037819, -0.7907805, 0.8229332
9: -12.4881582, -10.9556847, -12.4854565, -10.9393196, -0.9585166, 0.9502860

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6113

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3058313, upper bound: 0.3096843
time: 3.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3058313, upper bound: 0.3122078
time: 3.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.5092010, -10.3327703, -12.5595694, -10.3304691, -1.3351903, 1.3650362
1: 3.1650887, 4.4048729, 3.1498728, 4.4159746, -0.6596816, 0.6509708
2: -4.9405637, -3.7723846, -4.9670000, -3.7745914, -0.7247138, 0.7599808
3: -12.7444630, -10.8542843, -12.7671967, -10.8198147, -1.0414242, 1.0843437
4: -2.4249458, -0.9109068, -2.4483333, -0.9141688, -0.9941459, 1.0195420
5: -10.0805016, -8.6301575, -10.0826712, -8.5609083, -0.7424828, 0.7043022
6: -8.0949306, -6.3762999, -8.1258936, -6.3401184, -1.0791006, 1.0448296
7: -2.7784581, -1.9237614, -2.7871656, -1.9216783, -0.5067427, 0.4858779
8: -3.7931404, -2.3901978, -3.8591928, -2.3921003, -0.8025988, 0.8365532
9: -12.4915428, -10.9541588, -12.4935894, -10.9285564, -0.9669256, 0.9602213

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6113

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3103066, upper bound: 0.3103053
time: 3.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3103066, upper bound: 0.3128212
time: 3.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.5032082, -10.3374529, -12.5649128, -10.2927046, -1.3511205, 1.3639035
1: 3.1770535, 4.3952265, 3.1117172, 4.4168072, -0.6512846, 0.6586462
2: -4.9134779, -3.7920423, -4.9934330, -3.7673151, -0.7660424, 0.7286927
3: -12.7391062, -10.8620558, -12.7681141, -10.8145790, -1.0396174, 1.0666648
4: -2.4030488, -0.9233379, -2.5356345, -0.9013765, -0.9968395, 1.0113204
5: -10.0713997, -8.6408176, -10.0835190, -8.5548077, -0.7339574, 0.7022403
6: -8.0712376, -6.3921690, -8.1479902, -6.3453484, -1.0546598, 1.0558081
7: -2.7771819, -1.9302824, -2.7972021, -1.8765411, -0.5033576, 0.4991297
8: -3.7865248, -2.3921337, -3.8915586, -2.3886175, -0.8067083, 0.8276547
9: -12.4762735, -10.9640045, -12.4982224, -10.8963280, -0.9601717, 0.9540315

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3100584, upper bound: 0.3110185
time: 4.86 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3100584, upper bound: 0.3089696
time: 3.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.5087662, -10.3477097, -12.5126848, -10.3144026, -1.3364558, 1.3442495
1: 3.1660347, 4.4036565, 3.1239796, 4.4188757, -0.6584463, 0.6513615
2: -4.9400182, -3.7758119, -4.9900436, -3.7735031, -0.7539489, 0.7580003
3: -12.7347088, -10.8556108, -12.7536526, -10.8454094, -1.0309118, 1.0719807
4: -2.4268661, -0.9078383, -2.5484309, -0.9054160, -1.0025098, 1.0258343
5: -10.0772018, -8.6308403, -10.0804501, -8.5640287, -0.7408513, 0.7024754
6: -8.0929127, -6.3836627, -8.1218357, -6.3480806, -1.0663865, 1.0359514
7: -2.7797186, -1.9253938, -2.7935591, -1.8772876, -0.5074484, 0.5000182
8: -3.7923975, -2.3962889, -3.8688860, -2.3987875, -0.7951591, 0.8188255
9: -12.4900799, -10.9555140, -12.5005350, -10.9050789, -0.9647653, 0.9624592

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3060669, upper bound: 0.3118880
time: 3.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3060669, upper bound: 0.3098436
time: 3.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.5105495, -10.3326826, -12.5663586, -10.2904854, -1.3582199, 1.3720021
1: 3.1650028, 4.4064264, 3.1087551, 4.4235868, -0.6650937, 0.6629015
2: -4.9405823, -3.7703273, -5.0119352, -3.7648036, -0.7608883, 0.7703609
3: -12.7446899, -10.8541899, -12.7693119, -10.8096323, -1.0493735, 1.0834734
4: -2.4253788, -0.9076920, -2.5495744, -0.8993106, -1.0075889, 1.0323083
5: -10.0811577, -8.6300974, -10.0869446, -8.5474501, -0.7500601, 0.7086824
6: -8.0951958, -6.3754878, -8.1542091, -6.3337727, -1.0843959, 1.0568166
7: -2.7798884, -1.9237227, -2.7982550, -1.8741815, -0.5103062, 0.5042873
8: -3.7933383, -2.3892488, -3.8936281, -2.3871012, -0.8069832, 0.8324447
9: -12.4934654, -10.9539871, -12.5086746, -10.8942852, -0.9731498, 0.9723958

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3104866, upper bound: 0.3125332
time: 6.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3104866, upper bound: 0.3108355
time: 3.84 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 23.12 seconds
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 23.12
Output dim: 1, lower bound: -0.3100596, upper bound: 0.3089696
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 23.12
Output dim: 1, lower bound: -0.3100596, upper bound: 0.3089664
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 23.12
Output dim: 1, lower bound: -0.3104877, upper bound: 0.3104849
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 23.12
Output dim: 1, lower bound: -0.3104877, upper bound: 0.3104851
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 23.12
Output dim: 1, lower bound: -0.3058313, upper bound: 0.3096843
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 23.12
Output dim: 1, lower bound: -0.3058313, upper bound: 0.3122078
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 23.12
Output dim: 1, lower bound: -0.3103066, upper bound: 0.3103053
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 23.12
Output dim: 1, lower bound: -0.3103066, upper bound: 0.3128212
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 23.12
Output dim: 1, lower bound: -0.3100584, upper bound: 0.3110185
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 23.12
Output dim: 1, lower bound: -0.3100584, upper bound: 0.3089696
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 23.12
Output dim: 1, lower bound: -0.3060669, upper bound: 0.3118880
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 23.12
Output dim: 1, lower bound: -0.3060669, upper bound: 0.3098436
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 23.12
Output dim: 1, lower bound: -0.3104866, upper bound: 0.3125332
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 23.12
Output dim: 1, lower bound: -0.3104866, upper bound: 0.3108355

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.5098991, -10.3084040, -12.5058937, -10.3543921, -1.3148808, 1.3391185
1: 3.1254644, 4.4052472, 3.1648870, 4.4112635, -0.6629053, 0.6404713
2: -4.9839277, -3.7734060, -4.9452143, -3.7832747, -0.7250800, 0.7478327
3: -12.7358398, -10.8460054, -12.7515402, -10.8555136, -1.0235713, 1.0789819
4: -2.5264833, -0.9064560, -2.4471505, -0.9202743, -0.9950886, 1.0148022
5: -10.0790100, -8.6182413, -10.0761833, -8.5774479, -0.7356246, 0.7039580
6: -8.1200275, -6.3809261, -8.0934744, -6.3544621, -1.0709376, 1.0265238
7: -2.7845066, -1.8780582, -2.7824738, -1.9247766, -0.5083272, 0.4831021
8: -3.8257580, -2.3951864, -3.8345165, -2.4037819, -0.7941234, 0.8236774
9: -12.4977198, -10.9219112, -12.4854565, -10.9393196, -0.9650021, 0.9532145

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6193

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3014546, upper bound: 0.3106581
time: 3.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3014546, upper bound: 0.3122106
time: 3.60 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.5116835, -10.2933769, -12.5595694, -10.3304691, -1.3366404, 1.3668656
1: 3.1244097, 4.4080162, 3.1498728, 4.4159746, -0.6695321, 0.6518131
2: -4.9845052, -3.7679179, -4.9670000, -3.7745914, -0.7320490, 0.7602698
3: -12.7458229, -10.8445740, -12.7671967, -10.8198147, -1.0420886, 1.0904808
4: -2.5249727, -0.9063096, -2.4483333, -0.9141688, -1.0001578, 1.0212533
5: -10.0829716, -8.6174946, -10.0826712, -8.5609083, -0.7448304, 0.7101569
6: -8.1223640, -6.3727570, -8.1258936, -6.3401184, -1.0889523, 1.0473471
7: -2.7846875, -1.8763845, -2.7871656, -1.9216783, -0.5111692, 0.4873828
8: -3.8267059, -2.3881474, -3.8591928, -2.3921003, -0.8059512, 0.8372988
9: -12.5011063, -10.9203806, -12.4935894, -10.9285564, -0.9734085, 0.9631549

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6193

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3063455, upper bound: 0.3116363
time: 3.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3087310, upper bound: 0.3123602
time: 4.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3087310, upper bound: 0.3128208
time: 4.18 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.5054531, -10.3412571, -12.5663586, -10.2905340, -1.3476586, 1.3607018
1: 3.1847119, 4.4035759, 3.1087556, 4.4235849, -0.6450901, 0.6550788
2: -4.9352293, -3.7796521, -5.0119352, -3.7648454, -0.7515495, 0.7603872
3: -12.7415257, -10.8856211, -12.7692785, -10.8096323, -1.0404509, 1.0519679
4: -2.4202423, -0.9344673, -2.5495751, -0.8993111, -0.9975233, 1.0045896
5: -10.0226574, -8.6355715, -10.0869446, -8.5474510, -0.6919322, 0.7027134
6: -8.0489731, -6.3775148, -8.1542091, -6.3338041, -1.0375118, 1.0463789
7: -2.7785676, -1.9372101, -2.7982533, -1.8741813, -0.5036244, 0.4877561
8: -3.7881122, -2.4325495, -3.8936286, -2.3871012, -0.7916031, 0.7889537
9: -12.4644632, -10.9592104, -12.5086746, -10.8942881, -0.9386652, 0.9511291

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6193

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3064769, upper bound: 0.3112203
time: 3.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3064770, upper bound: 0.3112208
time: 6.55 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 28.80 seconds
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 28.80
Output dim: 1, lower bound: -0.3014546, upper bound: 0.3106581
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 28.80
Output dim: 1, lower bound: -0.3014546, upper bound: 0.3122106
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 28.80
Output dim: 1, lower bound: -0.3087310, upper bound: 0.3123602
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 28.80
Output dim: 1, lower bound: -0.3087310, upper bound: 0.3128208
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 28.80
Output dim: 1, lower bound: -0.3064769, upper bound: 0.3112203
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 28.80
Output dim: 1, lower bound: -0.3064770, upper bound: 0.3112208

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.5098991, -10.3084040, -12.5058918, -10.3544149, -1.3053470, 1.3300714
1: 3.1254644, 4.4052472, 3.1648893, 4.4112597, -0.6575527, 0.6393952
2: -4.9839277, -3.7734060, -4.9452133, -3.7832780, -0.7115684, 0.7444482
3: -12.7358398, -10.8460054, -12.7515335, -10.8555155, -1.0260706, 1.0789819
4: -2.5264833, -0.9064560, -2.4471488, -0.9202790, -0.9950881, 1.0180910
5: -10.0790100, -8.6182413, -10.0761833, -8.5774479, -0.7405808, 0.7039570
6: -8.1200275, -6.3809261, -8.0934725, -6.3544707, -1.0620089, 1.0253484
7: -2.7845066, -1.8780582, -2.7824733, -1.9247789, -0.4977341, 0.4831017
8: -3.8257580, -2.3951864, -3.8345160, -2.4037867, -0.7941226, 0.8446571
9: -12.4977198, -10.9219112, -12.4854536, -10.9393311, -0.9581997, 0.9519106

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2989563, upper bound: 0.3118544
time: 4.23 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2989563, upper bound: 0.3087157
time: 4.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.5116806, -10.2934628, -12.5523386, -10.3352509, -1.3250856, 1.3617668
1: 3.1244473, 4.4079776, 3.1618533, 4.4047613, -0.6596344, 0.6420627
2: -4.9844809, -3.7682192, -4.9397907, -3.7963002, -0.7115221, 0.7325213
3: -12.7458229, -10.8448048, -12.7617207, -10.8277426, -1.0204406, 1.0859311
4: -2.5249119, -0.9063096, -2.4259229, -0.9298229, -0.9884825, 0.9972386
5: -10.0828018, -8.6175051, -10.0729399, -8.5715752, -0.7335072, 0.6996644
6: -8.1222973, -6.3727632, -8.1019058, -6.3568578, -1.0711806, 1.0311929
7: -2.7846713, -1.8766043, -2.7844887, -1.9282269, -0.4943409, 0.4839113
8: -3.8263969, -2.3881493, -3.8526058, -2.3950019, -0.8022401, 0.8299688
9: -12.5010996, -10.9203959, -12.4761467, -10.9385490, -0.9628429, 0.9443572

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3063490, upper bound: 0.3120725
time: 3.81 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3063490, upper bound: 0.3100244
time: 4.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.5116835, -10.2933769, -12.5595684, -10.3304701, -1.3366394, 1.3678191
1: 3.1244097, 4.4080162, 3.1498747, 4.4159660, -0.6667113, 0.6509110
2: -4.9845052, -3.7679179, -4.9669847, -3.7745934, -0.7317033, 0.7418220
3: -12.7458229, -10.8445740, -12.7671976, -10.8198156, -1.0405762, 1.0904799
4: -2.5249727, -0.9063096, -2.4483199, -0.9141693, -0.9985738, 1.0162926
5: -10.0829716, -8.6174946, -10.0826683, -8.5609102, -0.7423667, 0.7101542
6: -8.1223640, -6.3727570, -8.1258907, -6.3401237, -1.0764792, 1.0471668
7: -2.7846875, -1.8763845, -2.7871649, -1.9216795, -0.5111676, 0.4942615
8: -3.8267059, -2.3881474, -3.8591928, -2.3921013, -0.8104048, 0.8372974
9: -12.5011063, -10.9203806, -12.4935837, -10.9285583, -0.9741402, 0.9631522

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3042546, upper bound: 0.3083484
time: 4.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3042546, upper bound: 0.3128199
time: 5.47 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 31.83 seconds
IS_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 31.83
Output dim: 1, lower bound: -0.2989563, upper bound: 0.3118544
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 31.83
Output dim: 1, lower bound: -0.2989563, upper bound: 0.3087157
IS_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 31.83
Output dim: 1, lower bound: -0.3063490, upper bound: 0.3120725
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 31.83
Output dim: 1, lower bound: -0.3063490, upper bound: 0.3100244
IS_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 31.83
Output dim: 1, lower bound: -0.3042546, upper bound: 0.3083484
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 31.83
Output dim: 1, lower bound: -0.3042546, upper bound: 0.3128199

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.5065899, -10.3020201, -12.5523386, -10.3353014, -1.3145342, 1.3504429
1: 3.1442146, 4.4051299, 3.1618533, 4.4047594, -0.6390305, 0.6342354
2: -4.9791384, -3.7775731, -4.9397907, -3.7963421, -0.7020960, 0.7225550
3: -12.7426558, -10.8763275, -12.7616882, -10.8277426, -1.0109022, 1.0544701
4: -2.5198276, -0.9330850, -2.4259224, -0.9298220, -0.9783397, 0.9696014
5: -10.0242996, -8.6229763, -10.0729418, -8.5715761, -0.6753737, 0.6938934
6: -8.0760612, -6.3747654, -8.1019068, -6.3568907, -1.0242167, 1.0209904
7: -2.7834241, -1.8900895, -2.7844868, -1.9282267, -0.4875931, 0.4669988
8: -3.8211713, -2.4314499, -3.8526068, -2.3950014, -0.7869004, 0.7864585
9: -12.4721384, -10.9256067, -12.4761467, -10.9385529, -0.9283450, 0.9230182

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6193

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3023361, upper bound: 0.3107611
time: 3.89 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3023362, upper bound: 0.3107646
time: 3.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.5626278, -10.2923288, -12.5595684, -10.3304701, -1.3435364, 1.3638628
1: 3.1108122, 4.4085207, 3.1498747, 4.4159660, -0.6730282, 0.6513214
2: -5.0055184, -3.7675691, -4.9669847, -3.7745934, -0.7363851, 0.7393544
3: -12.7463026, -10.8109932, -12.7671976, -10.8198156, -1.0353436, 1.0954328
4: -2.5271771, -0.9004273, -2.4483199, -0.9141693, -0.9999518, 1.0172498
5: -10.0834503, -8.6020432, -10.0826683, -8.5609102, -0.7409312, 0.7130376
6: -8.1512556, -6.3708744, -8.1258907, -6.3401237, -1.0825114, 1.0479987
7: -2.7891102, -1.8758249, -2.7871649, -1.9216795, -0.5130740, 0.4941446
8: -3.8499980, -2.3871775, -3.8591928, -2.3921013, -0.8144662, 0.8372062
9: -12.5040970, -10.9119291, -12.4935837, -10.9285583, -0.9764905, 0.9646200

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6193

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3004148, upper bound: 0.3067961
time: 4.31 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3004148, upper bound: 0.3128205
time: 4.36 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 27.19 seconds
IS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 27.19
Output dim: 1, lower bound: -0.3023361, upper bound: 0.3107611
IS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 10, time: 27.19
Output dim: 1, lower bound: -0.3023362, upper bound: 0.3107646
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 27.19
Output dim: 1, lower bound: -0.3004148, upper bound: 0.3067961
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 27.19
Output dim: 1, lower bound: -0.3004148, upper bound: 0.3128205

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.5626278, -10.2923288, -12.5595665, -10.3304958, -1.3340032, 1.3548851
1: 3.1108122, 4.4085207, 3.1498756, 4.4159636, -0.6676757, 0.6502497
2: -5.0055184, -3.7675691, -4.9669838, -3.7745955, -0.7228734, 0.7359793
3: -12.7463026, -10.8109932, -12.7671909, -10.8198185, -1.0378736, 1.0954325
4: -2.5271771, -0.9004273, -2.4483185, -0.9141748, -0.9999514, 1.0205245
5: -10.0834503, -8.6020432, -10.0826683, -8.5609131, -0.7458845, 0.7130368
6: -8.1512556, -6.3708744, -8.1258879, -6.3401308, -1.0735822, 1.0468712
7: -2.7891102, -1.8758249, -2.7871644, -1.9216826, -0.5024359, 0.4941440
8: -3.8499980, -2.3871775, -3.8591928, -2.3921046, -0.8144653, 0.8582242
9: -12.5040970, -10.9119291, -12.4935837, -10.9285698, -0.9696884, 0.9632950

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2993386, upper bound: 0.3124993
time: 4.17 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2993386, upper bound: 0.3104516
time: 4.21 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 20.99 seconds
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 20.99
Output dim: 1, lower bound: -0.2993386, upper bound: 0.3124993
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 20.99
Output dim: 1, lower bound: -0.2993386, upper bound: 0.3104516

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.5575371, -10.3008766, -12.5595665, -10.3305454, -1.3234324, 1.3435595
1: 3.1306753, 4.4056711, 3.1498752, 4.4159613, -0.6469800, 0.6424459
2: -5.0001631, -3.7769248, -4.9669838, -3.7746379, -0.7135627, 0.7260115
3: -12.7431374, -10.8425446, -12.7671585, -10.8198185, -1.0291049, 1.0639670
4: -2.5220981, -0.9272032, -2.4483182, -0.9141750, -0.9897885, 0.9928873
5: -10.0249481, -8.6075315, -10.0826664, -8.5609131, -0.6877414, 0.7070622
6: -8.1049519, -6.3728724, -8.1258888, -6.3401637, -1.0265992, 1.0364778
7: -2.7878575, -1.8893162, -2.7871628, -1.9216828, -0.4957287, 0.4772839
8: -3.8447638, -2.4304800, -3.8591928, -2.3921037, -0.7993412, 0.8147146
9: -12.4751215, -10.9171476, -12.4935837, -10.9285727, -0.9353256, 0.9419950

Time for backsubstitution: 12.50 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=2, k_high=3, k_mid=2, eps_mid=0.0078125, abs_max=0.6936578750610352
rel_dist={1: [-0.3128648053580161, 0.31286157369120016]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 1666.77 seconds
