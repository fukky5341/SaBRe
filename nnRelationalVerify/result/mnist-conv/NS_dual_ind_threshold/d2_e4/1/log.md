## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.5346226540000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0611053, 1.0611053)
1: (-9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1834817, 1.1834817)
2: (-4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4875827, 1.4875827)
3: (-11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4724336, 1.4724336)
4: (-8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5915928, 1.5915928)
5: (-0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3831000, 1.3831000)
6: (5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2248650, 1.2248650)
7: (-18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1333981, 1.1333976)
8: (-1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7780180, 1.7780180)
9: (-8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0668039, 1.0668039)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.81 + 33.33 = 56.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.5373092, upper bound: 0.5373085

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5373057
time: 3.54 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5373052, upper bound: 0.5373052
time: 3.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.37 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.37
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5373057
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.37
Output dim: 6, lower bound: -0.5373052, upper bound: 0.5373052

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.7332306, -3.2545981, -4.7338657, -3.2512143, -1.0583587, 1.0551679
1: -9.6261415, -7.8926196, -9.6277485, -7.8912416, -1.1797686, 1.1804724
2: -4.8868947, -3.2939901, -4.8912487, -3.2926872, -1.4809184, 1.4831171
3: -11.5036144, -9.6244411, -11.5047359, -9.6225996, -1.4702477, 1.4692516
4: -8.0182896, -6.0331798, -8.0193853, -6.0287986, -1.5853009, 1.5820498
5: -0.4142461, 1.0373187, -0.4151219, 1.0413597, -1.3801675, 1.3770638
6: 5.8290596, 7.1746392, 5.8219867, 7.1746693, -1.2157493, 1.2225838
7: -18.3063316, -16.2130280, -18.3083210, -16.2119331, -1.1292343, 1.1298628
8: -1.0611186, 0.7249036, -1.0620036, 0.7289176, -1.7737207, 1.7717280
9: -8.3870287, -6.9252753, -8.3875837, -6.9192224, -1.0632544, 1.0580244

Time for backsubstitution: 21.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5328673
time: 3.64 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5373057
time: 3.61 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.7620764, -3.2465339, -4.7340469, -3.2502491, -1.0759754, 1.0692582
1: -9.6346922, -7.8848681, -9.6282072, -7.8908520, -1.1884336, 1.1927881
2: -4.9031639, -3.2683783, -4.8924828, -3.2923195, -1.4965181, 1.5042901
3: -11.5137825, -9.6204147, -11.5050507, -9.6220722, -1.4818850, 1.4736729
4: -8.0546007, -6.0243802, -8.0196953, -6.0275483, -1.6125636, 1.6094661
5: -0.4412295, 1.0492529, -0.4153707, 1.0425110, -1.3971329, 1.3895788
6: 5.8082347, 7.2095165, 5.8199720, 7.1746774, -1.2367029, 1.2326741
7: -18.3103275, -16.1992168, -18.3088818, -16.2116222, -1.1379371, 1.1438727
8: -1.0775609, 0.7386322, -1.0622549, 0.7300658, -1.7919273, 1.7881212
9: -8.4240532, -6.9141598, -8.3877420, -6.9174995, -1.0761955, 1.0701394

Time for backsubstitution: 21.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5373059, upper bound: 0.5328667
time: 3.95 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5373059, upper bound: 0.5373051
time: 3.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.04 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 29.04
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5328673
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.04
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5373057
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.04
Output dim: 6, lower bound: -0.5373059, upper bound: 0.5328667
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.04
Output dim: 6, lower bound: -0.5373059, upper bound: 0.5373051

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.7332306, -3.2545981, -4.7620764, -3.2465339, -1.0632668, 1.0703640
1: -9.6261415, -7.8926196, -9.6346922, -7.8848681, -1.1858778, 1.1861434
2: -4.8868947, -3.2939901, -4.9031639, -3.2683783, -1.4983354, 1.4933743
3: -11.5036144, -9.6244411, -11.5137825, -9.6204147, -1.4723144, 1.4790549
4: -8.0182896, -6.0331798, -8.0546007, -6.0243802, -1.5876770, 1.6039891
5: -0.4142461, 1.0373187, -0.4412295, 1.0492529, -1.3879757, 1.3914738
6: 5.8290596, 7.1746392, 5.8082347, 7.2095165, -1.2236152, 1.2374911
7: -18.3063316, -16.2130280, -18.3103275, -16.1992168, -1.1403289, 1.1319613
8: -1.0611186, 0.7249036, -1.0775609, 0.7386322, -1.7829814, 1.7863245
9: -8.3870287, -6.9252753, -8.4240532, -6.9141598, -1.0677843, 1.0677860

Time for backsubstitution: 21.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5326531, upper bound: 0.5353186
time: 3.86 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328646, upper bound: 0.5373025
time: 3.75 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.7620764, -3.2465339, -4.7332306, -3.2545981, -1.0703640, 1.0632668
1: -9.6346922, -7.8848681, -9.6261415, -7.8926196, -1.1861434, 1.1858773
2: -4.9031639, -3.2683783, -4.8868947, -3.2939901, -1.4933743, 1.4983354
3: -11.5137825, -9.6204147, -11.5036144, -9.6244411, -1.4790549, 1.4723144
4: -8.0546007, -6.0243802, -8.0182896, -6.0331798, -1.6039891, 1.5876770
5: -0.4412295, 1.0492529, -0.4142461, 1.0373187, -1.3914738, 1.3879757
6: 5.8082347, 7.2095165, 5.8290596, 7.1746392, -1.2374911, 1.2236152
7: -18.3103275, -16.1992168, -18.3063316, -16.2130280, -1.1319618, 1.1403289
8: -1.0775609, 0.7386322, -1.0611186, 0.7249036, -1.7863245, 1.7829814
9: -8.4240532, -6.9141598, -8.3870287, -6.9252753, -1.0677857, 1.0677843

Time for backsubstitution: 21.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5370906, upper bound: 0.5308798
time: 4.07 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5373020, upper bound: 0.5328634
time: 3.77 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4.7620764, -3.2465339, -4.7620764, -3.2465339, -1.0749669, 1.0749671
1: -9.6346922, -7.8848681, -9.6346922, -7.8848681, -1.1984000, 1.1984000
2: -4.9031639, -3.2683783, -4.9031639, -3.2683783, -1.5028892, 1.5028887
3: -11.5137825, -9.6204147, -11.5137825, -9.6204147, -1.4794559, 1.4794559
4: -8.0546007, -6.0243802, -8.0546007, -6.0243802, -1.6160936, 1.6160936
5: -0.4412295, 1.0492529, -0.4412295, 1.0492529, -1.4035792, 1.4035158
6: 5.8082347, 7.2095165, 5.8082347, 7.2095165, -1.2445674, 1.2456303
7: -18.3103275, -16.1992168, -18.3103275, -16.1992168, -1.1414557, 1.1414557
8: -1.0775609, 0.7386322, -1.0775609, 0.7386322, -1.7919521, 1.7919521
9: -8.4240532, -6.9141598, -8.4240532, -6.9141598, -1.0730672, 1.0730677

Time for backsubstitution: 21.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5370915, upper bound: 0.5308798
time: 3.81 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5373030, upper bound: 0.5328635
time: 3.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.49 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.49
Output dim: 6, lower bound: -0.5326531, upper bound: 0.5353186
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.49
Output dim: 6, lower bound: -0.5328646, upper bound: 0.5373025
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.49
Output dim: 6, lower bound: -0.5370906, upper bound: 0.5308798
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.49
Output dim: 6, lower bound: -0.5373020, upper bound: 0.5328634
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.49
Output dim: 6, lower bound: -0.5370915, upper bound: 0.5308798
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.49
Output dim: 6, lower bound: -0.5373030, upper bound: 0.5328635

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.7314258, -3.2575948, -4.7615652, -3.2473886, -1.0602832, 1.0665157
1: -9.6188383, -7.8950686, -9.6326265, -7.8855562, -1.1775875, 1.1819277
2: -4.8808131, -3.2954841, -4.9014454, -3.2688007, -1.4914384, 1.4896526
3: -11.5002308, -9.6279354, -11.5128193, -9.6214113, -1.4670181, 1.4730215
4: -8.0088921, -6.0351691, -8.0519218, -6.0249429, -1.5775785, 1.5989270
5: -0.4124045, 1.0327108, -0.4407116, 1.0479475, -1.3811655, 1.3826652
6: 5.8320394, 7.1698112, 5.8090754, 7.2081466, -1.2189035, 1.2313910
7: -18.2920361, -16.2148533, -18.3062744, -16.1997337, -1.1255274, 1.1263227
8: -1.0590477, 0.7226987, -1.0769587, 0.7380075, -1.7789159, 1.7814727
9: -8.3802204, -6.9266338, -8.4221230, -6.9145389, -1.0605512, 1.0645423

Time for backsubstitution: 21.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5308787, upper bound: 0.5353156
time: 3.77 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5308787, upper bound: 0.5353151
time: 3.99 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.7427177, -3.2454305, -4.7620726, -3.2465417, -1.0755434, 1.0810668
1: -9.6300716, -7.8565598, -9.6346750, -7.8848724, -1.1903262, 1.2139733
2: -4.9002500, -3.2729273, -4.9031515, -3.2683816, -1.5103111, 1.5158062
3: -11.5197802, -9.6206064, -11.5137749, -9.6204252, -1.4968071, 1.4811049
4: -8.0353632, -6.0051775, -8.0545769, -6.0243850, -1.6024060, 1.6167545
5: -0.4241269, 1.0414745, -0.4412255, 1.0492388, -1.4065952, 1.3942699
6: 5.8030329, 7.1775441, 5.8082433, 7.2095094, -1.2397432, 1.2405105
7: -18.3091240, -16.1417809, -18.3102932, -16.1992226, -1.1420174, 1.1664732
8: -1.0744929, 0.7281022, -1.0775552, 0.7386274, -1.7998261, 1.7907071
9: -8.4032001, -6.8976917, -8.4240389, -6.9141631, -1.0822453, 1.0768082

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5317068, upper bound: 0.5370097
time: 3.89 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328613, upper bound: 0.5372979
time: 3.92 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.7602596, -3.2495265, -4.7327213, -3.2554517, -1.0672727, 1.0594645
1: -9.6273975, -7.8873100, -9.6240740, -7.8933086, -1.1778612, 1.1816626
2: -4.8970957, -3.2698770, -4.8851728, -3.2944117, -1.4864450, 1.4944692
3: -11.5104141, -9.6239157, -11.5026455, -9.6254368, -1.4737687, 1.4662757
4: -8.0451670, -6.0263734, -8.0156202, -6.0337420, -1.5938053, 1.5828362
5: -0.4393952, 1.0446494, -0.4137256, 1.0360119, -1.3844123, 1.3792863
6: 5.8112178, 7.2046905, 5.8299003, 7.1732697, -1.2330289, 1.2174599
7: -18.2960300, -16.2010441, -18.3022804, -16.2135429, -1.1171551, 1.1346874
8: -1.0754452, 0.7364316, -1.0605278, 0.7242770, -1.7822347, 1.7781377
9: -8.4172535, -6.9155054, -8.3850975, -6.9256582, -1.0605104, 1.0647907

Time for backsubstitution: 22.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5353158, upper bound: 0.5308779
time: 3.81 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5353158, upper bound: 0.5308778
time: 3.51 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.7715664, -3.2373648, -4.7332253, -3.2546053, -1.0779824, 1.0740645
1: -9.6387129, -7.8487806, -9.6261263, -7.8926249, -1.1906791, 1.2111497
2: -4.9165468, -3.2473269, -4.8868837, -3.2939944, -1.5050955, 1.5104685
3: -11.5299950, -9.6165466, -11.5036068, -9.6244507, -1.5035930, 1.4744048
4: -8.0716753, -5.9963799, -8.0182648, -6.0331850, -1.6193237, 1.6158834
5: -0.4511205, 1.0534856, -0.4142418, 1.0373049, -1.4011664, 1.3906546
6: 5.7822871, 7.2124214, 5.8290663, 7.1746316, -1.2594070, 1.2264180
7: -18.3131218, -16.1280022, -18.3062992, -16.2130318, -1.1337185, 1.1670449
8: -1.0909929, 0.7417817, -1.0611134, 0.7249002, -1.8023281, 1.7872834
9: -8.4401836, -6.8865857, -8.3870134, -6.9252787, -1.0819833, 1.0823958

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5361373, upper bound: 0.5325749
time: 3.79 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5372987, upper bound: 0.5328605
time: 4.32 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.7602596, -3.2495265, -4.7615652, -3.2473886, -1.0719266, 1.0711453
1: -9.6273975, -7.8873100, -9.6326265, -7.8855562, -1.1901193, 1.1941872
2: -4.8970957, -3.2698770, -4.9014454, -3.2688007, -1.4959602, 1.4991851
3: -11.5104141, -9.6239157, -11.5128193, -9.6214113, -1.4741497, 1.4734130
4: -8.0451670, -6.0263734, -8.0519218, -6.0249429, -1.6059518, 1.6112447
5: -0.4393952, 1.0446494, -0.4407116, 1.0479475, -1.3965187, 1.3947067
6: 5.8112178, 7.2046905, 5.8090754, 7.2081466, -1.2398934, 1.2394862
7: -18.2960300, -16.2010441, -18.3062744, -16.1997337, -1.1266475, 1.1358190
8: -1.0754452, 0.7364316, -1.0769587, 0.7380075, -1.7878919, 1.7871122
9: -8.4172535, -6.9155054, -8.4221230, -6.9145389, -1.0658350, 1.0700607

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5353161, upper bound: 0.5308779
time: 3.56 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5353168, upper bound: 0.5308778
time: 4.47 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.7715664, -3.2373648, -4.7620726, -3.2465417, -1.0871592, 1.0858045
1: -9.6387129, -7.8487806, -9.6346750, -7.8848724, -1.2029357, 1.2204075
2: -4.9165468, -3.2473269, -4.9031515, -3.2683816, -1.5147347, 1.5252886
3: -11.5299950, -9.6165466, -11.5137749, -9.6204252, -1.5039248, 1.4815474
4: -8.0716753, -5.9963799, -8.0545769, -6.0243850, -1.6308331, 1.6297269
5: -0.4511205, 1.0534856, -0.4412255, 1.0492388, -1.4132719, 1.4064198
6: 5.7822871, 7.2124214, 5.8082433, 7.2095094, -1.2608571, 1.2484322
7: -18.3131218, -16.1280022, -18.3102932, -16.1992226, -1.1432290, 1.1739695
8: -1.0909929, 0.7417817, -1.0775552, 0.7386274, -1.8087711, 1.7963233
9: -8.4401836, -6.8865857, -8.4240389, -6.9141631, -1.0874882, 1.0873051

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5361375, upper bound: 0.5325748
time: 3.60 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5372990, upper bound: 0.5328608
time: 3.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.41 seconds
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.41
Output dim: 6, lower bound: -0.5308787, upper bound: 0.5353156
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.41
Output dim: 6, lower bound: -0.5308787, upper bound: 0.5353151
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.41
Output dim: 6, lower bound: -0.5317068, upper bound: 0.5370097
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.41
Output dim: 6, lower bound: -0.5328613, upper bound: 0.5372979
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.41
Output dim: 6, lower bound: -0.5353158, upper bound: 0.5308779
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.41
Output dim: 6, lower bound: -0.5353158, upper bound: 0.5308778
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.41
Output dim: 6, lower bound: -0.5361373, upper bound: 0.5325749
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.41
Output dim: 6, lower bound: -0.5372987, upper bound: 0.5328605
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.41
Output dim: 6, lower bound: -0.5353161, upper bound: 0.5308779
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.41
Output dim: 6, lower bound: -0.5353168, upper bound: 0.5308778
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.41
Output dim: 6, lower bound: -0.5361375, upper bound: 0.5325748
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.41
Output dim: 6, lower bound: -0.5372990, upper bound: 0.5328608

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.7314258, -3.2575948, -4.7602596, -3.2495265, -1.0579929, 1.0650232
1: -9.6188383, -7.8950686, -9.6273975, -7.8873100, -1.1761332, 1.1764040
2: -4.8808131, -3.2954841, -4.8970957, -3.2698770, -1.4900746, 1.4850717
3: -11.5002308, -9.6279354, -11.5104141, -9.6239157, -1.4634800, 1.4702339
4: -8.0088921, -6.0351691, -8.0451670, -6.0263734, -1.5760384, 1.5923290
5: -0.4124045, 1.0327108, -0.4393952, 1.0446494, -1.3758941, 1.3793726
6: 5.8320394, 7.1698112, 5.8112178, 7.2046905, -1.2153044, 1.2292604
7: -18.2920361, -16.2148533, -18.2960300, -16.2010441, -1.1243978, 1.1160278
8: -1.0590477, 0.7226987, -1.0754452, 0.7364316, -1.7760406, 1.7793579
9: -8.3802204, -6.9266338, -8.4172535, -6.9155054, -1.0598183, 1.0597668

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5305899, upper bound: 0.5341492
time: 3.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5308756, upper bound: 0.5353138
time: 3.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.7314258, -3.2575948, -4.7715664, -3.2373648, -1.0719986, 1.0746479
1: -9.6188383, -7.8950686, -9.6387129, -7.8487806, -1.2033837, 1.1891613
2: -4.8808131, -3.2954841, -4.9165468, -3.2473269, -1.5041108, 1.5037436
3: -11.5002308, -9.6279354, -11.5299950, -9.6165466, -1.4705257, 1.4935665
4: -8.0088921, -6.0351691, -8.0716753, -5.9963799, -1.6063294, 1.6155467
5: -0.4124045, 1.0327108, -0.4511205, 1.0534856, -1.3859425, 1.3919773
6: 5.8320394, 7.1698112, 5.7822871, 7.2124214, -1.2219520, 1.2540689
7: -18.2920361, -16.2148533, -18.3131218, -16.1280022, -1.1526232, 1.1343226
8: -1.0590477, 0.7226987, -1.0909929, 0.7417817, -1.7843719, 1.7977657
9: -8.3802204, -6.9266338, -8.4401836, -6.8865857, -1.0754132, 1.0795529

Time for backsubstitution: 21.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5305899, upper bound: 0.5341511
time: 4.79 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5308756, upper bound: 0.5353139
time: 3.67 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.7420940, -3.2479591, -4.7602143, -3.2515600, -1.0693989, 1.0751908
1: -9.6294155, -7.8581786, -9.6333237, -7.8880668, -1.1859040, 1.2101703
2: -4.8965197, -3.2738619, -4.8956571, -3.2714643, -1.5008197, 1.5065150
3: -11.5184765, -9.6217556, -11.5112228, -9.6228313, -1.4933853, 1.4772191
4: -8.0285416, -6.0059338, -8.0411510, -6.0279684, -1.5911636, 1.6031418
5: -0.4225504, 1.0388007, -0.4377325, 1.0444579, -1.4002519, 1.3868294
6: 5.8045239, 7.1730256, 5.8128810, 7.2007518, -1.2294874, 1.2315519
7: -18.3056812, -16.1429920, -18.3034325, -16.2020950, -1.1356573, 1.1589520
8: -1.0734234, 0.7264895, -1.0757556, 0.7353244, -1.7954702, 1.7869473
9: -8.4020882, -6.8984923, -8.4214554, -6.9155283, -1.0795989, 1.0736437

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4558

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5317061, upper bound: 0.5366339
time: 3.53 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5317061, upper bound: 0.5370089
time: 3.77 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.7427182, -3.2454305, -4.7620730, -3.2465432, -1.0734148, 1.0804718
1: -9.6300697, -7.8565598, -9.6346731, -7.8848753, -1.1902947, 1.2143078
2: -4.9002485, -3.2729278, -4.9031477, -3.2683825, -1.5092607, 1.5118484
3: -11.5197802, -9.6206055, -11.5137730, -9.6204262, -1.4983792, 1.4811015
4: -8.0353594, -6.0051765, -8.0545692, -6.0243878, -1.6017818, 1.6086187
5: -0.4241252, 1.0414743, -0.4412231, 1.0492370, -1.4056268, 1.3938527
6: 5.8030329, 7.1775432, 5.8082438, 7.2095017, -1.2378283, 1.2405078
7: -18.3091240, -16.1417809, -18.3102875, -16.1992245, -1.1416841, 1.1624608
8: -1.0744948, 0.7281008, -1.0775537, 0.7386255, -1.8023081, 1.7899194
9: -8.4032001, -6.8976936, -8.4240379, -6.9141636, -1.0818987, 1.0776336

Time for backsubstitution: 22.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5325759, upper bound: 0.5361362
time: 3.91 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5325759, upper bound: 0.5372999
time: 3.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.7602596, -3.2495265, -4.7314258, -3.2575948, -1.0650229, 1.0579927
1: -9.6273975, -7.8873100, -9.6188383, -7.8950686, -1.1764045, 1.1761332
2: -4.8970957, -3.2698770, -4.8808131, -3.2954841, -1.4850717, 1.4900746
3: -11.5104141, -9.6239157, -11.5002308, -9.6279354, -1.4702339, 1.4634800
4: -8.0451670, -6.0263734, -8.0088921, -6.0351691, -1.5923290, 1.5760384
5: -0.4393952, 1.0446494, -0.4124045, 1.0327108, -1.3793731, 1.3758941
6: 5.8112178, 7.2046905, 5.8320394, 7.1698112, -1.2292604, 1.2153044
7: -18.2960300, -16.2010441, -18.2920361, -16.2148533, -1.1160278, 1.1243978
8: -1.0754452, 0.7364316, -1.0590477, 0.7226987, -1.7793579, 1.7760406
9: -8.4172535, -6.9155054, -8.3802204, -6.9266338, -1.0597668, 1.0598183

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5350213, upper bound: 0.5297014
time: 3.47 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5353128, upper bound: 0.5308767
time: 3.48 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.7602596, -3.2495265, -4.7427177, -3.2454305, -1.0784423, 1.0717294
1: -9.6273975, -7.8873100, -9.6300716, -7.8565598, -1.2062140, 1.1888113
2: -4.8970957, -3.2698770, -4.9002500, -3.2729273, -1.5094380, 1.5078144
3: -11.5104141, -9.6239157, -11.5197802, -9.6206064, -1.4772353, 1.4868112
4: -8.0451670, -6.0263734, -8.0353632, -6.0051775, -1.6071568, 1.6006241
5: -0.4393952, 1.0446494, -0.4241269, 1.0414745, -1.3887315, 1.3902054
6: 5.8112178, 7.2046905, 5.8030329, 7.1775441, -1.2375498, 1.2344050
7: -18.2960300, -16.2010441, -18.3091240, -16.1417809, -1.1520510, 1.1426172
8: -1.0754452, 0.7364316, -1.0744929, 0.7281022, -1.7876868, 1.7943430
9: -8.4172535, -6.9155054, -8.4032001, -6.8976917, -1.0698290, 1.0814900

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5350213, upper bound: 0.5297015
time: 3.55 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5353128, upper bound: 0.5308767
time: 3.51 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.7709427, -3.2398968, -4.7314062, -3.2596159, -1.0718381, 1.0540929
1: -9.6380882, -7.8503971, -9.6247177, -7.8958206, -1.1862822, 1.1921008
2: -4.9128122, -3.2482646, -4.8794651, -3.2970774, -1.4976783, 1.4740782
3: -11.5286922, -9.6177034, -11.5010595, -9.6268406, -1.5001884, 1.4714618
4: -8.0648270, -5.9971385, -8.0049009, -6.0367379, -1.6121531, 1.6023378
5: -0.4495529, 1.0508183, -0.4107319, 1.0325539, -1.3297558, 1.3839283
6: 5.7837687, 7.2079010, 5.8336897, 7.1658745, -1.2491612, 1.1747026
7: -18.3096809, -16.1292133, -18.2994347, -16.2158890, -1.1278400, 1.1595311
8: -1.0899186, 0.7401705, -1.0593195, 0.7216210, -1.7618895, 1.7835326
9: -8.4390583, -6.8873734, -8.3844728, -6.9266644, -1.0793226, 1.0343366

Time for backsubstitution: 21.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4558

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5361367, upper bound: 0.5321993
time: 3.62 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5361367, upper bound: 0.5325742
time: 3.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.7715659, -3.2373645, -4.7332258, -3.2546074, -1.0758023, 1.0740638
1: -9.6387129, -7.8487806, -9.6261244, -7.8926272, -1.1906486, 1.2114742
2: -4.9165449, -3.2473278, -4.8868794, -3.2939954, -1.5050931, 1.5064816
3: -11.5299950, -9.6165466, -11.5036049, -9.6244526, -1.5051708, 1.4744015
4: -8.0716734, -5.9963808, -8.0182571, -6.0331855, -1.6167865, 1.6078291
5: -0.4511197, 1.0534859, -0.4142404, 1.0373036, -1.4002123, 1.3904338
6: 5.7822876, 7.2124190, 5.8290687, 7.1746264, -1.2575011, 1.2249260
7: -18.3131180, -16.1279964, -18.3062935, -16.2130356, -1.1333899, 1.1630559
8: -1.0909925, 0.7417803, -1.0611115, 0.7248964, -1.8042774, 1.7865071
9: -8.4401836, -6.8865857, -8.3870125, -6.9252796, -1.0816214, 1.0832369

Time for backsubstitution: 22.05 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.14 + 561.93 = 618.07 seconds
