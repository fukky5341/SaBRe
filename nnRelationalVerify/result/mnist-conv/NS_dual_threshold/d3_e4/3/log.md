## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.872541919


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8442516, 1.8442519)
1: (-17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7918735, 2.7918735)
2: (-3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3578615, 2.3578615)
3: (-10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.6013856, 2.6013861)
4: (-12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7610822, 2.7610822)
5: (-4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0475702, 2.0475702)
6: (-3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2829180, 2.2829187)
7: (-9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1810565, 3.1810570)
8: (-2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0486369, 2.0486372)
9: (-4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.3078995, 2.3078992)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.93 + 39.82 = 63.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.8742898, upper bound: 0.8742899

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 471

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742861, upper bound: 0.8698156
time: 5.40 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742861, upper bound: 0.8742862
time: 8.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 14.03 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 14.03
Output dim: 0, lower bound: -0.8742861, upper bound: 0.8698156
NS_A2, status: Status.UNKNOWN, split count: 1, time: 14.03
Output dim: 0, lower bound: -0.8742861, upper bound: 0.8742862

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 6.6551008, 9.0135641, 6.6483846, 9.0208187, -1.7911239, 1.7899327
1: -17.4738808, -13.8005123, -17.4782677, -13.7856493, -2.6310873, 2.6196427
2: -3.2627926, -0.5239112, -3.2672889, -0.5199797, -2.3435087, 2.3436897
3: -10.8545094, -7.9736710, -10.8565397, -7.9592099, -2.4428215, 2.4299312
4: -12.5303125, -9.0268307, -12.5336456, -9.0226107, -2.7368712, 2.7304783
5: -4.9079671, -2.6729443, -4.9315729, -2.6704021, -1.8402214, 1.8614385
6: -3.0038815, -0.5669255, -3.0362353, -0.5638537, -2.0689974, 2.1008341
7: -9.3258781, -5.4602880, -9.3304539, -5.4337683, -2.9822493, 2.9613752
8: -2.5947638, -0.3486171, -2.5973587, -0.3460078, -2.0295734, 2.0295675
9: -4.4674053, -1.7695547, -4.4723735, -1.7615669, -2.2728467, 2.2686410

Time for backsubstitution: 20.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8722797, upper bound: 0.8698133
time: 4.74 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742836, upper bound: 0.8698130
time: 4.93 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 6.5904884, 9.0354805, 6.6375179, 9.0313339, -1.8851533, 1.8454711
1: -17.5395660, -13.7612858, -17.4851456, -13.7643471, -2.8484921, 2.7954130
2: -3.2866726, -0.4902341, -3.2758987, -0.5142205, -2.3659143, 2.3812420
3: -10.9079895, -7.9272223, -10.8677921, -7.9381189, -2.6484585, 2.6109838
4: -12.5829248, -8.9888020, -12.5387926, -9.0154057, -2.8057680, 2.7976289
5: -4.9724522, -2.5807762, -4.9653707, -2.6635835, -2.0558090, 2.0907953
6: -3.0983093, -0.4341698, -3.0826259, -0.5545940, -2.2864566, 2.3228679
7: -9.4304600, -5.3878288, -9.3434505, -5.3957224, -3.2355380, 3.1883569
8: -2.6340671, -0.3340940, -2.6018806, -0.3418708, -2.0750732, 2.0553889
9: -4.5135827, -1.7374012, -4.4801068, -1.7481221, -2.3314004, 2.3200212

Time for backsubstitution: 20.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 471

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8722797, upper bound: 0.8742847
time: 7.63 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742836, upper bound: 0.8742866
time: 7.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 36.26 seconds
NS_A1_A1, status: Status.VERIFIED, split count: 2, time: 36.26
Output dim: 0, lower bound: -0.8722797, upper bound: 0.8698133
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 36.26
Output dim: 0, lower bound: -0.8742836, upper bound: 0.8698130
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 36.26
Output dim: 0, lower bound: -0.8722797, upper bound: 0.8742847
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 36.26
Output dim: 0, lower bound: -0.8742836, upper bound: 0.8742866

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: 6.6523457, 9.0279541, 6.6483898, 9.0208197, -1.7914786, 1.8040123
1: -17.4762974, -13.7852192, -17.4782562, -13.7856503, -2.6307154, 2.6351571
2: -3.2660036, -0.5133848, -3.2672853, -0.5199825, -2.3465419, 2.3550711
3: -10.8567801, -7.9611397, -10.8565378, -7.9592118, -2.4440632, 2.4453132
4: -12.6000023, -9.0257072, -12.5336437, -9.0226345, -2.7848182, 2.7187796
5: -4.9100413, -2.6617532, -4.9315672, -2.6704035, -1.8405037, 1.8729239
6: -3.0051947, -0.5469370, -3.0362306, -0.5638537, -2.0678930, 2.1194656
7: -9.4240227, -5.4593954, -9.3304520, -5.4338074, -3.0250664, 2.9472060
8: -2.5963988, -0.3227386, -2.5973530, -0.3460088, -2.0259266, 2.0383558
9: -4.5145383, -1.7673821, -4.4723716, -1.7615879, -2.2968650, 2.2629371

Time for backsubstitution: 22.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 471

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8699019, upper bound: 0.8698126
time: 5.21 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8699019, upper bound: 0.8698164
time: 6.90 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: 6.5967355, 9.0346031, 6.6405220, 9.0309143, -1.8781757, 1.8413749
1: -17.5305214, -13.7629423, -17.4807930, -13.7651358, -2.8385363, 2.7893825
2: -3.2812908, -0.4936302, -3.2733004, -0.5158422, -2.3589578, 2.3751948
3: -10.9046402, -7.9296236, -10.8661757, -7.9392834, -2.6437244, 2.6071739
4: -12.5802689, -9.0147047, -12.5375166, -9.0278845, -2.7905130, 2.7703247
5: -4.9658089, -2.5817404, -4.9621696, -2.6640487, -2.0479035, 2.0857315
6: -3.0887156, -0.4356256, -3.0780017, -0.5552907, -2.2762194, 2.3162513
7: -9.4276257, -5.4237032, -9.3420916, -5.4130049, -3.2125063, 3.1510077
8: -2.6234274, -0.3358102, -2.5967550, -0.3427033, -2.0633216, 2.0482078
9: -4.5114355, -1.7562871, -4.4790773, -1.7572376, -2.3187506, 2.3001580

Time for backsubstitution: 22.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 471

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8718215, upper bound: 0.8739664
time: 8.29 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8722782, upper bound: 0.8742828
time: 7.03 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: 6.5878334, 9.0499210, 6.6375237, 9.0313330, -1.8855031, 1.8595874
1: -17.5419102, -13.7459974, -17.4851360, -13.7643471, -2.8480501, 2.8109226
2: -3.2899225, -0.4797206, -3.2758949, -0.5142236, -2.3688793, 2.3862183
3: -10.9102325, -7.9146242, -10.8677893, -7.9381218, -2.6496077, 2.6267524
4: -12.6525650, -8.9876604, -12.5387888, -9.0154305, -2.8298120, 2.7859211
5: -4.9745359, -2.5696206, -4.9653630, -2.6635838, -2.0559921, 2.0928979
6: -3.0996425, -0.4142132, -3.0826194, -0.5545945, -2.2854505, 2.3274643
7: -9.5286961, -5.3869090, -9.3434496, -5.3957644, -3.2489023, 3.1742334
8: -2.6356587, -0.3081903, -2.6018753, -0.3418732, -2.0714240, 2.0655837
9: -4.5606909, -1.7352256, -4.4801044, -1.7481432, -2.3382781, 2.3142765

Time for backsubstitution: 22.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 471

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8738254, upper bound: 0.8739663
time: 4.93 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742821, upper bound: 0.8742833
time: 6.98 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 34.78 seconds
NS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 34.78
Output dim: 0, lower bound: -0.8699019, upper bound: 0.8698126
NS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 34.78
Output dim: 0, lower bound: -0.8699019, upper bound: 0.8698164
NS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 34.78
Output dim: 0, lower bound: -0.8718215, upper bound: 0.8739664
NS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 34.78
Output dim: 0, lower bound: -0.8722782, upper bound: 0.8742828
NS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 34.78
Output dim: 0, lower bound: -0.8738254, upper bound: 0.8739663
NS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 34.78
Output dim: 0, lower bound: -0.8742821, upper bound: 0.8742833

## BFS NS instance: NS_A2_A1_A1

### Backsubstitution after applying NS history:
0: 6.6018014, 9.0313683, 6.6431580, 9.0293522, -1.8720462, 1.8361030
1: -17.5267963, -13.7792149, -17.4790154, -13.7735777, -2.8263540, 2.7711701
2: -3.2754140, -0.5011847, -3.2703350, -0.5197541, -2.3485589, 2.3651071
3: -10.9012451, -7.9479966, -10.8645592, -7.9483585, -2.6303544, 2.5870214
4: -12.5739088, -9.0200882, -12.5344257, -9.0313540, -2.7784677, 2.7615380
5: -4.9622688, -2.5904756, -4.9604430, -2.6682453, -2.0385232, 2.0738146
6: -3.0802796, -0.4380994, -3.0738943, -0.5564680, -2.2672005, 2.3096755
7: -9.4128742, -5.4308100, -9.3350506, -5.4170637, -3.1943426, 3.1378136
8: -2.6182928, -0.3412924, -2.5943027, -0.3453274, -2.0555005, 2.0397696
9: -4.4954944, -1.7594358, -4.4714704, -1.7589010, -2.3008454, 2.2897420

Time for backsubstitution: 22.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 471

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 859

## Relational analysis of NS_A2_A1_A1_B1

### Relational analysis result of NS_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8706827, upper bound: 0.8708997
time: 8.03 seconds

## Relational analysis of NS_A2_A1_A1_B2

### Relational analysis result of NS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8706827, upper bound: 0.8728254
time: 5.26 seconds

## BFS NS instance: NS_A2_A1_A2

### Backsubstitution after applying NS history:
0: 6.5807586, 9.0383606, 6.6405268, 9.0309105, -1.8906565, 1.8446646
1: -17.5756226, -13.7522535, -17.4807873, -13.7651625, -2.8596277, 2.7950969
2: -3.2919698, -0.4854952, -3.2732944, -0.5158520, -2.3660235, 2.3856380
3: -10.9421940, -7.9183407, -10.8661728, -7.9393063, -2.6698432, 2.6154871
4: -12.5902748, -9.0009670, -12.5375109, -9.0278931, -2.8078938, 2.7869720
5: -4.9871950, -2.5759869, -4.9621668, -2.6640615, -2.0701318, 2.0899239
6: -3.0936782, -0.4225092, -3.0779941, -0.5552931, -2.2832823, 2.3240471
7: -9.4361486, -5.3810720, -9.3420830, -5.4130149, -3.2183809, 3.1943984
8: -2.6337156, -0.3302522, -2.5967488, -0.3427114, -2.0727506, 2.0532172
9: -4.5263004, -1.7307678, -4.4790554, -1.7572411, -2.3293033, 2.3076739

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 471

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 859

## Relational analysis of NS_A2_A1_A2_B1

### Relational analysis result of NS_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8711423, upper bound: 0.8712157
time: 7.60 seconds

## Relational analysis of NS_A2_A1_A2_B2

### Relational analysis result of NS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8711423, upper bound: 0.8731461
time: 13.20 seconds

## BFS NS instance: NS_A2_A2_A1

### Backsubstitution after applying NS history:
0: 6.5929012, 9.0466766, 6.6401596, 9.0297718, -1.8793712, 1.8543012
1: -17.5381908, -13.7622662, -17.4833584, -13.7727890, -2.8358717, 2.7927117
2: -3.2840815, -0.4872589, -3.2729352, -0.5181378, -2.3584642, 2.3761287
3: -10.9068356, -7.9329395, -10.8661718, -7.9472113, -2.6362348, 2.6065931
4: -12.6462040, -8.9930449, -12.5357018, -9.0188999, -2.8178291, 2.7771320
5: -4.9709973, -2.5783663, -4.9636364, -2.6677804, -2.0465927, 2.0809722
6: -3.0912058, -0.4166851, -3.0785112, -0.5557709, -2.2764220, 2.3208423
7: -9.5139322, -5.3940191, -9.3364067, -5.3998232, -3.2307463, 3.1610370
8: -2.6305223, -0.3136730, -2.5994225, -0.3444963, -2.0636091, 2.0571904
9: -4.5447464, -1.7383752, -4.4724979, -1.7498055, -2.3203745, 2.3038583

Time for backsubstitution: 21.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 471

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 859

## Relational analysis of NS_A2_A2_A1_B1

### Relational analysis result of NS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726867, upper bound: 0.8708999
time: 8.79 seconds

## Relational analysis of NS_A2_A2_A1_B2

### Relational analysis result of NS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726867, upper bound: 0.8728258
time: 5.61 seconds

## BFS NS instance: NS_A2_A2_A2

### Backsubstitution after applying NS history:
0: 6.5718470, 9.0536900, 6.6375289, 9.0313301, -1.8979697, 1.8628874
1: -17.5870018, -13.7353029, -17.4851303, -13.7643719, -2.8692741, 2.8166380
2: -3.3006539, -0.4715407, -3.2758884, -0.5142332, -2.3759127, 2.3966551
3: -10.9477787, -7.9032879, -10.8677864, -7.9381456, -2.6757398, 2.6350827
4: -12.6625738, -8.9739227, -12.5387831, -9.0154390, -2.8400388, 2.8025680
5: -4.9959183, -2.5638604, -4.9653625, -2.6635966, -2.0782127, 2.0971043
6: -3.1046126, -0.4010997, -3.0826111, -0.5545983, -2.2925215, 2.3352027
7: -9.5372572, -5.3442793, -9.3434391, -5.3957739, -3.2547998, 3.2176213
8: -2.6459394, -0.3026347, -2.6018701, -0.3418813, -2.0808525, 2.0708330
9: -4.5755563, -1.7097167, -4.4800839, -1.7481475, -2.3488336, 2.3218386

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 471

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 859

## Relational analysis of NS_A2_A2_A2_B1

### Relational analysis result of NS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8731463, upper bound: 0.8712145
time: 5.31 seconds

## Relational analysis of NS_A2_A2_A2_B2

### Relational analysis result of NS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8731463, upper bound: 0.8731467
time: 12.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 39.88 seconds
NS_A2_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 39.88
Output dim: 0, lower bound: -0.8706827, upper bound: 0.8708997
NS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 39.88
Output dim: 0, lower bound: -0.8706827, upper bound: 0.8728254
NS_A2_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 39.88
Output dim: 0, lower bound: -0.8711423, upper bound: 0.8712157
NS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 39.88
Output dim: 0, lower bound: -0.8711423, upper bound: 0.8731461
NS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 39.88
Output dim: 0, lower bound: -0.8726867, upper bound: 0.8708999
NS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 39.88
Output dim: 0, lower bound: -0.8726867, upper bound: 0.8728258
NS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 39.88
Output dim: 0, lower bound: -0.8731463, upper bound: 0.8712145
NS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 39.88
Output dim: 0, lower bound: -0.8731463, upper bound: 0.8731467

## BFS NS instance: NS_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: 6.6018028, 9.0313683, 6.6431642, 9.0293522, -1.8696172, 1.8326459
1: -17.5267982, -13.7793112, -17.4790077, -13.7737617, -2.8229375, 2.7600470
2: -3.2754138, -0.5011899, -3.2703331, -0.5197650, -2.3434572, 2.3629944
3: -10.9012442, -7.9480009, -10.8645563, -7.9483657, -2.6303415, 2.5875883
4: -12.5739050, -9.0200891, -12.5344162, -9.0313549, -2.7784615, 2.7612901
5: -4.9622669, -2.5906110, -4.9604406, -2.6685071, -2.0373712, 2.0747733
6: -3.0802755, -0.4380999, -3.0738862, -0.5564699, -2.2671947, 2.3044238
7: -9.4128714, -5.4308138, -9.3350458, -5.4170699, -3.1917305, 3.1378064
8: -2.6182866, -0.3412938, -2.5942955, -0.3453274, -2.0528893, 2.0331943
9: -4.4954910, -1.7594352, -4.4714632, -1.7589008, -2.2993844, 2.2857814

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 471

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5773

## Relational analysis of NS_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of NS_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A2_A1_A1_B2_B1

### Relational analysis result of NS_A2_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8706827, upper bound: 0.8708236
time: 5.44 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2

### Relational analysis result of NS_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8706827, upper bound: 0.8728254
time: 5.13 seconds

## BFS NS instance: NS_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: 6.5807614, 9.0383596, 6.6405320, 9.0309086, -1.8882294, 1.8412075
1: -17.5756207, -13.7523518, -17.4807835, -13.7653465, -2.8561635, 2.7839742
2: -3.2919695, -0.4855006, -3.2732913, -0.5158623, -2.3609209, 2.3835282
3: -10.9421940, -7.9183426, -10.8661709, -7.9393110, -2.6695170, 2.6160555
4: -12.5902719, -9.0009689, -12.5374994, -9.0278940, -2.8078876, 2.7867246
5: -4.9871922, -2.5761256, -4.9621658, -2.6643248, -2.0689774, 2.0908930
6: -3.0936744, -0.4225092, -3.0779850, -0.5552950, -2.2832775, 2.3187943
7: -9.4361458, -5.3810759, -9.3420773, -5.4130192, -3.2157726, 3.1943908
8: -2.6337128, -0.3302526, -2.5967417, -0.3427124, -2.0701389, 2.0466433
9: -4.5262961, -1.7307675, -4.4790487, -1.7572424, -2.3278430, 2.3036885

Time for backsubstitution: 21.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 471

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5773

## Relational analysis of NS_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A2_A1_A2_B2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8711423, upper bound: 0.8711475
time: 8.12 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8711423, upper bound: 0.8731474
time: 15.15 seconds

## BFS NS instance: NS_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: 6.5980573, 9.0455761, 6.6508737, 9.0239601, -1.8664179, 1.8422496
1: -17.5348549, -13.7644424, -17.4756050, -13.7780304, -2.8197889, 2.7819381
2: -3.2822855, -0.4926096, -3.2671659, -0.5290657, -2.3452120, 2.3623354
3: -10.9047804, -7.9369764, -10.8617868, -7.9558029, -2.6219325, 2.5948572
4: -12.6395073, -8.9943008, -12.5217161, -9.0240936, -2.8043904, 2.7617278
5: -4.9692254, -2.5793104, -4.9596157, -2.6713574, -2.0387845, 2.0738356
6: -3.0848968, -0.4178653, -3.0664608, -0.5618215, -2.2630882, 2.3068469
7: -9.5111942, -5.3992362, -9.3256187, -5.4095416, -3.2175875, 3.1445942
8: -2.6230545, -0.3143387, -2.5854836, -0.3489947, -2.0506449, 2.0434771
9: -4.5392232, -1.7390479, -4.4617839, -1.7545620, -2.3081820, 2.2912862

Time for backsubstitution: 22.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 471

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of NS_A2_A2_A1_B1_B1

### Relational analysis result of NS_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726740, upper bound: 0.8707471
time: 5.89 seconds

## Relational analysis of NS_A2_A2_A1_B1_B2

### Relational analysis result of NS_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726740, upper bound: 0.8707459
time: 5.20 seconds

## BFS NS instance: NS_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: 6.5929055, 9.0466747, 6.6401653, 9.0297709, -1.8769417, 1.8508453
1: -17.5381851, -13.7623606, -17.4833546, -13.7729740, -2.8324556, 2.7815895
2: -3.2840796, -0.4872637, -3.2729330, -0.5181470, -2.3533621, 2.3740153
3: -10.9068356, -7.9329429, -10.8661718, -7.9472184, -2.6362219, 2.6071610
4: -12.6461992, -8.9930439, -12.5356894, -9.0189028, -2.8159161, 2.7768841
5: -4.9709959, -2.5785043, -4.9636350, -2.6680431, -2.0454397, 2.0819356
6: -3.0912011, -0.4166870, -3.0785024, -0.5557723, -2.2764163, 2.3155944
7: -9.5139275, -5.3940239, -9.3364048, -5.3998313, -3.2281466, 3.1610308
8: -2.6305180, -0.3136730, -2.5994167, -0.3444967, -2.0609980, 2.0504482
9: -4.5447426, -1.7383759, -4.4724917, -1.7498064, -2.3189135, 2.2998977

Time for backsubstitution: 22.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 471

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5773

## Relational analysis of NS_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of NS_A2_A2_A1_B2_B1

### Relational analysis result of NS_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726740, upper bound: 0.8726754
time: 8.26 seconds

## Relational analysis of NS_A2_A2_A1_B2_B2

### Relational analysis result of NS_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726740, upper bound: 0.8726741
time: 4.89 seconds

## BFS NS instance: NS_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: 6.5770016, 9.0525780, 6.6482477, 9.0255184, -1.8850183, 1.8508220
1: -17.5836678, -13.7374830, -17.4773808, -13.7696095, -2.8576670, 2.8058586
2: -3.2988541, -0.4768901, -3.2701261, -0.5251572, -2.3626709, 2.3829482
3: -10.9457321, -7.9073348, -10.8633995, -7.9467435, -2.6633320, 2.6233315
4: -12.6558657, -8.9751768, -12.5247965, -9.0206299, -2.8265851, 2.7871757
5: -4.9941492, -2.5648031, -4.9613428, -2.6671734, -2.0704021, 2.0899737
6: -3.0983083, -0.4022741, -3.0705626, -0.5606451, -2.2791905, 2.3212001
7: -9.5345192, -5.3494892, -9.3326530, -5.4054933, -3.2416301, 3.2011843
8: -2.6384940, -0.3033018, -2.5879302, -0.3463793, -2.0678720, 2.0571005
9: -4.5699978, -1.7103846, -4.4693537, -1.7529030, -2.3366103, 2.3092184

Time for backsubstitution: 22.62 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 63.75 + 542.47 = 606.22 seconds
