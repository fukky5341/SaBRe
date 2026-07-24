## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.719649471


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4588056, 1.4588056)
1: (-10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6418862, 1.6418862)
2: (-4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3488832, 1.3488827)
3: (-5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7874470, 1.7874465)
4: (-13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5689108, 1.5689108)
5: (-3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9303412, 0.9303412)
6: (-10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3711376, 1.3711374)
7: (-9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0479746, 2.0479746)
8: (9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5339589, 1.5339584)
9: (-7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8485889, 1.8485889)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.44 + 36.95 = 59.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.7232643, upper bound: 0.7232643

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7230495, upper bound: 0.7203430
time: 5.93 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232583, upper bound: 0.7232581
time: 6.89 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.89 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.89
Output dim: 8, lower bound: -0.7230495, upper bound: 0.7203430
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.89
Output dim: 8, lower bound: -0.7232583, upper bound: 0.7232581

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -8.0714731, -6.1792231, -8.0721741, -6.1762385, -1.4550529, 1.4531260
1: -10.4551840, -8.2894650, -10.4557533, -8.2872000, -1.6391869, 1.6374912
2: -4.7407713, -2.8069377, -4.7412596, -2.8047905, -1.3464637, 1.3445578
3: -5.6454954, -3.3557651, -5.6523795, -3.3553774, -1.7744493, 1.7810197
4: -12.9839735, -10.3730383, -12.9954071, -10.3716221, -1.5470557, 1.5571027
5: -3.3169928, -1.8149633, -3.3170977, -1.8114383, -0.9273677, 0.9239957
6: -10.5893393, -8.5212250, -10.5894642, -8.5142450, -1.3647666, 1.3571944
7: -9.0587940, -6.7383318, -9.0749073, -6.7382846, -2.0182753, 2.0347171
8: 9.8044977, 11.6911249, 9.8037453, 11.6943741, -1.5282474, 1.5255032
9: -7.3174295, -4.8442230, -7.3231111, -4.8436503, -1.8378916, 1.8428564

Time for backsubstitution: 21.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7230319, upper bound: 0.7177675
time: 6.39 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7230452, upper bound: 0.7203383
time: 6.42 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.0807810, -6.1731977, -8.0727243, -6.1738725, -1.4673944, 1.4580107
1: -10.4610443, -8.2833080, -10.4562073, -8.2853956, -1.6466446, 1.6435723
2: -4.7507782, -2.8016305, -4.7416334, -2.8030887, -1.3584361, 1.3486667
3: -5.6620226, -3.3412220, -5.6578507, -3.3550720, -1.7874269, 1.8011689
4: -13.0052757, -10.3377876, -13.0044613, -10.3705053, -1.5605559, 1.5855004
5: -3.3271499, -1.8082898, -3.3171818, -1.8086388, -0.9402673, 0.9277978
6: -10.6000547, -8.5052185, -10.5895653, -8.5086956, -1.3819652, 1.3710992
7: -9.0936813, -6.7009816, -9.0877161, -6.7382483, -2.0437613, 2.0708897
8: 9.7911587, 11.6980762, 9.8031473, 11.6969585, -1.5445657, 1.5341702
9: -7.3306956, -4.8341541, -7.3276329, -4.8431969, -1.8493257, 1.8572955

Time for backsubstitution: 20.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232430, upper bound: 0.7206853
time: 6.07 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232541, upper bound: 0.7232550
time: 4.46 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.09 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 31.09
Output dim: 8, lower bound: -0.7230319, upper bound: 0.7177675
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.09
Output dim: 8, lower bound: -0.7230452, upper bound: 0.7203383
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.09
Output dim: 8, lower bound: -0.7232430, upper bound: 0.7206853
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.09
Output dim: 8, lower bound: -0.7232541, upper bound: 0.7232550

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -8.0606775, -6.1799927, -8.0492935, -6.1778812, -1.4406042, 1.4276280
1: -10.4536781, -8.2903404, -10.4525423, -8.2891207, -1.6210732, 1.6204104
2: -4.7327185, -2.8076420, -4.7241783, -2.8062992, -1.3371840, 1.3262854
3: -5.6434312, -3.3572130, -5.6479220, -3.3584373, -1.7494769, 1.7561178
4: -12.9834614, -10.3787031, -12.9943132, -10.3836002, -1.5350728, 1.5521488
5: -3.3168366, -1.8266690, -3.3168011, -1.8362863, -0.9016972, 0.9115005
6: -10.5889664, -8.5379105, -10.5886736, -8.5495539, -1.3263683, 1.3380287
7: -9.0512199, -6.7391620, -9.0589066, -6.7400513, -2.0050883, 2.0138640
8: 9.8098698, 11.6909380, 9.8151245, 11.6939726, -1.5176373, 1.5096078
9: -7.3158903, -4.8457146, -7.3197823, -4.8468213, -1.8185644, 1.8228326

Time for backsubstitution: 20.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5832

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7209746, upper bound: 0.7175045
time: 5.61 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7230299, upper bound: 0.7177658
time: 6.71 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -8.0714636, -6.1792231, -8.0782347, -6.1419559, -1.4705815, 1.4493883
1: -10.4551783, -8.2894650, -10.4656858, -8.2759399, -1.6594605, 1.6383119
2: -4.7407627, -2.8069386, -4.7441840, -2.7735219, -1.3709860, 1.3399744
3: -5.6454930, -3.3557675, -5.6732383, -3.3531134, -1.7692118, 1.8083363
4: -12.9839745, -10.3730507, -13.0133190, -10.3639421, -1.5526068, 1.5702523
5: -3.3169920, -1.8149812, -3.3502960, -1.8095354, -0.9164188, 0.9376132
6: -10.5893402, -8.5212479, -10.6415777, -8.4985466, -1.3709574, 1.3826184
7: -9.0587835, -6.7383327, -9.0928860, -6.7119718, -2.0419526, 2.0495853
8: 9.8045053, 11.6911240, 9.7988462, 11.7070541, -1.5415602, 1.5276704
9: -7.3174295, -4.8442273, -7.3509769, -4.8395100, -1.8361235, 1.8686070

Time for backsubstitution: 20.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5832

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7209846, upper bound: 0.7200745
time: 5.72 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7230432, upper bound: 0.7203366
time: 5.82 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.0699835, -6.1739655, -8.0498524, -6.1755128, -1.4530005, 1.4325914
1: -10.4595413, -8.2841797, -10.4530001, -8.2873230, -1.6285267, 1.6264720
2: -4.7427278, -2.8023458, -4.7245569, -2.8046002, -1.3491616, 1.3303933
3: -5.6599636, -3.3426673, -5.6533775, -3.3581328, -1.7625308, 1.7762280
4: -13.0047617, -10.3434420, -13.0033674, -10.3824654, -1.5485492, 1.5782320
5: -3.3269942, -1.8199943, -3.3168843, -1.8334868, -0.9145968, 0.9153025
6: -10.5996838, -8.5218487, -10.5887728, -8.5439568, -1.3433938, 1.3517311
7: -9.0861626, -6.7018123, -9.0717535, -6.7400136, -2.0306187, 2.0490789
8: 9.7965260, 11.6978893, 9.8145266, 11.6965570, -1.5339503, 1.5182743
9: -7.3291626, -4.8356371, -7.3242855, -4.8463602, -1.8299928, 1.8372521

Time for backsubstitution: 20.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5832

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7211874, upper bound: 0.7204231
time: 5.17 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232410, upper bound: 0.7206838
time: 8.10 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.0807667, -6.1731997, -8.0787544, -6.1395888, -1.4768262, 1.4543364
1: -10.4610415, -8.2833109, -10.4661407, -8.2741013, -1.6669140, 1.6443973
2: -4.7507715, -2.8016338, -4.7445564, -2.7718129, -1.3760340, 1.3440862
3: -5.6620197, -3.3412247, -5.6787777, -3.3528092, -1.7821908, 1.8248885
4: -13.0052729, -10.3377981, -13.0223751, -10.3628740, -1.5661032, 1.5871812
5: -3.3271499, -1.8083093, -3.3503792, -1.8067348, -0.9293222, 0.9414189
6: -10.6000538, -8.5052395, -10.6416788, -8.4930000, -1.3882108, 1.3965129
7: -9.0936699, -6.7009835, -9.1056995, -6.7119355, -2.0674567, 2.0856757
8: 9.7911644, 11.6980762, 9.7982502, 11.7096386, -1.5515079, 1.5363359
9: -7.3306971, -4.8341560, -7.3555927, -4.8390565, -1.8475466, 1.8758931

Time for backsubstitution: 20.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5832

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7211933, upper bound: 0.7229886
time: 6.55 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232521, upper bound: 0.7232516
time: 6.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.78 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 33.78
Output dim: 8, lower bound: -0.7209746, upper bound: 0.7175045
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.78
Output dim: 8, lower bound: -0.7230299, upper bound: 0.7177658
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 33.78
Output dim: 8, lower bound: -0.7209846, upper bound: 0.7200745
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.78
Output dim: 8, lower bound: -0.7230432, upper bound: 0.7203366
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 33.78
Output dim: 8, lower bound: -0.7211874, upper bound: 0.7204231
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.78
Output dim: 8, lower bound: -0.7232410, upper bound: 0.7206838
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 33.78
Output dim: 8, lower bound: -0.7211933, upper bound: 0.7229886
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.78
Output dim: 8, lower bound: -0.7232521, upper bound: 0.7232516

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.0599661, -6.1822262, -8.0488644, -6.1789608, -1.4354038, 1.4238563
1: -10.4516087, -8.3025255, -10.4515047, -8.2948685, -1.5652423, 1.6070209
2: -4.7281909, -2.8091278, -4.7220469, -2.8070745, -1.3316350, 1.3376937
3: -5.6404724, -3.3586569, -5.6464849, -3.3592587, -1.7947130, 1.7520771
4: -12.9762936, -10.3813810, -12.9909153, -10.3850517, -1.5257983, 1.5366707
5: -3.3157675, -1.8282133, -3.3161721, -1.8370224, -0.8744307, 0.9077575
6: -10.5878925, -8.5437098, -10.5880699, -8.5523129, -1.3110785, 1.3313229
7: -9.0490303, -6.7426567, -9.0578079, -6.7419686, -2.0007024, 2.0650859
8: 9.8167086, 11.6898403, 9.8183813, 11.6932459, -1.5102172, 1.5049567
9: -7.3150520, -4.8505979, -7.3192616, -4.8491278, -1.8127227, 1.8147178

Time for backsubstitution: 20.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7209721, upper bound: 0.7158296
time: 5.67 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7209721, upper bound: 0.7175029
time: 4.68 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.0621700, -6.1784344, -8.0492916, -6.1778822, -1.4423099, 1.4286923
1: -10.4669847, -8.2894211, -10.4525423, -8.2891226, -1.6343727, 1.6177416
2: -4.7331171, -2.8029432, -4.7241774, -2.8063021, -1.3366132, 1.3313484
3: -5.6450996, -3.3539329, -5.6479220, -3.3584387, -1.7503362, 1.7592354
4: -12.9844170, -10.3680458, -12.9943075, -10.3836002, -1.5330637, 1.5625861
5: -3.3176014, -1.8261527, -3.3168008, -1.8362867, -0.9033945, 0.9108536
6: -10.5949688, -8.5368319, -10.5886707, -8.5495558, -1.3321671, 1.3364749
7: -9.0556011, -6.7372403, -9.0589056, -6.7400517, -2.0100527, 2.0147390
8: 9.8079472, 11.6961727, 9.8151321, 11.6939716, -1.5172486, 1.5150199
9: -7.3213110, -4.8447065, -7.3197832, -4.8468237, -1.8215570, 1.8223248

Time for backsubstitution: 20.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7230278, upper bound: 0.7160929
time: 4.86 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7230274, upper bound: 0.7177632
time: 6.40 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.0707502, -6.1814570, -8.0778103, -6.1430330, -1.4652572, 1.4456177
1: -10.4531059, -8.3016510, -10.4646368, -8.2816877, -1.6036348, 1.6249080
2: -4.7362380, -2.8084259, -4.7420530, -2.7742963, -1.3654151, 1.3512936
3: -5.6425376, -3.3572102, -5.6717978, -3.3539290, -1.8143396, 1.8042741
4: -12.9768066, -10.3757343, -13.0099192, -10.3653984, -1.5433207, 1.5530888
5: -3.3159239, -1.8165264, -3.3496661, -1.8102738, -0.8891807, 0.9337199
6: -10.5882645, -8.5270462, -10.6409807, -8.5013046, -1.3556924, 1.3758919
7: -9.0565948, -6.7418246, -9.0918064, -6.7138901, -2.0372219, 2.1007676
8: 9.8113403, 11.6900272, 9.8020992, 11.7063255, -1.5341353, 1.5230160
9: -7.3165889, -4.8491116, -7.3504524, -4.8418307, -1.8302646, 1.8602867

Time for backsubstitution: 21.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7209821, upper bound: 0.7183996
time: 7.67 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7209821, upper bound: 0.7200721
time: 6.31 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.0729561, -6.1776724, -8.0782356, -6.1419582, -1.4713387, 1.4504790
1: -10.4684858, -8.2885485, -10.4656858, -8.2759428, -1.6644697, 1.6356440
2: -4.7411604, -2.8022399, -4.7441797, -2.7735224, -1.3703756, 1.3450365
3: -5.6471510, -3.3524866, -5.6732388, -3.3531151, -1.7700081, 1.8114533
4: -12.9849300, -10.3623896, -13.0133123, -10.3639460, -1.5505962, 1.5719337
5: -3.3177564, -1.8144655, -3.3502958, -1.8095380, -0.9181168, 0.9369358
6: -10.5953407, -8.5201712, -10.6415768, -8.4985523, -1.3767500, 1.3810550
7: -9.0631657, -6.7364206, -9.0928850, -6.7119741, -2.0451579, 2.0504556
8: 9.8025799, 11.6963577, 9.7988482, 11.7070541, -1.5411887, 1.5330830
9: -7.3228445, -4.8432260, -7.3509774, -4.8395133, -1.8391275, 1.8670382

Time for backsubstitution: 21.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7230407, upper bound: 0.7186567
time: 7.11 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7230408, upper bound: 0.7203353
time: 6.99 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.0692673, -6.1761990, -8.0494251, -6.1765938, -1.4478006, 1.4288197
1: -10.4574814, -8.2963657, -10.4519615, -8.2930737, -1.5726914, 1.6130819
2: -4.7382030, -2.8038292, -4.7224250, -2.8053751, -1.3436131, 1.3418012
3: -5.6570072, -3.3441060, -5.6519413, -3.3589535, -1.8077388, 1.7721915
4: -12.9975967, -10.3461208, -12.9999704, -10.3839197, -1.5392733, 1.5611060
5: -3.3259261, -1.8215400, -3.3162551, -1.8342242, -0.8873329, 0.9115577
6: -10.5986090, -8.5276499, -10.5881729, -8.5467148, -1.3281488, 1.3450220
7: -9.0839691, -6.7053061, -9.0706558, -6.7419333, -2.0262384, 2.0998530
8: 9.8033609, 11.6967897, 9.8177834, 11.6958275, -1.5265326, 1.5136237
9: -7.3283186, -4.8405232, -7.3237634, -4.8486643, -1.8241496, 1.8291330

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7211849, upper bound: 0.7187465
time: 7.90 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7211849, upper bound: 0.7204189
time: 5.77 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.0714741, -6.1724110, -8.0498514, -6.1755142, -1.4547067, 1.4336548
1: -10.4728413, -8.2832575, -10.4529953, -8.2873268, -1.6418204, 1.6238017
2: -4.7431288, -2.7976451, -4.7245541, -2.8046021, -1.3485923, 1.3354588
3: -5.6616230, -3.3393867, -5.6533775, -3.3581324, -1.7633996, 1.7793450
4: -13.0057201, -10.3327847, -13.0033636, -10.3824692, -1.5465403, 1.5799056
5: -3.3277566, -1.8194783, -3.3168840, -1.8334882, -0.9162936, 0.9146552
6: -10.6056824, -8.5207672, -10.5887737, -8.5439587, -1.3491888, 1.3501806
7: -9.0905495, -6.6998901, -9.0717525, -6.7400160, -2.0355902, 2.0499406
8: 9.7945957, 11.7031231, 9.8145313, 11.6965561, -1.5335770, 1.5236864
9: -7.3345799, -4.8346319, -7.3242831, -4.8463635, -1.8329849, 1.8367381

Time for backsubstitution: 21.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232385, upper bound: 0.7190092
time: 5.26 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232385, upper bound: 0.7206808
time: 5.82 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.0800543, -6.1754303, -8.0783262, -6.1406651, -1.4715028, 1.4505658
1: -10.4589748, -8.2954960, -10.4650936, -8.2798500, -1.6110830, 1.6309929
2: -4.7462502, -2.8031173, -4.7424269, -2.7725859, -1.3704588, 1.3554053
3: -5.6590667, -3.3426630, -5.6773367, -3.3536243, -1.8272905, 1.8205762
4: -12.9981098, -10.3404799, -13.0189800, -10.3643303, -1.5568178, 1.5699710
5: -3.3260808, -1.8098536, -3.3497500, -1.8074725, -0.9018782, 0.9375241
6: -10.5989828, -8.5110397, -10.6410809, -8.4957590, -1.3729873, 1.3897817
7: -9.0914793, -6.7044730, -9.1046181, -6.7138510, -2.0627198, 2.1364183
8: 9.7979965, 11.6969795, 9.8015079, 11.7089109, -1.5440459, 1.5316801
9: -7.3298516, -4.8390479, -7.3550682, -4.8413749, -1.8416872, 1.8675690

Time for backsubstitution: 21.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7211908, upper bound: 0.7213121
time: 6.02 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7211908, upper bound: 0.7229857
time: 6.14 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.0822601, -6.1716490, -8.0787525, -6.1395893, -1.4775844, 1.4554245
1: -10.4743385, -8.2823877, -10.4661407, -8.2741051, -1.6688824, 1.6417260
2: -4.7511716, -2.7969317, -4.7445536, -2.7718143, -1.3754241, 1.3491507
3: -5.6636724, -3.3379426, -5.6787739, -3.3528087, -1.7829962, 1.8257880
4: -13.0062351, -10.3271360, -13.0223694, -10.3628759, -1.5640941, 1.5888581
5: -3.3279133, -1.8077917, -3.3503790, -1.8067358, -0.9310207, 0.9407421
6: -10.6060543, -8.5041599, -10.6416788, -8.4930067, -1.3939996, 1.3949519
7: -9.0980587, -6.6990676, -9.1056986, -6.7119360, -2.0706711, 2.0865309
8: 9.7892323, 11.7033091, 9.7982550, 11.7096367, -1.5511885, 1.5417461
9: -7.3361130, -4.8331604, -7.3555918, -4.8390598, -1.8505502, 1.8743253

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232496, upper bound: 0.7215771
time: 5.96 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232496, upper bound: 0.7232506
time: 6.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 33.20 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7209721, upper bound: 0.7158296
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7209721, upper bound: 0.7175029
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7230278, upper bound: 0.7160929
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7230274, upper bound: 0.7177632
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7209821, upper bound: 0.7183996
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7209821, upper bound: 0.7200721
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7230407, upper bound: 0.7186567
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7230408, upper bound: 0.7203353
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7211849, upper bound: 0.7187465
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7211849, upper bound: 0.7204189
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7232385, upper bound: 0.7190092
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7232385, upper bound: 0.7206808
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7211908, upper bound: 0.7213121
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7211908, upper bound: 0.7229857
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7232496, upper bound: 0.7215771
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.20
Output dim: 8, lower bound: -0.7232496, upper bound: 0.7232506

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.0594549, -6.1873493, -8.0477562, -6.1899729, -1.4240007, 1.4179523
1: -10.4509974, -8.3118973, -10.4501820, -8.3150444, -1.5445018, 1.5964255
2: -4.7278781, -2.8193183, -4.7213702, -2.8287644, -1.3093972, 1.3271408
3: -5.6350698, -3.3591478, -5.6348662, -3.3603218, -1.7882910, 1.7399821
4: -12.9755974, -10.3844223, -12.9894218, -10.3915663, -1.5179031, 1.5308166
5: -3.3125572, -1.8284235, -3.3092542, -1.8374764, -0.8705752, 0.9004321
6: -10.5877619, -8.5474548, -10.5877781, -8.5603352, -1.3004003, 1.3254066
7: -9.0476046, -6.7446017, -9.0547390, -6.7461438, -1.9946556, 2.0590138
8: 9.8189754, 11.6896486, 9.8232517, 11.6928349, -1.5058360, 1.4980721
9: -7.3141055, -4.8511348, -7.3171964, -4.8502803, -1.8103261, 1.8117576

Time for backsubstitution: 21.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.7184167, upper bound: 0.7158309
time: 6.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.7184169, upper bound: 0.7158295
time: 5.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.0599632, -6.1822352, -8.0665188, -6.1785836, -1.4295859, 1.4409924
1: -10.4516106, -8.3025379, -10.4802370, -8.2934446, -1.5562696, 1.6314840
2: -4.7281904, -2.8091402, -4.7498975, -2.8062611, -1.3214350, 1.3572404
3: -5.6404667, -3.3586566, -5.6483612, -3.3436878, -1.8101683, 1.7483559
4: -12.9762955, -10.3813848, -12.9982376, -10.3836231, -1.5240684, 1.5427282
5: -3.3157659, -1.8282148, -3.3163209, -1.8272979, -0.8841467, 0.9043944
6: -10.5878906, -8.5437164, -10.5999031, -8.5486565, -1.3135228, 1.3421991
7: -9.0490255, -6.7426591, -9.0645971, -6.7410707, -1.9989758, 2.0708518
8: 9.8167133, 11.6898384, 9.8168907, 11.6998014, -1.5157938, 1.5050917
9: -7.3150516, -4.8505993, -7.3249822, -4.8488684, -1.8129673, 1.8202457

Time for backsubstitution: 21.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.7184169, upper bound: 0.7175029
time: 6.90 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.7184167, upper bound: 0.7175014
time: 6.77 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 59.40 + 545.77 = 605.17 seconds
