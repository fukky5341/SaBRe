## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.8650754865
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.6260853, -9.4460907, -12.6260853, -9.4460907, -3.1799946, 3.1799946)
1: (-11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.4480700, 2.4480703)
2: (-8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9626331, 1.9626331)
3: (-7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.6083369, 2.6083369)
4: (-3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864)
5: (-5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222)
6: (-16.9029446, -13.7977066, -16.9029446, -13.7977066, -3.0488920, 3.0488920)
7: (-4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329)
8: (-5.2317653, -2.9253664, -5.2317653, -2.9253664, -2.1087542, 2.1087539)
9: (4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.5658846, 1.5658846)

## BASE Result
execution time: IAR + LP analysis = 14.04 + 33.87 = 47.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -1.2026377, upper bound: 1.2026362


# Binary Search by BASE starts (time budget: 3552.09 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.3057825565338135
rel_dist={9: [-0.8660068367423657, 0.8660078437084611]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.1616536378860474
rel_dist={9: [-0.6705902224208895, 0.6705927823695186]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=1.2096967697143555
rel_dist={9: [-0.7384599414700519, 0.738462472703862]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=1.2577396631240845
rel_dist={9: [-0.803270111495487, 0.8032729720962895]}

## Binary Search Result
Binary search time: 196.94 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual_ind) starts
Time budget: 3355.15 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4628
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4628

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0382476, upper bound: 1.0414131
time: 3.69 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0414107, upper bound: 1.0414132
time: 3.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.42 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.42
Output dim: 9, lower bound: -1.0382476, upper bound: 1.0414131
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.42
Output dim: 9, lower bound: -1.0414107, upper bound: 1.0414132

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.6142950, -9.4646616, -12.6255264, -9.4495239, -2.9467697, 2.9380937
1: -11.7295780, -9.1830034, -11.7349663, -9.1785412, -2.1982975, 2.2002537
2: -8.1562185, -6.2025928, -8.1618805, -6.2004538, -1.9387226, 1.9018209
3: -7.7039814, -5.1228108, -7.7198353, -5.1153197, -2.4302778, 2.4365635
4: -3.6705718, -1.3458829, -3.6762388, -1.3429496, -2.3276222, 2.3303559
5: -5.9458489, -3.8334849, -5.9528761, -3.8289294, -2.1169195, 2.1193912
6: -16.8938866, -13.8075314, -16.9025002, -13.7995644, -2.6401625, 2.6424117
7: -4.6812468, -2.2691436, -4.6864653, -2.2596817, -2.4215651, 2.4173217
8: -5.2248793, -2.9452572, -5.2311535, -2.9289222, -1.8238239, 1.8383367
9: 4.4159822, 5.9672208, 4.4072819, 5.9710965, -1.4396167, 1.4427493

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369809, upper bound: 1.0369700
time: 3.59 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0382464, upper bound: 1.0414119
time: 3.52 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -12.6260834, -9.4460993, -12.6260853, -9.4460907, -2.9576669, 2.9519141
1: -11.7355270, -9.1776171, -11.7355261, -9.1776123, -2.2070508, 2.2048526
2: -8.1624203, -6.1997900, -8.1624212, -6.1997881, -1.9458046, 1.9588850
3: -7.7232952, -5.1149702, -7.7233071, -5.1149702, -2.4360867, 2.4484644
4: -3.6771927, -1.3426094, -3.6771948, -1.3426085, -2.3345842, 2.3345854
5: -5.9543266, -3.8286080, -5.9543295, -3.8286073, -2.1257193, 2.1257215
6: -16.9029465, -13.7977123, -16.9029446, -13.7977066, -2.6517196, 2.6528037
7: -4.6868644, -2.2577372, -4.6868649, -2.2577319, -2.4291325, 2.4291277
8: -5.2317648, -2.9253731, -5.2317653, -2.9253664, -1.8601379, 1.8565118
9: 4.4055557, 5.9714375, 4.4055529, 5.9714375, -1.4457321, 1.4499098

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0414107, upper bound: 1.0382500
time: 3.58 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0414107, upper bound: 1.0414132
time: 3.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.89 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.89
Output dim: 9, lower bound: -1.0369809, upper bound: 1.0369700
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.89
Output dim: 9, lower bound: -1.0382464, upper bound: 1.0414119
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.89
Output dim: 9, lower bound: -1.0414107, upper bound: 1.0382500
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.89
Output dim: 9, lower bound: -1.0414107, upper bound: 1.0414132

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.6127205, -9.4679451, -12.6150150, -9.4680367, -2.9205952, 2.9181724
1: -11.7275772, -9.1851664, -11.7224598, -9.1907425, -2.1835642, 2.1832883
2: -8.1528530, -6.2059212, -8.1392603, -6.2174459, -1.9173803, 1.8761783
3: -7.6956253, -5.1244884, -7.6764216, -5.1337681, -2.4039493, 2.3926134
4: -3.6689939, -1.3531744, -3.6561694, -1.3796194, -2.2893746, 2.3029950
5: -5.9446893, -3.8381221, -5.9433832, -3.8529432, -2.0917461, 2.1052611
6: -16.8916492, -13.8142185, -16.8815880, -13.8336105, -2.6037951, 2.6116626
7: -4.6708336, -2.2724094, -4.6343160, -2.2939410, -2.3768926, 2.3619065
8: -5.2208276, -2.9473310, -5.2098694, -2.9440174, -1.8021178, 1.8173337
9: 4.4195633, 5.9667583, 4.4265347, 5.9674120, -1.4265766, 1.4222927

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338045, upper bound: 1.0369678
time: 3.60 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338045, upper bound: 1.0369678
time: 7.40 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.6142950, -9.4646616, -12.6255217, -9.4495277, -2.9428825, 2.9366155
1: -11.7295780, -9.1830034, -11.7349644, -9.1785440, -2.1982946, 2.2059555
2: -8.1562185, -6.2025928, -8.1618776, -6.2004566, -1.9387183, 1.9028769
3: -7.7039814, -5.1228108, -7.7198267, -5.1153212, -2.4302759, 2.4213932
4: -3.6705718, -1.3458829, -3.6762371, -1.3429542, -2.3276176, 2.3303542
5: -5.9458489, -3.8334849, -5.9528742, -3.8289349, -2.1169140, 2.1193893
6: -16.8938866, -13.8075314, -16.9024963, -13.7995739, -2.6269939, 2.6424093
7: -4.6812468, -2.2691436, -4.6864548, -2.2596841, -2.4215627, 2.4173112
8: -5.2248793, -2.9452572, -5.2311497, -2.9289269, -1.8210714, 1.8306718
9: 4.4159822, 5.9672208, 4.4072857, 5.9710965, -1.4367754, 1.4460238

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338045, upper bound: 1.0401465
time: 3.57 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338046, upper bound: 1.0414106
time: 3.64 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -12.6260834, -9.4460993, -12.6142950, -9.4646616, -2.9382501, 2.9522972
1: -11.7355270, -9.1776171, -11.7295780, -9.1830034, -2.2007942, 2.1994359
2: -8.1624203, -6.1997900, -8.1562185, -6.2025928, -1.9010615, 1.9371979
3: -7.7232952, -5.1149702, -7.7039814, -5.1228108, -2.4401488, 2.4306400
4: -3.6771927, -1.3426094, -3.6705718, -1.3458829, -2.3313098, 2.3279624
5: -5.9543266, -3.8286080, -5.9458489, -3.8334849, -2.1208417, 2.1172409
6: -16.9029465, -13.7977123, -16.8938866, -13.8075314, -2.6419930, 2.6430650
7: -4.6868644, -2.2577372, -4.6812468, -2.2691436, -2.4177208, 2.4235096
8: -5.2317648, -2.9253731, -5.2248793, -2.9452572, -1.8389502, 1.8275771
9: 4.4055557, 5.9714375, 4.4159822, 5.9672208, -1.4447441, 1.4400333

Time for backsubstitution: 13.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369675, upper bound: 1.0369833
time: 3.56 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0414093, upper bound: 1.0382490
time: 3.60 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -12.6260834, -9.4460993, -12.6260834, -9.4460993, -2.9519100, 2.9519095
1: -11.7355270, -9.1776171, -11.7355270, -9.1776171, -2.2048516, 2.2048516
2: -8.1624203, -6.1997900, -8.1624203, -6.1997900, -1.9588785, 1.9588783
3: -7.7232952, -5.1149702, -7.7232952, -5.1149702, -2.4360862, 2.4360862
4: -3.6771927, -1.3426094, -3.6771927, -1.3426094, -2.3345833, 2.3345833
5: -5.9543266, -3.8286080, -5.9543266, -3.8286080, -2.1257186, 2.1257186
6: -16.9029465, -13.7977123, -16.9029465, -13.7977123, -2.6528008, 2.6528008
7: -4.6868644, -2.2577372, -4.6868644, -2.2577372, -2.4291272, 2.4291272
8: -5.2317648, -2.9253731, -5.2317648, -2.9253731, -1.8565104, 1.8565102
9: 4.4055557, 5.9714375, 4.4055557, 5.9714375, -1.4457304, 1.4457304

Time for backsubstitution: 14.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369676, upper bound: 1.0369834
time: 3.76 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0414095, upper bound: 1.0382490
time: 3.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.63 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.63
Output dim: 9, lower bound: -1.0338045, upper bound: 1.0369678
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.63
Output dim: 9, lower bound: -1.0338045, upper bound: 1.0369678
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.63
Output dim: 9, lower bound: -1.0338045, upper bound: 1.0401465
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.63
Output dim: 9, lower bound: -1.0338046, upper bound: 1.0414106
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.63
Output dim: 9, lower bound: -1.0369675, upper bound: 1.0369833
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.63
Output dim: 9, lower bound: -1.0414093, upper bound: 1.0382490
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.63
Output dim: 9, lower bound: -1.0369676, upper bound: 1.0369834
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.63
Output dim: 9, lower bound: -1.0414095, upper bound: 1.0382490

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.6037884, -9.4831924, -12.6150150, -9.4680367, -2.9074249, 2.8985443
1: -11.7170792, -9.1952362, -11.7224598, -9.1907425, -2.1718283, 2.1736960
2: -8.1336479, -6.2195520, -8.1392603, -6.2174459, -1.8982735, 1.8616037
3: -7.6605272, -5.1412668, -7.6764216, -5.1337681, -2.3697319, 2.3760476
4: -3.6505313, -1.3825490, -3.6561694, -1.3796194, -2.2709119, 2.2736204
5: -5.9363489, -3.8574817, -5.9433832, -3.8529432, -2.0834057, 2.0859015
6: -16.8729839, -13.8415785, -16.8815880, -13.8336105, -2.5825715, 2.5847590
7: -4.6291065, -2.3034220, -4.6343160, -2.2939410, -2.3351655, 2.3308940
8: -5.2035871, -2.9603701, -5.2098694, -2.9440174, -1.7875657, 1.8018596
9: 4.4352264, 5.9635563, 4.4265347, 5.9674120, -1.4113655, 1.4144979

Time for backsubstitution: 14.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338046, upper bound: 1.0338077
time: 3.51 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338046, upper bound: 1.0369700
time: 3.41 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.6142950, -9.4646645, -12.6150150, -9.4680367, -2.9214211, 2.9225235
1: -11.7295742, -9.1830063, -11.7224598, -9.1907425, -2.1843824, 2.1863086
2: -8.1562166, -6.2025957, -8.1392603, -6.2174459, -1.9201846, 1.8798883
3: -7.7039728, -5.1228151, -7.6764216, -5.1337681, -2.4121213, 2.3942037
4: -3.6705706, -1.3458877, -3.6561694, -1.3796194, -2.2909513, 2.3102818
5: -5.9458466, -3.8334899, -5.9433832, -3.8529432, -2.0929034, 2.1098933
6: -16.8938847, -13.8075418, -16.8815880, -13.8336105, -2.6066847, 2.6182320
7: -4.6812353, -2.2691472, -4.6343160, -2.2939410, -2.3872943, 2.3651688
8: -5.2248750, -2.9452572, -5.2098694, -2.9440174, -1.8064651, 1.8166864
9: 4.4159861, 5.9672213, 4.4265347, 5.9674120, -1.4309165, 1.4203382

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338045, upper bound: 1.0338057
time: 3.74 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338045, upper bound: 1.0369701
time: 3.54 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.6037884, -9.4831924, -12.6255217, -9.4495277, -2.9312925, 2.9126310
1: -11.7170792, -9.1952362, -11.7349644, -9.1785440, -2.1844072, 2.1863046
2: -8.1336479, -6.2195520, -8.1618776, -6.2004566, -1.9165573, 1.8832831
3: -7.6605272, -5.1412668, -7.7198267, -5.1153212, -2.3878803, 2.4184000
4: -3.6505313, -1.3825490, -3.6762371, -1.3429542, -2.3075771, 2.2936881
5: -5.9363489, -3.8574817, -5.9528742, -3.8289349, -2.1074140, 2.0953925
6: -16.8729839, -13.8415785, -16.9024963, -13.7995739, -2.6160417, 2.6089315
7: -4.6291065, -2.3034220, -4.6864548, -2.2596841, -2.3694224, 2.3830328
8: -5.2035871, -2.9603701, -5.2311497, -2.9289269, -1.8021677, 1.8207569
9: 4.4352264, 5.9635563, 4.4072857, 5.9710965, -1.4172208, 1.4340585

Time for backsubstitution: 14.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338045, upper bound: 1.0369840
time: 3.57 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338046, upper bound: 1.0401463
time: 3.56 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.6142950, -9.4646645, -12.6255217, -9.4495277, -2.9428797, 2.9342055
1: -11.7295742, -9.1830063, -11.7349644, -9.1785440, -2.2039957, 2.2059524
2: -8.1562166, -6.2025957, -8.1618776, -6.2004566, -1.9397740, 1.9028728
3: -7.7039728, -5.1228151, -7.7198267, -5.1153212, -2.4151053, 2.4213912
4: -3.6705706, -1.3458877, -3.6762371, -1.3429542, -2.3276165, 2.3303494
5: -5.9458466, -3.8334899, -5.9528742, -3.8289349, -2.1169116, 2.1193843
6: -16.8938847, -13.8075418, -16.9024963, -13.7995739, -2.6269910, 2.6292410
7: -4.6812353, -2.2691472, -4.6864548, -2.2596841, -2.4065971, 2.4173076
8: -5.2248750, -2.9452572, -5.2311497, -2.9289269, -1.8161016, 1.8306684
9: 4.4159861, 5.9672213, 4.4072857, 5.9710965, -1.4428909, 1.4460222

Time for backsubstitution: 14.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338046, upper bound: 1.0360118
time: 6.18 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338045, upper bound: 1.0369679
time: 7.28 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.6155796, -9.4646130, -12.6127205, -9.4679451, -2.9183254, 2.9261198
1: -11.7230167, -9.1898079, -11.7275772, -9.1851664, -2.1838169, 2.1847093
2: -8.1397877, -6.2167826, -8.1528530, -6.2059212, -1.8754356, 1.9158430
3: -7.6798906, -5.1334224, -7.6956253, -5.1244884, -2.3961949, 2.4043121
4: -3.6571157, -1.3792810, -3.6689939, -1.3531744, -2.3039412, 2.2897129
5: -5.9448371, -3.8526239, -5.9446893, -3.8381221, -2.1067150, 2.0920653
6: -16.8820305, -13.8317547, -16.8916492, -13.8142185, -2.6112466, 2.6066976
7: -4.6347136, -2.2919960, -4.6708336, -2.2724094, -2.3623042, 2.3788376
8: -5.2104826, -2.9404635, -5.2208276, -2.9473310, -1.8179665, 1.8058743
9: 4.4248109, 5.9677482, 4.4195633, 5.9667583, -1.4242859, 1.4269905

Time for backsubstitution: 13.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369674, upper bound: 1.0338069
time: 3.84 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369675, upper bound: 1.0369834
time: 3.78 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.6260815, -9.4461040, -12.6142950, -9.4646616, -2.9367728, 2.9484076
1: -11.7355242, -9.1776180, -11.7295780, -9.1830034, -2.2064967, 2.1994331
2: -8.1624174, -6.1997933, -8.1562185, -6.2025928, -1.9021182, 1.9371941
3: -7.7232876, -5.1149745, -7.7039814, -5.1228108, -2.4249783, 2.4306381
4: -3.6771913, -1.3426149, -3.6705718, -1.3458829, -2.3313084, 2.3279569
5: -5.9543257, -3.8286128, -5.9458489, -3.8334849, -2.1208408, 2.1172361
6: -16.9029388, -13.7977200, -16.8938866, -13.8075314, -2.6419902, 2.6298969
7: -4.6868525, -2.2577424, -4.6812468, -2.2691436, -2.4177089, 2.4235044
8: -5.2317591, -2.9253755, -5.2248793, -2.9452572, -1.8313043, 1.8248253
9: 4.4055595, 5.9714379, 4.4159822, 5.9672208, -1.4480143, 1.4371935

Time for backsubstitution: 13.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401438, upper bound: 1.0338070
time: 3.78 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401439, upper bound: 1.0382468
time: 3.88 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.6155796, -9.4646130, -12.6245117, -9.4493790, -2.9319916, 2.9255958
1: -11.7230167, -9.1898079, -11.7335243, -9.1797705, -2.1878762, 2.1901062
2: -8.1397877, -6.2167826, -8.1590424, -6.2031207, -1.9329867, 1.9375365
3: -7.6798906, -5.1334224, -7.7149506, -5.1166463, -2.3920546, 2.4097486
4: -3.6571157, -1.3792810, -3.6756058, -1.3499019, -2.3072138, 2.2963247
5: -5.9448371, -3.8526239, -5.9531703, -3.8332491, -2.1115880, 2.1005464
6: -16.8820305, -13.8317547, -16.9007072, -13.8043985, -2.6220808, 2.6164105
7: -4.6347136, -2.2919960, -4.6764493, -2.2609990, -2.3737147, 2.3844533
8: -5.2104826, -2.9404635, -5.2277107, -2.9274454, -1.8355305, 1.8345790
9: 4.4248109, 5.9677482, 4.4091396, 5.9709697, -1.4252703, 1.4326971

Time for backsubstitution: 13.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369675, upper bound: 1.0338070
time: 3.84 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369675, upper bound: 1.0369814
time: 6.42 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.6260815, -9.4461040, -12.6260834, -9.4460993, -2.9504309, 2.9480224
1: -11.7355242, -9.1776180, -11.7355270, -9.1776171, -2.2105541, 2.2048495
2: -8.1624174, -6.1997933, -8.1624203, -6.1997900, -1.9599342, 1.9588742
3: -7.7232876, -5.1149745, -7.7232952, -5.1149702, -2.4209166, 2.4360843
4: -3.6771913, -1.3426149, -3.6771927, -1.3426094, -2.3345819, 2.3345778
5: -5.9543257, -3.8286128, -5.9543266, -3.8286080, -2.1257176, 2.1257138
6: -16.9029388, -13.7977200, -16.9029465, -13.7977123, -2.6527975, 2.6396325
7: -4.6868525, -2.2577424, -4.6868644, -2.2577372, -2.4112754, 2.4291220
8: -5.2317591, -2.9253755, -5.2317648, -2.9253731, -1.8488760, 1.8537579
9: 4.4055595, 5.9714379, 4.4055557, 5.9714375, -1.4489746, 1.4428911

Time for backsubstitution: 14.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401440, upper bound: 1.0338052
time: 3.74 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401440, upper bound: 1.0382473
time: 3.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.71 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0338046, upper bound: 1.0338077
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0338046, upper bound: 1.0369700
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0338045, upper bound: 1.0338057
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0338045, upper bound: 1.0369701
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0338045, upper bound: 1.0369840
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0338046, upper bound: 1.0401463
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0338046, upper bound: 1.0360118
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0338045, upper bound: 1.0369679
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0369674, upper bound: 1.0338069
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0369675, upper bound: 1.0369834
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0401438, upper bound: 1.0338070
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0401439, upper bound: 1.0382468
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0369675, upper bound: 1.0338070
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0369675, upper bound: 1.0369814
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0401440, upper bound: 1.0338052
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -1.0401440, upper bound: 1.0382473

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.6037884, -9.4831924, -12.6037884, -9.4831924, -2.8937535, 2.8937535
1: -11.7170792, -9.1952362, -11.7170792, -9.1952362, -2.1666803, 2.1666799
2: -8.1336479, -6.2195520, -8.1336479, -6.2195520, -1.8542871, 1.8542871
3: -7.6605272, -5.1412668, -7.6605272, -5.1412668, -2.3617754, 2.3617754
4: -3.6505313, -1.3825490, -3.6505313, -1.3825490, -2.2679822, 2.2679822
5: -5.9363489, -3.8574817, -5.9363489, -3.8574817, -2.0788672, 2.0788672
6: -16.8729839, -13.8415785, -16.8729839, -13.8415785, -2.5757537, 2.5757535
7: -4.6291065, -2.3034220, -4.6291065, -2.3034220, -2.3256845, 2.3256845
8: -5.2035871, -2.9603701, -5.2035871, -2.9603701, -1.7704682, 1.7704680
9: 4.4352264, 5.9635563, 4.4352264, 5.9635563, -1.4067433, 1.4067432

Time for backsubstitution: 14.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338011, upper bound: 1.0327712
time: 3.49 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338011, upper bound: 1.0338042
time: 3.54 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.6037884, -9.4831924, -12.6155796, -9.4646130, -2.9129500, 2.8986979
1: -11.7170792, -9.1952362, -11.7230167, -9.1898079, -2.1729741, 2.1742249
2: -8.1336479, -6.2195520, -8.1397877, -6.2167826, -1.8967361, 1.8608608
3: -7.6605272, -5.1412668, -7.6798906, -5.1334224, -2.3700948, 2.3796291
4: -3.6505313, -1.3825490, -3.6571157, -1.3792810, -2.2712502, 2.2745667
5: -5.9363489, -3.8574817, -5.9448371, -3.8526239, -2.0837250, 2.0873554
6: -16.8729839, -13.8415785, -16.8820305, -13.8317547, -2.5854740, 2.5843432
7: -4.6291065, -2.3034220, -4.6347136, -2.2919960, -2.3371105, 2.3312917
8: -5.2035871, -2.9603701, -5.2104826, -2.9404635, -1.7913215, 1.8024924
9: 4.4352264, 5.9635563, 4.4248109, 5.9677482, -1.4117794, 1.4164912

Time for backsubstitution: 14.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338012, upper bound: 1.0359336
time: 3.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338012, upper bound: 1.0369665
time: 3.46 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.6142950, -9.4646645, -12.6037884, -9.4831924, -2.9077497, 2.9176421
1: -11.7295742, -9.1830063, -11.7170792, -9.1952362, -2.1792336, 2.1792929
2: -8.1562166, -6.2025957, -8.1336479, -6.2195520, -1.8759317, 1.8725715
3: -7.7039728, -5.1228151, -7.6605272, -5.1412668, -2.4041648, 2.3799319
4: -3.6705706, -1.3458877, -3.6505313, -1.3825490, -2.2880216, 2.3046436
5: -5.9458466, -3.8334899, -5.9363489, -3.8574817, -2.0883648, 2.1028590
6: -16.8938847, -13.8075418, -16.8729839, -13.8415785, -2.5998664, 2.6092265
7: -4.6812353, -2.2691472, -4.6291065, -2.3034220, -2.3778133, 2.3599594
8: -5.2248750, -2.9452572, -5.2035871, -2.9603701, -1.7893672, 1.7850883
9: 4.4159861, 5.9672213, 4.4352264, 5.9635563, -1.4262941, 1.4125830

Time for backsubstitution: 13.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369773, upper bound: 1.0327712
time: 3.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369775, upper bound: 1.0338042
time: 3.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.6142950, -9.4646645, -12.6155796, -9.4646130, -2.9269462, 2.9226770
1: -11.7295742, -9.1830063, -11.7230167, -9.1898079, -2.1855268, 2.1868372
2: -8.1562166, -6.2025957, -8.1397877, -6.2167826, -1.9186473, 1.8791454
3: -7.7039728, -5.1228151, -7.6798906, -5.1334224, -2.4124842, 2.3977852
4: -3.6705706, -1.3458877, -3.6571157, -1.3792810, -2.2912896, 2.3112280
5: -5.9458466, -3.8334899, -5.9448371, -3.8526239, -2.0932226, 2.1113472
6: -16.8938847, -13.8075418, -16.8820305, -13.8317547, -2.6095862, 2.6178160
7: -4.6812353, -2.2691472, -4.6347136, -2.2919960, -2.3892393, 2.3655665
8: -5.2248750, -2.9452572, -5.2104826, -2.9404635, -1.8102214, 1.8173194
9: 4.4159861, 5.9672213, 4.4248109, 5.9677482, -1.4313304, 1.4223315

Time for backsubstitution: 13.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369773, upper bound: 1.0359335
time: 3.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369773, upper bound: 1.0369665
time: 3.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.6037884, -9.4831924, -12.6142950, -9.4646645, -2.9176426, 2.9077499
1: -11.7170792, -9.1952362, -11.7295742, -9.1830063, -2.1792927, 2.1792336
2: -8.1336479, -6.2195520, -8.1562166, -6.2025957, -1.8725717, 1.8759313
3: -7.6605272, -5.1412668, -7.7039728, -5.1228151, -2.3799324, 2.4041648
4: -3.6505313, -1.3825490, -3.6705706, -1.3458877, -2.3046436, 2.2880216
5: -5.9363489, -3.8574817, -5.9458466, -3.8334899, -2.1028590, 2.0883648
6: -16.8729839, -13.8415785, -16.8938847, -13.8075418, -2.6092262, 2.5998664
7: -4.6291065, -2.3034220, -4.6812353, -2.2691472, -2.3599594, 2.3778133
8: -5.2035871, -2.9603701, -5.2248750, -2.9452572, -1.7850883, 1.7893677
9: 4.4352264, 5.9635563, 4.4159861, 5.9672213, -1.4125829, 1.4262941

Time for backsubstitution: 14.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338013, upper bound: 1.0359474
time: 3.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338013, upper bound: 1.0369805
time: 3.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.6037884, -9.4831924, -12.6260815, -9.4461040, -2.9368181, 2.9127879
1: -11.7170792, -9.1952362, -11.7355242, -9.1776180, -2.1855459, 2.1868455
2: -8.1336479, -6.2195520, -8.1624174, -6.1997933, -1.9150333, 1.8825240
3: -7.6605272, -5.1412668, -7.7232876, -5.1149745, -2.3882422, 2.4219849
4: -3.6505313, -1.3825490, -3.6771913, -1.3426149, -2.3079164, 2.2946422
5: -5.9363489, -3.8574817, -5.9543257, -3.8286128, -2.1077361, 2.0968440
6: -16.8729839, -13.8415785, -16.9029388, -13.7977200, -2.6189437, 2.6085122
7: -4.6291065, -2.3034220, -4.6868525, -2.2577424, -2.3713641, 2.3834305
8: -5.2035871, -2.9603701, -5.2317591, -2.9253755, -1.8059220, 1.8213708
9: 4.4352264, 5.9635563, 4.4055595, 5.9714379, -1.4176390, 1.4360528

Time for backsubstitution: 14.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338011, upper bound: 1.0391099
time: 3.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0338011, upper bound: 1.0401427
time: 3.45 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.6142950, -9.4646645, -12.6142950, -9.4646645, -2.9292293, 2.9292293
1: -11.7295742, -9.1830063, -11.7295742, -9.1830063, -2.1988826, 2.1988821
2: -8.1562166, -6.2025957, -8.1562166, -6.2025957, -1.8955214, 1.8955214
3: -7.7039728, -5.1228151, -7.7039728, -5.1228151, -2.4071565, 2.4071569
4: -3.6705706, -1.3458877, -3.6705706, -1.3458877, -2.3246830, 2.3246830
5: -5.9458466, -3.8334899, -5.9458466, -3.8334899, -2.1123567, 2.1123567
6: -16.8938847, -13.8075418, -16.8938847, -13.8075418, -2.6201766, 2.6201766
7: -4.6812353, -2.2691472, -4.6812353, -2.2691472, -2.4120882, 2.4120882
8: -5.2248750, -2.9452572, -5.2248750, -2.9452572, -1.7990077, 1.7990079
9: 4.4159861, 5.9672213, 4.4159861, 5.9672213, -1.4382796, 1.4382796

Time for backsubstitution: 14.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0382430, upper bound: 1.0349796
time: 3.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0382429, upper bound: 1.0360105
time: 3.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.6142950, -9.4646645, -12.6260815, -9.4461040, -2.9484053, 2.9343624
1: -11.7295742, -9.1830063, -11.7355242, -9.1776180, -2.2051353, 2.2064934
2: -8.1562166, -6.2025957, -8.1624174, -6.1997933, -1.9382505, 1.9021139
3: -7.7039728, -5.1228151, -7.7232876, -5.1149745, -2.4154677, 2.4249763
4: -3.6705706, -1.3458877, -3.6771913, -1.3426149, -2.3279557, 2.3313036
5: -5.9458466, -3.8334899, -5.9543257, -3.8286128, -2.1172338, 2.1208358
6: -16.8938847, -13.8075418, -16.9029388, -13.7977200, -2.6298940, 2.6288228
7: -4.6812353, -2.2691472, -4.6868525, -2.2577424, -2.4086213, 2.4177053
8: -5.2248750, -2.9452572, -5.2317591, -2.9253755, -1.8198559, 1.8313010
9: 4.4159861, 5.9672213, 4.4055595, 5.9714379, -1.4433019, 1.4480126

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0382431, upper bound: 1.0381417
time: 3.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0382431, upper bound: 1.0391728
time: 3.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.6155796, -9.4646130, -12.6037884, -9.4831924, -2.8986983, 2.9129496
1: -11.7230167, -9.1898079, -11.7170792, -9.1952362, -2.1742249, 2.1729734
2: -8.1397877, -6.2167826, -8.1336479, -6.2195520, -1.8608608, 1.8967364
3: -7.6798906, -5.1334224, -7.6605272, -5.1412668, -2.3796291, 2.3700945
4: -3.6571157, -1.3792810, -3.6505313, -1.3825490, -2.2745667, 2.2712502
5: -5.9448371, -3.8526239, -5.9363489, -3.8574817, -2.0873554, 2.0837250
6: -16.8820305, -13.8317547, -16.8729839, -13.8415785, -2.5843427, 2.5854738
7: -4.6347136, -2.2919960, -4.6291065, -2.3034220, -2.3312917, 2.3371105
8: -5.2104826, -2.9404635, -5.2035871, -2.9603701, -1.8024924, 1.7913220
9: 4.4248109, 5.9677482, 4.4352264, 5.9635563, -1.4164913, 1.4117795

Time for backsubstitution: 13.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369640, upper bound: 1.0327705
time: 3.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369640, upper bound: 1.0338035
time: 3.96 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.6155796, -9.4646130, -12.6142950, -9.4646645, -2.9226766, 2.9269462
1: -11.7230167, -9.1898079, -11.7295742, -9.1830063, -2.1868372, 2.1855271
2: -8.1397877, -6.2167826, -8.1562166, -6.2025957, -1.8791454, 1.9186473
3: -7.6798906, -5.1334224, -7.7039728, -5.1228151, -2.3977852, 2.4124842
4: -3.6571157, -1.3792810, -3.6705706, -1.3458877, -2.3112280, 2.2912896
5: -5.9448371, -3.8526239, -5.9458466, -3.8334899, -2.1113472, 2.0932226
6: -16.8820305, -13.8317547, -16.8938847, -13.8075418, -2.6178162, 2.6095865
7: -4.6347136, -2.2919960, -4.6812353, -2.2691472, -2.3655665, 2.3892393
8: -5.2104826, -2.9404635, -5.2248750, -2.9452572, -1.8173201, 1.8102214
9: 4.4248109, 5.9677482, 4.4159861, 5.9672213, -1.4223316, 1.4313304

Time for backsubstitution: 14.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369640, upper bound: 1.0359469
time: 3.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369640, upper bound: 1.0369799
time: 3.80 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.6260815, -9.4461040, -12.6037884, -9.4831924, -2.9127879, 2.9368186
1: -11.7355242, -9.1776180, -11.7170792, -9.1952362, -2.1868458, 2.1855464
2: -8.1624174, -6.1997933, -8.1336479, -6.2195520, -1.8825240, 1.9150331
3: -7.7232876, -5.1149745, -7.6605272, -5.1412668, -2.4219847, 2.3882422
4: -3.6771913, -1.3426149, -3.6505313, -1.3825490, -2.2946422, 2.3079164
5: -5.9543257, -3.8286128, -5.9363489, -3.8574817, -2.0968440, 2.1077361
6: -16.9029388, -13.7977200, -16.8729839, -13.8415785, -2.6085122, 2.6189435
7: -4.6868525, -2.2577424, -4.6291065, -2.3034220, -2.3834305, 2.3713641
8: -5.2317591, -2.9253755, -5.2035871, -2.9603701, -1.8213708, 1.8059218
9: 4.4055595, 5.9714379, 4.4352264, 5.9635563, -1.4360528, 1.4176389

Time for backsubstitution: 13.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401404, upper bound: 1.0327704
time: 3.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401404, upper bound: 1.0338034
time: 3.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.6260815, -9.4461040, -12.6142950, -9.4646645, -2.9343624, 2.9484053
1: -11.7355242, -9.1776180, -11.7295742, -9.1830063, -2.2064934, 2.2051351
2: -8.1624174, -6.1997933, -8.1562166, -6.2025957, -1.9021142, 1.9382505
3: -7.7232876, -5.1149745, -7.7039728, -5.1228151, -2.4249763, 2.4154675
4: -3.6771913, -1.3426149, -3.6705706, -1.3458877, -2.3313036, 2.3279557
5: -5.9543257, -3.8286128, -5.9458466, -3.8334899, -2.1208358, 2.1172338
6: -16.9029388, -13.7977200, -16.8938847, -13.8075418, -2.6288228, 2.6298945
7: -4.6868525, -2.2577424, -4.6812353, -2.2691472, -2.4177053, 2.4086211
8: -5.2317591, -2.9253755, -5.2248750, -2.9452572, -1.8313012, 1.8198562
9: 4.4055595, 5.9714379, 4.4159861, 5.9672213, -1.4480126, 1.4433019

Time for backsubstitution: 13.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401404, upper bound: 1.0349788
time: 3.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401404, upper bound: 1.0360098
time: 3.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.6155796, -9.4646130, -12.6155796, -9.4646130, -2.9123545, 2.9123538
1: -11.7230167, -9.1898079, -11.7230167, -9.1898079, -2.1783133, 2.1783135
2: -8.1397877, -6.2167826, -8.1397877, -6.2167826, -1.9184117, 1.9184120
3: -7.6798906, -5.1334224, -7.6798906, -5.1334224, -2.3754950, 2.3754950
4: -3.6571157, -1.3792810, -3.6571157, -1.3792810, -2.2778347, 2.2778347
5: -5.9448371, -3.8526239, -5.9448371, -3.8526239, -2.0922132, 2.0922132
6: -16.8820305, -13.8317547, -16.8820305, -13.8317547, -2.5951762, 2.5951762
7: -4.6347136, -2.2919960, -4.6347136, -2.2919960, -2.3427176, 2.3427176
8: -5.2104826, -2.9404635, -5.2104826, -2.9404635, -1.8200703, 1.8200703
9: 4.4248109, 5.9677482, 4.4248109, 5.9677482, -1.4174740, 1.4174738

Time for backsubstitution: 13.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369641, upper bound: 1.0327705
time: 3.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369641, upper bound: 1.0338036
time: 4.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.6155796, -9.4646130, -12.6260815, -9.4461040, -2.9363451, 2.9264331
1: -11.7230167, -9.1898079, -11.7355242, -9.1776180, -2.1908951, 2.1909344
2: -8.1397877, -6.2167826, -8.1624174, -6.1997933, -1.9366965, 1.9403396
3: -7.6798906, -5.1334224, -7.7232876, -5.1149745, -2.3936424, 2.4179313
4: -3.6571157, -1.3792810, -3.6771913, -1.3426149, -2.3145008, 2.2979102
5: -5.9448371, -3.8526239, -5.9543257, -3.8286128, -2.1162243, 2.1017017
6: -16.8820305, -13.8317547, -16.9029388, -13.7977200, -2.6286502, 2.6193178
7: -4.6347136, -2.2919960, -4.6868525, -2.2577424, -2.3769712, 2.3948565
8: -5.2104826, -2.9404635, -5.2317591, -2.9253755, -1.8348753, 1.8389487
9: 4.4248109, 5.9677482, 4.4055595, 5.9714379, -1.4233272, 1.4370332

Time for backsubstitution: 14.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369641, upper bound: 1.0359468
time: 3.50 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369641, upper bound: 1.0369797
time: 4.16 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.6260815, -9.4461040, -12.6155796, -9.4646130, -2.9264326, 2.9363456
1: -11.7355242, -9.1776180, -11.7230167, -9.1898079, -2.1909347, 2.1908948
2: -8.1624174, -6.1997933, -8.1397877, -6.2167826, -1.9403396, 1.9366968
3: -7.7232876, -5.1149745, -7.6798906, -5.1334224, -2.4179311, 2.3936427
4: -3.6771913, -1.3426149, -3.6571157, -1.3792810, -2.2979102, 2.3145008
5: -5.9543257, -3.8286128, -5.9448371, -3.8526239, -2.1017017, 2.1162243
6: -16.9029388, -13.7977200, -16.8820305, -13.8317547, -2.6193175, 2.6286502
7: -4.6868525, -2.2577424, -4.6347136, -2.2919960, -2.3948565, 2.3769712
8: -5.2317591, -2.9253755, -5.2104826, -2.9404635, -1.8389487, 1.8348753
9: 4.4055595, 5.9714379, 4.4248109, 5.9677482, -1.4370332, 1.4233272

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401404, upper bound: 1.0327685
time: 4.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401404, upper bound: 1.0338014
time: 5.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.6260815, -9.4461040, -12.6260815, -9.4461040, -2.9480190, 2.9480190
1: -11.7355242, -9.1776180, -11.7355242, -9.1776180, -2.2105513, 2.2105515
2: -8.1624174, -6.1997933, -8.1624174, -6.1997933, -1.9599299, 1.9599295
3: -7.7232876, -5.1149745, -7.7232876, -5.1149745, -2.4209146, 2.4209146
4: -3.6771913, -1.3426149, -3.6771913, -1.3426149, -2.3345764, 2.3345764
5: -5.9543257, -3.8286128, -5.9543257, -3.8286128, -2.1257129, 2.1257129
6: -16.9029388, -13.7977200, -16.9029388, -13.7977200, -2.6396296, 2.6396296
7: -4.6868525, -2.2577424, -4.6868525, -2.2577424, -2.4112725, 2.4112725
8: -5.2317591, -2.9253755, -5.2317591, -2.9253755, -1.8488731, 1.8488731
9: 4.4055595, 5.9714379, 4.4055595, 5.9714379, -1.4489727, 1.4489727

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401405, upper bound: 1.0349771
time: 3.93 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401405, upper bound: 1.0360098
time: 3.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 21.85 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0338011, upper bound: 1.0327712
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0338011, upper bound: 1.0338042
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0338012, upper bound: 1.0359336
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0338012, upper bound: 1.0369665
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0369773, upper bound: 1.0327712
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0369775, upper bound: 1.0338042
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0369773, upper bound: 1.0359335
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0369773, upper bound: 1.0369665
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0338013, upper bound: 1.0359474
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0338013, upper bound: 1.0369805
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0338011, upper bound: 1.0391099
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0338011, upper bound: 1.0401427
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0382430, upper bound: 1.0349796
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0382429, upper bound: 1.0360105
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0382431, upper bound: 1.0381417
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0382431, upper bound: 1.0391728
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0369640, upper bound: 1.0327705
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0369640, upper bound: 1.0338035
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0369640, upper bound: 1.0359469
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0369640, upper bound: 1.0369799
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0401404, upper bound: 1.0327704
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0401404, upper bound: 1.0338034
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0401404, upper bound: 1.0349788
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0401404, upper bound: 1.0360098
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0369641, upper bound: 1.0327705
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0369641, upper bound: 1.0338036
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0369641, upper bound: 1.0359468
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0369641, upper bound: 1.0369797
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0401404, upper bound: 1.0327685
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0401404, upper bound: 1.0338014
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0401405, upper bound: 1.0349771
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.85
Output dim: 9, lower bound: -1.0401405, upper bound: 1.0360098

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.5894165, -9.4834394, -12.6022205, -9.4832220, -2.8793812, 2.8918233
1: -11.7147350, -9.1957560, -11.7168236, -9.1952944, -2.1624837, 2.1641095
2: -8.1254482, -6.2202072, -8.1327581, -6.2196226, -1.8456724, 1.8528092
3: -7.6508093, -5.1419401, -7.6594734, -5.1413403, -2.3515773, 2.3598206
4: -3.6498561, -1.3846881, -3.6504588, -1.3827853, -2.2670708, 2.2657707
5: -5.9322414, -3.8579407, -5.9359026, -3.8575306, -2.0747108, 2.0779619
6: -16.8632393, -13.8422756, -16.8719273, -13.8416557, -2.5660748, 2.5743883
7: -4.6281543, -2.3042984, -4.6290002, -2.3035164, -2.3246379, 2.3247018
8: -5.2031207, -2.9636369, -5.2035379, -2.9607282, -1.7692049, 1.7664088
9: 4.4361553, 5.9620638, 4.4353271, 5.9633942, -1.4049182, 1.4040962

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0327689, upper bound: 1.0327697
time: 4.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0327689, upper bound: 1.0327695
time: 3.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.6064825, -9.4676933, -12.6037760, -9.4831924, -2.8956718, 2.9090948
1: -11.7186003, -9.1918821, -11.7170773, -9.1952381, -2.1702275, 2.1684642
2: -8.1345882, -6.2094717, -8.1336441, -6.2195516, -1.8539729, 1.8641622
3: -7.6623507, -5.1274810, -7.6605229, -5.1412687, -2.3616600, 2.3750451
4: -3.6542990, -1.3816853, -3.6505308, -1.3825505, -2.2717485, 2.2688456
5: -5.9377508, -3.8513942, -5.9363470, -3.8574817, -2.0802691, 2.0849528
6: -16.8748703, -13.8297348, -16.8729782, -13.8415794, -2.5762744, 2.5876102
7: -4.6312318, -2.3003809, -4.6291051, -2.3034220, -2.3278098, 2.3287241
8: -5.2076907, -2.9575229, -5.2035871, -2.9603720, -1.7743139, 1.7739285
9: 4.4320097, 5.9639344, 4.4352255, 5.9635563, -1.4091716, 1.4078504

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0327689, upper bound: 1.0338043
time: 3.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0327689, upper bound: 1.0338043
time: 3.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.5894165, -9.4834394, -12.6140137, -9.4646358, -2.8985786, 2.8967581
1: -11.7147350, -9.1957560, -11.7227640, -9.1898651, -2.1687760, 2.1716549
2: -8.1254482, -6.2202072, -8.1388969, -6.2168603, -1.8881063, 1.8593814
3: -7.6508093, -5.1419401, -7.6788368, -5.1334953, -2.3598967, 2.3776741
4: -3.6498561, -1.3846881, -3.6570425, -1.3795154, -2.2703407, 2.2723544
5: -5.9322414, -3.8579407, -5.9443889, -3.8526735, -2.0795679, 2.0864482
6: -16.8632393, -13.8422756, -16.8809738, -13.8318310, -2.5757957, 2.5829792
7: -4.6281543, -2.3042984, -4.6346068, -2.2920923, -2.3360620, 2.3303084
8: -5.2031207, -2.9636369, -5.2104325, -2.9408207, -1.7900577, 1.7984319
9: 4.4361553, 5.9620638, 4.4249110, 5.9675846, -1.4099545, 1.4138442

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0327681, upper bound: 1.0359316
time: 4.49 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0327681, upper bound: 1.0359317
time: 3.49 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.6064825, -9.4676933, -12.6155710, -9.4646130, -2.9148679, 2.9140830
1: -11.7186003, -9.1918821, -11.7230158, -9.1898079, -2.1765208, 2.1760099
2: -8.1345882, -6.2094717, -8.1397820, -6.2167845, -1.8964219, 1.8707356
3: -7.6623507, -5.1274810, -7.6798863, -5.1334229, -2.3699799, 2.3928988
4: -3.6542990, -1.3816853, -3.6571159, -1.3792822, -2.2750168, 2.2754307
5: -5.9377508, -3.8513942, -5.9448333, -3.8526249, -2.0851259, 2.0934391
6: -16.8748703, -13.8297348, -16.8820267, -13.8317585, -2.5859954, 2.5962012
7: -4.6312318, -2.3003809, -4.6347113, -2.2919977, -2.3392341, 2.3343303
8: -5.2076907, -2.9575229, -5.2104816, -2.9404659, -1.7951679, 1.8059893
9: 4.4320097, 5.9639344, 4.4248114, 5.9677482, -1.4142079, 1.4175990

Time for backsubstitution: 14.46 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=1.4499115943908691
rel_dist={9: [-1.041417153698955, 1.0414178245879198]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4628
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4628

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9239484, upper bound: 0.9264222
time: 4.49 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264198, upper bound: 0.9264224
time: 3.48 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.14 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.14
Output dim: 9, lower bound: -0.9239484, upper bound: 0.9264222
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.14
Output dim: 9, lower bound: -0.9264198, upper bound: 0.9264224

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.6142950, -9.4646616, -12.6253052, -9.4509029, -2.6947808, 2.6887937
1: -11.7295780, -9.1830034, -11.7347383, -9.1789150, -2.0372033, 2.0393543
2: -8.1562185, -6.2025928, -8.1616602, -6.2007189, -1.7923832, 1.7554693
3: -7.7039814, -5.1228108, -7.7184439, -5.1154594, -2.2634583, 2.2687924
4: -3.6705718, -1.3458829, -3.6758542, -1.3430860, -2.2141824, 2.2343931
5: -5.9458489, -3.8334849, -5.9522924, -3.8290606, -2.0369411, 2.0388162
6: -16.8938866, -13.8075314, -16.9023170, -13.8003120, -2.3748555, 2.3780632
7: -4.6812468, -2.2691436, -4.6863050, -2.2604613, -2.2580323, 2.2724099
8: -5.2248793, -2.9452572, -5.2309074, -2.9303536, -1.6538091, 1.6723459
9: 4.4159822, 5.9672208, 4.4079752, 5.9709582, -1.3434095, 1.3458604

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9226713, upper bound: 0.9227775
time: 3.37 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9239474, upper bound: 0.9264189
time: 4.78 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -12.6260834, -9.4460993, -12.6260853, -9.4460907, -2.7087617, 2.7005694
1: -11.7355270, -9.1776171, -11.7355261, -9.1776123, -2.0463715, 2.0438831
2: -8.1624203, -6.1997900, -8.1624212, -6.1997881, -1.7995596, 1.8117037
3: -7.7232952, -5.1149702, -7.7233071, -5.1149702, -2.2680392, 2.2821369
4: -3.6771927, -1.3426094, -3.6771948, -1.3426085, -2.2216625, 2.2216020
5: -5.9543266, -3.8286080, -5.9543295, -3.8286073, -2.0483327, 2.0534730
6: -16.9029465, -13.7977123, -16.9029446, -13.7977066, -2.3876100, 2.3872631
7: -4.6868644, -2.2577372, -4.6868649, -2.2577319, -2.2669582, 2.2630723
8: -5.2317648, -2.9253731, -5.2317653, -2.9253664, -1.6943960, 1.6902850
9: 4.4055557, 5.9714375, 4.4055529, 5.9714375, -1.3488095, 1.3538238

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264198, upper bound: 0.9239509
time: 4.38 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264198, upper bound: 0.9264202
time: 5.44 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.56 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.56
Output dim: 9, lower bound: -0.9226713, upper bound: 0.9227775
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.56
Output dim: 9, lower bound: -0.9239474, upper bound: 0.9264189
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.56
Output dim: 9, lower bound: -0.9264198, upper bound: 0.9239509
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.56
Output dim: 9, lower bound: -0.9264198, upper bound: 0.9264202

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.6120644, -9.4693012, -12.6147919, -9.4694204, -2.6676407, 2.6670756
1: -11.7267456, -9.1860628, -11.7222338, -9.1911182, -2.0216632, 2.0211430
2: -8.1514482, -6.2072983, -8.1390467, -6.2177081, -1.7697654, 1.7282948
3: -7.6921673, -5.1251917, -7.6750240, -5.1339102, -2.2337437, 2.2241783
4: -3.6683304, -1.3561938, -3.6557870, -1.3797555, -2.1755252, 2.2059128
5: -5.9442081, -3.8400362, -5.9428005, -3.8530736, -2.0093989, 2.0119989
6: -16.8907185, -13.8169785, -16.8814087, -13.8343544, -2.3372781, 2.3445997
7: -4.6665182, -2.2737746, -4.6341553, -2.2947202, -2.2130609, 2.2163284
8: -5.2191539, -2.9481945, -5.2096214, -2.9454460, -1.6302924, 1.6504636
9: 4.4210367, 5.9665642, 4.4272251, 5.9672756, -1.3285568, 1.3250313

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9226687, upper bound: 0.9219869
time: 3.52 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9226687, upper bound: 0.9227749
time: 3.73 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.6142950, -9.4646616, -12.6253042, -9.4509058, -2.6902132, 2.6873159
1: -11.7295780, -9.1830034, -11.7347364, -9.1789179, -2.0372000, 2.0446298
2: -8.1562185, -6.2025928, -8.1616564, -6.2007232, -1.7923794, 1.7564447
3: -7.7039814, -5.1228108, -7.7184362, -5.1154609, -2.2634563, 2.2516026
4: -3.6705718, -1.3458829, -3.6758518, -1.3430908, -2.1989756, 2.2343919
5: -5.9458489, -3.8334849, -5.9522915, -3.8290660, -2.0352731, 2.0331900
6: -16.8938866, -13.8075314, -16.9023170, -13.8003178, -2.3599379, 2.3780606
7: -4.6812468, -2.2691436, -4.6862931, -2.2604654, -2.2580295, 2.2426963
8: -5.2248793, -2.9452572, -5.2309031, -2.9303546, -1.6510568, 1.6628838
9: 4.4159822, 5.9672208, 4.4079790, 5.9709578, -1.3405669, 1.3481113

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203038, upper bound: 0.9251456
time: 3.51 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203039, upper bound: 0.9264214
time: 3.45 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -12.6260834, -9.4460993, -12.6142950, -9.4646616, -2.6893458, 2.7025347
1: -11.7355270, -9.1776171, -11.7295780, -9.1830034, -2.0401154, 2.0388010
2: -8.1624203, -6.1997900, -8.1562185, -6.2025928, -1.7550535, 1.7909529
3: -7.7232952, -5.1149702, -7.7039814, -5.1228108, -2.2738209, 2.2639675
4: -3.6771927, -1.3426094, -3.6705718, -1.3458829, -2.2351179, 2.2150555
5: -5.9543266, -3.8286080, -5.9458489, -3.8334849, -2.0419345, 2.0363634
6: -16.9029465, -13.7977123, -16.8938866, -13.8075314, -2.3778834, 2.3789289
7: -4.6868644, -2.2577372, -4.6812468, -2.2691436, -2.2731442, 2.2608697
8: -5.2317648, -2.9253731, -5.2248793, -2.9452572, -1.6732078, 1.6590772
9: 4.4055557, 5.9714375, 4.4159822, 5.9672208, -1.3486581, 1.3439943

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9227747, upper bound: 0.9226739
time: 3.50 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264185, upper bound: 0.9239498
time: 4.10 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -12.6260834, -9.4460993, -12.6260834, -9.4460993, -2.7005649, 2.7005646
1: -11.7355270, -9.1776171, -11.7355270, -9.1776171, -2.0438824, 2.0438821
2: -8.1624203, -6.1997900, -8.1624203, -6.1997900, -1.8116970, 1.8116970
3: -7.7232952, -5.1149702, -7.7232952, -5.1149702, -2.2680387, 2.2680385
4: -3.6771927, -1.3426094, -3.6771927, -1.3426094, -2.2216616, 2.2216609
5: -5.9543266, -3.8286080, -5.9543266, -3.8286080, -2.0534668, 2.0534668
6: -16.9029465, -13.7977123, -16.9029465, -13.7977123, -2.3872607, 2.3872602
7: -4.6868644, -2.2577372, -4.6868644, -2.2577372, -2.2630701, 2.2630699
8: -5.2317648, -2.9253731, -5.2317648, -2.9253731, -1.6902831, 1.6902833
9: 4.4055557, 5.9714375, 4.4055557, 5.9714375, -1.3488078, 1.3488078

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9227751, upper bound: 0.9226740
time: 3.44 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264187, upper bound: 0.9239476
time: 5.05 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.16 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.16
Output dim: 9, lower bound: -0.9226687, upper bound: 0.9219869
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.16
Output dim: 9, lower bound: -0.9226687, upper bound: 0.9227749
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.16
Output dim: 9, lower bound: -0.9203038, upper bound: 0.9251456
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.16
Output dim: 9, lower bound: -0.9203039, upper bound: 0.9264214
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.16
Output dim: 9, lower bound: -0.9227747, upper bound: 0.9226739
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.16
Output dim: 9, lower bound: -0.9264185, upper bound: 0.9239498
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.16
Output dim: 9, lower bound: -0.9227751, upper bound: 0.9226740
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.16
Output dim: 9, lower bound: -0.9264187, upper bound: 0.9239476

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.5976992, -9.4695463, -12.6113777, -9.4694767, -2.6532254, 2.6632946
1: -11.7244072, -9.1865749, -11.7216768, -9.1912432, -2.0171938, 2.0180740
2: -8.1432428, -6.2079744, -8.1371021, -6.2178779, -1.7610607, 1.7257118
3: -7.6824608, -5.1258593, -7.6727190, -5.1340685, -2.2234468, 2.2209275
4: -3.6676550, -1.3583300, -3.6556249, -1.3802676, -2.1741524, 2.2034144
5: -5.9400983, -3.8405018, -5.9418221, -3.8531821, -2.0002279, 2.0058551
6: -16.8809757, -13.8176746, -16.8790970, -13.8345194, -2.3275578, 2.3419950
7: -4.6655698, -2.2746515, -4.6339245, -2.2949286, -2.2111425, 2.2147343
8: -5.2186933, -2.9514589, -5.2095122, -2.9462285, -1.6285336, 1.6463044
9: 4.4219666, 5.9650669, 4.4274464, 5.9669218, -1.3264027, 1.3221812

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9226688, upper bound: 0.9195163
time: 3.47 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9226687, upper bound: 0.9219869
time: 3.63 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.6147518, -9.4537964, -12.6147728, -9.4694185, -2.6672082, 2.6824629
1: -11.7282667, -9.1826954, -11.7222319, -9.1911173, -2.0246511, 2.0229499
2: -8.1523743, -6.1972394, -8.1390400, -6.2177105, -1.7682528, 1.7381659
3: -7.6939936, -5.1114044, -7.6750164, -5.1339111, -2.2323351, 2.2374446
4: -3.6720974, -1.3553286, -3.6557865, -1.3797574, -2.1792336, 2.2066283
5: -5.9456120, -3.8339415, -5.9427962, -3.8530738, -2.0135260, 2.0145209
6: -16.8926010, -13.8051357, -16.8814011, -13.8343573, -2.3362546, 2.3564558
7: -4.6686397, -2.2707405, -4.6341543, -2.2947209, -2.2140274, 2.2195315
8: -5.2232618, -2.9453602, -5.2096205, -2.9454489, -1.6341515, 1.6533737
9: 4.4178209, 5.9669390, 4.4272265, 5.9672756, -1.3309805, 1.3257561

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9226688, upper bound: 0.9203046
time: 3.49 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9226687, upper bound: 0.9227729
time: 3.82 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.6037884, -9.4831924, -12.6253042, -9.4509058, -2.6793036, 2.6633315
1: -11.7170792, -9.1952362, -11.7347364, -9.1789179, -2.0233126, 2.0254056
2: -8.1336479, -6.2195520, -8.1616564, -6.2007232, -1.7702184, 1.7369297
3: -7.6605272, -5.1412668, -7.7184362, -5.1154609, -2.2210608, 2.2490792
4: -3.6505313, -1.3825490, -3.6758518, -1.3430908, -2.1960545, 2.1974199
5: -5.9363489, -3.8574817, -5.9522915, -3.8290660, -2.0193338, 2.0080376
6: -16.8729839, -13.8415785, -16.9023170, -13.8003178, -2.3507347, 2.3445826
7: -4.6291065, -2.3034220, -4.6862931, -2.2604654, -2.2051587, 2.2421856
8: -5.2035871, -2.9603701, -5.2309031, -2.9303546, -1.6321535, 1.6547663
9: 4.4352264, 5.9635563, 4.4079790, 5.9709578, -1.3210123, 1.3371693

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203039, upper bound: 0.9226720
time: 4.82 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203038, upper bound: 0.9251437
time: 3.58 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.6142950, -9.4646645, -12.6253042, -9.4509058, -2.6902108, 2.6842253
1: -11.7295742, -9.1830063, -11.7347364, -9.1789179, -2.0424747, 2.0446267
2: -8.1562166, -6.2025957, -8.1616564, -6.2007232, -1.7933564, 1.7564404
3: -7.7039728, -5.1228151, -7.7184362, -5.1154609, -2.2462659, 2.2516007
4: -3.6705706, -1.3458877, -3.6758518, -1.3430908, -2.1989741, 2.2191842
5: -5.9458466, -3.8334899, -5.9522915, -3.8290660, -2.0352683, 2.0371523
6: -16.8938847, -13.8075418, -16.9023170, -13.8003178, -2.3599355, 2.3631439
7: -4.6812353, -2.2691472, -4.6862931, -2.2604654, -2.2283173, 2.2426929
8: -5.2248750, -2.9452572, -5.2309031, -2.9303546, -1.6442966, 1.6628802
9: 4.4159861, 5.9672213, 4.4079790, 5.9709578, -1.3456602, 1.3481098

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203039, upper bound: 0.9223324
time: 3.82 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203039, upper bound: 0.9227754
time: 3.74 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.6155796, -9.4646130, -12.6120644, -9.4693012, -2.6676245, 2.6753919
1: -11.7230167, -9.1898079, -11.7267456, -9.1860628, -2.0218863, 2.0232708
2: -8.1397877, -6.2167826, -8.1514482, -6.2072983, -1.7278907, 1.7683218
3: -7.6798906, -5.1334224, -7.6921673, -5.1251917, -2.2292013, 2.2342556
4: -3.6571157, -1.3792810, -3.6683304, -1.3561938, -2.2066274, 2.1763971
5: -5.9448371, -3.8526239, -5.9442081, -3.8400362, -2.0151186, 2.0088181
6: -16.8820305, -13.8317547, -16.8907185, -13.8169785, -2.3444262, 2.3413508
7: -4.6347136, -2.2919960, -4.6665182, -2.2737746, -2.2170620, 2.2158995
8: -5.2104826, -2.9404635, -5.2191539, -2.9481945, -1.6513479, 1.6355641
9: 4.4248109, 5.9677482, 4.4210367, 5.9665642, -1.3278270, 1.3291378

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9219857, upper bound: 0.9226694
time: 3.58 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9227721, upper bound: 0.9226712
time: 3.48 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.6260815, -9.4461040, -12.6142950, -9.4646616, -2.6878676, 2.6979659
1: -11.7355242, -9.1776180, -11.7295780, -9.1830034, -2.0453911, 2.0387983
2: -8.1624174, -6.1997933, -8.1562185, -6.2025928, -1.7560306, 1.7909489
3: -7.7232876, -5.1149745, -7.7039814, -5.1228108, -2.2566314, 2.2639656
4: -3.6771913, -1.3426149, -3.6705718, -1.3458829, -2.2351170, 2.1998487
5: -5.9543257, -3.8286128, -5.9458489, -3.8334849, -2.0363083, 2.0347013
6: -16.9029388, -13.7977200, -16.8938866, -13.8075314, -2.3778806, 2.3640122
7: -4.6868525, -2.2577424, -4.6812468, -2.2691436, -2.2434316, 2.2608674
8: -5.2317591, -2.9253755, -5.2248793, -2.9452572, -1.6637714, 1.6563253
9: 4.4055595, 5.9714379, 4.4159822, 5.9672208, -1.3509033, 1.3411545

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9251429, upper bound: 0.9203047
time: 3.57 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9251429, upper bound: 0.9203064
time: 3.72 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.6155796, -9.4646130, -12.6238537, -9.4507341, -2.6788492, 2.6732810
1: -11.7230167, -9.1898079, -11.7326880, -9.1806612, -2.0257010, 2.0282848
2: -8.1397877, -6.2167826, -8.1576328, -6.2044988, -1.7842698, 1.7890804
3: -7.6798906, -5.1334224, -7.7114973, -5.1173482, -2.2233419, 2.2383127
4: -3.6571157, -1.3792810, -3.6749406, -1.3529215, -2.1931467, 2.1829913
5: -5.9448371, -3.8526239, -5.9526863, -3.8351665, -2.0266442, 2.0259249
6: -16.8820305, -13.8317547, -16.8997669, -13.8071594, -2.3538294, 2.3496578
7: -4.6347136, -2.2919960, -4.6721344, -2.2623625, -2.2070661, 2.2180948
8: -5.2104826, -2.9404635, -5.2260351, -2.9283085, -1.6684289, 1.6665387
9: 4.4248109, 5.9677482, 4.4106135, 5.9707747, -1.3279734, 1.3339624

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9219860, upper bound: 0.9226688
time: 3.85 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9227725, upper bound: 0.9226712
time: 3.62 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.6260815, -9.4461040, -12.6260834, -9.4460993, -2.6990862, 2.6959982
1: -11.7355242, -9.1776180, -11.7355270, -9.1776171, -2.0491586, 2.0438800
2: -8.1624174, -6.1997933, -8.1624203, -6.1997900, -1.8126736, 1.8116927
3: -7.7232876, -5.1149745, -7.7232952, -5.1149702, -2.2508488, 2.2680366
4: -3.6771913, -1.3426149, -3.6771927, -1.3426094, -2.2216606, 2.2064550
5: -5.9543257, -3.8286128, -5.9543266, -3.8286080, -2.0478406, 2.0518093
6: -16.9029388, -13.7977200, -16.9029465, -13.7977123, -2.3872569, 2.3723433
7: -4.6868525, -2.2577424, -4.6868644, -2.2577372, -2.2333560, 2.2630675
8: -5.2317591, -2.9253755, -5.2317648, -2.9253731, -1.6808586, 1.6875310
9: 4.4055595, 5.9714379, 4.4055557, 5.9714375, -1.3510270, 1.3459685

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9251430, upper bound: 0.9203045
time: 3.80 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9251431, upper bound: 0.9239481
time: 3.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.03 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9226688, upper bound: 0.9195163
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9226687, upper bound: 0.9219869
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9226688, upper bound: 0.9203046
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9226687, upper bound: 0.9227729
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9203039, upper bound: 0.9226720
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9203038, upper bound: 0.9251437
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9203039, upper bound: 0.9223324
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9203039, upper bound: 0.9227754
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9219857, upper bound: 0.9226694
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9227721, upper bound: 0.9226712
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9251429, upper bound: 0.9203047
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9251429, upper bound: 0.9203064
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9219860, upper bound: 0.9226688
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9227725, upper bound: 0.9226712
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9251430, upper bound: 0.9203045
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.03
Output dim: 9, lower bound: -0.9251431, upper bound: 0.9239481

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.5976992, -9.4695463, -12.6003685, -9.4832525, -2.6417789, 2.6579912
1: -11.7244072, -9.1865749, -11.7165241, -9.1953592, -2.0125079, 2.0113165
2: -8.1432428, -6.2079744, -8.1317043, -6.2197070, -1.7172122, 1.7187350
3: -7.6824608, -5.1258593, -7.6582232, -5.1414270, -2.2156391, 2.2077529
4: -3.6676550, -1.3583300, -3.6503699, -1.3830605, -2.1887827, 2.1978514
5: -5.9400983, -3.8405018, -5.9353724, -3.8575897, -1.9942923, 1.9980166
6: -16.8809757, -13.8176746, -16.8706741, -13.8417435, -2.3219109, 2.3332031
7: -4.6655698, -2.2746515, -4.6288733, -2.3036284, -2.2200484, 2.2094009
8: -5.2186933, -2.9514589, -5.2034769, -2.9611506, -1.6129503, 1.6122582
9: 4.4219666, 5.9650669, 4.4354448, 5.9632034, -1.3219476, 1.3152769

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9195182
time: 3.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9195181
time: 3.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.5976992, -9.4695463, -12.6121578, -9.4646702, -2.6609769, 2.6638434
1: -11.7244072, -9.1865749, -11.7224617, -9.1899309, -2.0187998, 2.0188181
2: -8.1432428, -6.2079744, -8.1378431, -6.2169514, -1.7596178, 1.7253067
3: -7.6824608, -5.1258593, -7.6775846, -5.1335816, -2.2239599, 2.2259505
4: -3.6676550, -1.3583300, -3.6569541, -1.3797917, -2.1750250, 2.2041285
5: -5.9400983, -3.8405018, -5.9438577, -3.8527329, -1.9996476, 2.0089753
6: -16.8809757, -13.8176746, -16.8797188, -13.8319235, -2.3316312, 2.3418214
7: -4.6655698, -2.2746515, -4.6344824, -2.2922044, -2.2139802, 2.2154675
8: -5.2186933, -2.9514589, -5.2103753, -2.9412460, -1.6338046, 1.6471825
9: 4.4219666, 5.9650669, 4.4250298, 5.9673929, -1.3269842, 1.3249774

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203012, upper bound: 0.9219884
time: 3.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9219885
time: 3.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.6147518, -9.4537964, -12.6037731, -9.4831924, -2.6557627, 2.6771047
1: -11.7282667, -9.1826954, -11.7170773, -9.1952381, -2.0199661, 2.0161924
2: -8.1523743, -6.1972394, -8.1336412, -6.2195525, -1.7243886, 1.7311890
3: -7.6939936, -5.1114044, -7.6605206, -5.1412678, -2.2245259, 2.2242694
4: -3.6720974, -1.3553286, -3.6505313, -1.3825512, -2.1938624, 2.2010646
5: -5.9456120, -3.8339415, -5.9363456, -3.8574815, -2.0075917, 2.0066826
6: -16.8926010, -13.8051357, -16.8729801, -13.8415794, -2.3306072, 2.3476653
7: -4.6686397, -2.2707405, -4.6291046, -2.3034220, -2.2229886, 2.2141995
8: -5.2232618, -2.9453602, -5.2035871, -2.9603739, -1.6185687, 1.6192989
9: 4.4178209, 5.9669390, 4.4352255, 5.9635563, -1.3265257, 1.3188516

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9203045
time: 3.49 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9203045
time: 3.32 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.6147518, -9.4537964, -12.6155643, -9.4646091, -2.6749597, 2.6830120
1: -11.7282667, -9.1826954, -11.7230158, -9.1898088, -2.0262585, 2.0236928
2: -8.1523743, -6.1972394, -8.1397800, -6.2167830, -1.7668085, 1.7377617
3: -7.6939936, -5.1114044, -7.6798830, -5.1334219, -2.2328463, 2.2424681
4: -3.6720974, -1.3553286, -3.6571162, -1.3792825, -2.1801052, 2.2073421
5: -5.9456120, -3.8339415, -5.9448323, -3.8526244, -2.0129457, 2.0176404
6: -16.8926010, -13.8051357, -16.8820229, -13.8317566, -2.3403277, 2.3562813
7: -4.6686397, -2.2707405, -4.6347113, -2.2919981, -2.2168655, 2.2202644
8: -5.2232618, -2.9453602, -5.2104816, -2.9404669, -1.6394219, 1.6542580
9: 4.4178209, 5.9669390, 4.4248104, 5.9677467, -1.3315616, 1.3285526

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203012, upper bound: 0.9227747
time: 3.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9227748
time: 3.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.6037884, -9.4831924, -12.6142950, -9.4646645, -2.6678796, 2.6579874
1: -11.7170792, -9.1952362, -11.7295742, -9.1830063, -2.0186577, 2.0185986
2: -8.1336479, -6.2195520, -8.1562166, -6.2025957, -1.7265632, 1.7299228
3: -7.6605272, -5.1412668, -7.7039728, -5.1228151, -2.2132597, 2.2374923
4: -3.6505313, -1.3825490, -3.6705706, -1.3458877, -2.2106972, 2.1918280
5: -5.9363489, -3.8574817, -5.9458466, -3.8334899, -2.0133796, 2.0002084
6: -16.8729839, -13.8415785, -16.8938847, -13.8075418, -2.3450904, 2.3357303
7: -4.6291065, -2.3034220, -4.6812353, -2.2691472, -2.2141376, 2.2368276
8: -5.2035871, -2.9603701, -5.2248750, -2.9452572, -1.6165879, 1.6208675
9: 4.4352264, 5.9635563, 4.4159861, 5.9672213, -1.3165438, 1.3302553

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9218854
time: 3.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9226718
time: 3.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.6037884, -9.4831924, -12.6260815, -9.4461040, -2.6870561, 2.6638832
1: -11.7170792, -9.1952362, -11.7355242, -9.1776180, -2.0249109, 2.0261664
2: -8.1336479, -6.2195520, -8.1624174, -6.1997933, -1.7687883, 1.7365155
3: -7.6605272, -5.1412668, -7.7232876, -5.1149745, -2.2215695, 2.2497528
4: -3.6505313, -1.3825490, -3.6771913, -1.3426149, -2.1969280, 2.1981454
5: -5.9363489, -3.8574817, -5.9543257, -3.8286128, -2.0187564, 2.0111556
6: -16.8729839, -13.8415785, -16.9029388, -13.7977200, -2.3548074, 2.3444026
7: -4.6291065, -2.3034220, -4.6868525, -2.2577424, -2.2079964, 2.2418001
8: -5.2035871, -2.9603701, -5.2317591, -2.9253755, -1.6374216, 1.6556284
9: 4.4352264, 5.9635563, 4.4055595, 5.9714379, -1.3216000, 1.3399668

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9243563
time: 3.48 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9251427
time: 3.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.6142950, -9.4646645, -12.6142950, -9.4646645, -2.6787877, 2.6787877
1: -11.7295742, -9.1830063, -11.7295742, -9.1830063, -2.0378213, 2.0378208
2: -8.1562166, -6.2025957, -8.1562166, -6.2025957, -1.7494338, 1.7494338
3: -7.7039728, -5.1228151, -7.7039728, -5.1228151, -2.2384648, 2.2384648
4: -3.6705706, -1.3458877, -3.6705706, -1.3458877, -2.2135925, 2.2135928
5: -5.9458466, -3.8334899, -5.9458466, -3.8334899, -2.0293193, 2.0293190
6: -16.8938847, -13.8075418, -16.8938847, -13.8075418, -2.3542922, 2.3542917
7: -4.6812353, -2.2691472, -4.6812353, -2.2691472, -2.2373352, 2.2373352
8: -5.2248750, -2.9452572, -5.2248750, -2.9452572, -1.6287177, 1.6287177
9: 4.4159861, 5.9672213, 4.4159861, 5.9672213, -1.3412156, 1.3412156

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9239446, upper bound: 0.9215363
time: 4.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9239447, upper bound: 0.9223293
time: 4.03 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.6142950, -9.4646645, -12.6260815, -9.4461040, -2.6979632, 2.6847785
1: -11.7295742, -9.1830063, -11.7355242, -9.1776180, -2.0440741, 2.0453877
2: -8.1562166, -6.2025957, -8.1624174, -6.1997933, -1.7919264, 1.7560265
3: -7.7039728, -5.1228151, -7.7232876, -5.1149745, -2.2467752, 2.2566292
4: -3.6705706, -1.3458877, -3.6771913, -1.3426149, -2.1998472, 2.2199092
5: -5.9458466, -3.8334899, -5.9543257, -3.8286128, -2.0346966, 2.0402725
6: -16.8938847, -13.8075418, -16.9029388, -13.7977200, -2.3640096, 2.3629649
7: -4.6812353, -2.2691472, -4.6868525, -2.2577424, -2.2311544, 2.2434280
8: -5.2248750, -2.9452572, -5.2317591, -2.9253755, -1.6495659, 1.6637683
9: 4.4159861, 5.9672213, 4.4055595, 5.9714379, -1.3462379, 1.3509018

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9239447, upper bound: 0.9240064
time: 5.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9239447, upper bound: 0.9248020
time: 4.16 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.6121578, -9.4646702, -12.5976992, -9.4695463, -2.6638436, 2.6609766
1: -11.7224617, -9.1899309, -11.7244072, -9.1865749, -2.0188184, 2.0188000
2: -8.1378431, -6.2169514, -8.1432428, -6.2079744, -1.7253065, 1.7596176
3: -7.6775846, -5.1335816, -7.6824608, -5.1258593, -2.2259502, 2.2239594
4: -3.6569541, -1.3797917, -3.6676550, -1.3583300, -2.2041283, 2.1750250
5: -5.9438577, -3.8527329, -5.9400983, -3.8405018, -2.0089755, 1.9996476
6: -16.8797188, -13.8319235, -16.8809757, -13.8176746, -2.3418214, 2.3316312
7: -4.6344824, -2.2922044, -4.6655698, -2.2746515, -2.2154675, 2.2139800
8: -5.2103753, -2.9412460, -5.2186933, -2.9514589, -1.6471825, 1.6338046
9: 4.4250298, 5.9673929, 4.4219666, 5.9650669, -1.3249776, 1.3269842

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9219858, upper bound: 0.9218847
time: 3.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9219858, upper bound: 0.9226713
time: 3.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.6155643, -9.4646091, -12.6147518, -9.4537964, -2.6830120, 2.6749597
1: -11.7230158, -9.1898088, -11.7282667, -9.1826954, -2.0236931, 2.0262585
2: -8.1397800, -6.2167830, -8.1523743, -6.1972394, -1.7377617, 1.7668090
3: -7.6798830, -5.1334219, -7.6939936, -5.1114044, -2.2424679, 2.2328463
4: -3.6571162, -1.3792825, -3.6720974, -1.3553286, -2.2073421, 2.1801054
5: -5.9448323, -3.8526244, -5.9456120, -3.8339415, -2.0176401, 2.0129459
6: -16.8820229, -13.8317566, -16.8926010, -13.8051357, -2.3562813, 2.3403277
7: -4.6347113, -2.2919981, -4.6686397, -2.2707405, -2.2202649, 2.2168653
8: -5.2104816, -2.9404669, -5.2232618, -2.9453602, -1.6542580, 1.6394222
9: 4.4248104, 5.9677467, 4.4178209, 5.9669390, -1.3285527, 1.3315616

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9227722, upper bound: 0.9218847
time: 3.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9227722, upper bound: 0.9226713
time: 3.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.6260815, -9.4461040, -12.6037884, -9.4831924, -2.6638832, 2.6870561
1: -11.7355242, -9.1776180, -11.7170792, -9.1952362, -2.0261660, 2.0249114
2: -8.1624174, -6.1997933, -8.1336479, -6.2195520, -1.7365155, 1.7687881
3: -7.7232876, -5.1149745, -7.6605272, -5.1412668, -2.2497528, 2.2215698
4: -3.6771913, -1.3426149, -3.6505313, -1.3825490, -2.1981459, 2.1969278
5: -5.9543257, -3.8286128, -5.9363489, -3.8574817, -2.0111561, 2.0187566
6: -16.9029388, -13.7977200, -16.8729839, -13.8415785, -2.3444026, 2.3548074
7: -4.6868525, -2.2577424, -4.6291065, -2.3034220, -2.2418003, 2.2079959
8: -5.2317591, -2.9253755, -5.2035871, -2.9603701, -1.6556284, 1.6374216
9: 4.4055595, 5.9714379, 4.4352264, 5.9635563, -1.3399668, 1.3215998

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9251401, upper bound: 0.9195174
time: 3.97 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9251401, upper bound: 0.9203037
time: 3.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.6260815, -9.4461040, -12.6142950, -9.4646645, -2.6847787, 2.6979635
1: -11.7355242, -9.1776180, -11.7295742, -9.1830063, -2.0453882, 2.0440738
2: -8.1624174, -6.1997933, -8.1562166, -6.2025957, -1.7560265, 1.7919264
3: -7.7232876, -5.1149745, -7.7039728, -5.1228151, -2.2566290, 2.2467754
4: -3.6771913, -1.3426149, -3.6705706, -1.3458877, -2.2199092, 2.1998470
5: -5.9543257, -3.8286128, -5.9458466, -3.8334899, -2.0402727, 2.0346963
6: -16.9029388, -13.7977200, -16.8938847, -13.8075418, -2.3629646, 2.3640099
7: -4.6868525, -2.2577424, -4.6812353, -2.2691472, -2.2434282, 2.2311542
8: -5.2317591, -2.9253755, -5.2248750, -2.9452572, -1.6637683, 1.6495659
9: 4.4055595, 5.9714379, 4.4159861, 5.9672213, -1.3509016, 1.3462379

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9251403, upper bound: 0.9215361
time: 3.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9251403, upper bound: 0.9203019
time: 3.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.6121578, -9.4646702, -12.6094646, -9.4509773, -2.6750641, 2.6588426
1: -11.7224617, -9.1899309, -11.7303524, -9.1811800, -2.0226212, 2.0238156
2: -8.1378431, -6.2169514, -8.1494246, -6.2052054, -1.7816849, 1.7803683
3: -7.6775846, -5.1335816, -7.7017908, -5.1180158, -2.2200937, 2.2280216
4: -3.6569541, -1.3797917, -3.6742599, -1.3550556, -2.1906486, 2.1816163
5: -5.9438577, -3.8527329, -5.9485779, -3.8356285, -2.0205021, 2.0167556
6: -16.8797188, -13.8319235, -16.8900261, -13.8078527, -2.3512237, 2.3399439
7: -4.6344824, -2.2922044, -4.6711888, -2.2632403, -2.2054863, 2.2161791
8: -5.2103753, -2.9412460, -5.2255793, -2.9315681, -1.6642632, 1.6648046
9: 4.4250298, 5.9673929, 4.4115477, 5.9692783, -1.3251359, 1.3318135

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9219859, upper bound: 0.9218824
time: 4.00 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9219859, upper bound: 0.9226689
time: 4.45 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.6155643, -9.4646091, -12.6265583, -9.4352283, -2.6942377, 2.6726398
1: -11.7230158, -9.1898088, -11.7342167, -9.1773052, -2.0274887, 2.0312762
2: -8.1397800, -6.2167830, -8.1585684, -6.1944714, -1.7941396, 1.7875659
3: -7.6798830, -5.1334219, -7.7133131, -5.1035633, -2.2366090, 2.2369149
4: -3.6571162, -1.3792825, -3.6787097, -1.3520534, -2.1938658, 2.1867039
5: -5.9448323, -3.8526244, -5.9540911, -3.8290641, -2.0291743, 2.0300541
6: -16.8820229, -13.8317566, -16.9016609, -13.7953157, -2.3656847, 2.3486421
7: -4.6347113, -2.2919981, -4.6742630, -2.2593372, -2.2102947, 2.2190731
8: -5.2104816, -2.9404669, -5.2301440, -2.9254723, -1.6713362, 1.6703967
9: 4.4248104, 5.9677467, 4.4074039, 5.9711523, -1.3287058, 1.3363930

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9227724, upper bound: 0.9218847
time: 3.47 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9227724, upper bound: 0.9226713
time: 3.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.6260815, -9.4461040, -12.6155796, -9.4646130, -2.6750884, 2.6850009
1: -11.7355242, -9.1776180, -11.7230167, -9.1898079, -2.0299649, 2.0299253
2: -8.1624174, -6.1997933, -8.1397877, -6.2167826, -1.7931581, 1.7895155
3: -7.7232876, -5.1149745, -7.6798906, -5.1334224, -2.2498841, 2.2255950
4: -3.6771913, -1.3426149, -3.6571157, -1.3792810, -2.1846848, 2.2034960
5: -5.9543257, -3.8286128, -5.9448371, -3.8526239, -2.0226655, 2.0358691
6: -16.9029388, -13.7977200, -16.8820305, -13.8317547, -2.3537774, 2.3631096
7: -4.6868525, -2.2577424, -4.6347136, -2.2919960, -2.2329245, 2.2101793
8: -5.2317591, -2.9253755, -5.2104826, -2.9404635, -1.6727219, 1.6686485
9: 4.4055595, 5.9714379, 4.4248109, 5.9677482, -1.3401103, 1.3264046

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9251404, upper bound: 0.9195174
time: 3.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9251404, upper bound: 0.9203037
time: 3.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.6260815, -9.4461040, -12.6260815, -9.4461040, -2.6959953, 2.6959951
1: -11.7355242, -9.1776180, -11.7355242, -9.1776180, -2.0491552, 2.0491555
2: -8.1624174, -6.1997933, -8.1624174, -6.1997933, -1.8126693, 1.8126690
3: -7.7232876, -5.1149745, -7.7232876, -5.1149745, -2.2508478, 2.2508476
4: -3.6771913, -1.3426149, -3.6771913, -1.3426149, -2.2064543, 2.2064543
5: -5.9543257, -3.8286128, -5.9543257, -3.8286128, -2.0518045, 2.0518048
6: -16.9029388, -13.7977200, -16.9029388, -13.7977200, -2.3723404, 2.3723402
7: -4.6868525, -2.2577424, -4.6868525, -2.2577424, -2.2333541, 2.2333541
8: -5.2317591, -2.9253755, -5.2317591, -2.9253755, -1.6808558, 1.6808559
9: 4.4055595, 5.9714379, 4.4055595, 5.9714379, -1.3510251, 1.3510251

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9251405, upper bound: 0.9215354
time: 4.23 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9251405, upper bound: 0.9203036
time: 3.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.83 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9195182
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9195181
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9203012, upper bound: 0.9219884
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9219885
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9203045
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9203045
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9203012, upper bound: 0.9227747
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9227748
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9218854
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9226718
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9243563
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9203013, upper bound: 0.9251427
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9239446, upper bound: 0.9215363
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9239447, upper bound: 0.9223293
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9239447, upper bound: 0.9240064
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9239447, upper bound: 0.9248020
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9219858, upper bound: 0.9218847
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9219858, upper bound: 0.9226713
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9227722, upper bound: 0.9218847
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9227722, upper bound: 0.9226713
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9251401, upper bound: 0.9195174
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9251401, upper bound: 0.9203037
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9251403, upper bound: 0.9215361
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9251403, upper bound: 0.9203019
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9219859, upper bound: 0.9218824
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9219859, upper bound: 0.9226689
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9227724, upper bound: 0.9218847
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9227724, upper bound: 0.9226713
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9251404, upper bound: 0.9195174
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9251404, upper bound: 0.9203037
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9251405, upper bound: 0.9215354
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.83
Output dim: 9, lower bound: -0.9251405, upper bound: 0.9203036

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.5894165, -9.4834394, -12.6003685, -9.4832525, -2.6295700, 2.6402194
1: -11.7147350, -9.1957560, -11.7165241, -9.1953592, -2.0015731, 2.0029657
2: -8.1254482, -6.2202072, -8.1317043, -6.2197070, -1.6995940, 1.7056999
3: -7.6508093, -5.1419401, -7.6582232, -5.1414270, -2.1847944, 2.1918480
4: -3.6498561, -1.3846881, -3.6503699, -1.3830605, -2.1723604, 2.1712332
5: -5.9322414, -3.8579407, -5.9353724, -3.8575897, -1.9790568, 1.9820848
6: -16.8632393, -13.8422756, -16.8706741, -13.8417435, -2.3018980, 2.3090117
7: -4.6281543, -2.3042984, -4.6288733, -2.3036284, -2.1819983, 2.1823325
8: -5.2031207, -2.9636369, -5.2034769, -2.9611506, -1.6001940, 1.5978017
9: 4.4361553, 5.9620638, 4.4354448, 5.9632034, -1.3085608, 1.3078561

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9195210, upper bound: 0.9195181
time: 3.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9195210, upper bound: 0.9195163
time: 3.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.5999279, -9.4649096, -12.6003685, -9.4832525, -2.6435742, 2.6641107
1: -11.7272396, -9.1835213, -11.7165241, -9.1953592, -2.0141234, 2.0155921
2: -8.1480112, -6.2032909, -8.1317043, -6.2197070, -1.7212329, 1.7239807
3: -7.6942725, -5.1234794, -7.6582232, -5.1414270, -2.2272000, 2.2100103
4: -3.6698940, -1.3480227, -3.6503699, -1.3830605, -2.1904559, 2.2081997
5: -5.9417353, -3.8339572, -5.9353724, -3.8575897, -1.9910402, 2.0072374
6: -16.8841419, -13.8082352, -16.8706741, -13.8417435, -2.3260112, 2.3424833
7: -4.6802902, -2.2700262, -4.6288733, -2.3036284, -2.2349105, 2.2125401
8: -5.2244139, -2.9485188, -5.2034769, -2.9611506, -1.6191077, 1.6124239
9: 4.4169159, 5.9657230, 4.4354448, 5.9632034, -1.3281047, 1.3136928

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9195210, upper bound: 0.9195182
time: 3.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9195210, upper bound: 0.9195181
time: 3.49 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.5894165, -9.4834394, -12.6121578, -9.4646702, -2.6487679, 2.6460106
1: -11.7147350, -9.1957560, -11.7224617, -9.1899309, -2.0078645, 2.0104671
2: -8.1254482, -6.2202072, -8.1378431, -6.2169514, -1.7417908, 1.7122719
3: -7.6508093, -5.1419401, -7.6775846, -5.1335816, -2.1931152, 2.2100465
4: -3.6498561, -1.3846881, -3.6569541, -1.3797917, -2.1585855, 2.1775105
5: -5.9322414, -3.8579407, -5.9438577, -3.8527329, -1.9844117, 1.9930434
6: -16.8632393, -13.8422756, -16.8797188, -13.8319235, -2.3116183, 2.3176301
7: -4.6281543, -2.3042984, -4.6344824, -2.2922044, -2.1759386, 2.1883988
8: -5.2031207, -2.9636369, -5.2103753, -2.9412460, -1.6210482, 1.6325841
9: 4.4361553, 5.9620638, 4.4250298, 5.9673929, -1.3135974, 1.3175566

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9195202, upper bound: 0.9219882
time: 3.41 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9195202, upper bound: 0.9219865
time: 3.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.5999279, -9.4649096, -12.6121578, -9.4646702, -2.6627722, 2.6699910
1: -11.7272396, -9.1835213, -11.7224617, -9.1899309, -2.0204148, 2.0230932
2: -8.1480112, -6.2032909, -8.1378431, -6.2169514, -1.7636986, 1.7305529
3: -7.6942725, -5.1234794, -7.6775846, -5.1335816, -2.2349243, 2.2282085
4: -3.6698940, -1.3480227, -3.6569541, -1.3797917, -2.1767044, 2.2144768
5: -5.9417353, -3.8339572, -5.9438577, -3.8527329, -1.9963956, 2.0181954
6: -16.8841419, -13.8082352, -16.8797188, -13.8319235, -2.3357315, 2.3511016
7: -4.6802902, -2.2700262, -4.6344824, -2.2922044, -2.2272048, 2.2186067
8: -5.2244139, -2.9485188, -5.2103753, -2.9412460, -1.6399615, 1.6474133
9: 4.4169159, 5.9657230, 4.4250298, 5.9673929, -1.3331411, 1.3233933

Time for backsubstitution: 14.50 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=1.353825569152832
rel_dist={9: [-0.9264248152800034, 0.9264272454017544]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4628
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4628

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8638846, upper bound: 0.8660053
time: 3.99 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8660025, upper bound: 0.8660036
time: 4.19 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.38 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.38
Output dim: 9, lower bound: -0.8638846, upper bound: 0.8660053
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.38
Output dim: 9, lower bound: -0.8660025, upper bound: 0.8660036

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.6142950, -9.4646616, -12.6251688, -9.4517803, -2.5684857, 2.5640891
1: -11.7295780, -9.1830034, -11.7345943, -9.1791496, -1.9565930, 1.9588745
2: -8.1562185, -6.2025928, -8.1615210, -6.2008877, -1.7192013, 1.6822443
3: -7.7039814, -5.1228108, -7.7175589, -5.1155481, -2.1800284, 2.1847124
4: -3.6705718, -1.3458829, -3.6756074, -1.3431737, -2.1501484, 2.1697161
5: -5.9458489, -3.8334849, -5.9519234, -3.8291423, -1.9798365, 1.9813049
6: -16.8938866, -13.8075314, -16.9022064, -13.8007851, -2.2420440, 2.2458565
7: -4.6812468, -2.2691436, -4.6862006, -2.2609551, -2.1705303, 2.1844645
8: -5.2248793, -2.9452572, -5.2307491, -2.9312587, -1.5685992, 1.5893171
9: 4.4159822, 5.9672208, 4.4084139, 5.9708700, -1.2952828, 1.2973074

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8627765, upper bound: 0.8627618
time: 4.05 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8638837, upper bound: 0.8660023
time: 3.89 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -12.6260834, -9.4460993, -12.6260853, -9.4460907, -2.5843096, 2.5748968
1: -11.7355270, -9.1776171, -11.7355261, -9.1776123, -1.9660320, 1.9633982
2: -8.1624203, -6.1997900, -8.1624212, -6.1997881, -1.7264371, 1.7381129
3: -7.7232952, -5.1149702, -7.7233071, -5.1149702, -2.1840158, 2.1989732
4: -3.6771927, -1.3426094, -3.6771948, -1.3426085, -2.1577559, 2.1577277
5: -5.9543266, -3.8286080, -5.9543295, -3.8286073, -1.9913883, 1.9963298
6: -16.9029465, -13.7977123, -16.9029446, -13.7977066, -2.2555552, 2.2544928
7: -4.6868644, -2.2577372, -4.6868649, -2.2577319, -2.1799722, 2.1758604
8: -5.2317648, -2.9253731, -5.2317653, -2.9253664, -1.6115246, 1.6071715
9: 4.4055557, 5.9714375, 4.4055529, 5.9714375, -1.3003483, 1.3057810

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8660026, upper bound: 0.8638856
time: 3.89 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8660026, upper bound: 0.8660033
time: 3.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.60 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 22.60
Output dim: 9, lower bound: -0.8627765, upper bound: 0.8627618
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.60
Output dim: 9, lower bound: -0.8638837, upper bound: 0.8660023
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.60
Output dim: 9, lower bound: -0.8660026, upper bound: 0.8638856
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.60
Output dim: 9, lower bound: -0.8660026, upper bound: 0.8660033

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.6142960, -9.4646616, -12.6251640, -9.4517813, -2.5635796, 2.5626116
1: -11.7295771, -9.1830034, -11.7345915, -9.1791544, -1.9565525, 1.9639375
2: -8.1562195, -6.2025924, -8.1615171, -6.2008905, -1.7191906, 1.6831818
3: -7.7039828, -5.1228123, -7.7175508, -5.1155500, -2.1800265, 2.1665132
4: -3.6705716, -1.3458819, -3.6756060, -1.3431780, -2.1340470, 2.1697137
5: -5.9458485, -3.8334854, -5.9519229, -3.8291481, -1.9770360, 1.9756780
6: -16.8938866, -13.8075333, -16.9022007, -13.8007889, -2.2262521, 2.2458544
7: -4.6812463, -2.2691441, -4.6861901, -2.2609582, -2.1705284, 2.1530030
8: -5.2248807, -2.9452553, -5.2307444, -2.9312615, -1.5658467, 1.5789549
9: 4.4159808, 5.9672213, 4.4084177, 5.9708705, -1.2924399, 1.2990463

Time for backsubstitution: 13.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8606429, upper bound: 0.8648971
time: 4.20 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8606429, upper bound: 0.8660043
time: 4.23 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -12.6260834, -9.4460993, -12.6142950, -9.4646616, -2.5648932, 2.5776534
1: -11.7355270, -9.1776171, -11.7295780, -9.1830034, -1.9597759, 1.9584837
2: -8.1624203, -6.1997900, -8.1562185, -6.2025928, -1.6820493, 1.7178304
3: -7.7232952, -5.1149702, -7.7039814, -5.1228108, -2.1906576, 2.1806312
4: -3.6771927, -1.3426094, -3.6705718, -1.3458829, -2.1706619, 2.1511815
5: -5.9543266, -3.8286080, -5.9458489, -3.8334849, -1.9849901, 1.9793494
6: -16.9029465, -13.7977123, -16.8938866, -13.8075314, -2.2458291, 2.2468610
7: -4.6868644, -2.2577372, -4.6812468, -2.2691436, -2.1853347, 2.1738837
8: -5.2317648, -2.9253731, -5.2248793, -2.9452572, -1.5903363, 1.5748271
9: 4.4055557, 5.9714375, 4.4159822, 5.9672208, -1.3006151, 1.2959749

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8627606, upper bound: 0.8627774
time: 3.84 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8660013, upper bound: 0.8638864
time: 4.99 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -12.6260834, -9.4460993, -12.6260834, -9.4460993, -2.5748920, 2.5748923
1: -11.7355270, -9.1776171, -11.7355270, -9.1776171, -1.9633970, 1.9633973
2: -8.1624203, -6.1997900, -8.1624203, -6.1997900, -1.7381067, 1.7381063
3: -7.7232952, -5.1149702, -7.7232952, -5.1149702, -2.1840153, 2.1840148
4: -3.6771927, -1.3426094, -3.6771927, -1.3426094, -2.1577539, 2.1577535
5: -5.9543266, -3.8286080, -5.9543266, -3.8286080, -1.9963236, 1.9963238
6: -16.9029465, -13.7977123, -16.9029465, -13.7977123, -2.2544901, 2.2544901
7: -4.6868644, -2.2577372, -4.6868644, -2.2577372, -2.1758585, 2.1758580
8: -5.2317648, -2.9253731, -5.2317648, -2.9253731, -1.6071699, 1.6071699
9: 4.4055557, 5.9714375, 4.4055557, 5.9714375, -1.3003464, 1.3003466

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8627609, upper bound: 0.8627791
time: 4.01 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8660015, upper bound: 0.8638844
time: 7.00 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.76 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 25.76
Output dim: 9, lower bound: -0.8606429, upper bound: 0.8648971
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.76
Output dim: 9, lower bound: -0.8606429, upper bound: 0.8660043
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 25.76
Output dim: 9, lower bound: -0.8627606, upper bound: 0.8627774
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.76
Output dim: 9, lower bound: -0.8660013, upper bound: 0.8638864
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 25.76
Output dim: 9, lower bound: -0.8627609, upper bound: 0.8627791
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.76
Output dim: 9, lower bound: -0.8660015, upper bound: 0.8638844

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.6142950, -9.4646645, -12.6251640, -9.4517813, -2.5635762, 2.5591807
1: -11.7295742, -9.1830063, -11.7345915, -9.1791544, -1.9616532, 1.9639337
2: -8.1562166, -6.2025957, -8.1615171, -6.2008905, -1.7201352, 1.6831775
3: -7.7039728, -5.1228151, -7.7175508, -5.1155500, -2.1618257, 2.1665111
4: -3.6705706, -1.3458877, -3.6756060, -1.3431780, -2.1340451, 2.1536133
5: -5.9458466, -3.8334899, -5.9519229, -3.8291481, -1.9770312, 1.9785087
6: -16.8938847, -13.8075418, -16.9022007, -13.8007889, -2.2262487, 2.2300625
7: -4.6812353, -2.2691472, -4.6861901, -2.2609582, -2.1390681, 2.1530004
8: -5.2248750, -2.9452572, -5.2307444, -2.9312615, -1.5581911, 1.5789514
9: 4.4159861, 5.9672213, 4.4084177, 5.9708705, -1.2970231, 1.2990451

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8606428, upper bound: 0.8606447
time: 3.90 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8606429, upper bound: 0.8627616
time: 4.06 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.6260815, -9.4461040, -12.6142960, -9.4646616, -2.5634160, 2.5727456
1: -11.7355242, -9.1776180, -11.7295771, -9.1830034, -1.9648385, 1.9584432
2: -8.1624174, -6.1997933, -8.1562195, -6.2025924, -1.6829867, 1.7178197
3: -7.7232876, -5.1149745, -7.7039828, -5.1228123, -2.1724582, 2.1806290
4: -3.6771913, -1.3426149, -3.6705716, -1.3458819, -2.1706591, 2.1350803
5: -5.9543257, -3.8286128, -5.9458485, -3.8334854, -1.9793634, 1.9765558
6: -16.9029388, -13.7977200, -16.8938866, -13.8075333, -2.2458267, 2.2310705
7: -4.6868525, -2.2577424, -4.6812463, -2.2691441, -2.1538739, 2.1738818
8: -5.2317591, -2.9253755, -5.2248807, -2.9452553, -1.5800056, 1.5720752
9: 4.4055595, 5.9714379, 4.4159808, 5.9672213, -1.3023477, 1.2931350

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8648944, upper bound: 0.8606433
time: 5.21 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8648944, upper bound: 0.8606454
time: 4.58 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.6260815, -9.4461040, -12.6260834, -9.4461012, -2.5734138, 2.5699868
1: -11.7355242, -9.1776180, -11.7355270, -9.1776142, -1.9684601, 1.9633577
2: -8.1624174, -6.1997933, -8.1624203, -6.1997881, -1.7390432, 1.7380953
3: -7.7232876, -5.1149745, -7.7232962, -5.1149721, -2.1658173, 2.1840129
4: -3.6771913, -1.3426149, -3.6771927, -1.3426101, -2.1577539, 2.1416533
5: -5.9543257, -3.8286128, -5.9543266, -3.8286083, -1.9906979, 1.9935348
6: -16.9029388, -13.7977200, -16.9029427, -13.7977133, -2.2544866, 2.2386990
7: -4.6868525, -2.2577424, -4.6868644, -2.2577386, -2.1443968, 2.1758556
8: -5.2317591, -2.9253755, -5.2317648, -2.9253736, -1.5968511, 1.6044171
9: 4.4055595, 5.9714379, 4.4055562, 5.9714379, -1.3020532, 1.2975068

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8648946, upper bound: 0.8606433
time: 6.29 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8648945, upper bound: 0.8638863
time: 4.55 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.42 seconds
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.42
Output dim: 9, lower bound: -0.8606428, upper bound: 0.8606447
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 25.42
Output dim: 9, lower bound: -0.8606429, upper bound: 0.8627616
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.42
Output dim: 9, lower bound: -0.8648944, upper bound: 0.8606433
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 25.42
Output dim: 9, lower bound: -0.8648944, upper bound: 0.8606454
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.42
Output dim: 9, lower bound: -0.8648946, upper bound: 0.8606433
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 25.42
Output dim: 9, lower bound: -0.8648945, upper bound: 0.8638863
Binary search (step 2): status=Status.VERIFIED, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=1.3057825565338135
rel_dist={9: [-0.8660068367423657, 0.8660078437084611]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 1858.99 seconds
