## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.16694256000000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4273009, 0.4273009)
1: (-11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3401191, 0.3401189)
2: (-11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4596653, 0.4596653)
3: (-10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3955762, 0.3955760)
4: (-2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2465289, 0.2465290)
5: (-9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3903337, 0.3903337)
6: (-12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3224721, 0.3224721)
7: (-6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2837343, 0.2837342)
8: (-0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3891225, 0.3891225)
9: (2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3820839, 0.3820841)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.73 + 34.15 = 55.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.1738982, upper bound: 0.1738985

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 4640
type: A, layer: 1, pos: 5752

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1738866, upper bound: 0.1730411
time: 3.53 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1738976, upper bound: 0.1738972
time: 3.43 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.11 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.11
Output dim: 9, lower bound: -0.1738866, upper bound: 0.1730411
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.11
Output dim: 9, lower bound: -0.1738976, upper bound: 0.1738972

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -10.2468166, -9.1725855, -10.2508316, -9.1683722, -0.4192097, 0.4190769
1: -11.1884527, -10.3611889, -11.1897335, -10.3596973, -0.3376069, 0.3376667
2: -11.1811485, -10.3391953, -11.1822586, -10.3385725, -0.4580255, 0.4583812
3: -10.6043768, -9.8126945, -10.6064930, -9.8110046, -0.3905654, 0.3922265
4: -2.8171291, -2.1812935, -2.8186769, -2.1794705, -0.2433788, 0.2422197
5: -9.9185095, -8.8984604, -9.9213305, -8.8958168, -0.3851390, 0.3851323
6: -12.9212761, -12.0871592, -12.9223022, -12.0861521, -0.3205166, 0.3205142
7: -6.0027871, -5.3160534, -6.0050840, -5.3135734, -0.2791915, 0.2791814
8: -0.7677855, -0.1047544, -0.7726889, -0.1003079, -0.3802133, 0.3802338
9: 2.6518168, 3.3024251, 2.6495242, 3.3045654, -0.3778257, 0.3778811

Time for backsubstitution: 20.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4640
type: B, layer: 1, pos: 5752

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730408, upper bound: 0.1730407
time: 3.26 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730408, upper bound: 0.1730411
time: 4.31 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -10.2548389, -9.1682301, -10.2548409, -9.1682301, -0.4186273, 0.4273005
1: -11.1897335, -10.3585377, -11.1897326, -10.3585377, -0.3401191, 0.3385999
2: -11.1824350, -10.3379650, -11.1824360, -10.3379669, -0.4596024, 0.4583659
3: -10.6066322, -9.8093472, -10.6066303, -9.8093472, -0.3942647, 0.3950126
4: -2.8202233, -2.1794238, -2.8202238, -2.1794238, -0.2438347, 0.2460912
5: -9.9238539, -8.8957748, -9.9238539, -8.8957729, -0.3867793, 0.3903334
6: -12.9232407, -12.0861320, -12.9232416, -12.0861320, -0.3210566, 0.3224721
7: -6.0050840, -5.3113313, -6.0050840, -5.3113294, -0.2837329, 0.2795016
8: -0.7771072, -0.1003079, -0.7771058, -0.1003079, -0.3811164, 0.3891213
9: 2.6494017, 3.3066108, 2.6494017, 3.3066115, -0.3820844, 0.3796787

Time for backsubstitution: 21.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4640
type: B, layer: 1, pos: 5752

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730408, upper bound: 0.1738865
time: 3.30 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730410, upper bound: 0.1738869
time: 4.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.18 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.18
Output dim: 9, lower bound: -0.1730408, upper bound: 0.1730407
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.18
Output dim: 9, lower bound: -0.1730408, upper bound: 0.1730411
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.18
Output dim: 9, lower bound: -0.1730408, upper bound: 0.1738865
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.18
Output dim: 9, lower bound: -0.1730410, upper bound: 0.1738869

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -10.2468166, -9.1725855, -10.2468166, -9.1725855, -0.4150596, 0.4150598
1: -11.1884527, -10.3611889, -11.1884527, -10.3611889, -0.3363683, 0.3363686
2: -11.1811485, -10.3391953, -11.1811485, -10.3391953, -0.4574065, 0.4574065
3: -10.6043768, -9.8126945, -10.6043768, -9.8126945, -0.3890634, 0.3890631
4: -2.8171291, -2.1812935, -2.8171291, -2.1812935, -0.2411704, 0.2411704
5: -9.9185095, -8.8984604, -9.9185095, -8.8984604, -0.3825078, 0.3825078
6: -12.9212761, -12.0871592, -12.9212761, -12.0871592, -0.3195193, 0.3195193
7: -6.0027871, -5.3160534, -6.0027871, -5.3160534, -0.2768834, 0.2768834
8: -0.7677855, -0.1047544, -0.7677855, -0.1047544, -0.3757563, 0.3757560
9: 2.6518168, 3.3024251, 2.6518168, 3.3024251, -0.3756833, 0.3756835

Time for backsubstitution: 20.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4640
type: A, layer: 1, pos: 5752

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4640

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727342, upper bound: 0.1730402
time: 3.56 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730399, upper bound: 0.1730402
time: 3.48 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -10.2468166, -9.1725855, -10.2548389, -9.1682301, -0.4192710, 0.4230881
1: -11.1884527, -10.3611889, -11.1897335, -10.3585377, -0.3387635, 0.3377235
2: -11.1811485, -10.3391953, -11.1824350, -10.3379650, -0.4586363, 0.4583735
3: -10.6043768, -9.8126945, -10.6066322, -9.8093472, -0.3922725, 0.3910553
4: -2.8171291, -2.1812935, -2.8202233, -2.1794238, -0.2430170, 0.2442442
5: -9.9185095, -8.8984604, -9.9238539, -8.8957748, -0.3851674, 0.3876727
6: -12.9212761, -12.0871592, -12.9232407, -12.0861320, -0.3205285, 0.3214624
7: -6.0027871, -5.3160534, -6.0050840, -5.3113313, -0.2814341, 0.2791816
8: -0.7677855, -0.1047544, -0.7771072, -0.1003079, -0.3802176, 0.3846579
9: 2.6518168, 3.3024251, 2.6494017, 3.3066108, -0.3798709, 0.3778958

Time for backsubstitution: 20.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4640
type: A, layer: 1, pos: 5752

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4640

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727342, upper bound: 0.1730406
time: 3.64 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730397, upper bound: 0.1730406
time: 3.58 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -10.2548389, -9.1682301, -10.2468166, -9.1725855, -0.4230886, 0.4192712
1: -11.1897335, -10.3585377, -11.1884527, -10.3611889, -0.3377235, 0.3387635
2: -11.1824350, -10.3379650, -11.1811485, -10.3391953, -0.4583731, 0.4586363
3: -10.6066322, -9.8093472, -10.6043768, -9.8126945, -0.3910551, 0.3922727
4: -2.8202233, -2.1794238, -2.8171291, -2.1812935, -0.2442443, 0.2430171
5: -9.9238539, -8.8957748, -9.9185095, -8.8984604, -0.3876729, 0.3851674
6: -12.9232407, -12.0861320, -12.9212761, -12.0871592, -0.3214624, 0.3205285
7: -6.0050840, -5.3113313, -6.0027871, -5.3160534, -0.2791815, 0.2814341
8: -0.7771072, -0.1003079, -0.7677855, -0.1047544, -0.3846579, 0.3802176
9: 2.6494017, 3.3066108, 2.6518168, 3.3024251, -0.3778963, 0.3798714

Time for backsubstitution: 21.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4640
type: A, layer: 1, pos: 5752

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4640

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727340, upper bound: 0.1738853
time: 3.33 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730397, upper bound: 0.1738853
time: 3.44 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -10.2548389, -9.1682301, -10.2548389, -9.1682301, -0.4186273, 0.4186275
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3386002, 0.3386002
2: -11.1824350, -10.3379650, -11.1824350, -10.3379650, -0.4583659, 0.4583659
3: -10.6066322, -9.8093472, -10.6066322, -9.8093472, -0.3950124, 0.3950124
4: -2.8202233, -2.1794238, -2.8202233, -2.1794238, -0.2438341, 0.2438344
5: -9.9238539, -8.8957748, -9.9238539, -8.8957748, -0.3867798, 0.3867798
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3210564, 0.3210566
7: -6.0050840, -5.3113313, -6.0050840, -5.3113313, -0.2795017, 0.2795017
8: -0.7771072, -0.1003079, -0.7771072, -0.1003079, -0.3811159, 0.3811159
9: 2.6494017, 3.3066108, 2.6494017, 3.3066108, -0.3796787, 0.3796787

Time for backsubstitution: 21.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4640
type: A, layer: 1, pos: 5752

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4640

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727342, upper bound: 0.1738971
time: 5.01 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730397, upper bound: 0.1738971
time: 5.00 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.62 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.62
Output dim: 9, lower bound: -0.1727342, upper bound: 0.1730402
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.62
Output dim: 9, lower bound: -0.1730399, upper bound: 0.1730402
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.62
Output dim: 9, lower bound: -0.1727342, upper bound: 0.1730406
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.62
Output dim: 9, lower bound: -0.1730397, upper bound: 0.1730406
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.62
Output dim: 9, lower bound: -0.1727340, upper bound: 0.1738853
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.62
Output dim: 9, lower bound: -0.1730397, upper bound: 0.1738853
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.62
Output dim: 9, lower bound: -0.1727342, upper bound: 0.1738971
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.62
Output dim: 9, lower bound: -0.1730397, upper bound: 0.1738971

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -10.2464275, -9.1766014, -10.2466812, -9.1739902, -0.4134696, 0.4109027
1: -11.1882133, -10.3617258, -11.1883688, -10.3613968, -0.3356640, 0.2704221
2: -11.1804161, -10.3418016, -11.1808825, -10.3401070, -0.4458070, 0.4539914
3: -10.6036568, -9.8137445, -10.6041193, -9.8130665, -0.3324749, 0.3873973
4: -2.8155243, -2.1821926, -2.8165662, -2.1816237, -0.2390784, 0.2247275
5: -9.9179182, -8.8986006, -9.9182940, -8.8985119, -0.3231732, 0.3813813
6: -12.9208107, -12.0914869, -12.9211168, -12.0886774, -0.3122838, 0.3150859
7: -6.0011001, -5.3166857, -6.0021901, -5.3162766, -0.2750380, 0.2382250
8: -0.7665071, -0.1077681, -0.7673349, -0.1058111, -0.2560074, 0.3724067
9: 2.6528292, 3.3022037, 2.6521759, 3.3023460, -0.3746166, 0.3712556

Time for backsubstitution: 21.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4640
type: B, layer: 1, pos: 5752

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4640

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727345, upper bound: 0.1727345
time: 3.35 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727345, upper bound: 0.1730402
time: 3.19 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -10.2576799, -9.1724415, -10.2468138, -9.1725941, -0.4261408, 0.4150038
1: -11.1894722, -10.3609524, -11.1884489, -10.3611917, -0.3368495, 0.3366985
2: -11.1870193, -10.3390503, -11.1811466, -10.3391962, -0.4634929, 0.4572520
3: -10.6065340, -9.8122807, -10.6043739, -9.8126945, -0.3916841, 0.3894153
4: -2.8171892, -2.1781166, -2.8171258, -2.1812949, -0.2410243, 0.2454966
5: -9.9187126, -8.8975039, -9.9185085, -8.8984585, -0.3827538, 0.3827422
6: -12.9341965, -12.0871382, -12.9212780, -12.0871735, -0.3314620, 0.3191652
7: -6.0036077, -5.3122935, -6.0027809, -5.3160582, -0.2771711, 0.2809722
8: -0.7742023, -0.1047745, -0.7677822, -0.1047683, -0.3827987, 0.3753505
9: 2.6517177, 3.3045449, 2.6518207, 3.3024251, -0.3755841, 0.3778570

Time for backsubstitution: 21.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4640
type: B, layer: 1, pos: 5752

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4640

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730403, upper bound: 0.1727345
time: 3.26 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730403, upper bound: 0.1730402
time: 3.18 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -10.2464275, -9.1766014, -10.2547073, -9.1696339, -0.4176812, 0.4189317
1: -11.1882133, -10.3617258, -11.1896515, -10.3587408, -0.3380585, 0.2717491
2: -11.1804161, -10.3418016, -11.1821690, -10.3388834, -0.4470372, 0.4549556
3: -10.6036568, -9.8137445, -10.6063738, -9.8097200, -0.3356347, 0.3893852
4: -2.8155243, -2.1821926, -2.8196604, -2.1797562, -0.2409226, 0.2277886
5: -9.9179182, -8.8986006, -9.9236355, -8.8958302, -0.3257998, 0.3865476
6: -12.9208107, -12.0914869, -12.9230785, -12.0876484, -0.3132918, 0.3170300
7: -6.0011001, -5.3166857, -6.0044889, -5.3115497, -0.2795889, 0.2405238
8: -0.7665071, -0.1077681, -0.7766619, -0.1013660, -0.2604523, 0.3813090
9: 2.6528292, 3.3022037, 2.6497583, 3.3065324, -0.3788066, 0.3734586

Time for backsubstitution: 21.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4640
type: B, layer: 1, pos: 5752

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4640

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1735797, upper bound: 0.1727349
time: 3.91 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1735797, upper bound: 0.1730406
time: 3.89 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -10.2576799, -9.1724415, -10.2548389, -9.1682415, -0.4265716, 0.4230323
1: -11.1894722, -10.3609524, -11.1897335, -10.3585405, -0.3392441, 0.3380532
2: -11.1870193, -10.3390503, -11.1824322, -10.3379688, -0.4647217, 0.4582186
3: -10.6065340, -9.8122807, -10.6066284, -9.8093491, -0.3948932, 0.3914075
4: -2.8171892, -2.1781166, -2.8202217, -2.1794271, -0.2428707, 0.2472380
5: -9.9187126, -8.8975039, -9.9238491, -8.8957758, -0.3854134, 0.3879073
6: -12.9341965, -12.0871382, -12.9232416, -12.0861454, -0.3315055, 0.3211086
7: -6.0036077, -5.3122935, -6.0050778, -5.3113332, -0.2817221, 0.2829307
8: -0.7742023, -0.1047745, -0.7771006, -0.1003208, -0.3872595, 0.3842528
9: 2.6517177, 3.3045449, 2.6494040, 3.3066115, -0.3797727, 0.3800688

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4640
type: B, layer: 1, pos: 5752

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4640

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1738854, upper bound: 0.1727349
time: 4.00 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1738854, upper bound: 0.1730406
time: 3.86 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -10.2544527, -9.1722450, -10.2466812, -9.1739902, -0.4214578, 0.4151142
1: -11.1894951, -10.3590813, -11.1883688, -10.3613968, -0.3370187, 0.3381526
2: -11.1816788, -10.3405752, -11.1808825, -10.3401070, -0.4560575, 0.4552207
3: -10.6058855, -9.8103971, -10.6041193, -9.8130665, -0.3896332, 0.3906064
4: -2.8186185, -2.1803331, -2.8165662, -2.1816237, -0.2421521, 0.2416723
5: -9.9232521, -8.8959160, -9.9182940, -8.8985119, -0.3865321, 0.3840415
6: -12.9227753, -12.0904608, -12.9211168, -12.0886774, -0.3197207, 0.3160954
7: -6.0033975, -5.3119688, -6.0021901, -5.3162766, -0.2773364, 0.2805399
8: -0.7758126, -0.1033216, -0.7673349, -0.1058111, -0.3829799, 0.3768678
9: 2.6504140, 3.3063841, 2.6521759, 3.3023460, -0.3768287, 0.3793061

Time for backsubstitution: 21.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4640
type: B, layer: 1, pos: 5752

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4640

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727344, upper bound: 0.1735796
time: 3.45 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727344, upper bound: 0.1738853
time: 3.40 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10.2657013, -9.1680889, -10.2468138, -9.1725941, -0.4285717, 0.4192140
1: -11.1907539, -10.3582954, -11.1884489, -10.3611917, -0.3382041, 0.3390992
2: -11.1883011, -10.3378277, -11.1811466, -10.3391962, -0.4644585, 0.4584818
3: -10.6087799, -9.8089304, -10.6043739, -9.8126945, -0.3936749, 0.3926294
4: -2.8202839, -2.1762519, -2.8171258, -2.1812949, -0.2440982, 0.2466617
5: -9.9240608, -8.8948212, -9.9185085, -8.8984585, -0.3879192, 0.3854036
6: -12.9361610, -12.0861092, -12.9212780, -12.0871735, -0.3321463, 0.3201745
7: -6.0059042, -5.3075709, -6.0027809, -5.3160582, -0.2794695, 0.2842320
8: -0.7835188, -0.1003289, -0.7677822, -0.1047683, -0.3898895, 0.3798118
9: 2.6493034, 3.3087296, 2.6518207, 3.3024251, -0.3777957, 0.3820436

Time for backsubstitution: 21.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4640
type: B, layer: 1, pos: 5752

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4640

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730403, upper bound: 0.1735796
time: 3.41 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730403, upper bound: 0.1738853
time: 3.34 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -10.2544527, -9.1722450, -10.2547073, -9.1696339, -0.4169965, 0.4144711
1: -11.1894951, -10.3590813, -11.1896515, -10.3587408, -0.3378947, 0.3379893
2: -11.1816788, -10.3405752, -11.1821690, -10.3388834, -0.4560618, 0.4549518
3: -10.6058855, -9.8103971, -10.6063738, -9.8097200, -0.3936057, 0.3933468
4: -2.8186185, -2.1803331, -2.8196604, -2.1797562, -0.2417418, 0.2424952
5: -9.9232521, -8.8959160, -9.9236355, -8.8958302, -0.3856387, 0.3856547
6: -12.9227753, -12.0904608, -12.9230785, -12.0876484, -0.3193145, 0.3166234
7: -6.0033975, -5.3119688, -6.0044889, -5.3115497, -0.2776563, 0.2786072
8: -0.7758126, -0.1033216, -0.7766619, -0.1013660, -0.3794379, 0.3777678
9: 2.6504140, 3.3063841, 2.6497583, 3.3065324, -0.3786135, 0.3791132

Time for backsubstitution: 21.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4640
type: B, layer: 1, pos: 5752

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4640

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727513, upper bound: 0.1735912
time: 3.25 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727513, upper bound: 0.1738967
time: 3.29 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -10.2657013, -9.1680889, -10.2548389, -9.1682415, -0.4294548, 0.4185710
1: -11.1907539, -10.3582954, -11.1897335, -10.3585405, -0.3390803, 0.3389354
2: -11.1883011, -10.3378277, -11.1824322, -10.3379688, -0.4644532, 0.4582114
3: -10.6087799, -9.8089304, -10.6066284, -9.8093491, -0.3976312, 0.3953667
4: -2.8202839, -2.1762519, -2.8202217, -2.1794271, -0.2436879, 0.2481587
5: -9.9240608, -8.8948212, -9.9238491, -8.8957758, -0.3870258, 0.3870153
6: -12.9361610, -12.0861092, -12.9232416, -12.0861454, -0.3326063, 0.3207023
7: -6.0059042, -5.3075709, -6.0050778, -5.3113332, -0.2797890, 0.2835897
8: -0.7835188, -0.1003289, -0.7771006, -0.1003208, -0.3881578, 0.3807120
9: 2.6493034, 3.3087296, 2.6494040, 3.3066115, -0.3795791, 0.3818510

Time for backsubstitution: 21.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4640
type: B, layer: 1, pos: 5752

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4640

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730568, upper bound: 0.1735909
time: 3.44 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730568, upper bound: 0.1738964
time: 3.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.47 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1727345, upper bound: 0.1727345
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1727345, upper bound: 0.1730402
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1730403, upper bound: 0.1727345
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1730403, upper bound: 0.1730402
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1735797, upper bound: 0.1727349
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1735797, upper bound: 0.1730406
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1738854, upper bound: 0.1727349
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1738854, upper bound: 0.1730406
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1727344, upper bound: 0.1735796
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1727344, upper bound: 0.1738853
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1730403, upper bound: 0.1735796
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1730403, upper bound: 0.1738853
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1727513, upper bound: 0.1735912
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1727513, upper bound: 0.1738967
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1730568, upper bound: 0.1735909
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 9, lower bound: -0.1730568, upper bound: 0.1738964

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -10.2464275, -9.1766014, -10.2464275, -9.1766014, -0.4108114, 0.4108114
1: -11.1882133, -10.3617258, -11.1882133, -10.3617258, -0.2701657, 0.2701656
2: -11.1804161, -10.3418016, -11.1804161, -10.3418016, -0.4439077, 0.4439077
3: -10.6036568, -9.8137445, -10.6036568, -9.8137445, -0.3316469, 0.3316469
4: -2.8155243, -2.1821926, -2.8155243, -2.1821926, -0.2235334, 0.2235334
5: -9.9179182, -8.8986006, -9.9179182, -8.8986006, -0.3228539, 0.3228540
6: -12.9208107, -12.0914869, -12.9208107, -12.0914869, -0.3094506, 0.3094506
7: -6.0011001, -5.3166857, -6.0011001, -5.3166857, -0.2370971, 0.2370971
8: -0.7665071, -0.1077681, -0.7665071, -0.1077681, -0.2540426, 0.2540425
9: 2.6528292, 3.3022037, 2.6528292, 3.3022037, -0.3706145, 0.3706143

Time for backsubstitution: 21.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5752

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5752

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727268, upper bound: 0.1727347
time: 3.37 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727345, upper bound: 0.1727347
time: 3.22 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -10.2464275, -9.1766014, -10.2576799, -9.1724415, -0.4149697, 0.4220622
1: -11.1882133, -10.3617258, -11.1894722, -10.3609600, -0.3362019, 0.2712206
2: -11.1804161, -10.3418016, -11.1870108, -10.3390503, -0.4466629, 0.4594932
3: -10.6036568, -9.8137445, -10.6065331, -9.8122826, -0.3333116, 0.3895297
4: -2.8155243, -2.1821926, -2.8171895, -2.1781173, -0.2429665, 0.2252085
5: -9.9179182, -8.8986006, -9.9187117, -8.8975096, -0.3239233, 0.3820124
6: -12.9208107, -12.0914869, -12.9341974, -12.0871391, -0.3138204, 0.3270959
7: -6.0011001, -5.3166857, -6.0035992, -5.3122940, -0.2792270, 0.2393417
8: -0.7665071, -0.1077681, -0.7742019, -0.1047773, -0.2570796, 0.3796558
9: 2.6528292, 3.3022037, 2.6517239, 3.3045444, -0.3768682, 0.3716304

Time for backsubstitution: 20.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5752

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5752

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727268, upper bound: 0.1730402
time: 3.44 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727345, upper bound: 0.1730402
time: 3.22 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -10.2576799, -9.1724415, -10.2464275, -9.1766014, -0.4220631, 0.4149697
1: -11.1894722, -10.3609524, -11.1882133, -10.3617258, -0.2712204, 0.3362107
2: -11.1870193, -10.3390503, -11.1804161, -10.3418016, -0.4595051, 0.4466619
3: -10.6065340, -9.8122807, -10.6036568, -9.8137445, -0.3895292, 0.3333154
4: -2.8171892, -2.1781166, -2.8155243, -2.1821926, -0.2252100, 0.2429676
5: -9.9187126, -8.8975039, -9.9179182, -8.8986006, -0.3820128, 0.3239324
6: -12.9341965, -12.0871382, -12.9208107, -12.0914869, -0.3270956, 0.3138216
7: -6.0036077, -5.3122935, -6.0011001, -5.3166857, -0.2393482, 0.2792281
8: -0.7742023, -0.1047745, -0.7665071, -0.1077681, -0.3796573, 0.2570817
9: 2.6517177, 3.3045449, 2.6528292, 3.3022037, -0.3716345, 0.3768680

Time for backsubstitution: 21.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5752

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5752

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730324, upper bound: 0.1727344
time: 3.47 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730401, upper bound: 0.1727344
time: 3.34 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -10.2576799, -9.1724415, -10.2576799, -9.1724415, -0.4254062, 0.4254060
1: -11.1894722, -10.3609524, -11.1894722, -10.3609524, -0.3372493, 0.3372493
2: -11.1870193, -10.3390503, -11.1870193, -10.3390503, -0.4636955, 0.4636955
3: -10.6065340, -9.8122807, -10.6065340, -9.8122807, -0.3923163, 0.3923161
4: -2.8171892, -2.1781166, -2.8171892, -2.1781166, -0.2455870, 0.2455870
5: -9.9187126, -8.8975039, -9.9187126, -8.8975039, -0.3834238, 0.3834238
6: -12.9341965, -12.0871382, -12.9341965, -12.0871382, -0.3303394, 0.3303394
7: -6.0036077, -5.3122935, -6.0036077, -5.3122935, -0.2794497, 0.2794498
8: -0.7742023, -0.1047745, -0.7742023, -0.1047745, -0.3798890, 0.3798890
9: 2.6517177, 3.3045449, 2.6517177, 3.3045449, -0.3770308, 0.3770308

Time for backsubstitution: 21.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5752

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5752

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730326, upper bound: 0.1727344
time: 3.25 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730403, upper bound: 0.1727344
time: 3.38 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -10.2464275, -9.1766014, -10.2544527, -9.1722450, -0.4150226, 0.4188011
1: -11.1882133, -10.3617258, -11.1894951, -10.3590813, -0.3377972, 0.2714930
2: -11.1804161, -10.3418016, -11.1816788, -10.3405752, -0.4451385, 0.4541287
3: -10.6036568, -9.8137445, -10.6058855, -9.8103971, -0.3348069, 0.3887661
4: -2.8155243, -2.1821926, -2.8186185, -2.1803331, -0.2404706, 0.2265944
5: -9.9179182, -8.8986006, -9.9232521, -8.8959160, -0.3254805, 0.3859985
6: -12.9208107, -12.0914869, -12.9227753, -12.0904608, -0.3104589, 0.3168881
7: -6.0011001, -5.3166857, -6.0033975, -5.3119688, -0.2794075, 0.2393959
8: -0.7665071, -0.1077681, -0.7758126, -0.1033216, -0.2584877, 0.3809340
9: 2.6528292, 3.3022037, 2.6504140, 3.3063841, -0.3786654, 0.3728173

Time for backsubstitution: 21.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5752

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5752

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1735719, upper bound: 0.1727352
time: 3.47 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1735794, upper bound: 0.1727352
time: 3.82 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.88 + 550.12 = 605.99 seconds
