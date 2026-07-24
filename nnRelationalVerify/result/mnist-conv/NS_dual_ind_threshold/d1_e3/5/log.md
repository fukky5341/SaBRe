## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.16436890799999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4758203, 0.4758205)
1: (-10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4666057, 0.4666057)
2: (-8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4128773, 0.4128771)
3: (-10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3284395, 0.3284395)
4: (-9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3763745, 0.3763745)
5: (7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3506265, 0.3506265)
6: (-4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3361690, 0.3361690)
7: (-13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4507301, 0.4507303)
8: (0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2773076, 0.2773077)
9: (-6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4804311, 0.4804308)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.60 + 33.56 = 56.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.1660289, upper bound: 0.1660293

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651902, upper bound: 0.1660287
time: 3.00 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660282, upper bound: 0.1660287
time: 3.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.29 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.29
Output dim: 5, lower bound: -0.1651902, upper bound: 0.1660287
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.29
Output dim: 5, lower bound: -0.1660282, upper bound: 0.1660287

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -9.1664829, -8.3243504, -9.1675816, -8.3232307, -0.4735425, 0.4733636
1: -10.3149166, -9.4458456, -10.3158188, -9.4444799, -0.4641585, 0.4639802
2: -8.5248184, -7.8177791, -8.5259857, -7.8151259, -0.4095929, 0.4081028
3: -10.0622196, -9.3906841, -10.0635166, -9.3887920, -0.3252097, 0.3246911
4: -9.9653015, -9.3388252, -9.9671946, -9.3378487, -0.3729515, 0.3739338
5: 7.7569065, 8.3272171, 7.7539816, 8.3283243, -0.3452764, 0.3471427
6: -4.2340689, -3.5515242, -4.2371969, -3.5506365, -0.3309658, 0.3325107
7: -13.7521219, -12.8542557, -13.7577066, -12.8517342, -0.4404287, 0.4434590
8: 0.9244347, 1.3759155, 0.9234161, 1.3785205, -0.2735031, 0.2725682
9: -6.6224918, -6.0463724, -6.6238995, -6.0457287, -0.4780879, 0.4779959

Time for backsubstitution: 20.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 836

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630095, upper bound: 0.1660238
time: 3.54 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651853, upper bound: 0.1660238
time: 3.49 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -9.1680317, -8.3224335, -9.1680298, -8.3224335, -0.4758115, 0.4749398
1: -10.3162136, -9.4436121, -10.3162117, -9.4436102, -0.4670138, 0.4665947
2: -8.5261040, -7.8132086, -8.5261040, -7.8132086, -0.4128771, 0.4100657
3: -10.0638485, -9.3873940, -10.0638485, -9.3873920, -0.3283318, 0.3259737
4: -9.9685497, -9.3377151, -9.9685497, -9.3377151, -0.3738165, 0.3763752
5: 7.7518911, 8.3284492, 7.7518911, 8.3284502, -0.3472960, 0.3506258
6: -4.2394090, -3.5505166, -4.2394104, -3.5505152, -0.3340129, 0.3357153
7: -13.7619133, -12.8516092, -13.7619114, -12.8516083, -0.4416375, 0.4505296
8: 0.9233375, 1.3804221, 0.9233375, 1.3804226, -0.2769003, 0.2750397
9: -6.6247835, -6.0456257, -6.6247845, -6.0456238, -0.4803705, 0.4825499

Time for backsubstitution: 20.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660284, upper bound: 0.1651903
time: 3.21 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660283, upper bound: 0.1660287
time: 3.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.32 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.32
Output dim: 5, lower bound: -0.1630095, upper bound: 0.1660238
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.32
Output dim: 5, lower bound: -0.1651853, upper bound: 0.1660238
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.32
Output dim: 5, lower bound: -0.1660284, upper bound: 0.1651903
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.32
Output dim: 5, lower bound: -0.1660283, upper bound: 0.1660287

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -9.1654873, -8.3243923, -9.1654978, -8.3245544, -0.4712207, 0.4712667
1: -10.3146038, -9.4458590, -10.3150797, -9.4449110, -0.4633474, 0.4632368
2: -8.5240669, -7.8178115, -8.5244274, -7.8161378, -0.4078362, 0.4065232
3: -10.0620975, -9.3907070, -10.0631294, -9.3888521, -0.3249772, 0.3242522
4: -9.9650755, -9.3388948, -9.9667158, -9.3382130, -0.3730426, 0.3736050
5: 7.7569752, 8.3264427, 7.7551646, 8.3267632, -0.3446846, 0.3461404
6: -4.2339802, -3.5524323, -4.2358131, -3.5525026, -0.3290100, 0.3301857
7: -13.7520962, -12.8543072, -13.7577152, -12.8518858, -0.4399247, 0.4428220
8: 0.9244499, 1.3758059, 0.9236422, 1.3782363, -0.2731966, 0.2721932
9: -6.6211495, -6.0463853, -6.6211491, -6.0472841, -0.4752092, 0.4752722

Time for backsubstitution: 20.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 836

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1630096, upper bound: 0.1638481
time: 3.24 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630096, upper bound: 0.1660240
time: 3.55 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -9.1664810, -8.3243504, -9.1675787, -8.3232307, -0.4735413, 0.4709957
1: -10.3149157, -9.4458456, -10.3158169, -9.4444799, -0.4641590, 0.4633839
2: -8.5248165, -7.8177814, -8.5259848, -7.8151278, -0.4095917, 0.4066746
3: -10.0622196, -9.3906851, -10.0635176, -9.3887911, -0.3252608, 0.3246796
4: -9.9653006, -9.3388262, -9.9671926, -9.3378496, -0.3728127, 0.3745699
5: 7.7569065, 8.3272161, 7.7539811, 8.3283243, -0.3450720, 0.3470953
6: -4.2340698, -3.5515258, -4.2371969, -3.5506372, -0.3288765, 0.3325095
7: -13.7521229, -12.8543558, -13.7577066, -12.8519506, -0.4402568, 0.4434586
8: 0.9244351, 1.3759151, 0.9234161, 1.3785195, -0.2733232, 0.2725687
9: -6.6224918, -6.0463734, -6.6238966, -6.0457287, -0.4780865, 0.4749093

Time for backsubstitution: 21.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 836

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651855, upper bound: 0.1638481
time: 5.35 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651854, upper bound: 0.1660238
time: 3.92 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -9.1680317, -8.3224335, -9.1664829, -8.3243504, -0.4737639, 0.4743896
1: -10.3162136, -9.4436121, -10.3149166, -9.4458456, -0.4643087, 0.4651928
2: -8.5261040, -7.8132086, -8.5248184, -7.8177791, -0.4082568, 0.4115522
3: -10.0638485, -9.3873940, -10.0622196, -9.3906841, -0.3249265, 0.3266633
4: -9.9685497, -9.3377151, -9.9653015, -9.3388252, -0.3753378, 0.3730559
5: 7.7518911, 8.3284492, 7.7569065, 8.3272171, -0.3493376, 0.3454280
6: -4.2394090, -3.5505166, -4.2340689, -3.5515242, -0.3348105, 0.3306022
7: -13.7619133, -12.8516092, -13.7521219, -12.8542557, -0.4471509, 0.4403389
8: 0.9233375, 1.3804221, 0.9244347, 1.3759155, -0.2722744, 0.2757823
9: -6.6247835, -6.0456257, -6.6224918, -6.0463724, -0.4793553, 0.4776967

Time for backsubstitution: 21.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 836

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660230, upper bound: 0.1630094
time: 3.85 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660230, upper bound: 0.1651852
time: 3.80 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4749389, 0.4749389
1: -10.3162136, -9.4436121, -10.3162136, -9.4436121, -0.4670134, 0.4670134
2: -8.5261040, -7.8132086, -8.5261040, -7.8132086, -0.4100652, 0.4100652
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3259737, 0.3259737
4: -9.9685497, -9.3377151, -9.9685497, -9.3377151, -0.3738160, 0.3738160
5: 7.7518911, 8.3284492, 7.7518911, 8.3284492, -0.3472953, 0.3472954
6: -4.2394090, -3.5505166, -4.2394090, -3.5505166, -0.3340129, 0.3340127
7: -13.7619133, -12.8516092, -13.7619133, -12.8516092, -0.4416380, 0.4416382
8: 0.9233375, 1.3804221, 0.9233375, 1.3804221, -0.2750393, 0.2750392
9: -6.6247835, -6.0456257, -6.6247835, -6.0456257, -0.4825492, 0.4825492

Time for backsubstitution: 21.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 836

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660234, upper bound: 0.1630094
time: 3.41 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660234, upper bound: 0.1651852
time: 3.23 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.62 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 28.62
Output dim: 5, lower bound: -0.1630096, upper bound: 0.1638481
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 5, lower bound: -0.1630096, upper bound: 0.1660240
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 5, lower bound: -0.1651855, upper bound: 0.1638481
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 5, lower bound: -0.1651854, upper bound: 0.1660238
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 5, lower bound: -0.1660230, upper bound: 0.1630094
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 5, lower bound: -0.1660230, upper bound: 0.1651852
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 5, lower bound: -0.1660234, upper bound: 0.1630094
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 5, lower bound: -0.1660234, upper bound: 0.1651852

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -9.1664801, -8.3243513, -9.1654978, -8.3245544, -0.4722123, 0.4713044
1: -10.3149176, -9.4458456, -10.3150797, -9.4449110, -0.4637070, 0.4632523
2: -8.5248175, -7.8177814, -8.5244274, -7.8161378, -0.4085886, 0.4065490
3: -10.0622196, -9.3906851, -10.0631294, -9.3888521, -0.3251585, 0.3242395
4: -9.9652996, -9.3388252, -9.9667158, -9.3382130, -0.3727531, 0.3737714
5: 7.7569065, 8.3272171, 7.7551646, 8.3267632, -0.3446987, 0.3464875
6: -4.2340698, -3.5515270, -4.2358131, -3.5525026, -0.3291047, 0.3311172
7: -13.7521191, -12.8544769, -13.7577152, -12.8518858, -0.4399321, 0.4430499
8: 0.9244351, 1.3759155, 0.9236422, 1.3782363, -0.2732202, 0.2723308
9: -6.6224899, -6.0463729, -6.6211491, -6.0472841, -0.4765320, 0.4752846

Time for backsubstitution: 20.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630096, upper bound: 0.1651854
time: 5.11 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630096, upper bound: 0.1660238
time: 4.32 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.1643972, -8.3256760, -9.1675787, -8.3232307, -0.4714816, 0.4720337
1: -10.3141785, -9.4462776, -10.3158169, -9.4444799, -0.4634304, 0.4635282
2: -8.5232620, -7.8187904, -8.5259848, -7.8151278, -0.4080408, 0.4070978
3: -10.0618305, -9.3907433, -10.0635176, -9.3887911, -0.3247573, 0.3246398
4: -9.9648209, -9.3391895, -9.9671926, -9.3378496, -0.3727880, 0.3737359
5: 7.7580891, 8.3256569, 7.7539811, 8.3283243, -0.3446240, 0.3465011
6: -4.2326860, -3.5533931, -4.2371969, -3.5506372, -0.3295722, 0.3306496
7: -13.7521305, -12.8544130, -13.7577066, -12.8519506, -0.4400206, 0.4429626
8: 0.9246607, 1.3756309, 0.9234161, 1.3785195, -0.2732654, 0.2722852
9: -6.6197419, -6.0479279, -6.6238966, -6.0457287, -0.4753761, 0.4764409

Time for backsubstitution: 20.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1630096, upper bound: 0.1630091
time: 4.60 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1630096, upper bound: 0.1638477
time: 4.41 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -9.1664801, -8.3243513, -9.1675787, -8.3232307, -0.4711750, 0.4709959
1: -10.3149176, -9.4458456, -10.3158169, -9.4444799, -0.4635568, 0.4633832
2: -8.5248175, -7.8177814, -8.5259848, -7.8151278, -0.4081647, 0.4066744
3: -10.0622196, -9.3906851, -10.0635176, -9.3887911, -0.3252606, 0.3247418
4: -9.9652996, -9.3388252, -9.9671926, -9.3378496, -0.3735867, 0.3745699
5: 7.7569065, 8.3272171, 7.7539811, 8.3283243, -0.3450720, 0.3469341
6: -4.2340698, -3.5515270, -4.2371969, -3.5506372, -0.3288765, 0.3304212
7: -13.7521191, -12.8544769, -13.7577066, -12.8519506, -0.4402568, 0.4432864
8: 0.9244351, 1.3759155, 0.9234161, 1.3785195, -0.2733232, 0.2723856
9: -6.6224899, -6.0463729, -6.6238966, -6.0457287, -0.4750009, 0.4749084

Time for backsubstitution: 20.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630096, upper bound: 0.1651848
time: 4.53 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630095, upper bound: 0.1660238
time: 3.54 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -9.1659460, -8.3237572, -9.1654873, -8.3243923, -0.4716663, 0.4720683
1: -10.3154764, -9.4440441, -10.3146038, -9.4458590, -0.4635663, 0.4643812
2: -8.5245476, -7.8142195, -8.5240669, -7.8178115, -0.4066763, 0.4097962
3: -10.0634613, -9.3874531, -10.0620975, -9.3907070, -0.3244882, 0.3264308
4: -9.9680710, -9.3380785, -9.9650755, -9.3388948, -0.3750091, 0.3731465
5: 7.7530737, 8.3268890, 7.7569752, 8.3264427, -0.3471055, 0.3448355
6: -4.2380238, -3.5523829, -4.2339802, -3.5524323, -0.3324852, 0.3286462
7: -13.7619171, -12.8517656, -13.7520962, -12.8543072, -0.4464307, 0.4398353
8: 0.9235644, 1.3801394, 0.9244499, 1.3758059, -0.2718992, 0.2754762
9: -6.6220341, -6.0471787, -6.6211495, -6.0463853, -0.4766321, 0.4748173

Time for backsubstitution: 20.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 836

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1638473, upper bound: 0.1630101
time: 3.49 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1638473, upper bound: 0.1630100
time: 3.75 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -9.1680288, -8.3224344, -9.1664810, -8.3243504, -0.4713957, 0.4743881
1: -10.3162136, -9.4436131, -10.3149157, -9.4458456, -0.4637184, 0.4651928
2: -8.5261040, -7.8132076, -8.5248165, -7.8177814, -0.4068291, 0.4115515
3: -10.0638494, -9.3873940, -10.0622196, -9.3906851, -0.3249151, 0.3267143
4: -9.9685488, -9.3377151, -9.9653006, -9.3388262, -0.3759739, 0.3729179
5: 7.7518921, 8.3284492, 7.7569065, 8.3272161, -0.3484461, 0.3452237
6: -4.2394104, -3.5505173, -4.2340698, -3.5515258, -0.3348093, 0.3285131
7: -13.7619133, -12.8518276, -13.7521229, -12.8543558, -0.4468358, 0.4401672
8: 0.9233379, 1.3804221, 0.9244351, 1.3759151, -0.2722740, 0.2756054
9: -6.6247816, -6.0456252, -6.6224918, -6.0463734, -0.4762692, 0.4776943

Time for backsubstitution: 20.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 836

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638473, upper bound: 0.1651860
time: 3.27 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638472, upper bound: 0.1651858
time: 3.58 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -9.1659460, -8.3237572, -9.1670351, -8.3224735, -0.4728429, 0.4726191
1: -10.3154764, -9.4440441, -10.3158998, -9.4436264, -0.4662709, 0.4662006
2: -8.5245476, -7.8142195, -8.5253525, -7.8132405, -0.4084880, 0.4083066
3: -10.0634613, -9.3874531, -10.0637264, -9.3874168, -0.3255359, 0.3257412
4: -9.9680710, -9.3380785, -9.9683247, -9.3377848, -0.3734868, 0.3739066
5: 7.7530737, 8.3268890, 7.7519608, 8.3276768, -0.3462899, 0.3467028
6: -4.2380238, -3.5523829, -4.2393212, -3.5514228, -0.3316875, 0.3320563
7: -13.7619171, -12.8517656, -13.7618866, -12.8516560, -0.4410005, 0.4411340
8: 0.9235644, 1.3801394, 0.9233546, 1.3803144, -0.2746637, 0.2747331
9: -6.6220341, -6.0471787, -6.6234422, -6.0456371, -0.4798260, 0.4796698

Time for backsubstitution: 20.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 836

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1638475, upper bound: 0.1630094
time: 3.36 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1638475, upper bound: 0.1630089
time: 4.18 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -9.1680288, -8.3224344, -9.1680307, -8.3224344, -0.4725718, 0.4749382
1: -10.3162136, -9.4436131, -10.3162136, -9.4436131, -0.4664221, 0.4670126
2: -8.5261040, -7.8132076, -8.5261040, -7.8132095, -0.4086375, 0.4100649
3: -10.0638494, -9.3873940, -10.0638494, -9.3873940, -0.3259621, 0.3260244
4: -9.9685488, -9.3377151, -9.9685497, -9.3377161, -0.3744521, 0.3736780
5: 7.7518921, 8.3284492, 7.7518911, 8.3284502, -0.3472486, 0.3470829
6: -4.2394104, -3.5505173, -4.2394090, -3.5505173, -0.3340118, 0.3319235
7: -13.7619133, -12.8518276, -13.7619133, -12.8517065, -0.4416380, 0.4414656
8: 0.9233379, 1.3804221, 0.9233384, 1.3804221, -0.2750394, 0.2748628
9: -6.6247816, -6.0456252, -6.6247826, -6.0456247, -0.4794617, 0.4825478

Time for backsubstitution: 20.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 836

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638474, upper bound: 0.1651853
time: 3.08 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638474, upper bound: 0.1651853
time: 3.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 27.26 seconds
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.26
Output dim: 5, lower bound: -0.1630096, upper bound: 0.1651854
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.26
Output dim: 5, lower bound: -0.1630096, upper bound: 0.1660238
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 27.26
Output dim: 5, lower bound: -0.1630096, upper bound: 0.1630091
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 27.26
Output dim: 5, lower bound: -0.1630096, upper bound: 0.1638477
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.26
Output dim: 5, lower bound: -0.1630096, upper bound: 0.1651848
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.26
Output dim: 5, lower bound: -0.1630095, upper bound: 0.1660238
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 27.26
Output dim: 5, lower bound: -0.1638473, upper bound: 0.1630101
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 27.26
Output dim: 5, lower bound: -0.1638473, upper bound: 0.1630100
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.26
Output dim: 5, lower bound: -0.1638473, upper bound: 0.1651860
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.26
Output dim: 5, lower bound: -0.1638472, upper bound: 0.1651858
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 27.26
Output dim: 5, lower bound: -0.1638475, upper bound: 0.1630094
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 27.26
Output dim: 5, lower bound: -0.1638475, upper bound: 0.1630089
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.26
Output dim: 5, lower bound: -0.1638474, upper bound: 0.1651853
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.26
Output dim: 5, lower bound: -0.1638474, upper bound: 0.1651853

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -9.1664801, -8.3243513, -9.1643972, -8.3256760, -0.4710121, 0.4702811
1: -10.3149176, -9.4458456, -10.3141785, -9.4462776, -0.4625287, 0.4622512
2: -8.5248175, -7.8177814, -8.5232620, -7.8187904, -0.4059279, 0.4053810
3: -10.0622196, -9.3906851, -10.0618305, -9.3907433, -0.3232074, 0.3228062
4: -9.9652996, -9.3388252, -9.9648209, -9.3391895, -0.3718214, 0.3718574
5: 7.7569065, 8.3272171, 7.7580891, 8.3256569, -0.3435614, 0.3434873
6: -4.2340698, -3.5515270, -4.2326860, -3.5533931, -0.3278369, 0.3283048
7: -13.7521191, -12.8544769, -13.7521305, -12.8544130, -0.4372180, 0.4373050
8: 0.9244351, 1.3759155, 0.9246607, 1.3756309, -0.2708731, 0.2709193
9: -6.6224899, -6.0463729, -6.6197419, -6.0479279, -0.4755049, 0.4743481

Time for backsubstitution: 20.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630042, upper bound: 0.1651727
time: 3.18 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630042, upper bound: 0.1651797
time: 3.62 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9.1664801, -8.3243513, -9.1659460, -8.3237572, -0.4730597, 0.4717038
1: -10.3149176, -9.4458456, -10.3154764, -9.4440441, -0.4647408, 0.4635813
2: -8.5248175, -7.8177814, -8.5245476, -7.8142195, -0.4105487, 0.4067020
3: -10.0622196, -9.3906851, -10.0634613, -9.3874531, -0.3266121, 0.3244755
4: -9.9652996, -9.3388252, -9.9680710, -9.3380785, -0.3728573, 0.3751752
5: 7.7569065, 8.3272171, 7.7530737, 8.3268890, -0.3448498, 0.3471680
6: -4.2340698, -3.5515270, -4.2380238, -3.5523829, -0.3287408, 0.3334165
7: -13.7521191, -12.8544769, -13.7619171, -12.8517656, -0.4398427, 0.4464810
8: 0.9244351, 1.3759155, 0.9235644, 1.3801394, -0.2754996, 0.2720366
9: -6.6224899, -6.0463729, -6.6220341, -6.0471787, -0.4761400, 0.4766440

Time for backsubstitution: 20.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630042, upper bound: 0.1660116
time: 3.58 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630042, upper bound: 0.1660179
time: 3.65 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -9.1664801, -8.3243513, -9.1664801, -8.3243513, -0.4699743, 0.4699740
1: -10.3149176, -9.4458456, -10.3149176, -9.4458456, -0.4623775, 0.4623780
2: -8.5248175, -7.8177814, -8.5248175, -7.8177814, -0.4055052, 0.4055052
3: -10.0622196, -9.3906851, -10.0622196, -9.3906851, -0.3233093, 0.3233093
4: -9.9652996, -9.3388252, -9.9652996, -9.3388252, -0.3726559, 0.3726559
5: 7.7569065, 8.3272171, 7.7569065, 8.3272171, -0.3439353, 0.3439351
6: -4.2340698, -3.5515270, -4.2340698, -3.5515270, -0.3276086, 0.3276086
7: -13.7521191, -12.8544769, -13.7521191, -12.8544769, -0.4375415, 0.4375415
8: 0.9244351, 1.3759155, 0.9244351, 1.3759155, -0.2709742, 0.2709739
9: -6.6224899, -6.0463729, -6.6224899, -6.0463729, -0.4739723, 0.4739723

Time for backsubstitution: 20.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630042, upper bound: 0.1651731
time: 3.47 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630042, upper bound: 0.1651796
time: 3.49 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.1664801, -8.3243513, -9.1680288, -8.3224344, -0.4720216, 0.4713957
1: -10.3149176, -9.4458456, -10.3162136, -9.4436131, -0.4645905, 0.4637179
2: -8.5248175, -7.8177814, -8.5261040, -7.8132076, -0.4101248, 0.4068289
3: -10.0622196, -9.3906851, -10.0638494, -9.3873940, -0.3267140, 0.3249772
4: -9.9652996, -9.3388252, -9.9685488, -9.3377151, -0.3736913, 0.3759737
5: 7.7569065, 8.3272171, 7.7518921, 8.3284492, -0.3452234, 0.3475139
6: -4.2340698, -3.5515270, -4.2394104, -3.5505173, -0.3285131, 0.3327210
7: -13.7521191, -12.8544769, -13.7619133, -12.8518276, -0.4401674, 0.4467688
8: 0.9244351, 1.3759155, 0.9233379, 1.3804221, -0.2756054, 0.2720914
9: -6.6224899, -6.0463729, -6.6247816, -6.0456252, -0.4746079, 0.4762681

Time for backsubstitution: 20.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630040, upper bound: 0.1660115
time: 4.05 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630040, upper bound: 0.1660178
time: 4.18 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -9.1680288, -8.3224344, -9.1643972, -8.3256760, -0.4724333, 0.4723282
1: -10.3162136, -9.4436131, -10.3141785, -9.4462776, -0.4638572, 0.4644642
2: -8.5261040, -7.8132076, -8.5232620, -7.8187904, -0.4072518, 0.4100008
3: -10.0638494, -9.3873940, -10.0618305, -9.3907433, -0.3248752, 0.3262111
4: -9.9685488, -9.3377151, -9.9648209, -9.3391895, -0.3751395, 0.3728931
5: 7.7518921, 8.3284492, 7.7580891, 8.3256569, -0.3468652, 0.3447757
6: -4.2394104, -3.5505173, -4.2326860, -3.5533931, -0.3329493, 0.3292089
7: -13.7619133, -12.8518276, -13.7521305, -12.8544130, -0.4463375, 0.4399309
8: 0.9233379, 1.3804221, 0.9246607, 1.3756309, -0.2719905, 0.2755446
9: -6.6247816, -6.0456252, -6.6197419, -6.0479279, -0.4776678, 0.4749842

Time for backsubstitution: 20.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638415, upper bound: 0.1651728
time: 4.81 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638414, upper bound: 0.1651799
time: 4.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9.1680288, -8.3224344, -9.1664801, -8.3243513, -0.4713957, 0.4720218
1: -10.3162136, -9.4436131, -10.3149176, -9.4458456, -0.4637184, 0.4645908
2: -8.5261040, -7.8132076, -8.5248175, -7.8177814, -0.4068289, 0.4101248
3: -10.0638494, -9.3873940, -10.0622196, -9.3906851, -0.3249772, 0.3267140
4: -9.9685488, -9.3377151, -9.9652996, -9.3388252, -0.3759737, 0.3736911
5: 7.7518921, 8.3284492, 7.7569065, 8.3272171, -0.3476183, 0.3452233
6: -4.2394104, -3.5505173, -4.2340698, -3.5515270, -0.3327210, 0.3285131
7: -13.7619133, -12.8518276, -13.7521191, -12.8544769, -0.4468355, 0.4401674
8: 0.9233379, 1.3804221, 0.9244351, 1.3759155, -0.2720914, 0.2756054
9: -6.6247816, -6.0456252, -6.6224899, -6.0463729, -0.4762678, 0.4746075

Time for backsubstitution: 20.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638413, upper bound: 0.1651734
time: 4.99 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638413, upper bound: 0.1651802
time: 3.10 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.16 + 551.54 = 607.70 seconds
