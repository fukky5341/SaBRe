## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.259609539


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5734241, 0.5734241)
1: (-4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5444160, 0.5444160)
2: (-5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4155605, 0.4155604)
3: (-10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4519312, 0.4519312)
4: (4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5382333, 0.5382330)
5: (-7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3759611, 0.3759613)
6: (-3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5690289, 0.5690289)
7: (-6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6293364, 0.6293364)
8: (-3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4772696, 0.4772696)
9: (-6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5227687, 0.5227687)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.66 + 34.37 = 56.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.2676387, upper bound: 0.2676391

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2676213, upper bound: 0.2665052
time: 3.70 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2676386, upper bound: 0.2676390
time: 3.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.69 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.69
Output dim: 4, lower bound: -0.2676213, upper bound: 0.2665052
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.69
Output dim: 4, lower bound: -0.2676386, upper bound: 0.2676390

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -7.1057281, -6.2920442, -7.1057281, -6.2920437, -0.5734220, 0.5734119
1: -4.9336247, -4.0876627, -4.9336262, -4.0876632, -0.5444155, 0.5444164
2: -5.6065383, -4.8446727, -5.6065388, -4.8446698, -0.4155605, 0.4155604
3: -10.7152195, -9.9662924, -10.7152185, -9.9662876, -0.4519305, 0.4519308
4: 4.3862052, 5.0951376, 4.3862052, 5.0951319, -0.5382323, 0.5382385
5: -7.9719658, -7.2899961, -7.9719667, -7.2899981, -0.3759617, 0.3759611
6: -3.2702127, -2.2950273, -3.2702138, -2.2950287, -0.5690284, 0.5690286
7: -6.3882127, -5.3162727, -6.3882108, -5.3162708, -0.6293359, 0.6293337
8: -3.3521700, -2.5929770, -3.3521700, -2.5929770, -0.4772713, 0.4772696
9: -6.7731881, -5.9340925, -6.7731886, -5.9340925, -0.5227678, 0.5227578

Time for backsubstitution: 20.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1489
type: A, layer: 3, pos: 1158
type: B, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1166
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1788
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 2397
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: A, layer: 3, pos: 1237

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 2146

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2631323
time: 4.00 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2642505, upper bound: 0.2631347
time: 3.80 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -7.1057281, -6.2920442, -7.1057286, -6.2920437, -0.5734239, 0.5734107
1: -4.9336257, -4.0876632, -4.9336252, -4.0876632, -0.5444171, 0.5444160
2: -5.6065378, -4.8446698, -5.6065383, -4.8446698, -0.4155605, 0.4155611
3: -10.7152195, -9.9662857, -10.7152185, -9.9662857, -0.4519312, 0.4519324
4: 4.3862057, 5.0951319, 4.3862057, 5.0951324, -0.5382419, 0.5382333
5: -7.9719663, -7.2899961, -7.9719667, -7.2899966, -0.3759611, 0.3759623
6: -3.2702122, -2.2950277, -3.2702131, -2.2950277, -0.5690279, 0.5690284
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6293364, 0.6293354
8: -3.3521690, -2.5929761, -3.3521690, -2.5929775, -0.4772704, 0.4772716
9: -6.7731891, -5.9340935, -6.7731895, -5.9340916, -0.5227687, 0.5227562

Time for backsubstitution: 21.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1998
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1158
type: A, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1166
type: B, layer: 3, pos: 1166
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: A, layer: 3, pos: 1237

Time for candidate selection: 0.44 seconds

### Candidate
type: B, layer: 3, pos: 2146

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2642654, upper bound: 0.2599351
time: 3.78 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2642679, upper bound: 0.2642679
time: 6.57 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.83 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 31.83
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2631323
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 31.83
Output dim: 4, lower bound: -0.2642505, upper bound: 0.2631347
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.83
Output dim: 4, lower bound: -0.2642654, upper bound: 0.2599351
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.83
Output dim: 4, lower bound: -0.2642679, upper bound: 0.2642679

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -7.1040239, -6.3016939, -7.1046515, -6.2947102, -0.5680261, 0.5614276
1: -4.9265709, -4.1055174, -4.9310088, -4.0930181, -0.5188859, 0.5150902
2: -5.6021481, -4.8480887, -5.6057863, -4.8455601, -0.4104278, 0.4087943
3: -10.7147493, -9.9734716, -10.7151003, -9.9678879, -0.4474320, 0.4417074
4: 4.4075670, 5.0967073, 4.3923445, 5.0951071, -0.5144391, 0.5169017
5: -7.9698524, -7.3177462, -7.9719191, -7.2975159, -0.3456051, 0.3417432
6: -3.2394617, -2.3095107, -3.2613039, -2.2957196, -0.5369077, 0.5446727
7: -6.3792620, -5.3224087, -6.3856320, -5.3179893, -0.6126313, 0.6126411
8: -3.3442302, -2.5945168, -3.3503389, -2.5937629, -0.4640884, 0.4687586
9: -6.7697401, -5.9505219, -6.7722459, -5.9387207, -0.5042591, 0.5020385

Time for backsubstitution: 21.69 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1158
type: B, layer: 3, pos: 1158
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 1166
type: A, layer: 3, pos: 1166
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 2397
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: B, layer: 3, pos: 1237

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 2146

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2588013
time: 4.19 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2631323
time: 3.98 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -7.1054392, -6.2968388, -7.1057110, -6.2923408, -0.5735533, 0.5703435
1: -4.9329510, -4.0926147, -4.9335814, -4.0882335, -0.5440252, 0.5258569
2: -5.6062040, -4.8455529, -5.6065173, -4.8447247, -0.4102716, 0.4122267
3: -10.7150192, -9.9674225, -10.7152071, -9.9663601, -0.4489396, 0.4588954
4: 4.3905897, 5.0951314, 4.3864765, 5.0951304, -0.5417383, 0.5251808
5: -7.9719534, -7.2944074, -7.9719648, -7.2903786, -0.3702502, 0.3471594
6: -3.2694063, -2.2952042, -3.2701635, -2.2950420, -0.5476971, 0.5688610
7: -6.3875494, -5.3214750, -6.3881683, -5.3166542, -0.6334963, 0.6264458
8: -3.3481998, -2.5931854, -3.3519216, -2.5929890, -0.4747853, 0.4797256
9: -6.7729297, -5.9358687, -6.7731724, -5.9342413, -0.5363555, 0.5208025

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1158
type: B, layer: 3, pos: 1158
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1166
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 2397
type: A, layer: 3, pos: 1166
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: B, layer: 3, pos: 1237

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 2146

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2642482, upper bound: 0.2588014
time: 4.04 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2642482, upper bound: 0.2631344
time: 4.90 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -7.1046500, -6.2947102, -7.1040258, -6.3016925, -0.5614400, 0.5680137
1: -4.9310064, -4.0930176, -4.9265690, -4.1055183, -0.5150917, 0.5188859
2: -5.6057878, -4.8455601, -5.6021481, -4.8480854, -0.4087944, 0.4104279
3: -10.7151003, -9.9678860, -10.7147493, -9.9734669, -0.4417076, 0.4474339
4: 4.3923445, 5.0951066, 4.4075680, 5.0967016, -0.5169055, 0.5144401
5: -7.9719195, -7.2975159, -7.9698501, -7.3177490, -0.3417434, 0.3456056
6: -3.2613039, -2.2957208, -3.2394617, -2.3095117, -0.5446725, 0.5369084
7: -6.3856320, -5.3179893, -6.3792620, -5.3224049, -0.6126442, 0.6126313
8: -3.3503389, -2.5937638, -3.3442302, -2.5945177, -0.4687591, 0.4640884
9: -6.7722454, -5.9387207, -6.7697420, -5.9505219, -0.5020497, 0.5042474

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1158
type: A, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1166
type: B, layer: 3, pos: 1166
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 2397
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: A, layer: 3, pos: 1237

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 2146

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599348, upper bound: 0.2599350
time: 3.87 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599348, upper bound: 0.2599351
time: 3.93 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -7.1057105, -6.2923403, -7.1054411, -6.2968388, -0.5703557, 0.5735409
1: -4.9335823, -4.0882335, -4.9329510, -4.0926142, -0.5258577, 0.5440257
2: -5.6065173, -4.8447247, -5.6062050, -4.8455491, -0.4122269, 0.4102720
3: -10.7152081, -9.9663591, -10.7150192, -9.9674168, -0.4588954, 0.4489415
4: 4.3864751, 5.0951314, 4.3905897, 5.0951262, -0.5251837, 0.5417390
5: -7.9719639, -7.2903781, -7.9719529, -7.2944078, -0.3471599, 0.3702502
6: -3.2701631, -2.2950418, -3.2694054, -2.2952044, -0.5688608, 0.5476966
7: -6.3881674, -5.3166542, -6.3875499, -5.3214707, -0.6264493, 0.6334958
8: -3.3519216, -2.5929880, -3.3481994, -2.5931859, -0.4797258, 0.4747853
9: -6.7731729, -5.9342413, -6.7729301, -5.9358683, -0.5208135, 0.5363443

Time for backsubstitution: 21.21 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1158
type: A, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1166
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 2397
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: A, layer: 3, pos: 1237

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 2146

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599348, upper bound: 0.2642657
time: 4.02 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599348, upper bound: 0.2642682
time: 3.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.26 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 29.26
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2588013
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 29.26
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2631323
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 29.26
Output dim: 4, lower bound: -0.2642482, upper bound: 0.2588014
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 29.26
Output dim: 4, lower bound: -0.2642482, upper bound: 0.2631344
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.26
Output dim: 4, lower bound: -0.2599348, upper bound: 0.2599350
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.26
Output dim: 4, lower bound: -0.2599348, upper bound: 0.2599351
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.26
Output dim: 4, lower bound: -0.2599348, upper bound: 0.2642657
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.26
Output dim: 4, lower bound: -0.2599348, upper bound: 0.2642682

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -7.1040239, -6.3016939, -7.1040249, -6.3016930, -0.5603795, 0.5603685
1: -4.9265709, -4.1055174, -4.9265695, -4.1055183, -0.4994965, 0.4994968
2: -5.6021481, -4.8480887, -5.6021481, -4.8480864, -0.4060905, 0.4060873
3: -10.7147493, -9.9734716, -10.7147493, -9.9734669, -0.4401257, 0.4401252
4: 4.4075670, 5.0967073, 4.4075670, 5.0967016, -0.5034258, 0.5034308
5: -7.9698524, -7.3177462, -7.9698501, -7.3177462, -0.3238716, 0.3238708
6: -3.2394617, -2.3095107, -3.2394612, -2.3095117, -0.5222764, 0.5222762
7: -6.3792620, -5.3224087, -6.3792629, -5.3224058, -0.6030509, 0.6030483
8: -3.3442302, -2.5945168, -3.3442297, -2.5945172, -0.4602213, 0.4602201
9: -6.7697401, -5.9505219, -6.7697411, -5.9505205, -0.4923480, 0.4923383

Time for backsubstitution: 21.73 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1158
type: B, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 1166
type: A, layer: 3, pos: 1166
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1788
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 2397
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: B, layer: 3, pos: 1237

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2146

## Relational analysis of NS_A1_A1_B1_A1

### Relational analysis result of NS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2589743
time: 4.07 seconds

## Relational analysis of NS_A1_A1_B1_A2

### Relational analysis result of NS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2588014
time: 3.99 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -7.1040239, -6.3016939, -7.1054411, -6.2968373, -0.5686569, 0.5614541
1: -4.9265709, -4.1055174, -4.9329515, -4.0926142, -0.5261960, 0.5159979
2: -5.6021481, -4.8480887, -5.6062036, -4.8455491, -0.4076247, 0.4106089
3: -10.7147493, -9.9734716, -10.7150192, -9.9674168, -0.4442849, 0.4407420
4: 4.4075670, 5.0967073, 4.3905902, 5.0951262, -0.5025969, 0.5245473
5: -7.9698524, -7.3177462, -7.9719529, -7.2944078, -0.3579168, 0.3360929
6: -3.2394617, -2.3095107, -3.2694061, -2.2952061, -0.5373335, 0.5536158
7: -6.3792620, -5.3224087, -6.3875504, -5.3214707, -0.6154928, 0.6111956
8: -3.3442302, -2.5945168, -3.3481989, -2.5931854, -0.4625294, 0.4704273
9: -6.7697401, -5.9505219, -6.7729311, -5.9358683, -0.5110326, 0.4965880

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1158
type: B, layer: 3, pos: 1158
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1166
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 2397
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: B, layer: 3, pos: 1237

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 2146

## Relational analysis of NS_A1_A1_B2_A1

### Relational analysis result of NS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2631323
time: 4.15 seconds

## Relational analysis of NS_A1_A1_B2_A2

### Relational analysis result of NS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599171, upper bound: 0.2631325
time: 4.07 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -7.1054392, -6.2968388, -7.1040249, -6.3016930, -0.5614645, 0.5686460
1: -4.9329510, -4.0926147, -4.9265695, -4.1055183, -0.5159969, 0.5261964
2: -5.6062040, -4.8455529, -5.6021481, -4.8480864, -0.4106090, 0.4076217
3: -10.7150192, -9.9674225, -10.7147493, -9.9734669, -0.4407423, 0.4442849
4: 4.3905897, 5.0951314, 4.4075670, 5.0967016, -0.5245416, 0.5026026
5: -7.9719534, -7.2944074, -7.9698501, -7.3177462, -0.3360940, 0.3579161
6: -3.2694063, -2.2952042, -3.2394612, -2.3095117, -0.5536165, 0.5373328
7: -6.3875494, -5.3214750, -6.3792629, -5.3224058, -0.6111984, 0.6154902
8: -3.3481998, -2.5931854, -3.3442297, -2.5945172, -0.4704285, 0.4625275
9: -6.7729297, -5.9358687, -6.7697411, -5.9505205, -0.4965973, 0.5110232

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1158
type: A, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1166
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 2397
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: A, layer: 3, pos: 1237

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 2146

## Relational analysis of NS_A1_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2588014
time: 3.98 seconds

## Relational analysis of NS_A1_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2642481, upper bound: 0.2588014
time: 3.99 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -7.1054392, -6.2968388, -7.1054411, -6.2968373, -0.5708504, 0.5708408
1: -4.9329510, -4.0926147, -4.9329515, -4.0926142, -0.5255733, 0.5255741
2: -5.6062040, -4.8455529, -5.6062036, -4.8455491, -0.4081165, 0.4081167
3: -10.7150192, -9.9674225, -10.7150192, -9.9674168, -0.4575069, 0.4575069
4: 4.3905897, 5.0951314, 4.3905902, 5.0951262, -0.5389104, 0.5389171
5: -7.9719534, -7.2944074, -7.9719529, -7.2944078, -0.3467164, 0.3467160
6: -3.2694063, -2.2952042, -3.2694061, -2.2952061, -0.5475559, 0.5475557
7: -6.3875494, -5.3214750, -6.3875504, -5.3214707, -0.6314356, 0.6314330
8: -3.3481998, -2.5931854, -3.3481989, -2.5931854, -0.4776144, 0.4776134
9: -6.7729297, -5.9358687, -6.7729311, -5.9358683, -0.5348663, 0.5348570

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1158
type: B, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1166
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: B, layer: 3, pos: 1237

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2146

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599171, upper bound: 0.2631325
time: 4.11 seconds

## Relational analysis of NS_A1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2642505, upper bound: 0.2631347
time: 4.10 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.1040249, -6.3016930, -7.1040258, -6.3016925, -0.5603809, 0.5603673
1: -4.9265699, -4.1055183, -4.9265690, -4.1055183, -0.4994981, 0.4994967
2: -5.6021481, -4.8480844, -5.6021481, -4.8480854, -0.4060876, 0.4060907
3: -10.7147503, -9.9734659, -10.7147493, -9.9734669, -0.4401255, 0.4401271
4: 4.4075680, 5.0967021, 4.4075680, 5.0967016, -0.5034349, 0.5034256
5: -7.9698496, -7.3177462, -7.9698501, -7.3177490, -0.3238714, 0.3238720
6: -3.2394598, -2.3095117, -3.2394617, -2.3095117, -0.5222769, 0.5222769
7: -6.3792624, -5.3224044, -6.3792620, -5.3224049, -0.6030519, 0.6030502
8: -3.3442292, -2.5945168, -3.3442302, -2.5945177, -0.4602203, 0.4602222
9: -6.7697420, -5.9505215, -6.7697420, -5.9505219, -0.4923494, 0.4923365

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1998
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1158
type: A, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1166
type: B, layer: 3, pos: 1166
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1788
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: A, layer: 3, pos: 1237

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 2146

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2601077, upper bound: 0.2599351
time: 4.25 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599348, upper bound: 0.2599351
time: 3.91 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.1054406, -6.2968397, -7.1040258, -6.3016925, -0.5614667, 0.5686438
1: -4.9329524, -4.0926142, -4.9265690, -4.1055183, -0.5159991, 0.5261962
2: -5.6062036, -4.8455501, -5.6021481, -4.8480854, -0.4106090, 0.4076250
3: -10.7150192, -9.9674158, -10.7147493, -9.9734669, -0.4407425, 0.4442866
4: 4.3905897, 5.0951262, 4.4075680, 5.0967016, -0.5245512, 0.5025971
5: -7.9719529, -7.2944078, -7.9698501, -7.3177490, -0.3360935, 0.3579168
6: -3.2694051, -2.2952037, -3.2394617, -2.3095117, -0.5536160, 0.5373337
7: -6.3875518, -5.3214707, -6.3792620, -5.3224049, -0.6111991, 0.6154921
8: -3.3481998, -2.5931864, -3.3442302, -2.5945177, -0.4704278, 0.4625297
9: -6.7729301, -5.9358683, -6.7697420, -5.9505219, -0.4965987, 0.5110211

Time for backsubstitution: 22.02 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1158
type: A, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1166
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 2397
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: A, layer: 3, pos: 1237

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 2146

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2642654, upper bound: 0.2599351
time: 3.83 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2642654, upper bound: 0.2599351
time: 3.90 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.1040249, -6.3016930, -7.1054411, -6.2968388, -0.5686581, 0.5614531
1: -4.9265699, -4.1055183, -4.9329510, -4.0926142, -0.5261974, 0.5159974
2: -5.6021481, -4.8480844, -5.6062050, -4.8455491, -0.4076223, 0.4106094
3: -10.7147503, -9.9734659, -10.7150192, -9.9674168, -0.4442849, 0.4407437
4: 4.4075680, 5.0967021, 4.3905897, 5.0951262, -0.5026059, 0.5245421
5: -7.9698496, -7.3177462, -7.9719529, -7.2944078, -0.3579164, 0.3360939
6: -3.2394598, -2.3095117, -3.2694054, -2.2952044, -0.5373328, 0.5536163
7: -6.3792624, -5.3224044, -6.3875499, -5.3214707, -0.6154940, 0.6111977
8: -3.3442292, -2.5945168, -3.3481994, -2.5931859, -0.4625282, 0.4704292
9: -6.7697420, -5.9505215, -6.7729301, -5.9358683, -0.5110340, 0.4965862

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1158
type: B, layer: 3, pos: 1158
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1166
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 2397
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: B, layer: 3, pos: 1237

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 2146

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599348, upper bound: 0.2599351
time: 3.88 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599348, upper bound: 0.2642658
time: 3.96 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.1054406, -6.2968397, -7.1054411, -6.2968388, -0.5708530, 0.5708392
1: -4.9329524, -4.0926142, -4.9329510, -4.0926142, -0.5255752, 0.5255737
2: -5.6062036, -4.8455501, -5.6062050, -4.8455491, -0.4081167, 0.4081172
3: -10.7150192, -9.9674158, -10.7150192, -9.9674168, -0.4575071, 0.4575090
4: 4.3905897, 5.0951262, 4.3905897, 5.0951262, -0.5389199, 0.5389109
5: -7.9719529, -7.2944078, -7.9719529, -7.2944078, -0.3467164, 0.3467169
6: -3.2694051, -2.2952037, -3.2694054, -2.2952044, -0.5475550, 0.5475557
7: -6.3875518, -5.3214707, -6.3875499, -5.3214707, -0.6314361, 0.6314349
8: -3.3481998, -2.5931864, -3.3481994, -2.5931859, -0.4776139, 0.4776149
9: -6.7729301, -5.9358683, -6.7729301, -5.9358683, -0.5348682, 0.5348556

Time for backsubstitution: 22.01 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1998
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1158
type: A, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1166
type: B, layer: 3, pos: 1166
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1788
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: A, layer: 3, pos: 1237

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 2146

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2642654, upper bound: 0.2599351
time: 3.88 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2642679, upper bound: 0.2642680
time: 5.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.17 seconds
NS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2589743
NS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2588014
NS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2631323
NS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2599171, upper bound: 0.2631325
NS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2588014
NS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2642481, upper bound: 0.2588014
NS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2599171, upper bound: 0.2631325
NS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2642505, upper bound: 0.2631347
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2601077, upper bound: 0.2599351
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2599348, upper bound: 0.2599351
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2642654, upper bound: 0.2599351
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2642654, upper bound: 0.2599351
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2599348, upper bound: 0.2599351
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2599348, upper bound: 0.2642658
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2642654, upper bound: 0.2599351
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.17
Output dim: 4, lower bound: -0.2642679, upper bound: 0.2642680

## BFS NS instance: NS_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.1040239, -6.3016939, -7.1040249, -6.3016930, -0.5603795, 0.5603685
1: -4.9265709, -4.1055174, -4.9265695, -4.1055183, -0.4994965, 0.4994968
2: -5.6021481, -4.8480887, -5.6021481, -4.8480864, -0.4060905, 0.4060873
3: -10.7147493, -9.9734716, -10.7147493, -9.9734669, -0.4401257, 0.4401252
4: 4.4075670, 5.0967073, 4.4075670, 5.0967016, -0.5034258, 0.5034308
5: -7.9698524, -7.3177462, -7.9698501, -7.3177462, -0.3238716, 0.3238708
6: -3.2394617, -2.3095107, -3.2394612, -2.3095117, -0.5222764, 0.5222762
7: -6.3792620, -5.3224087, -6.3792629, -5.3224058, -0.6030509, 0.6030483
8: -3.3442302, -2.5945168, -3.3442297, -2.5945172, -0.4602213, 0.4602201
9: -6.7697401, -5.9505219, -6.7697411, -5.9505205, -0.4923480, 0.4923383

Time for backsubstitution: 21.50 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1158
type: B, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 1166
type: A, layer: 3, pos: 1166
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1788
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 2397
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: B, layer: 3, pos: 1237

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 2146

## Relational analysis of NS_A1_A1_B1_A1_B1

### Relational analysis result of NS_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2600898, upper bound: 0.2588013
time: 4.30 seconds

## Relational analysis of NS_A1_A1_B1_A1_B2

### Relational analysis result of NS_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2588014
time: 4.36 seconds

## BFS NS instance: NS_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.1054392, -6.2968388, -7.1040249, -6.3016930, -0.5614645, 0.5686460
1: -4.9329510, -4.0926147, -4.9265695, -4.1055183, -0.5159969, 0.5261964
2: -5.6062040, -4.8455529, -5.6021481, -4.8480864, -0.4106090, 0.4076217
3: -10.7150192, -9.9674225, -10.7147493, -9.9734669, -0.4407423, 0.4442849
4: 4.3905897, 5.0951314, 4.4075670, 5.0967016, -0.5245416, 0.5026026
5: -7.9719534, -7.2944074, -7.9698501, -7.3177462, -0.3360940, 0.3579161
6: -3.2694063, -2.2952042, -3.2394612, -2.3095117, -0.5536165, 0.5373328
7: -6.3875494, -5.3214750, -6.3792629, -5.3224058, -0.6111984, 0.6154902
8: -3.3481998, -2.5931854, -3.3442297, -2.5945172, -0.4704285, 0.4625275
9: -6.7729297, -5.9358687, -6.7697411, -5.9505205, -0.4965973, 0.5110232

Time for backsubstitution: 21.20 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1158
type: A, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1166
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 2397
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: A, layer: 3, pos: 1237

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 2146

## Relational analysis of NS_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2600898, upper bound: 0.2588013
time: 4.39 seconds

## Relational analysis of NS_A1_A1_B1_A2_B2

### Relational analysis result of NS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2588014
time: 4.44 seconds

## BFS NS instance: NS_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.1040239, -6.3016939, -7.1054411, -6.2968373, -0.5686569, 0.5614541
1: -4.9265709, -4.1055174, -4.9329515, -4.0926142, -0.5261960, 0.5159979
2: -5.6021481, -4.8480887, -5.6062036, -4.8455491, -0.4076247, 0.4106089
3: -10.7147493, -9.9734716, -10.7150192, -9.9674168, -0.4442849, 0.4407420
4: 4.4075670, 5.0967073, 4.3905902, 5.0951262, -0.5025969, 0.5245473
5: -7.9698524, -7.3177462, -7.9719529, -7.2944078, -0.3579168, 0.3360929
6: -3.2394617, -2.3095107, -3.2694061, -2.2952061, -0.5373335, 0.5536158
7: -6.3792620, -5.3224087, -6.3875504, -5.3214707, -0.6154928, 0.6111956
8: -3.3442302, -2.5945168, -3.3481989, -2.5931854, -0.4625294, 0.4704273
9: -6.7697401, -5.9505219, -6.7729311, -5.9358683, -0.5110326, 0.4965880

Time for backsubstitution: 21.11 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1158
type: B, layer: 3, pos: 1158
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1166
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 2397
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: B, layer: 3, pos: 1237

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 2146

## Relational analysis of NS_A1_A1_B2_A1_B1

### Relational analysis result of NS_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2588014
time: 4.30 seconds

## Relational analysis of NS_A1_A1_B2_A1_B2

### Relational analysis result of NS_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2631323
time: 3.90 seconds

## BFS NS instance: NS_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.1054392, -6.2968388, -7.1054411, -6.2968373, -0.5708504, 0.5708408
1: -4.9329510, -4.0926147, -4.9329515, -4.0926142, -0.5255733, 0.5255741
2: -5.6062040, -4.8455529, -5.6062036, -4.8455491, -0.4081165, 0.4081167
3: -10.7150192, -9.9674225, -10.7150192, -9.9674168, -0.4575069, 0.4575069
4: 4.3905897, 5.0951314, 4.3905902, 5.0951262, -0.5389104, 0.5389171
5: -7.9719534, -7.2944074, -7.9719529, -7.2944078, -0.3467164, 0.3467160
6: -3.2694063, -2.2952042, -3.2694061, -2.2952061, -0.5475559, 0.5475557
7: -6.3875494, -5.3214750, -6.3875504, -5.3214707, -0.6314356, 0.6314330
8: -3.3481998, -2.5931854, -3.3481989, -2.5931854, -0.4776144, 0.4776134
9: -6.7729297, -5.9358687, -6.7729311, -5.9358683, -0.5348663, 0.5348570

Time for backsubstitution: 21.22 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 1489
type: B, layer: 3, pos: 1489
type: A, layer: 3, pos: 1158
type: B, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1166
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: B, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: B, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: B, layer: 3, pos: 1237

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 2146

## Relational analysis of NS_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2588014
time: 4.28 seconds

## Relational analysis of NS_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599169, upper bound: 0.2631323
time: 3.92 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.02 + 551.22 = 607.24 seconds
