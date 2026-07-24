## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 547.332881116455


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-131.2357788, 514.9561768, -131.2357788, 514.9561768, -646.1919556, 646.1919556)
1: (-85.4777908, 295.0576172, -85.4777908, 295.0576172, -380.5354004, 380.5354004)
2: (-46.5778465, 267.0964355, -46.5778465, 267.0964355, -313.6742859, 313.6742859)
3: (-62.6862679, 401.5845032, -62.6862679, 401.5845032, -464.2707825, 464.2707825)
4: (-84.1672974, 324.8345337, -84.1672974, 324.8345337, -409.0018311, 409.0018311)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.98 + 1.96 = 3.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -547.3383545, upper bound: 547.3383545

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3380039, upper bound: 547.3379198
time: 0.73 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3378287, upper bound: 547.3378287
time: 0.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.56 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.56
Output dim: 0, lower bound: -547.3380039, upper bound: 547.3379198
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.56
Output dim: 0, lower bound: -547.3378287, upper bound: 547.3378287

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -120.2437515, 472.3688660, -128.2485199, 503.3413391, -623.5850830, 600.6173706
1: -78.1661453, 270.1556396, -83.4768143, 288.2567444, -366.4228821, 353.6324158
2: -42.5786781, 244.8407440, -45.4966469, 261.0160828, -303.5947571, 290.3373718
3: -57.4809189, 367.3250732, -61.2812729, 392.2051392, -449.6860352, 428.6063538
4: -77.0038528, 297.4574890, -82.2214355, 317.3574219, -394.3612671, 379.6788940

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3377658, upper bound: 547.3376566
time: 0.61 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3373124, upper bound: 547.3368812
time: 0.92 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -149.9540710, 589.7162476, -126.4450989, 496.4124451, -646.3665161, 716.1612549
1: -97.3518753, 337.5323486, -82.2974243, 284.2470398, -381.5989075, 419.8297729
2: -53.1631165, 305.9652405, -44.8499107, 257.4466858, -310.6098022, 350.8151550
3: -71.7943268, 458.7635803, -60.4117203, 386.6867981, -458.4811401, 519.1752930
4: -96.0736160, 371.9808350, -81.0377197, 312.9695740, -409.0431824, 453.0184937

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3367133, upper bound: 547.3372258
time: 0.84 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3365165, upper bound: 547.3365165
time: 0.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.63 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.63
Output dim: 0, lower bound: -547.3377658, upper bound: 547.3376566
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.63
Output dim: 0, lower bound: -547.3373124, upper bound: 547.3368812
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.63
Output dim: 0, lower bound: -547.3367133, upper bound: 547.3372258
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.63
Output dim: 0, lower bound: -547.3365165, upper bound: 547.3365165

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -114.8700562, 451.4649963, -109.2387009, 430.7756348, -545.6456299, 560.7035522
1: -74.6642303, 257.9971008, -71.2149887, 246.4134064, -321.0776367, 329.2120667
2: -40.6660080, 233.8542938, -38.7378044, 223.3826294, -264.0485840, 272.5920715
3: -54.9001541, 350.7150574, -51.9269485, 334.7366943, -389.6368408, 402.6419983
4: -73.5123825, 284.0310669, -70.0383072, 271.1769104, -344.6893005, 354.0692749

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372928, upper bound: 547.3357953
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372928, upper bound: 547.3376474
time: 0.94 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -119.3078537, 468.7784119, -125.5525818, 492.9270630, -612.2349243, 594.3309937
1: -77.5551224, 268.0227356, -81.7019958, 282.0599976, -359.6151123, 349.7246704
2: -42.2567635, 242.9011841, -44.5616150, 255.3820648, -297.6388245, 287.4627991
3: -57.0367165, 364.4077148, -59.9952965, 383.7647705, -440.8014832, 424.4029541
4: -76.4175262, 295.0973511, -80.5323868, 310.4957275, -386.9132690, 375.6296997

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3367335, upper bound: 547.3349158
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372218, upper bound: 547.3367831
time: 0.65 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -144.6641083, 569.3541870, -106.9246902, 421.7014771, -566.3656006, 676.2786865
1: -93.9324036, 325.6292114, -69.6471176, 240.9900970, -334.9224548, 395.2763367
2: -51.2848206, 295.1861877, -37.8863716, 218.5022888, -269.7870789, 333.0725708
3: -69.2381287, 442.5281067, -50.8403091, 327.3419495, -396.5800781, 493.3684082
4: -92.6474991, 358.8392639, -68.4782791, 265.2011108, -357.8485413, 427.3175354

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3354154
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3371738
time: 0.75 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -148.9254608, 585.6918335, -123.3423080, 484.3466797, -633.2719116, 709.0341187
1: -96.6671829, 335.1721802, -80.2481689, 277.1153870, -373.7825317, 415.4203186
2: -52.7988968, 303.8163147, -43.7705116, 250.9599762, -303.7588806, 347.5867920
3: -71.3112640, 455.5359192, -58.9310722, 376.9502258, -448.2614746, 514.4669189
4: -95.4253006, 369.3507690, -79.0862350, 305.0727844, -400.4980774, 448.4369812

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3360193, upper bound: 547.3346250
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3363783, upper bound: 547.3363783
time: 0.95 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.58 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -547.3372928, upper bound: 547.3357953
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -547.3372928, upper bound: 547.3376474
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -547.3367335, upper bound: 547.3349158
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -547.3372218, upper bound: 547.3367831
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3354154
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3371738
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -547.3360193, upper bound: 547.3346250
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -547.3363783, upper bound: 547.3363783

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -101.7468109, 400.0260620, -102.0607910, 402.6747742, -504.4215698, 502.0867920
1: -66.2440262, 228.8151703, -66.6406937, 230.4448090, -296.6888428, 295.4558105
2: -36.1155357, 207.4273834, -36.2671585, 208.9207764, -245.0363007, 243.6945496
3: -48.6522446, 310.9599304, -48.5294342, 313.0802002, -361.7324524, 359.4893799
4: -65.2491608, 251.8498688, -65.5410233, 253.5947418, -318.8438721, 317.3907776

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372928, upper bound: 547.3357953
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372928, upper bound: 547.3357953
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -130.8084564, 511.9103699, -106.9988556, 422.2120972, -553.0204468, 618.9092407
1: -85.4175110, 292.1023865, -69.7554169, 241.3184357, -326.7358704, 361.8577881
2: -46.6538773, 264.3400574, -37.9299622, 218.7617950, -265.4156189, 302.2699585
3: -62.3905907, 397.9736328, -50.8272324, 327.8374329, -390.2279968, 448.8008118
4: -83.5466995, 321.9670105, -68.5824814, 265.5649109, -349.1116028, 390.5494690

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3377348, upper bound: 547.3376474
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3377348, upper bound: 547.3376474
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -106.2125778, 417.4243469, -118.3611145, 464.8594360, -571.0719604, 535.7853394
1: -69.1554413, 238.8853607, -77.1027298, 266.0827637, -335.2382202, 315.9880066
2: -37.7155075, 216.5227661, -42.0824051, 240.8968506, -278.6123352, 258.6051636
3: -50.8041229, 324.7277222, -56.5980988, 362.0694275, -412.8735352, 381.3258057
4: -68.1722107, 262.9846497, -76.0326996, 292.8941650, -361.0663757, 339.0173035

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3367335, upper bound: 547.3349158
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3367335, upper bound: 547.3349158
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -136.6076050, 534.8322144, -123.4257736, 484.6921082, -621.2996826, 658.2579956
1: -89.2275620, 305.2999573, -80.3167267, 277.1981506, -366.4256897, 385.6166992
2: -48.7108498, 276.3085327, -43.8006058, 250.9826660, -299.6935120, 320.1091309
3: -65.1412506, 415.9751892, -58.9605713, 377.1750183, -442.3162842, 474.9356995
4: -87.2921219, 336.5792847, -79.1453400, 305.1659851, -392.4580994, 415.7246094

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3367335, upper bound: 547.3367831
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3367335, upper bound: 547.3367831
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -131.4029388, 516.9334106, -99.7419891, 393.5411377, -524.9440308, 616.6754150
1: -85.3287811, 296.0358276, -65.0498047, 224.9743042, -310.3031006, 361.0855713
2: -46.6489143, 268.4205322, -35.4058456, 203.9978790, -250.6467896, 303.8263855
3: -62.8861237, 402.2091064, -47.4319839, 305.5853882, -368.4714966, 449.6410828
4: -84.2279053, 326.1688843, -63.9612389, 247.5621185, -331.7900391, 390.1300659

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3354154
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3354154
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -159.9563293, 629.0576782, -104.6987228, 413.1952515, -573.1514893, 733.7562866
1: -104.6211319, 359.6234741, -68.1928024, 235.9178314, -340.5389404, 427.8162842
2: -57.2650452, 325.6665039, -37.0836449, 213.9009399, -271.1659241, 362.7501221
3: -76.6513214, 489.8793030, -49.7444534, 320.4734192, -397.1247559, 539.6237793
4: -102.5937119, 396.8489075, -67.0298004, 259.6133728, -362.2070007, 463.8787231

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3371738
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366024, upper bound: 547.3371738
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -135.7447662, 533.5726929, -116.0589447, 455.8789062, -591.6236572, 649.6315918
1: -88.1159515, 305.7598877, -75.5915833, 260.9002075, -349.0161438, 381.3514709
2: -48.1927681, 277.2132568, -41.2625313, 236.2594452, -284.4522095, 318.4757996
3: -65.0040283, 415.4676208, -55.4940605, 354.9456787, -419.9497070, 470.9616699
4: -87.0627594, 336.8804626, -74.5283508, 287.2092285, -374.2719727, 411.4088135

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3360193, upper bound: 547.3346250
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3360193, upper bound: 547.3346250
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -164.1940460, 645.4486694, -121.1929398, 476.0309753, -640.2250366, 766.6416016
1: -107.3507919, 369.1069336, -78.8483658, 272.1968384, -379.5476379, 447.9552917
2: -58.7749672, 334.2385864, -42.9997673, 246.5059509, -305.2808838, 377.2383423
3: -78.7037201, 502.8334351, -57.8828316, 370.2843628, -448.9880981, 560.7162476
4: -105.3541489, 407.3412781, -77.6855087, 299.6747742, -405.0289307, 485.0267944

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3363783, upper bound: 547.3363783
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3363783, upper bound: 547.3363783
time: 0.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.69 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3372928, upper bound: 547.3357953
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3372928, upper bound: 547.3357953
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3377348, upper bound: 547.3376474
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3377348, upper bound: 547.3376474
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3367335, upper bound: 547.3349158
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3367335, upper bound: 547.3349158
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3367335, upper bound: 547.3367831
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3367335, upper bound: 547.3367831
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3354154
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3354154
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3371738
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3366024, upper bound: 547.3371738
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3360193, upper bound: 547.3346250
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3360193, upper bound: 547.3346250
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3363783, upper bound: 547.3363783
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -547.3363783, upper bound: 547.3363783

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -101.7468109, 400.0260620, -96.4393692, 380.5977783, -482.3446045, 496.4654236
1: -66.2440262, 228.8151703, -63.0231209, 217.8451080, -284.0891418, 291.8382568
2: -36.1155357, 207.4273834, -34.2901497, 197.5977936, -233.7133179, 241.7175293
3: -48.6522446, 310.9599304, -45.8679962, 295.8847046, -344.5369568, 356.8279419
4: -65.2491608, 251.8498688, -61.9480782, 239.7389679, -304.9881287, 313.7978821

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3370897, upper bound: 547.3357953
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3370897, upper bound: 547.3357953
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -101.7468109, 400.0260620, -129.6836090, 511.8005981, -613.5474243, 529.7096558
1: -66.2440262, 228.8151703, -84.1925430, 292.8567810, -359.1007690, 313.0076904
2: -36.1155357, 207.4273834, -45.9043770, 265.9435425, -302.0590820, 253.3317566
3: -48.6522446, 310.9599304, -61.6851807, 397.3250122, -445.9772644, 372.6451111
4: -65.2491608, 251.8498688, -82.8035736, 322.5731812, -387.8223267, 334.6533508

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3370897, upper bound: 547.3357953
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3370897, upper bound: 547.3357953
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -130.8084564, 511.9103699, -101.3439713, 400.0878906, -530.8963013, 613.2543335
1: -85.4175110, 292.1023865, -66.1144409, 228.6697845, -314.0871887, 358.2167969
2: -46.6538773, 264.3400574, -35.9420013, 207.3851013, -254.0389709, 300.2820435
3: -62.3905907, 397.9736328, -48.1574783, 310.5730591, -372.9635925, 446.1311035
4: -83.5466995, 321.9670105, -64.9799576, 251.6443024, -335.1910095, 386.9469604

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3328848, upper bound: 547.3335991
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3369751, upper bound: 547.3371212
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -130.8084564, 511.9103699, -134.5446014, 531.3707275, -662.1791992, 646.4549561
1: -85.4175110, 292.1023865, -87.3451385, 303.7068176, -389.1243286, 379.4475098
2: -46.6538773, 264.3400574, -47.5780449, 275.7478943, -322.4017334, 311.9180603
3: -62.3905907, 397.9736328, -63.9985275, 412.1109619, -474.5015564, 461.9721069
4: -83.5466995, 321.9670105, -85.8856888, 334.5320435, -418.0787354, 407.8526917

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3334114, upper bound: 547.3335991
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3369751, upper bound: 547.3371212
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -106.2125778, 417.4243469, -110.8594284, 435.7467651, -541.9592896, 528.2837524
1: -69.1554413, 238.8853607, -72.1158981, 249.1473846, -318.3028259, 311.0011902
2: -37.7155075, 216.5227661, -39.3291664, 225.7836151, -263.4991150, 255.8519287
3: -50.8041229, 324.7277222, -53.0228004, 338.7244263, -389.5285645, 377.7505188
4: -68.1722107, 262.9846497, -71.1107101, 274.2623291, -342.4345398, 334.0953674

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3365022, upper bound: 547.3347234
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366869, upper bound: 547.3347600
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -106.2125778, 417.4243469, -139.5514832, 548.5408325, -654.7534180, 556.9757690
1: -69.1554413, 238.8853607, -90.5423355, 314.0679626, -383.2233887, 329.4277039
2: -37.7155075, 216.5227661, -49.5176811, 284.7007751, -322.4162598, 266.0404358
3: -50.8041229, 324.7277222, -66.8542480, 426.7710876, -477.5751953, 391.5819397
4: -68.1722107, 262.9846497, -89.4759216, 346.0180054, -414.1902161, 352.4605408

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3365022, upper bound: 547.3347234
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366869, upper bound: 547.3347600
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -136.6076050, 534.8322144, -115.9449921, 455.7554321, -592.3629761, 650.7772217
1: -89.2275620, 305.2999573, -75.3574982, 260.3305969, -349.5581665, 380.6574707
2: -48.7108498, 276.3085327, -41.0605659, 235.9186707, -284.6295166, 317.3690491
3: -65.1412506, 415.9751892, -55.4156113, 353.9572754, -419.0985107, 471.3907166
4: -87.2921219, 336.5792847, -74.2495117, 286.6217957, -373.9138489, 410.8287964

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3330310, upper bound: 547.3324798
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3350607, upper bound: 547.3344690
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -136.6076050, 534.8322144, -144.4652405, 568.3351440, -704.9426880, 679.2974243
1: -89.2275620, 305.2999573, -93.7231522, 324.9772949, -414.2048645, 399.0231018
2: -48.7108498, 276.3085327, -51.2042618, 294.5749512, -343.2857971, 327.5127869
3: -65.1412506, 415.9751892, -69.1744690, 441.6322632, -506.7734985, 485.1496277
4: -87.2921219, 336.5792847, -92.5596542, 358.0667419, -445.3588562, 429.1389465

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3330310, upper bound: 547.3324798
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3350607, upper bound: 547.3344690
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -131.4029388, 516.9334106, -96.4393692, 380.5977783, -512.0006104, 613.3728027
1: -85.3287811, 296.0358276, -63.0231209, 217.8451080, -303.1738586, 359.0589294
2: -46.6489143, 268.4205322, -34.2901497, 197.5977936, -244.2467041, 302.7106934
3: -62.8861237, 402.2091064, -45.8679962, 295.8847046, -358.7708130, 448.0770874
4: -84.2279053, 326.1688843, -61.9480782, 239.7389679, -323.9668579, 388.1169128

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354647, upper bound: 547.3352581
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3354154
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3354154
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -131.4029388, 516.9334106, -129.6836090, 511.8005981, -643.2034912, 646.6170044
1: -85.3287811, 296.0358276, -84.1925430, 292.8567810, -378.1855164, 380.2283325
2: -46.6489143, 268.4205322, -45.9043770, 265.9435425, -312.5924072, 314.3248596
3: -62.8861237, 402.2091064, -61.6851807, 397.3250122, -460.2110901, 463.8942871
4: -84.2279053, 326.1688843, -82.8035736, 322.5731812, -406.8010559, 408.9724121

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354647, upper bound: 547.3352581
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3354154
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3354154
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -159.9563293, 629.0576782, -101.3439713, 400.0878906, -560.0441895, 730.4016113
1: -104.6211319, 359.6234741, -66.1144409, 228.6697845, -333.2908630, 425.7379150
2: -57.2650452, 325.6665039, -35.9420013, 207.3851013, -264.6501160, 361.6084900
3: -76.6513214, 489.8793030, -48.1574783, 310.5730591, -387.2243652, 538.0366821
4: -102.5937119, 396.8489075, -64.9799576, 251.6443024, -354.2379456, 461.8288574

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354647, upper bound: 547.3369614
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3365569, upper bound: 547.3370798
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -159.9563293, 629.0576782, -134.5446014, 531.3707275, -691.3270264, 763.6022949
1: -104.6211319, 359.6234741, -87.3451385, 303.7068176, -408.3279419, 446.9686279
2: -57.2650452, 325.6665039, -47.5780449, 275.7478943, -333.0128784, 373.2445374
3: -76.6513214, 489.8793030, -63.9985275, 412.1109619, -488.7622681, 553.8777466
4: -102.5937119, 396.8489075, -85.8856888, 334.5320435, -437.1256714, 482.7345886

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366024, upper bound: 547.3369614
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3365569, upper bound: 547.3370798
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -135.7447662, 533.5726929, -110.8270721, 435.6192932, -571.3640747, 644.3997803
1: -88.1159515, 305.7598877, -72.0950089, 249.0726166, -337.1885071, 377.8548889
2: -48.1927681, 277.2132568, -39.3180733, 225.7137909, -273.9065552, 316.5313416
3: -65.0040283, 415.4676208, -53.0078392, 338.6245117, -403.6285400, 468.4754639
4: -87.0627594, 336.8804626, -71.0910034, 274.1790466, -361.2418213, 407.9714661

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3352836, upper bound: 547.3345123
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3360193, upper bound: 547.3346250
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3360193, upper bound: 547.3346250
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -135.7447662, 533.5726929, -139.8666992, 549.8269653, -685.5717163, 673.4392700
1: -88.1159515, 305.7598877, -90.7610779, 314.8559570, -402.9718933, 396.5209656
2: -48.1927681, 277.2132568, -49.6208420, 285.3995972, -333.5923767, 326.8340454
3: -65.0040283, 415.4676208, -67.0060120, 427.8593445, -492.8633728, 482.4736328
4: -87.0627594, 336.8804626, -89.6881104, 346.8556213, -433.9183960, 426.5685730

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3352836, upper bound: 547.3345123
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3360193, upper bound: 547.3346250
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3360193, upper bound: 547.3346250
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -164.1940460, 645.4486694, -115.9105072, 455.6186523, -619.8126831, 761.3591919
1: -107.3507919, 369.1069336, -75.3351135, 260.2509766, -367.6017761, 444.4420471
2: -58.7749672, 334.2385864, -41.0487099, 235.8446960, -294.6195984, 375.2872925
3: -78.7037201, 502.8334351, -55.3996429, 353.8505554, -432.5542603, 558.2330933
4: -105.3541489, 407.3412781, -74.2283783, 286.5332336, -391.8873901, 481.5695801

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3352836, upper bound: 547.3362303
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3363395, upper bound: 547.3363395
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -164.1940460, 645.4486694, -144.7616730, 569.5354004, -733.7293091, 790.2102051
1: -107.3507919, 369.1069336, -93.9299774, 325.7182312, -433.0690308, 463.0368652
2: -58.7749672, 334.2385864, -51.3017082, 295.2307739, -354.0057373, 385.5402527
3: -78.7037201, 502.8334351, -69.3167572, 442.6552734, -521.3589478, 572.1502075
4: -105.3541489, 407.3412781, -92.7593155, 358.8540344, -464.2081909, 500.1005859

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3363783, upper bound: 547.3362303
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3363395, upper bound: 547.3363395
time: 0.67 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.71 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3370897, upper bound: 547.3357953
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3370897, upper bound: 547.3357953
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3370897, upper bound: 547.3357953
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3370897, upper bound: 547.3357953
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3328848, upper bound: 547.3335991
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3369751, upper bound: 547.3371212
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3334114, upper bound: 547.3335991
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3369751, upper bound: 547.3371212
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3365022, upper bound: 547.3347234
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3366869, upper bound: 547.3347600
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3365022, upper bound: 547.3347234
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3366869, upper bound: 547.3347600
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3330310, upper bound: 547.3324798
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3350607, upper bound: 547.3344690
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3330310, upper bound: 547.3324798
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3350607, upper bound: 547.3344690
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3354154
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3354154
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3354154
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3362947, upper bound: 547.3354154
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3354647, upper bound: 547.3369614
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3365569, upper bound: 547.3370798
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3366024, upper bound: 547.3369614
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3365569, upper bound: 547.3370798
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3360193, upper bound: 547.3346250
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3360193, upper bound: 547.3346250
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3360193, upper bound: 547.3346250
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3360193, upper bound: 547.3346250
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3352836, upper bound: 547.3362303
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3363395, upper bound: 547.3363395
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3363783, upper bound: 547.3362303
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -547.3363395, upper bound: 547.3363395

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -90.8096466, 358.4926758, -96.4393692, 380.5977783, -471.4073792, 454.9320374
1: -59.4543037, 205.3186035, -63.0231209, 217.8451080, -277.2994080, 268.3417358
2: -32.3506165, 186.2490387, -34.2901497, 197.5977936, -229.9484100, 220.5391846
3: -43.1944962, 278.8751221, -45.8679962, 295.8847046, -339.0791931, 324.7431030
4: -58.4242172, 225.9353790, -61.9480782, 239.7389679, -298.1631775, 287.8834229

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372614, upper bound: 547.3356899
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372028, upper bound: 547.3357195
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -104.9053345, 412.3982544, -96.4393692, 380.5977783, -485.5031128, 508.8376160
1: -68.3061523, 235.8987885, -63.0231209, 217.8451080, -286.1512451, 298.9219055
2: -37.2648659, 213.7928925, -34.2901497, 197.5977936, -234.8626556, 248.0830383
3: -50.1921921, 320.6970825, -45.8679962, 295.8847046, -346.0768738, 366.5650635
4: -67.3625336, 259.6656799, -61.9480782, 239.7389679, -307.1015015, 321.6137390

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372614, upper bound: 547.3356899
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372028, upper bound: 547.3357195
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -90.8096466, 358.4926758, -129.6836090, 511.8005981, -602.6102295, 488.1762695
1: -59.4543037, 205.3186035, -84.1925430, 292.8567810, -352.3110962, 289.5111389
2: -32.3506165, 186.2490387, -45.9043770, 265.9435425, -298.2941589, 232.1534119
3: -43.1944962, 278.8751221, -61.6851807, 397.3250122, -440.5195007, 340.5602417
4: -58.4242172, 225.9353790, -82.8035736, 322.5731812, -380.9974060, 308.7389526

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3355872, upper bound: 547.3357805
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3355872, upper bound: 547.3357953
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -104.9053345, 412.3982544, -129.6836090, 511.8005981, -616.7059326, 542.0818481
1: -68.3061523, 235.8987885, -84.1925430, 292.8567810, -361.1629028, 320.0913391
2: -37.2648659, 213.7928925, -45.9043770, 265.9435425, -303.2084045, 259.6972656
3: -50.1921921, 320.6970825, -61.6851807, 397.3250122, -447.5171204, 382.3822327
4: -67.3625336, 259.6656799, -82.8035736, 322.5731812, -389.9357300, 342.4692383

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3355872, upper bound: 547.3357805
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3355872, upper bound: 547.3357953
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -107.2346878, 419.3368530, -95.5530396, 377.7424927, -484.9771729, 514.8898926
1: -70.1070633, 240.1096802, -62.3974838, 215.6972351, -285.8042908, 302.5071106
2: -38.2458763, 218.8977661, -33.8631287, 195.6786041, -233.9244843, 252.7608948
3: -51.0672989, 326.0219727, -45.3543930, 292.8720398, -343.9393311, 371.3763733
4: -68.3189621, 265.0378418, -61.2876549, 237.3321686, -305.6511230, 326.3255005

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3328182, upper bound: 547.3340839
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3330526, upper bound: 547.3338611
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -126.7876663, 497.1045532, -100.2099152, 395.7824707, -522.5700073, 597.3142700
1: -82.7326508, 283.1000366, -65.3497849, 226.0776825, -308.8103027, 348.4497986
2: -45.1754112, 256.3040771, -35.4947853, 205.0923767, -250.2677917, 291.7988586
3: -60.5243111, 385.6567078, -47.5942955, 306.9897156, -367.5140076, 433.2509766
4: -81.0041122, 312.1524658, -64.2221909, 248.7963104, -329.8003540, 376.3746033

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366264, upper bound: 547.3366370
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364651, upper bound: 547.3366459
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -107.2346878, 419.3368530, -127.8780136, 505.5718994, -612.8065796, 547.2147827
1: -70.1070633, 240.1096802, -82.9466171, 288.7554932, -358.8625488, 323.0562439
2: -38.2458763, 218.8977661, -45.1404762, 262.3791504, -300.6250000, 264.0381775
3: -51.0672989, 326.0219727, -60.7493134, 391.4693604, -442.5366516, 386.7713013
4: -68.3189621, 265.0378418, -81.4941711, 318.0466309, -386.3656006, 346.5320129

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3333480, upper bound: 547.3335991
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3333480, upper bound: 547.3335741
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -126.7876663, 497.1045532, -133.5971222, 527.7496948, -654.5373535, 630.7015381
1: -82.7326508, 283.1000366, -86.7242508, 301.5817871, -384.3144226, 369.8242798
2: -45.1754112, 256.3040771, -47.2173424, 273.8723450, -319.0476990, 303.5214233
3: -60.5243111, 385.6567078, -63.5287399, 409.1637573, -469.6880493, 449.1853943
4: -81.0041122, 312.1524658, -85.2486191, 332.2081909, -413.2123108, 397.4010925

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359373, upper bound: 547.3363198
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359785, upper bound: 547.3360645
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -98.4271927, 386.9835510, -109.0334015, 428.5697021, -526.9968872, 496.0169373
1: -64.0776749, 221.1417084, -70.9543686, 245.0179596, -309.0956421, 292.0960693
2: -34.9019394, 200.3048248, -38.6821251, 222.0232544, -256.9251709, 238.9869537
3: -47.2334480, 300.3842468, -52.2051697, 333.0820007, -380.3154602, 352.5894165
4: -63.2654495, 243.3714294, -69.9772797, 269.7257080, -332.9911499, 313.3486938

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372821, upper bound: 547.3358891
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372821, upper bound: 547.3358891
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -100.1811142, 394.0117798, -106.2044144, 417.7816162, -517.9627075, 500.2161865
1: -65.2803192, 225.3317719, -69.1030045, 238.6009216, -303.8812256, 294.4347229
2: -35.6435089, 204.1735687, -37.6732674, 216.2370758, -251.8805847, 241.8468323
3: -48.1156311, 306.4182129, -50.8133507, 324.3704834, -372.4861145, 357.2315674
4: -64.5369110, 248.1072998, -68.1009827, 262.6513367, -327.1882324, 316.2082825

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372821, upper bound: 547.3358891
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372821, upper bound: 547.3358891
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -98.4271927, 386.9835510, -138.0657196, 542.7623901, -641.1895752, 525.0492554
1: -64.0776749, 221.1417084, -89.5052109, 310.7470398, -374.8247070, 310.6469116
2: -34.9019394, 200.3048248, -48.9533844, 281.7014465, -316.6033936, 249.2582092
3: -47.2334480, 300.3842468, -66.1586380, 422.1430969, -469.3765564, 366.5428772
4: -63.2654495, 243.3714294, -88.5312576, 342.2604675, -405.5259094, 331.9026794

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3350402, upper bound: 547.3346785
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3350402, upper bound: 547.3347234
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -100.1811142, 394.0117798, -133.6356506, 525.6682129, -625.8493042, 527.6473999
1: -65.2803192, 225.3317719, -86.8473816, 300.7377930, -366.0181274, 312.1790771
2: -35.6435089, 204.1735687, -47.4720116, 272.6615601, -308.3050232, 251.6455536
3: -48.1156311, 306.4182129, -64.1145554, 408.8748169, -456.9904480, 370.5327759
4: -64.5369110, 248.1072998, -85.7354889, 331.5742188, -396.1111145, 333.8427734

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3353150, upper bound: 547.3346883
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3350402, upper bound: 547.3347600
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -109.9572601, 429.5555725, -108.3563309, 426.4428711, -536.4000854, 537.9119263
1: -71.8554001, 245.9788666, -70.3810654, 243.3423157, -315.1976929, 316.3598328
2: -39.2032204, 224.2291412, -38.3059044, 220.6537476, -259.8569641, 262.5350342
3: -52.3541946, 334.0786438, -51.7105675, 330.5642090, -382.9183960, 385.7892151
4: -70.0258942, 271.5775757, -69.2806091, 267.8924255, -337.9183044, 340.8581238

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -547.3317073, upper bound: 547.3288519
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -547.3317297, upper bound: 547.3288084
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -131.8047791, 516.9647217, -114.7557373, 451.2975464, -583.1022949, 631.7203979
1: -86.0522385, 294.5384521, -74.5846786, 257.6455994, -343.6977844, 369.1231384
2: -46.9709282, 266.6426086, -40.6222649, 233.5068512, -280.4777527, 307.2648315
3: -62.9246216, 401.3212585, -54.8461838, 350.2829590, -413.2075806, 456.1674500
4: -84.2716599, 324.8121948, -73.4791870, 283.6816406, -367.9533081, 398.2913513

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366264, upper bound: 547.3366203
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3361904, upper bound: 547.3361904
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -109.9572601, 429.5555725, -135.1990967, 532.2200317, -642.1773071, 564.7546387
1: -71.8554001, 245.9788666, -87.4540939, 304.1077881, -375.9631958, 333.4329529
2: -39.2032204, 224.2291412, -47.7946358, 275.9065552, -315.1097717, 272.0237732
3: -52.3541946, 334.0786438, -64.6726456, 412.7523499, -465.1065369, 398.7512817
4: -70.0258942, 271.5775757, -86.4052811, 335.0356445, -405.0615234, 357.9828491

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -547.3314566, upper bound: 547.3324798
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -547.3314566, upper bound: 547.3324798
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -131.8047791, 516.9647217, -143.3908691, 564.2894897, -696.0941772, 660.3554077
1: -86.0522385, 294.5384521, -93.0086746, 322.5877380, -408.6399841, 387.5470886
2: -46.9709282, 266.6426086, -50.8005524, 292.4559326, -339.4267883, 317.4430847
3: -62.9246216, 401.3212585, -68.6533966, 438.3307495, -501.2553711, 469.9746704
4: -84.2716599, 324.8121948, -91.8477936, 355.4141541, -439.6858215, 416.6599731

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3349584, upper bound: 547.3344690
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3350098, upper bound: 547.3344373
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -125.8159790, 496.3499146, -96.4393692, 380.5977783, -506.4137573, 592.7893066
1: -81.7188110, 284.3707581, -63.0231209, 217.8451080, -299.5639038, 347.3938599
2: -44.5908165, 258.3686523, -34.2901497, 197.5977936, -242.1886139, 292.6588135
3: -59.8746719, 385.7695312, -45.8679962, 295.8847046, -355.7593689, 431.6375122
4: -80.4250565, 313.3394775, -61.9480782, 239.7389679, -320.1640320, 375.2875366

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364253, upper bound: 547.3354364
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364031, upper bound: 547.3351695
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -133.9065552, 526.3733521, -96.4393692, 380.5977783, -514.5042114, 622.8127441
1: -86.9048157, 301.5799561, -63.0231209, 217.8451080, -304.7499390, 364.6030884
2: -47.5429344, 273.3964844, -34.2901497, 197.5977936, -245.1407318, 307.6865845
3: -64.1577377, 409.7696533, -45.8679962, 295.8847046, -360.0424500, 455.6376343
4: -85.9159164, 332.2004700, -61.9480782, 239.7389679, -325.6548767, 394.1485291

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364253, upper bound: 547.3354364
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364031, upper bound: 547.3351695
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -125.8159790, 496.3499146, -129.6836090, 511.8005981, -637.6165771, 626.0335083
1: -81.7188110, 284.3707581, -84.1925430, 292.8567810, -374.5755615, 368.5632629
2: -44.5908165, 258.3686523, -45.9043770, 265.9435425, -310.5343628, 304.2730103
3: -59.8746719, 385.7695312, -61.6851807, 397.3250122, -457.1996765, 447.4546814
4: -80.4250565, 313.3394775, -82.8035736, 322.5731812, -402.9982300, 396.1430359

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3346301, upper bound: 547.3353750
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3346301, upper bound: 547.3354154
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -133.9065552, 526.3733521, -129.6836090, 511.8005981, -645.7071533, 656.0569458
1: -86.9048157, 301.5799561, -84.1925430, 292.8567810, -379.7615967, 385.7724915
2: -47.5429344, 273.3964844, -45.9043770, 265.9435425, -313.4864807, 319.3007812
3: -64.1577377, 409.7696533, -61.6851807, 397.3250122, -461.4827271, 471.4548340
4: -85.9159164, 332.2004700, -82.8035736, 322.5731812, -408.4890747, 415.0040283

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3346301, upper bound: 547.3353750
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3346301, upper bound: 547.3354154
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -150.1253052, 590.0617065, -98.3461761, 388.4212036, -538.5464478, 688.4077759
1: -98.2356033, 337.2728577, -64.1951828, 222.0284729, -320.2640686, 401.4680481
2: -53.8128510, 305.4613342, -34.9009399, 201.3956299, -255.2084808, 340.3622437
3: -71.9460449, 459.3550720, -46.7352180, 301.5195618, -373.4656067, 506.0903015
4: -96.3227921, 372.2178955, -63.0794601, 244.3205414, -340.6433411, 435.2973633

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354050, upper bound: 547.3369851
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366647, upper bound: 547.3367602
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -224.5697479, 872.3598022, -100.8659592, 398.2401123, -622.8098145, 973.2257080
1: -148.1078339, 503.8217468, -65.8069534, 227.5930328, -375.7008362, 569.6286011
2: -80.8760681, 455.5897827, -35.7766228, 206.4030457, -287.2790833, 491.3663940
3: -106.9006195, 688.5737305, -47.9338570, 309.1147766, -416.0153809, 736.5075684
4: -143.5261078, 556.8256836, -64.6820908, 250.4506226, -393.9767456, 621.5077515

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354440, upper bound: 547.3363413
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354440, upper bound: 547.3361890
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -150.1253052, 590.0617065, -131.5169525, 519.5538940, -669.6790771, 721.5786743
1: -98.2356033, 337.2728577, -85.3750305, 296.9805298, -395.2161255, 422.6478271
2: -53.8128510, 305.4613342, -46.5150070, 269.6862488, -323.4990845, 351.9763184
3: -71.9460449, 459.3550720, -62.5499229, 402.8939209, -474.8399658, 521.9047852
4: -96.3227921, 372.2178955, -83.9400330, 327.1040039, -423.4267883, 456.1579285

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364881, upper bound: 547.3369614
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364881, upper bound: 547.3369614
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -224.5697479, 872.3598022, -133.9595337, 529.0986328, -753.6683960, 1006.3193359
1: -148.1078339, 503.8217468, -86.9676437, 302.3864136, -450.4942017, 590.7893677
2: -80.8760681, 455.5897827, -47.3741074, 274.5415649, -355.4176331, 502.9638977
3: -106.9006195, 688.5737305, -63.7233658, 410.3180847, -517.2185669, 752.2971191
4: -143.5261078, 556.8256836, -85.5194626, 333.0676575, -476.5937500, 642.3450928

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3352313, upper bound: 547.3363239
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3353892, upper bound: 547.3361578
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -125.8159790, 496.3499146, -110.8270721, 435.6192932, -561.4353027, 607.1770020
1: -81.7188110, 284.3707581, -72.0950089, 249.0726166, -330.7913818, 356.4657288
2: -44.5908165, 258.3686523, -39.3180733, 225.7137909, -270.3045959, 297.6867371
3: -59.8746719, 385.7695312, -53.0078392, 338.6245117, -398.4991760, 438.7773743
4: -80.4250565, 313.3394775, -71.0910034, 274.1790466, -354.6040955, 384.4304810

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3356296, upper bound: 547.3351306
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3363548, upper bound: 547.3353980
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -133.9065552, 526.3733521, -110.8270721, 435.6192932, -569.5258789, 637.2004395
1: -86.9048157, 301.5799561, -72.0950089, 249.0726166, -335.9773865, 373.6749573
2: -47.5429344, 273.3964844, -39.3180733, 225.7137909, -273.2567139, 312.7145386
3: -64.1577377, 409.7696533, -53.0078392, 338.6245117, -402.7822571, 462.7774963
4: -85.9159164, 332.2004700, -71.0910034, 274.1790466, -360.0949402, 403.2914734

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3356296, upper bound: 547.3343606
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3363547, upper bound: 547.3353975
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -125.8159790, 496.3499146, -139.8666992, 549.8269653, -675.6429443, 636.2164917
1: -81.7188110, 284.3707581, -90.7610779, 314.8559570, -396.5747681, 375.1318054
2: -44.5908165, 258.3686523, -49.6208420, 285.3995972, -329.9904175, 307.9894714
3: -59.8746719, 385.7695312, -67.0060120, 427.8593445, -487.7340088, 452.7754822
4: -80.4250565, 313.3394775, -89.6881104, 346.8556213, -427.2806702, 403.0275879

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3344989, upper bound: 547.3344989
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3344989, upper bound: 547.3346250
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -133.9065552, 526.3733521, -139.8666992, 549.8269653, -683.7335205, 666.2400513
1: -86.9048157, 301.5799561, -90.7610779, 314.8559570, -401.7607422, 392.3410339
2: -47.5429344, 273.3964844, -49.6208420, 285.3995972, -332.9425354, 323.0172424
3: -64.1577377, 409.7696533, -67.0060120, 427.8593445, -492.0170593, 476.7756348
4: -85.9159164, 332.2004700, -89.6881104, 346.8556213, -432.7715149, 421.8885803

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3344989, upper bound: 547.3344989
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3344989, upper bound: 547.3346250
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -154.4664917, 606.8662720, -112.7403564, 443.3281250, -597.7946167, 719.6066284
1: -101.0314941, 347.0222778, -73.2854004, 253.2202301, -354.2517090, 420.3076782
2: -55.3634567, 314.2730713, -39.9392052, 229.5038147, -284.8672485, 354.2122498
3: -74.0511169, 472.6681824, -53.8877068, 344.2264404, -418.2775574, 526.5559082
4: -99.1551132, 383.0088806, -72.2004852, 278.7688293, -377.9239197, 455.2093201

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364873, upper bound: 547.3367285
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366440, upper bound: 547.3369708
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -228.8635559, 889.2682495, -115.4814911, 453.9769592, -682.8405151, 1004.7495728
1: -150.9119263, 513.4939575, -75.0594711, 259.2864685, -410.1983948, 588.5534058
2: -82.4273376, 464.3058777, -40.8994789, 234.9629364, -317.3902588, 505.2053528
3: -109.0018311, 701.8023682, -55.1985283, 352.5423279, -461.5441589, 757.0007324
4: -146.3686981, 567.5239258, -73.9611511, 285.4637451, -431.8324585, 641.4849854

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364446, upper bound: 547.3368320
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366134, upper bound: 547.3370676
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -154.4664917, 606.8662720, -141.4764404, 556.7131348, -711.1796265, 748.3427124
1: -101.0314941, 347.0222778, -91.7832718, 318.4092102, -419.4406433, 438.8055420
2: -55.3634567, 314.2730713, -50.1458244, 288.6412048, -344.0046692, 364.4188538
3: -74.0511169, 472.6681824, -67.7500534, 432.6445007, -506.6956177, 540.4182129
4: -99.1551132, 383.0088806, -90.6501770, 350.7543640, -449.9094849, 473.6590576

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3362303, upper bound: 547.3362303
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3362303, upper bound: 547.3362303
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -228.8635559, 889.2682495, -144.2252502, 567.4628296, -796.3264160, 1033.4934082
1: -150.9119263, 513.4939575, -93.5823669, 324.5025330, -475.4144592, 607.0762329
2: -82.4273376, 464.3058777, -51.1138496, 294.1170349, -376.5443726, 515.4197388
3: -109.0018311, 701.8023682, -69.0644073, 441.0106812, -550.0125122, 770.8666382
4: -146.3686981, 567.5239258, -92.4226990, 357.5018311, -503.8705444, 659.9466553

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3348828, upper bound: 547.3352557
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3351504, upper bound: 547.3351504
time: 0.74 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.01 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3372614, upper bound: 547.3356899
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3372028, upper bound: 547.3357195
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3372614, upper bound: 547.3356899
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3372028, upper bound: 547.3357195
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3355872, upper bound: 547.3357805
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3355872, upper bound: 547.3357953
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3355872, upper bound: 547.3357805
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3355872, upper bound: 547.3357953
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3328182, upper bound: 547.3340839
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3330526, upper bound: 547.3338611
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3366264, upper bound: 547.3366370
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3364651, upper bound: 547.3366459
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3333480, upper bound: 547.3335991
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3333480, upper bound: 547.3335741
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3359373, upper bound: 547.3363198
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3359785, upper bound: 547.3360645
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3372821, upper bound: 547.3358891
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3372821, upper bound: 547.3358891
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3372821, upper bound: 547.3358891
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3372821, upper bound: 547.3358891
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3350402, upper bound: 547.3346785
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3350402, upper bound: 547.3347234
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3353150, upper bound: 547.3346883
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3350402, upper bound: 547.3347600
NS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3317073, upper bound: 547.3288519
NS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3317297, upper bound: 547.3288084
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3366264, upper bound: 547.3366203
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3361904, upper bound: 547.3361904
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3314566, upper bound: 547.3324798
NS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3314566, upper bound: 547.3324798
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3349584, upper bound: 547.3344690
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3350098, upper bound: 547.3344373
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3364253, upper bound: 547.3354364
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3364031, upper bound: 547.3351695
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3364253, upper bound: 547.3354364
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3364031, upper bound: 547.3351695
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3346301, upper bound: 547.3353750
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3346301, upper bound: 547.3354154
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3346301, upper bound: 547.3353750
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3346301, upper bound: 547.3354154
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3354050, upper bound: 547.3369851
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3366647, upper bound: 547.3367602
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3354440, upper bound: 547.3363413
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3354440, upper bound: 547.3361890
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3364881, upper bound: 547.3369614
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3364881, upper bound: 547.3369614
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3352313, upper bound: 547.3363239
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3353892, upper bound: 547.3361578
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3356296, upper bound: 547.3351306
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3363548, upper bound: 547.3353980
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3356296, upper bound: 547.3343606
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3363547, upper bound: 547.3353975
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3344989, upper bound: 547.3344989
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3344989, upper bound: 547.3346250
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3344989, upper bound: 547.3344989
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3344989, upper bound: 547.3346250
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3364873, upper bound: 547.3367285
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3366440, upper bound: 547.3369708
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3364446, upper bound: 547.3368320
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3366134, upper bound: 547.3370676
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3362303, upper bound: 547.3362303
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3362303, upper bound: 547.3362303
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3348828, upper bound: 547.3352557
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -547.3351504, upper bound: 547.3351504

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -89.5874634, 353.9501343, -91.8193054, 362.9890747, -452.5764771, 445.7694397
1: -58.7119102, 202.6694489, -60.1101494, 207.6627808, -266.3746948, 262.7796021
2: -31.9195518, 183.8037262, -32.6430855, 188.3371887, -220.2567139, 216.4468079
3: -42.6447639, 275.2740784, -43.6970711, 281.9362488, -324.5809631, 318.9711609
4: -57.6809044, 223.0065918, -59.0527153, 228.5276642, -286.2085266, 282.0592041

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3371643, upper bound: 547.3356913
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3371643, upper bound: 547.3356913
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -85.7037277, 338.8433228, -91.5826187, 362.6981201, -448.4018555, 430.4259033
1: -56.0764313, 193.8061676, -59.9237213, 207.2911224, -263.3675537, 253.7298737
2: -30.5071716, 175.7915802, -32.5458412, 187.7585754, -218.2657318, 208.3374176
3: -40.7607117, 263.1013184, -43.6964493, 281.4983826, -322.2590027, 306.7977600
4: -55.1464577, 213.1529388, -59.1680679, 227.7280426, -282.8745117, 272.3209839

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3371471, upper bound: 547.3354813
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3369727, upper bound: 547.3354319
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -103.1669006, 405.5536194, -91.8193054, 362.9890747, -466.1559753, 497.3729248
1: -67.2005463, 231.9711456, -60.1101494, 207.6627808, -274.8633423, 292.0812683
2: -36.6487236, 210.2177582, -32.6430855, 188.3371887, -224.9859161, 242.8608398
3: -49.4173508, 315.3251343, -43.6970711, 281.9362488, -331.3536072, 359.0222168
4: -66.2855377, 255.3544006, -59.0527153, 228.5276642, -294.8132019, 314.4071045

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372921, upper bound: 547.3356899
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372921, upper bound: 547.3356899
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -100.2533417, 394.4410095, -91.5826187, 362.6981201, -462.9514465, 486.0235291
1: -65.2939835, 225.3596344, -59.9237213, 207.2911224, -272.5851135, 285.2832947
2: -35.6095314, 204.2514496, -32.5458412, 187.7585754, -223.3681030, 236.7972870
3: -47.9848747, 306.3495789, -43.6964493, 281.4983826, -329.4832153, 350.0460205
4: -64.3543091, 248.0574799, -59.1680679, 227.7280426, -292.0823364, 307.2255249

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372921, upper bound: 547.3357195
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372921, upper bound: 547.3357195
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -90.8096466, 358.4926758, -123.7317200, 488.3746033, -579.1842041, 482.2243042
1: -59.4543037, 205.3186035, -80.3372879, 279.5730896, -339.0274048, 285.6558838
2: -32.3506165, 186.2490387, -43.8221931, 253.9359741, -286.2865906, 230.0712280
3: -43.1944962, 278.8751221, -58.8230095, 379.2201538, -422.4146423, 337.6980591
4: -58.4242172, 225.9353790, -79.0109863, 307.9269714, -366.3511963, 304.9463501

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3355862, upper bound: 547.3356097
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3353484, upper bound: 547.3355292
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -90.8096466, 358.4926758, -152.2593842, 600.5090332, -691.3186646, 510.7520447
1: -59.4543037, 205.3186035, -99.4850616, 343.0602722, -402.5145874, 304.8036499
2: -32.3506165, 186.2490387, -54.4353294, 310.8546448, -343.2052612, 240.6843414
3: -43.1944962, 278.8751221, -72.5804977, 466.5309753, -509.7254639, 351.4555969
4: -58.4242172, 225.9353790, -97.4042969, 378.1645508, -436.5887756, 323.3396606

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3355862, upper bound: 547.3356097
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3353484, upper bound: 547.3355292
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -104.9053345, 412.3982544, -123.7317200, 488.3746033, -593.2799072, 536.1299438
1: -68.3061523, 235.8987885, -80.3372879, 279.5730896, -347.8792114, 316.2360840
2: -37.2648659, 213.7928925, -43.8221931, 253.9359741, -291.2008362, 257.6150818
3: -50.1921921, 320.6970825, -58.8230095, 379.2201538, -429.4123230, 379.5200806
4: -67.3625336, 259.6656799, -79.0109863, 307.9269714, -375.2894897, 338.6766663

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3358004, upper bound: 547.3356097
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3347818, upper bound: 547.3354084
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -104.9053345, 412.3982544, -152.2593842, 600.5090332, -705.4143677, 564.6575928
1: -68.3061523, 235.8987885, -99.4850616, 343.0602722, -411.3664246, 335.3838501
2: -37.2648659, 213.7928925, -54.4353294, 310.8546448, -348.1195068, 268.2282104
3: -50.1921921, 320.6970825, -72.5804977, 466.5309753, -516.7231445, 393.2775879
4: -67.3625336, 259.6656799, -97.4042969, 378.1645508, -445.5270996, 357.0699768

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3358004, upper bound: 547.3356097
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3347818, upper bound: 547.3354114
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -106.5748444, 416.5618591, -86.5983810, 342.1296082, -448.7044067, 503.1601868
1: -69.6774597, 238.5377502, -56.6561737, 195.5145416, -265.1920166, 295.1939087
2: -38.0137558, 217.4728241, -30.7643719, 177.4240875, -215.4378357, 248.2371826
3: -50.7474022, 323.8957520, -41.1664276, 265.3340759, -316.0814819, 365.0621948
4: -67.8901901, 263.3138733, -55.6773186, 215.0802155, -282.9703674, 318.9911804

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3315034, upper bound: 547.3333180
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3327989, upper bound: 547.3340700
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -102.9668961, 404.3468628, -97.5094833, 385.7641602, -488.7310486, 501.8563538
1: -67.4546890, 231.5068970, -63.6569824, 220.0126648, -287.4673462, 295.1638489
2: -36.8195724, 211.2067871, -34.5274506, 199.9499817, -236.7695465, 245.7341919
3: -49.1443214, 314.0945129, -46.1100349, 298.1800537, -347.3243713, 360.2045288
4: -65.6963272, 255.5479584, -62.1528473, 242.2696533, -307.9659729, 317.7008057

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3317267, upper bound: 547.3331670
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3329985, upper bound: 547.3337663
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -124.8937302, 489.7737732, -95.2088165, 376.6339111, -501.5276489, 584.9826050
1: -81.5348587, 278.8676453, -62.1849861, 215.0645905, -296.5994568, 341.0526123
2: -44.5094147, 252.4574280, -33.7070007, 195.0875244, -239.5969391, 286.1644287
3: -59.6769485, 379.8750305, -45.2367935, 291.8832092, -351.5601501, 425.1118164
4: -79.8457336, 307.5050354, -61.0695877, 236.6733704, -316.5191040, 368.5746155

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366264, upper bound: 547.3366370
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366264, upper bound: 547.3366370
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -122.2936172, 479.6667175, -96.6383514, 382.6831055, -504.9767151, 576.3050537
1: -79.8059845, 272.8672485, -63.1442604, 218.5038757, -298.3097229, 336.0115051
2: -43.5704613, 247.0317688, -34.2659302, 197.9298096, -241.5002289, 281.2976990
3: -58.4129372, 371.7391663, -46.0766373, 296.7181702, -355.1311035, 417.8157959
4: -78.1189194, 300.8817444, -62.3046532, 240.0960236, -318.2149353, 363.1863708

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3358858, upper bound: 547.3366459
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364651, upper bound: 547.3366459
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -104.5298004, 408.4801941, -119.2933350, 472.0322571, -576.5620728, 527.7735596
1: -68.3591309, 233.9506683, -77.3718414, 269.7975159, -338.1565857, 311.3224487
2: -37.2892494, 213.3641357, -42.1394577, 245.3053589, -282.5946045, 255.5035858
3: -49.7365723, 317.6042786, -56.6304932, 365.4660034, -415.2025757, 374.2347717
4: -66.5485153, 258.2643433, -75.9857101, 297.0999146, -363.6484375, 334.2500000

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3332713, upper bound: 547.3326722
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3333458, upper bound: 547.3335085
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3333458, upper bound: 547.3335741
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -106.4749832, 416.4595337, -174.6956940, 687.7030029, -794.1779785, 591.1550903
1: -69.6274185, 238.4247589, -113.7988052, 391.9395752, -461.5669861, 352.2235413
2: -37.9853973, 217.3544159, -62.0687332, 354.6847534, -392.6701355, 279.4231567
3: -50.7145004, 323.7453918, -83.4727783, 534.0474243, -584.7619019, 407.2181702
4: -67.8650131, 263.1584167, -112.2833023, 431.8788757, -499.7438965, 375.4417114

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3333291, upper bound: 547.3326686
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -547.3309948, upper bound: 547.3324400
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3333933, upper bound: 547.3335074
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -126.1416016, 494.4005127, -124.4002228, 490.9550781, -617.0966797, 618.8006592
1: -82.3161240, 281.5833740, -80.7705154, 280.7433472, -363.0594482, 362.3538818
2: -44.9468002, 254.9376984, -44.0293312, 255.0045624, -299.9513245, 298.9670105
3: -60.2110558, 383.5845032, -59.2123032, 380.7834473, -440.9945068, 442.7967834
4: -80.5869522, 310.4827881, -79.4454880, 309.2447510, -389.8316956, 389.9282837

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359295, upper bound: 547.3363198
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359295, upper bound: 547.3363198
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -117.0589066, 461.4503174, -135.7469788, 536.6527100, -653.7116089, 597.1972656
1: -76.3058395, 262.4494629, -88.1168976, 306.4034729, -382.7092896, 350.5663452
2: -41.7009201, 237.9354706, -47.9347801, 278.4168091, -320.1177063, 285.8702393
3: -56.0291061, 356.9515381, -64.3767090, 415.3428345, -471.3719482, 421.3282166
4: -74.8174744, 289.4003601, -86.2869186, 337.5748291, -412.3923035, 375.6872864

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359775, upper bound: 547.3360645
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359775, upper bound: 547.3360645
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -98.4271927, 386.9835510, -103.0867462, 405.2596741, -503.6868286, 490.0702820
1: -64.0776749, 221.1417084, -67.0494766, 231.3836365, -295.4612732, 288.1911621
2: -34.9019394, 200.3048248, -36.5163383, 209.5512695, -244.4532166, 236.8211670
3: -47.2334480, 300.3842468, -49.4677277, 314.3753357, -361.6087952, 349.8519897
4: -63.2654495, 243.3714294, -66.2045441, 254.6617737, -317.9271851, 309.5759888

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3371482, upper bound: 547.3359569
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3371482, upper bound: 547.3358911
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -98.4271927, 386.9835510, -104.9732361, 412.9386597, -511.3658447, 491.9567871
1: -64.0776749, 221.1417084, -68.3416901, 236.0200195, -300.0976868, 289.4833984
2: -34.9019394, 200.3048248, -37.2998619, 213.8298340, -248.7317810, 237.6046906
3: -47.2334480, 300.3842468, -50.3920975, 320.9343872, -368.1678467, 350.7763367
4: -63.2654495, 243.3714294, -67.5567245, 259.8527527, -323.1181946, 310.9281616

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3371482, upper bound: 547.3359569
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3371482, upper bound: 547.3358911
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -100.1811142, 394.0117798, -103.0867462, 405.2596741, -505.4407654, 497.0985107
1: -65.2803192, 225.3317719, -67.0494766, 231.3836365, -296.6639404, 292.3811951
2: -35.6435089, 204.1735687, -36.5163383, 209.5512695, -245.1947784, 240.6898956
3: -48.1156311, 306.4182129, -49.4677277, 314.3753357, -362.4909668, 355.8859253
4: -64.5369110, 248.1072998, -66.2045441, 254.6617737, -319.1986389, 314.3118286

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3371886, upper bound: 547.3358891
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3371886, upper bound: 547.3356952
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -100.1811142, 394.0117798, -104.9732361, 412.9386597, -513.1197510, 498.9850159
1: -65.2803192, 225.3317719, -68.3416901, 236.0200195, -301.3003540, 293.6734314
2: -35.6435089, 204.1735687, -37.2998619, 213.8298340, -249.4733429, 241.4733887
3: -48.1156311, 306.4182129, -50.3920975, 320.9343872, -369.0500183, 356.8103027
4: -64.5369110, 248.1072998, -67.5567245, 259.8527527, -324.3896484, 315.6640320

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3371886, upper bound: 547.3358891
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3371886, upper bound: 547.3356952
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -98.4271927, 386.9835510, -132.1155853, 519.3452148, -617.7723999, 519.0991211
1: -64.0776749, 221.1417084, -85.6588593, 297.4872131, -361.5648499, 306.8005371
2: -34.9019394, 200.3048248, -46.8790932, 269.7125549, -304.6145020, 247.1839142
3: -47.2334480, 300.3842468, -63.3149529, 404.0768127, -451.3102722, 363.6991882
4: -63.2654495, 243.3714294, -84.7642136, 327.6500244, -390.9154663, 328.1356506

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3350402, upper bound: 547.3345053
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -547.3307326, upper bound: 547.3297469
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -547.3328333, upper bound: 547.3324712
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -98.4271927, 386.9835510, -160.6406250, 631.4760742, -729.9032593, 547.6240845
1: -64.0776749, 221.1417084, -104.9343338, 360.8462219, -424.9238892, 326.0759888
2: -34.9019394, 200.3048248, -57.4849701, 326.6592407, -361.5611877, 257.7897034
3: -47.2334480, 300.3842468, -77.0296249, 491.4369507, -538.6703491, 377.4138794
4: -63.2654495, 243.3714294, -103.1011124, 398.1205750, -461.3860168, 346.4725342

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3350402, upper bound: 547.3345960
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -547.3307326, upper bound: 547.3298207
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -547.3307326, upper bound: 547.3325080
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -100.1811142, 394.0117798, -127.6769943, 502.2023010, -602.3834229, 521.6887817
1: -65.2803192, 225.3317719, -82.9951401, 287.4594116, -352.7397461, 308.3268127
2: -35.6435089, 204.1735687, -45.3928719, 260.6254883, -296.2689514, 249.5664062
3: -48.1156311, 306.4182129, -61.2624474, 390.7567444, -438.8723755, 367.6806335
4: -64.5369110, 248.1072998, -81.9618378, 316.9148254, -381.4517212, 330.0691528

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3353150, upper bound: 547.3345753
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3345242, upper bound: 547.3344055
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -100.1811142, 394.0117798, -156.1192017, 613.6267090, -713.8078003, 550.1309204
1: -65.2803192, 225.3317719, -102.1735306, 350.8883057, -416.1686401, 327.5052490
2: -35.6435089, 204.1735687, -55.9532738, 317.6725464, -353.3160400, 260.1268311
3: -48.1156311, 306.4182129, -74.9294586, 478.1170349, -526.2326660, 381.3476562
4: -64.5369110, 248.1072998, -100.2366486, 387.2436218, -451.7804871, 348.3439331

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3350402, upper bound: 547.3346612
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3345242, upper bound: 547.3345179
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -129.9359436, 509.7096558, -106.8494110, 420.2433167, -550.1792603, 616.5590820
1: -84.8714294, 290.3429260, -69.4529648, 239.5475769, -324.4190063, 359.7958679
2: -46.3157654, 262.8302002, -37.7885399, 216.9242706, -263.2400513, 300.6187439
3: -62.0908394, 395.5935974, -51.2322578, 325.5495911, -387.6404419, 446.8258667
4: -83.1314621, 320.2196350, -68.4926834, 263.6985779, -346.8300476, 388.7123108

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366264, upper bound: 547.3366203
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366264, upper bound: 547.3366166
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -127.0671768, 498.5882874, -110.0256500, 432.8702393, -559.9374390, 608.6139526
1: -82.9654541, 283.7599792, -71.5779877, 247.1764526, -330.1419067, 355.3379517
2: -45.2769012, 256.8755188, -39.0360413, 223.9271393, -269.2040405, 295.9114990
3: -60.6860390, 386.6567383, -52.7698326, 336.1256104, -396.8116150, 439.4265442
4: -81.2272415, 312.9201355, -70.6937180, 272.1633606, -353.3905640, 383.6137695

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3361904, upper bound: 547.3361904
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3355640, upper bound: 547.3361904
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -129.0005188, 505.7760315, -132.7577972, 522.7304688, -651.7309570, 638.5338135
1: -84.2535706, 288.2002869, -86.0528412, 298.9869080, -383.2404480, 374.2531128
2: -45.9823570, 260.9367676, -47.0543976, 271.1653442, -317.1477051, 307.9911194
3: -61.5715141, 392.6502380, -63.5731163, 405.9750061, -467.5465088, 456.2233582
4: -82.4732971, 317.8259583, -85.0203705, 329.3242798, -411.7975769, 402.8463135

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3349584, upper bound: 547.3344690
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3349584, upper bound: 547.3344690
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -130.7552338, 512.8577881, -201.0794678, 784.7261963, -915.4813843, 713.9372559
1: -85.3714218, 292.1010742, -132.1531372, 450.5318909, -535.9033203, 424.2542114
2: -46.6092911, 264.3993835, -72.2340622, 407.3519287, -453.9612122, 336.6334229
3: -62.4390984, 398.0343323, -96.2185593, 615.4087524, -677.8478394, 494.2528076
4: -83.6229019, 322.1116333, -129.1807251, 497.6040039, -581.2268677, 451.2923584

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3350098, upper bound: 547.3344373
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3350098, upper bound: 547.3344373
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -125.8159790, 496.3499146, -90.9978714, 359.5478516, -485.3638306, 587.3477173
1: -81.7188110, 284.3707581, -59.6296196, 205.7477570, -287.4665222, 344.0003662
2: -44.5908165, 258.3686523, -32.4103546, 186.4974518, -231.0882721, 290.7789917
3: -59.8746719, 385.7695312, -43.2274475, 279.6111145, -339.4857788, 428.9969482
4: -80.4250565, 313.3394775, -58.5640259, 226.3837433, -306.8088074, 371.9035034

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3371507, upper bound: 547.3351429
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3371507, upper bound: 547.3353485
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -123.4680176, 487.1864929, -96.5824509, 380.7461243, -504.2141113, 583.7689209
1: -80.1942444, 279.1048584, -63.2572670, 218.1713257, -298.3655701, 342.3621216
2: -43.7486649, 253.5775909, -34.3665581, 197.6884003, -241.4370422, 287.9440918
3: -58.7330017, 378.5921936, -45.9539909, 296.6196289, -355.3526306, 424.5461731
4: -78.9013901, 307.4920044, -62.1873322, 239.9322968, -318.8336487, 369.6793213

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3371507, upper bound: 547.3351429
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3371507, upper bound: 547.3353485
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -133.9065552, 526.3733521, -90.9978714, 359.5478516, -493.4543762, 617.3712158
1: -86.9048157, 301.5799561, -59.6296196, 205.7477570, -292.6525574, 361.2095642
2: -47.5429344, 273.3964844, -32.4103546, 186.4974518, -234.0403900, 305.8068237
3: -64.1577377, 409.7696533, -43.2274475, 279.6111145, -343.7688293, 452.9971008
4: -85.9159164, 332.2004700, -58.5640259, 226.3837433, -312.2996521, 390.7644958

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364031, upper bound: 547.3350726
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364031, upper bound: 547.3351695
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -132.0919647, 519.2390137, -96.5824509, 380.7461243, -512.8380737, 615.8214111
1: -85.7259903, 297.5038757, -63.2572670, 218.1713257, -303.8973083, 360.7611389
2: -46.8917694, 269.6893921, -34.3665581, 197.6884003, -244.5801697, 304.0558777
3: -63.2704849, 404.2208557, -45.9539909, 296.6196289, -359.8900757, 450.1748352
4: -84.7340012, 327.6663208, -62.1873322, 239.9322968, -324.6662903, 389.8536377

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364031, upper bound: 547.3350726
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364031, upper bound: 547.3351695
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -125.8159790, 496.3499146, -123.7317200, 488.3746033, -614.1905518, 620.0815430
1: -81.7188110, 284.3707581, -80.3372879, 279.5730896, -361.2918701, 364.7080383
2: -44.5908165, 258.3686523, -43.8221931, 253.9359741, -298.5267944, 302.1907959
3: -59.8746719, 385.7695312, -58.8230095, 379.2201538, -439.0948181, 444.5924988
4: -80.4250565, 313.3394775, -79.0109863, 307.9269714, -388.3520203, 392.3504639

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3355635, upper bound: 547.3351398
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3353484, upper bound: 547.3353570
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -125.8159790, 496.3499146, -152.2593842, 600.5090332, -726.3250122, 648.6091919
1: -81.7188110, 284.3707581, -99.4850616, 343.0602722, -424.7790833, 383.8558044
2: -44.5908165, 258.3686523, -54.4353294, 310.8546448, -355.4454651, 312.8039856
3: -59.8746719, 385.7695312, -72.5804977, 466.5309753, -526.4056396, 458.3500366
4: -80.4250565, 313.3394775, -97.4042969, 378.1645508, -458.5895996, 410.7437744

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3355635, upper bound: 547.3351429
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354705, upper bound: 547.3353570
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -133.9065552, 526.3733521, -123.7317200, 488.3746033, -622.2811279, 650.1051025
1: -86.9048157, 301.5799561, -80.3372879, 279.5730896, -366.4779053, 381.9172363
2: -47.5429344, 273.3964844, -43.8221931, 253.9359741, -301.4789124, 317.2185974
3: -64.1577377, 409.7696533, -58.8230095, 379.2201538, -443.3778992, 468.5926514
4: -85.9159164, 332.2004700, -79.0109863, 307.9269714, -393.8428650, 411.2114563

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3346301, upper bound: 547.3350552
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3345277, upper bound: 547.3351534
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -133.9065552, 526.3733521, -152.2593842, 600.5090332, -734.4155884, 678.6327515
1: -86.9048157, 301.5799561, -99.4850616, 343.0602722, -429.9650879, 401.0650024
2: -47.5429344, 273.3964844, -54.4353294, 310.8546448, -358.3975830, 327.8317871
3: -64.1577377, 409.7696533, -72.5804977, 466.5309753, -530.6887207, 482.3501587
4: -85.9159164, 332.2004700, -97.4042969, 378.1645508, -464.0804443, 429.6047668

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3346301, upper bound: 547.3350726
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3345277, upper bound: 547.3351695
time: 0.89 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.95 + 417.60 = 421.54 seconds
