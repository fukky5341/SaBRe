## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 380.961918313704


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-120.7598648, 310.2168884, -120.7598648, 310.2168884, -430.9767456, 430.9767456)
1: (-306.0167542, 469.4156189, -306.0167542, 469.4156189, -775.4323730, 775.4323730)
2: (-197.4686890, 458.7374573, -197.4686890, 458.7374573, -656.2061768, 656.2061768)
3: (-330.1389465, 542.4745483, -330.1389465, 542.4745483, -872.6135254, 872.6134644)
4: (-287.0395203, 524.3436890, -287.0395203, 524.3436890, -811.3831177, 811.3831177)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.86 + 1.96 = 2.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -380.9771574, upper bound: 380.9771574

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9771470, upper bound: 380.9766991
time: 1.15 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9766991, upper bound: 380.9766991
time: 0.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.83 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 0, lower bound: -380.9771470, upper bound: 380.9766991
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 0, lower bound: -380.9766991, upper bound: 380.9766991

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -102.1709671, 261.3860168, -114.3362198, 294.5241699, -396.6950684, 375.7222290
1: -258.5334167, 395.5080872, -290.0555725, 446.3215942, -704.8548584, 685.5635986
2: -166.9586182, 385.7630310, -186.8040314, 435.1654968, -602.1241455, 572.5670776
3: -278.8269348, 456.4924622, -312.8416443, 515.4390869, -794.2659912, 769.3339844
4: -243.0326538, 440.8841858, -271.7865601, 498.0593872, -741.0920410, 712.6705933

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9638806, upper bound: 380.9643346
time: 0.67 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9634022
time: 0.61 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -118.8791046, 305.5854492, -119.3134308, 306.6369019, -425.5159912, 424.8988647
1: -301.2553406, 462.6041870, -302.3125000, 464.1497498, -765.4050293, 764.9166870
2: -194.1865692, 452.0354309, -194.8987427, 453.5514832, -647.7380371, 646.9342041
3: -325.1773376, 534.5447998, -326.2976990, 536.3432617, -861.5205078, 860.8423462
4: -282.2735596, 516.6941528, -283.3926697, 518.4217529, -800.6953125, 800.0867920

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9688829, upper bound: 380.9686109
time: 0.65 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9670411
time: 0.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.18 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -380.9638806, upper bound: 380.9643346
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9634022
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -380.9688829, upper bound: 380.9686109
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9670411

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -99.9006424, 255.5218964, -110.9370041, 285.7468872, -385.6475220, 366.4588928
1: -252.6559448, 386.7456665, -281.2103577, 433.2129211, -685.8688354, 667.9560547
2: -163.2021637, 376.8170776, -181.0965881, 421.8009338, -585.0030518, 557.9136963
3: -272.7193298, 446.2372742, -303.6001587, 500.1722717, -772.8915405, 749.8373413
4: -237.7994080, 430.8985596, -263.8819275, 483.1522217, -720.9516602, 694.7805176

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9634022
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9634022
time: 0.60 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -97.9270630, 249.5894623, -132.3061829, 337.7371521, -435.6642151, 381.8955078
1: -246.6619720, 378.1087341, -332.5302429, 511.4418945, -758.1038818, 710.6389160
2: -159.5982208, 368.2834778, -215.3123322, 497.2321777, -656.8302612, 583.5958252
3: -267.0090332, 436.4848633, -361.0427246, 591.5975342, -858.6064453, 797.5274658
4: -233.1238403, 420.5596008, -315.0249023, 568.7626343, -801.8864746, 735.5844727

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9634022
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9634022
time: 0.65 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -116.4445114, 299.3266907, -115.4604187, 296.7470703, -413.1915894, 414.7871094
1: -294.8954773, 453.2035522, -292.2443542, 449.3006287, -744.1961060, 745.4478760
2: -190.0927124, 442.5434265, -188.4377594, 438.5581665, -628.6508179, 630.9810791
3: -318.5371399, 523.6215820, -315.7623901, 519.0971069, -837.6341553, 839.3839722
4: -276.6002197, 506.0847168, -274.4066162, 501.6541748, -778.2543945, 780.4913330

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9655942, upper bound: 380.9649652
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9672541, upper bound: 380.9659970
time: 0.81 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -114.7860565, 294.0978699, -133.7529602, 340.8833008, -455.6693726, 427.8507996
1: -289.4912109, 445.7116089, -336.0966187, 516.3527222, -805.8439331, 781.8081055
2: -187.1117554, 434.8902283, -217.6470032, 502.3092957, -689.4210205, 652.5372314
3: -313.7854919, 515.0421753, -365.0690918, 597.5002441, -911.2856445, 880.1111450
4: -272.7771606, 496.7930908, -318.3937683, 574.4002075, -847.1773682, 815.1868286

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9632818, upper bound: 380.9641155
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9652623, upper bound: 380.9652623
time: 0.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.10 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9634022
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9634022
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9634022
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9634022
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -380.9655942, upper bound: 380.9649652
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -380.9672541, upper bound: 380.9659970
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -380.9632818, upper bound: 380.9641155
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -380.9652623, upper bound: 380.9652623

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -98.4243774, 251.7500458, -110.9370041, 285.7468872, -384.1712646, 362.6870422
1: -248.8316956, 381.0985107, -281.2103577, 433.2129211, -682.0444946, 662.3088379
2: -160.7417755, 371.1033936, -181.0965881, 421.8009338, -582.5426636, 552.1999512
3: -268.7075195, 439.6619263, -303.6001587, 500.1722717, -768.8795776, 743.2620239
4: -234.3728943, 424.5164490, -263.8819275, 483.1522217, -717.5250244, 688.3983154

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9638806, upper bound: 380.9643346
time: 1.40 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9638806, upper bound: 380.9643346
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -121.2071915, 307.5454407, -110.9370041, 285.7468872, -406.9540710, 418.4824524
1: -303.9519043, 464.8384399, -281.2103577, 433.2129211, -737.1647949, 746.0488281
2: -197.3629303, 452.6755676, -181.0965881, 421.8009338, -619.1638184, 633.7721558
3: -329.8601685, 537.6703491, -303.6001587, 500.1722717, -830.0321655, 841.2705078
4: -288.7722778, 516.7805176, -263.8819275, 483.1522217, -771.9244385, 780.6624146

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9638806, upper bound: 380.9643346
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9638806, upper bound: 380.9643346
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -98.4243774, 251.7500458, -132.3061829, 337.7371521, -436.1615295, 384.0561523
1: -248.8316956, 381.0985107, -332.5302429, 511.4418945, -760.2735596, 713.6287842
2: -160.7417755, 371.1033936, -215.3123322, 497.2321777, -657.9737549, 586.4157104
3: -268.7075195, 439.6619263, -361.0427246, 591.5975342, -860.3048706, 800.7044678
4: -234.3728943, 424.5164490, -315.0249023, 568.7626343, -803.1354980, 739.5412598

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9616194, upper bound: 380.9616194
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9616194, upper bound: 380.9634022
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -121.2071915, 307.5454407, -132.3061829, 337.7371521, -458.9443359, 439.8515930
1: -303.9519043, 464.8384399, -332.5302429, 511.4418945, -815.3937988, 797.3686523
2: -197.3629303, 452.6755676, -215.3123322, 497.2321777, -694.5950317, 667.9879150
3: -329.8601685, 537.6703491, -361.0427246, 591.5975342, -921.4575195, 898.7128906
4: -288.7722778, 516.7805176, -315.0249023, 568.7626343, -857.5349121, 831.8053589

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9616194, upper bound: 380.9616194
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9616194, upper bound: 380.9634022
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -110.7866364, 284.8183594, -113.1279068, 290.9010315, -401.6876831, 397.9462585
1: -279.9613647, 431.7265320, -286.1373291, 440.5910645, -720.5524292, 717.8638916
2: -180.4087067, 420.9908447, -184.4208832, 429.9171143, -610.3257446, 605.4117432
3: -302.9371948, 498.4674072, -309.3737793, 508.9295654, -811.8667603, 807.8411865
4: -263.0373535, 481.4628296, -268.8120422, 491.7917480, -754.8291016, 750.2749023

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9655942, upper bound: 380.9648697
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9655942, upper bound: 380.9649652
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -113.2716904, 291.7255554, -113.5330505, 292.1243286, -405.3960266, 405.2586060
1: -287.0264282, 441.8933105, -287.4485779, 442.4291687, -729.4555664, 729.3417969
2: -184.9462128, 431.3915405, -185.2638702, 431.7685852, -616.7147827, 616.6553955
3: -310.0375366, 510.5404358, -310.5914001, 511.1463013, -821.1837158, 821.1316528
4: -268.9721985, 493.4302368, -269.7741089, 493.9593811, -762.9315186, 763.2043457

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9651701, upper bound: 380.9638768
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9672541, upper bound: 380.9659970
time: 1.17 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -108.7731247, 278.6733093, -130.9861755, 334.0053406, -442.7784729, 409.6594849
1: -273.8473816, 422.8240662, -329.0453186, 506.1594849, -780.0068359, 751.8692627
2: -176.9207458, 411.9698181, -212.9710999, 492.1043091, -669.0248413, 624.9409180
3: -297.3388062, 488.2400513, -357.5556030, 585.6395874, -882.9783936, 845.7955933
4: -258.4479370, 470.5967407, -311.7515259, 562.8529663, -821.3009033, 782.3480835

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9625808, upper bound: 380.9625808
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9625808, upper bound: 380.9641155
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -111.6881104, 286.6523132, -131.8983459, 336.4448242, -448.1329346, 418.5506592
1: -281.8409119, 434.6015015, -331.3848572, 509.7459717, -791.5868530, 765.9862061
2: -182.0905151, 423.9880066, -214.5076599, 495.8117981, -677.9023438, 638.4956665
3: -305.5063171, 502.2222595, -360.0471191, 589.8678589, -895.3740845, 862.2692871
4: -265.3152466, 484.4011230, -313.9131775, 567.0054321, -832.3206177, 798.3142700

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9627244, upper bound: 380.9623105
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9652623, upper bound: 380.9652623
time: 0.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.39 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9638806, upper bound: 380.9643346
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9638806, upper bound: 380.9643346
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9638806, upper bound: 380.9643346
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9638806, upper bound: 380.9643346
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9616194, upper bound: 380.9616194
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9616194, upper bound: 380.9634022
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9616194, upper bound: 380.9616194
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9616194, upper bound: 380.9634022
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9655942, upper bound: 380.9648697
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9655942, upper bound: 380.9649652
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9651701, upper bound: 380.9638768
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9672541, upper bound: 380.9659970
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9625808, upper bound: 380.9625808
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9625808, upper bound: 380.9641155
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9627244, upper bound: 380.9623105
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -380.9652623, upper bound: 380.9652623

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -98.4243774, 251.7500458, -98.4243774, 251.7500458, -350.1744385, 350.1744385
1: -248.8316956, 381.0985107, -248.8316956, 381.0985107, -629.9301758, 629.9301758
2: -160.7417755, 371.1033936, -160.7417755, 371.1033936, -531.8450928, 531.8450928
3: -268.7075195, 439.6619263, -268.7075195, 439.6619263, -708.3694458, 708.3693848
4: -234.3728943, 424.5164490, -234.3728943, 424.5164490, -658.8892212, 658.8892822

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9704964, upper bound: 380.9698301
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9704878, upper bound: 380.9691472
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -98.4243774, 251.7500458, -114.9623718, 295.5403442, -393.9647217, 366.7123718
1: -248.8316956, 381.0985107, -291.0227051, 447.5180969, -696.3497925, 672.1212158
2: -160.7417755, 371.1033936, -187.5861511, 436.8085022, -597.5501709, 558.6895752
3: -268.7075195, 439.6619263, -314.4665527, 517.0219727, -785.7294922, 754.1284180
4: -234.3728943, 424.5164490, -273.1453857, 499.6650391, -734.0377197, 697.6618652

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9704964, upper bound: 380.9698301
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9704878, upper bound: 380.9691472
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -121.2071915, 307.5454407, -98.4243774, 251.7500458, -372.9572449, 405.9698181
1: -303.9519043, 464.8384399, -248.8316956, 381.0985107, -685.0504150, 713.6701660
2: -197.3629303, 452.6755676, -160.7417755, 371.1033936, -568.4662476, 613.4172363
3: -329.8601685, 537.6703491, -268.7075195, 439.6619263, -769.5219116, 806.3778076
4: -288.7722778, 516.7805176, -234.3728943, 424.5164490, -713.2886963, 751.1533203

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9605104, upper bound: 380.9566559
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9633783, upper bound: 380.9632765
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -121.2071915, 307.5454407, -114.9623718, 295.5403442, -416.7475281, 422.5077820
1: -303.9519043, 464.8384399, -291.0227051, 447.5180969, -751.4699707, 755.8611450
2: -197.3629303, 452.6755676, -187.5861511, 436.8085022, -634.1713867, 640.2616577
3: -329.8601685, 537.6703491, -314.4665527, 517.0219727, -846.8820801, 852.1368408
4: -288.7722778, 516.7805176, -273.1453857, 499.6650391, -788.4371338, 789.9259033

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9605104, upper bound: 380.9566559
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9633783, upper bound: 380.9632765
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -98.4243774, 251.7500458, -132.8636780, 338.6278992, -437.0522766, 384.6137085
1: -248.8316956, 381.0985107, -333.7034302, 512.9467773, -761.7784424, 714.8019409
2: -160.7417755, 371.1033936, -216.0730896, 499.0194702, -659.7611084, 587.1763916
3: -268.7075195, 439.6619263, -362.6532288, 593.5738525, -862.2813721, 802.3150635
4: -234.3728943, 424.5164490, -316.2369995, 570.6166382, -804.9895020, 740.7534180

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9636996, upper bound: 380.9644672
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9635329, upper bound: 380.9663189
time: 1.15 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -121.2071915, 307.5454407, -132.8636780, 338.6278992, -459.8350830, 440.4091187
1: -303.9519043, 464.8384399, -333.7034302, 512.9467773, -816.8986816, 798.5418701
2: -197.3629303, 452.6755676, -216.0730896, 499.0194702, -696.3823242, 668.7485962
3: -329.8601685, 537.6703491, -362.6532288, 593.5738525, -922.6699219, 900.0441895
4: -288.7722778, 516.7805176, -316.2369995, 570.6166382, -859.3889160, 833.0175171

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9593867, upper bound: 380.9569701
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9606447, upper bound: 380.9623562
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -110.7866364, 284.8183594, -109.6963882, 282.0104370, -392.7970276, 394.5147400
1: -279.9613647, 431.7265320, -277.1208191, 427.4205933, -707.3818970, 708.8473511
2: -180.4087067, 420.9908447, -178.6779022, 416.6372986, -597.0460205, 599.6687622
3: -302.9371948, 498.4674072, -299.9077454, 493.4888306, -796.4260254, 798.3751221
4: -263.0373535, 481.4628296, -260.6233826, 476.6534424, -739.6907959, 742.0861816

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9650567, upper bound: 380.9643782
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9639097, upper bound: 380.9620685
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -110.7866364, 284.8183594, -112.2540283, 289.0718689, -399.8585205, 397.0723877
1: -279.9613647, 431.7265320, -284.2784119, 437.8852234, -717.8465576, 716.0049438
2: -180.4087067, 420.9908447, -183.1924438, 427.2842407, -607.6929321, 604.1832886
3: -302.9371948, 498.4674072, -307.1679993, 505.8885193, -808.8256836, 805.6353149
4: -263.0373535, 481.4628296, -266.7010803, 488.8782043, -751.9155273, 748.1638794

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9655942, upper bound: 380.9649652
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9655942, upper bound: 380.9649652
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -109.0301132, 281.0728149, -103.0452042, 264.7278137, -373.7579346, 384.1180115
1: -276.0675049, 425.9029846, -259.7313232, 400.3863525, -676.4537964, 685.6342773
2: -177.8213959, 415.2883301, -167.5460358, 390.6874084, -568.5087891, 582.8342896
3: -298.4519653, 491.9556580, -281.7455750, 462.1687012, -760.6204834, 773.7011108
4: -259.0552368, 475.3352356, -245.1014709, 446.5229797, -705.5781860, 720.4367065

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9614057, upper bound: 380.9615317
time: 1.22 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9651701, upper bound: 380.9638768
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -112.4986725, 289.7720642, -112.4619904, 289.4015198, -401.9002075, 402.2340698
1: -285.0617065, 439.0145264, -284.7319031, 438.4011230, -723.4628296, 723.7462769
2: -183.6560974, 428.5306702, -183.4857483, 427.7785645, -611.4346924, 612.0164185
3: -307.9367371, 507.1999207, -307.6806641, 506.4842529, -814.4210205, 814.8805542
4: -267.1188965, 490.1815491, -267.2108459, 489.4280701, -756.5469971, 757.3923340

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9628674, upper bound: 380.9627077
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9672541, upper bound: 380.9659970
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -108.7731247, 278.6733093, -128.2663422, 327.1898804, -435.9630127, 406.9396057
1: -273.8473816, 422.8240662, -322.0240479, 495.9188232, -769.7661743, 744.8481445
2: -176.9207458, 411.9698181, -208.3258514, 481.9090576, -658.8297119, 620.2956543
3: -297.3388062, 488.2400513, -350.1667480, 573.6081543, -870.8165283, 838.4066772
4: -258.4479370, 470.5967407, -305.2421875, 551.3351440, -809.7830811, 775.8388672

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9625808, upper bound: 380.9625843
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9625808, upper bound: 380.9625843
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -108.7731247, 278.6733093, -130.6303558, 333.4102783, -442.1833496, 409.3036499
1: -273.8473816, 422.8240662, -328.1747437, 505.2344971, -779.0819092, 750.9985962
2: -176.9207458, 411.9698181, -212.3650513, 491.3749390, -668.2955933, 624.3348389
3: -297.3388062, 488.2400513, -356.6177063, 584.6494141, -881.9559937, 844.8574829
4: -258.4479370, 470.5967407, -310.8424683, 561.9472046, -820.3951416, 781.4392090

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9625808, upper bound: 380.9641155
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9310774, upper bound: 380.9586681
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9255832, upper bound: 380.9293998
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -108.9349823, 279.7982178, -126.8789444, 323.4369202, -432.3718567, 406.6771545
1: -274.7210999, 424.3558044, -318.4849243, 490.1543274, -764.8753662, 742.8406982
2: -177.5135956, 413.8149109, -206.1419983, 476.7683105, -654.2819214, 619.9569092
3: -297.9356384, 490.3381348, -346.3067932, 567.1441650, -864.7557983, 836.5376587
4: -258.8384705, 472.8908997, -301.9306030, 545.0991821, -803.9374390, 774.8215332

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9626912, upper bound: 380.9623105
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9626912, upper bound: 380.9623105
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -111.0221024, 284.9909668, -130.6061554, 333.2128906, -444.2349243, 415.5971069
1: -280.1744690, 432.1013184, -328.0128479, 504.8550415, -785.0295410, 760.1141357
2: -180.9919586, 421.5588379, -212.3562469, 491.0599976, -672.0519409, 633.9150391
3: -303.6570129, 499.3389587, -356.4131470, 584.2122803, -887.4957275, 855.5419922
4: -263.7202759, 481.6189270, -310.8589172, 561.5164795, -825.2366333, 792.4778442

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9652623, upper bound: 380.9652516
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9652623, upper bound: 380.9652623
time: 4.06 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.61 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9704964, upper bound: 380.9698301
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9704878, upper bound: 380.9691472
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9704964, upper bound: 380.9698301
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9704878, upper bound: 380.9691472
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9605104, upper bound: 380.9566559
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9633783, upper bound: 380.9632765
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9605104, upper bound: 380.9566559
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9633783, upper bound: 380.9632765
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9636996, upper bound: 380.9644672
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9635329, upper bound: 380.9663189
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9593867, upper bound: 380.9569701
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9606447, upper bound: 380.9623562
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9650567, upper bound: 380.9643782
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9639097, upper bound: 380.9620685
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9655942, upper bound: 380.9649652
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9655942, upper bound: 380.9649652
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9614057, upper bound: 380.9615317
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9651701, upper bound: 380.9638768
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9628674, upper bound: 380.9627077
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9672541, upper bound: 380.9659970
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9625808, upper bound: 380.9625843
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9625808, upper bound: 380.9625843
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9310774, upper bound: 380.9586681
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9255832, upper bound: 380.9293998
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9626912, upper bound: 380.9623105
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9626912, upper bound: 380.9623105
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9652623, upper bound: 380.9652516
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -380.9652623, upper bound: 380.9652623

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -93.1684570, 237.2095032, -94.9713821, 242.1663361, -335.3347778, 332.1808777
1: -235.4190063, 358.5313110, -240.0233765, 366.2069702, -601.6258545, 598.5546875
2: -152.2551575, 349.3713684, -155.1737213, 356.7851257, -509.0402832, 504.5451050
3: -254.2992554, 413.7934265, -259.2516479, 422.6173401, -676.9166260, 673.0450439
4: -222.0640717, 399.2503967, -226.2753754, 407.8713989, -629.9354858, 625.5257568

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9614069, upper bound: 380.9621898
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9690631, upper bound: 380.9698759
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -101.4119110, 258.9053955, -96.2342758, 246.2473755, -347.6593018, 355.1396790
1: -255.5931091, 391.6606140, -243.1604462, 372.9624634, -628.5555420, 634.8210449
2: -165.8540039, 382.3430481, -157.1222076, 362.8078613, -528.6618652, 539.4652710
3: -276.6239624, 451.9491577, -262.7373047, 430.2107544, -706.8347168, 714.6864624
4: -241.6507111, 436.0446472, -229.2885590, 415.1838074, -656.8345337, 665.3330688

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9614576, upper bound: 380.9621240
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9690631, upper bound: 380.9690631
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -93.1684570, 237.2095032, -111.3061066, 285.3708191, -378.5392761, 348.5156250
1: -235.4190063, 358.5313110, -281.7118835, 431.7363586, -667.1552734, 640.2429810
2: -152.2551575, 349.3713684, -181.6745605, 421.5973816, -573.8522949, 531.0458984
3: -254.2992554, 413.7934265, -304.4714661, 498.9122925, -753.2113647, 718.2647705
4: -222.0640717, 399.2503967, -264.6067505, 481.9951172, -704.0592041, 663.8571777

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9639189, upper bound: 380.9656527
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9695402, upper bound: 380.9693951
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -101.4119110, 258.9053955, -112.3794327, 289.0913696, -390.5032654, 371.2848206
1: -255.5931091, 391.6606140, -284.2840576, 437.9413452, -693.5344238, 675.9445801
2: -165.8540039, 382.3430481, -183.2599030, 427.1511536, -593.0050049, 565.6029663
3: -276.6239624, 451.9491577, -307.3737793, 505.9075012, -782.5314941, 759.3228760
4: -241.6507111, 436.0446472, -267.0661316, 488.7619324, -730.4126587, 703.1107178

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9638124, upper bound: 380.9655841
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9695249, upper bound: 380.9687579
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -120.9119873, 306.7928162, -98.2270050, 251.2552338, -372.1672363, 405.0198364
1: -303.2066956, 463.7180481, -248.3337402, 380.3607178, -683.5673828, 712.0517578
2: -196.8792267, 451.5505066, -160.4170990, 370.3749695, -567.2542114, 611.9675903
3: -329.0675049, 536.3627930, -268.1704712, 438.8043518, -767.8717651, 804.5332031
4: -288.0696411, 515.5184937, -233.9027863, 423.6912231, -711.7608032, 749.4212646

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9629693, upper bound: 380.9627806
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9630050, upper bound: 380.9626888
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -120.9119873, 306.7928162, -114.7604370, 295.0382690, -415.9502563, 421.5532532
1: -303.2066956, 463.7180481, -290.5197144, 446.7566528, -749.9633789, 754.2377930
2: -196.8792267, 451.5505066, -187.2583771, 436.0684814, -632.9476929, 638.8088989
3: -329.0675049, 536.3627930, -313.9242249, 516.1424561, -845.2099609, 850.2869873
4: -288.0696411, 515.5184937, -272.6601257, 498.8272400, -786.8967896, 788.1785889

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9588994, upper bound: 380.9579576
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9592944, upper bound: 380.9580780
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -94.7528763, 242.3783722, -131.6704865, 335.6495972, -430.4024658, 374.0488281
1: -239.3380432, 367.1473083, -330.6920776, 508.5830383, -747.9210815, 697.8393555
2: -154.6408997, 357.2329102, -214.0756683, 494.5659180, -649.2066650, 571.3084717
3: -258.6221924, 423.4337769, -359.4293518, 588.4779663, -847.1001587, 782.8630981
4: -225.7157593, 408.7839355, -313.4220581, 565.6278076, -791.3435669, 722.2059937

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9620817, upper bound: 380.9631992
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9609251, upper bound: 380.9634192
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -97.3208618, 249.0142670, -132.3849487, 337.4258728, -434.7467346, 381.3992004
1: -245.8462524, 377.1168823, -332.3957825, 511.1782227, -757.0243530, 709.5126953
2: -158.7783356, 367.0519714, -215.2241211, 497.2371826, -656.0155029, 582.2759399
3: -265.6719360, 434.9741211, -361.3278809, 591.4869995, -857.1589355, 796.3018188
4: -231.7763519, 419.9033203, -315.1222534, 568.5837402, -800.3601074, 735.0255737

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9642555, upper bound: 380.9654761
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9634359, upper bound: 380.9655549
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -120.9119873, 306.7928162, -132.6845398, 338.1801147, -459.0921021, 439.4773560
1: -303.2066956, 463.7180481, -333.2601624, 512.2791748, -815.4858398, 796.9782104
2: -196.8792267, 451.5505066, -215.7806549, 498.3574524, -695.2365723, 667.3311768
3: -329.0675049, 536.3627930, -362.1752319, 592.7993164, -921.0461426, 898.1817627
4: -288.0696411, 515.5184937, -315.8061829, 569.8704224, -857.9399414, 831.3245850

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9574608, upper bound: 380.9573960
time: 1.14 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9556945, upper bound: 380.9560693
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -111.4800873, 284.3848267, -106.9254074, 273.8384094, -385.3184814, 391.3102417
1: -280.8716125, 430.1457825, -269.6250916, 414.6357117, -695.5070801, 699.7708740
2: -181.6865540, 419.6785583, -174.2463379, 404.4981689, -586.1846313, 593.9248657
3: -304.3847046, 497.0873718, -292.2195740, 478.9474182, -783.3320923, 789.3068848
4: -265.2142334, 479.4714050, -254.2575989, 462.3090210, -727.5230713, 733.7290039

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9601033, upper bound: 380.9593495
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9645996, upper bound: 380.9634975
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -109.1176300, 280.5442505, -108.8721466, 279.9075317, -389.0251160, 389.4163818
1: -275.5698853, 425.2963867, -274.9705505, 424.2618103, -699.8316650, 700.2669678
2: -177.6226501, 414.5187683, -177.3017273, 413.4623108, -591.0848389, 591.8204346
3: -298.3363342, 491.0006104, -297.6418457, 489.8186646, -788.1550293, 788.6424561
4: -259.1658020, 474.1903076, -258.7049866, 473.0849609, -732.2507324, 732.8952637

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9622745, upper bound: 380.9611700
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9622745, upper bound: 380.9620685
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -109.2510834, 280.9028931, -112.2540283, 289.0718689, -398.3229370, 393.1569214
1: -275.9576721, 425.8211670, -284.2784119, 437.8852234, -713.8427734, 710.0994263
2: -177.8346252, 415.0449829, -183.1924438, 427.2842407, -605.1187744, 598.2373657
3: -298.7307434, 491.6095581, -307.1679993, 505.8885193, -804.6191406, 798.7775879
4: -259.4568787, 474.8179321, -266.7010803, 488.8782043, -748.3350830, 741.5189209

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9620624, upper bound: 380.9626972
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9637496, upper bound: 380.9630777
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -127.2716217, 324.6364746, -112.2540283, 289.0718689, -416.3435059, 436.8905029
1: -319.3569336, 492.1195679, -284.2784119, 437.8852234, -757.2421875, 776.3978271
2: -206.5873566, 478.1732178, -183.1924438, 427.2842407, -633.8714600, 661.3656616
3: -347.4738159, 569.1818237, -307.1679993, 505.8885193, -853.3174438, 876.0156250
4: -302.8741760, 547.0573120, -266.7010803, 488.8782043, -791.7523804, 813.7584229

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9620624, upper bound: 380.9626972
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9637496, upper bound: 380.9630777
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -107.7066269, 277.7078247, -102.3524551, 262.9560242, -370.6625671, 380.0602722
1: -272.6918640, 420.8123474, -257.9591980, 397.6987000, -670.3905640, 678.7715454
2: -175.6369324, 410.3365173, -166.4091187, 388.0806580, -563.7175903, 576.7456055
3: -294.7832642, 486.0839539, -279.8246460, 459.0579224, -753.8411255, 765.9085083
4: -255.9264984, 469.6476440, -243.4642487, 443.5184326, -699.4449463, 713.1118774

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9649602, upper bound: 380.9635386
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9649602, upper bound: 380.9638768
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -107.1580582, 276.0041504, -109.6241760, 282.2971497, -389.4552002, 385.6283264
1: -271.0488892, 418.1104126, -277.4255676, 427.7940674, -698.8428955, 695.5358887
2: -174.5921173, 408.3463440, -178.7618408, 417.2605591, -591.8526001, 587.1080322
3: -292.9909668, 483.0111084, -299.8619385, 494.2233887, -787.2143555, 782.8729858
4: -254.3287201, 466.8004761, -260.4918518, 477.5166626, -731.8453979, 727.2923584

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9624216, upper bound: 380.9616209
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9624216, upper bound: 380.9627077
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -111.3981018, 286.9939575, -111.9047928, 287.9855957, -399.3836975, 398.8987427
1: -282.3124695, 434.8010559, -283.3146667, 436.2551270, -718.5676270, 718.1156616
2: -181.8485718, 424.4610291, -182.5737610, 425.6957703, -607.5443115, 607.0346680
3: -304.8905029, 502.3545837, -306.1287842, 504.0123901, -808.9028931, 808.4833374
4: -264.4818726, 485.5103149, -265.8785706, 487.0349121, -751.5166626, 751.3889160

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9666425, upper bound: 380.9659014
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9670028, upper bound: 380.9659970
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -109.2507782, 280.9021912, -128.2663422, 327.1898804, -436.4406433, 409.1684570
1: -275.9569092, 425.8200684, -322.0240479, 495.9188232, -771.8756714, 747.8441162
2: -177.8341370, 415.0439453, -208.3258514, 481.9090576, -659.7431641, 623.3697510
3: -298.7299194, 491.6083374, -350.1667480, 573.6081543, -872.1657715, 841.6782837
4: -259.4561157, 474.8166504, -305.2421875, 551.3351440, -810.7911377, 780.0587158

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9609505, upper bound: 380.9597657
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9609505, upper bound: 380.9625654
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -127.2834320, 324.6663208, -128.2663422, 327.1898804, -454.4733276, 452.9325867
1: -319.3868408, 492.1625366, -322.0240479, 495.9188232, -815.3054810, 814.1865845
2: -206.6075439, 478.2161560, -208.3258514, 481.9090576, -688.5166016, 686.5419922
3: -347.5069885, 569.2322998, -350.1667480, 573.6081543, -919.8941650, 918.0401611
4: -302.9024048, 547.1058960, -305.2421875, 551.3351440, -854.2375488, 852.3480835

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9609505, upper bound: 380.9597657
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9609505, upper bound: 380.9625654
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -109.1707840, 281.3984680, -126.8789444, 323.4369202, -432.6076965, 408.2774048
1: -276.3908691, 426.4184570, -318.4849243, 490.1543274, -766.5451660, 744.9033813
2: -178.0801544, 415.9350891, -206.1419983, 476.7683105, -654.8484497, 622.0769653
3: -298.7316895, 492.6340942, -346.3067932, 567.1441650, -865.5286865, 838.7558594
4: -259.3273315, 476.0366516, -301.9306030, 545.0991821, -804.4263306, 777.9672852

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9623267, upper bound: 380.9618267
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9609490, upper bound: 380.9602752
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9616224, upper bound: 380.9608774
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -127.6394730, 325.8866272, -126.8789444, 323.4369202, -451.0763855, 452.7655640
1: -320.5043335, 493.9557800, -318.4849243, 490.1543274, -810.6586914, 812.4405518
2: -207.3299866, 480.1855469, -206.1419983, 476.7683105, -684.0982666, 686.3275146
3: -348.5040283, 571.6269531, -346.3067932, 567.1441650, -914.3041992, 916.6005249
4: -303.6637573, 549.3306885, -301.9306030, 545.0991821, -848.7628174, 851.2612915

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9623267, upper bound: 380.9618267
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9609490, upper bound: 380.9602752
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9616224, upper bound: 380.9608774
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -107.8035202, 276.4328918, -127.9712219, 326.4171753, -434.2207031, 404.4041138
1: -273.4784851, 419.0770874, -321.3631287, 494.6373901, -768.1158447, 740.4401855
2: -176.7188721, 409.0323486, -208.0818329, 481.2188416, -657.9377441, 617.1141968
3: -295.1441040, 484.8772888, -349.1212158, 572.4176025, -867.3233643, 833.7404785
4: -255.8923340, 468.0235901, -304.5791626, 550.1826782, -806.0750122, 772.6027832

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9652012, upper bound: 380.9641451
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9652012, upper bound: 380.9641060
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -109.8429337, 281.9240723, -129.8141022, 331.1714478, -441.0143127, 411.7381592
1: -277.0534363, 427.5181885, -325.9428406, 501.8222656, -778.8757324, 753.4610596
2: -179.0109711, 416.9618530, -211.0251770, 487.9883728, -666.9993286, 627.9870605
3: -300.4399109, 493.9495544, -354.2476196, 580.6408081, -880.6747437, 847.9086304
4: -260.9820557, 476.4109192, -308.9997559, 558.0578003, -819.0398560, 785.4105835

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9652049, upper bound: 380.9652623
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9652049, upper bound: 380.9652049
time: 0.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.48 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9614069, upper bound: 380.9621898
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9690631, upper bound: 380.9698759
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9614576, upper bound: 380.9621240
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9690631, upper bound: 380.9690631
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9639189, upper bound: 380.9656527
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9695402, upper bound: 380.9693951
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9638124, upper bound: 380.9655841
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9695249, upper bound: 380.9687579
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9629693, upper bound: 380.9627806
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9630050, upper bound: 380.9626888
NS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9588994, upper bound: 380.9579576
NS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9592944, upper bound: 380.9580780
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9620817, upper bound: 380.9631992
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9609251, upper bound: 380.9634192
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9642555, upper bound: 380.9654761
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9634359, upper bound: 380.9655549
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9574608, upper bound: 380.9573960
NS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9556945, upper bound: 380.9560693
NS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9601033, upper bound: 380.9593495
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9645996, upper bound: 380.9634975
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9622745, upper bound: 380.9611700
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9622745, upper bound: 380.9620685
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9620624, upper bound: 380.9626972
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9637496, upper bound: 380.9630777
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9620624, upper bound: 380.9626972
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9637496, upper bound: 380.9630777
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9649602, upper bound: 380.9635386
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9649602, upper bound: 380.9638768
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9624216, upper bound: 380.9616209
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9624216, upper bound: 380.9627077
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9666425, upper bound: 380.9659014
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9670028, upper bound: 380.9659970
NS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9609505, upper bound: 380.9597657
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9609505, upper bound: 380.9625654
NS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9609505, upper bound: 380.9597657
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9609505, upper bound: 380.9625654
NS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9609490, upper bound: 380.9602752
NS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9616224, upper bound: 380.9608774
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9609490, upper bound: 380.9602752
NS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9616224, upper bound: 380.9608774
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9652012, upper bound: 380.9641451
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9652012, upper bound: 380.9641060
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9652049, upper bound: 380.9652623
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -380.9652049, upper bound: 380.9652049

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -87.9001770, 224.2659302, -86.6620789, 220.7617798, -308.6619263, 310.9280090
1: -222.2310791, 339.3752136, -219.1048737, 333.8877563, -556.1188354, 558.4799194
2: -143.6252899, 329.8266907, -141.5534515, 324.3289185, -467.9541931, 471.3801270
3: -240.1775818, 391.4868469, -236.9543457, 384.9630737, -625.1405029, 628.4411621
4: -209.6988220, 377.3805847, -206.8268127, 371.1541443, -580.8529663, 584.2073975

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9598569, upper bound: 380.9610990
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9587535, upper bound: 380.9593530
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9599721, upper bound: 380.9606485
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -92.2502365, 235.0676575, -93.2447357, 238.1340332, -330.3842773, 328.3123779
1: -233.1465149, 355.3613281, -235.7660980, 360.2433777, -593.3898315, 591.1274414
2: -150.6905823, 346.2301941, -152.2382812, 350.8646851, -501.5552673, 498.4684753
3: -251.8883514, 410.0918274, -254.7263641, 415.6498413, -667.5382080, 664.8180542
4: -219.7636566, 395.7245178, -221.9519806, 401.2299500, -620.9935913, 617.6764526

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9680355, upper bound: 380.9676640
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9661361, upper bound: 380.9665691
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -96.0199890, 245.8811188, -88.2354355, 225.5187225, -321.5386658, 334.1165466
1: -242.0563660, 372.2004395, -223.0597076, 341.6514893, -583.7078857, 595.2600708
2: -156.8143463, 362.8215332, -144.0153809, 331.4344788, -488.2487793, 506.8368530
3: -262.2317200, 429.2409363, -241.3102112, 393.7293701, -655.9609375, 670.5510254
4: -228.7015686, 414.2088928, -210.5525818, 379.6546631, -608.3562012, 624.7614746

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9614576, upper bound: 380.9621240
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9614576, upper bound: 380.9621240
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -100.3426895, 256.3020935, -94.2560501, 241.5178528, -341.8605347, 350.5581055
1: -252.9050140, 387.7775574, -238.1822510, 365.8987122, -618.8037109, 625.9598389
2: -164.1044922, 378.4673157, -153.7572327, 355.8392944, -519.9437866, 532.2245483
3: -273.7384338, 447.4559021, -257.4779968, 421.9725647, -695.7108765, 704.9338379
4: -239.1232147, 431.6676025, -224.3913879, 407.3416138, -646.4647827, 656.0589600

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9680284, upper bound: 380.9665363
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9660232, upper bound: 380.9660232
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -89.6922684, 228.6554260, -105.8434830, 271.6395569, -361.3317566, 334.4989014
1: -226.5989227, 345.7685547, -267.7184143, 410.9736328, -637.5725708, 613.4868774
2: -146.5439453, 336.4818726, -172.6267853, 401.0162048, -547.5601807, 509.1086426
3: -244.8912048, 398.9686890, -289.6285706, 474.8191528, -719.7102051, 688.5972900
4: -213.8498383, 384.7566528, -251.7194061, 458.5353088, -672.3851318, 636.4760742

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9613191, upper bound: 380.9629001
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9624830, upper bound: 380.9642185
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -92.6816483, 236.0197754, -110.4544373, 283.3033142, -375.9849548, 346.4742126
1: -234.2104950, 356.7430420, -279.5971069, 428.6102905, -662.8206177, 636.3401489
2: -151.4544067, 347.6357727, -180.2841949, 418.5647583, -570.0191650, 527.9198608
3: -253.0080719, 411.7141113, -302.1985474, 495.2814636, -748.2895508, 713.9125977
4: -220.8658295, 397.2747803, -262.5158997, 478.5465088, -699.4123535, 659.7906494

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9695402, upper bound: 380.9693951
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9684547, upper bound: 380.9686943
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -97.7655487, 250.0989380, -107.0585022, 275.6405334, -373.4060669, 357.1573486
1: -246.4573975, 378.4649963, -270.6826172, 417.5560608, -664.0134277, 649.1474609
2: -159.8209076, 369.2461853, -174.4622345, 407.0224915, -566.8433838, 543.7083130
3: -266.8627625, 436.5992737, -292.9433594, 482.2959290, -749.1584473, 729.5426025
4: -232.8991241, 421.2867737, -254.5028687, 465.7957458, -698.6948242, 675.7896729

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9611067, upper bound: 380.9614716
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9611067, upper bound: 380.9655841
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -100.8041611, 257.4061890, -111.3658218, 286.6156921, -387.4198608, 368.7720032
1: -254.0502930, 389.4108276, -281.7446289, 434.1961670, -688.2464600, 671.1554565
2: -164.8518372, 380.1102600, -181.5949249, 423.5009155, -588.3527832, 561.7050781
3: -274.9755554, 449.3414307, -304.6498718, 501.5562744, -776.5317383, 753.9913330
4: -240.2068329, 433.5175476, -264.6191101, 484.6097412, -724.8165894, 698.1366577

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9591452, upper bound: 380.9576734
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9644142, upper bound: 380.9643693
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -119.9620590, 304.3874207, -94.5898361, 241.9667969, -361.9288025, 398.9772644
1: -300.7395020, 460.1469727, -238.9221954, 366.5270996, -667.2664795, 699.0691528
2: -195.2836456, 447.9379883, -154.3707123, 356.6185608, -551.9021606, 602.3087158
3: -326.4889832, 532.1959839, -258.1803894, 422.7114868, -749.2004395, 790.3763428
4: -285.8507690, 511.4593811, -225.3290253, 408.0916748, -693.9423828, 736.7883911

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9618954, upper bound: 380.9610703
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9618954, upper bound: 380.9626498
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -120.2719040, 305.1986084, -97.1225739, 248.5167694, -368.7886658, 402.3211670
1: -301.5084839, 461.3930054, -245.3452148, 376.3760376, -677.8845215, 706.7382202
2: -195.7590027, 449.1813354, -158.4518280, 366.3197632, -562.0786743, 607.6331787
3: -327.3137817, 533.6177368, -265.1322632, 434.1132202, -761.4270020, 798.7499390
4: -286.5761108, 512.8379517, -231.3044128, 419.0727844, -705.6489258, 744.1423340

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9618954, upper bound: 380.9610734
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9618954, upper bound: 380.9626888
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -91.5183182, 233.0827484, -134.9739380, 342.2069092, -433.7252197, 368.0566711
1: -230.6600342, 352.7880554, -338.3598022, 517.8907471, -748.5507812, 691.1478271
2: -149.4139404, 343.4057922, -219.6260223, 503.5865479, -653.0004883, 563.0317993
3: -249.7111816, 407.0451050, -367.9349670, 599.6276245, -849.0315552, 774.9801025
4: -218.2285156, 392.5960999, -321.6629944, 575.6052246, -793.8337402, 714.2589722

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9618648, upper bound: 380.9629914
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9613607, upper bound: 380.9629259
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -94.2173309, 241.0102692, -129.6534119, 330.4045105, -424.6218262, 370.6636963
1: -237.9510651, 365.0980835, -325.4938354, 500.8055115, -738.7565918, 690.5918579
2: -153.7579193, 355.1668396, -210.7363739, 486.6373291, -640.3952637, 565.9031982
3: -257.1510620, 421.0662842, -353.9420776, 579.4769897, -836.5508423, 775.0083618
4: -224.4734802, 406.4511414, -308.6896362, 556.7409668, -781.2144165, 715.1407471

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9602188, upper bound: 380.9618453
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9593732, upper bound: 380.9595326
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -94.2555161, 240.1184998, -135.6339874, 343.7994080, -438.0548706, 375.7525024
1: -237.5848541, 363.2815857, -339.9566345, 520.2672119, -757.8520508, 703.2380981
2: -153.8350067, 353.8515625, -220.6919098, 506.0193787, -659.8543701, 574.5434570
3: -257.1889954, 419.2667847, -369.7017822, 602.3565063, -859.2424927, 788.9683838
4: -224.6871490, 404.4181824, -323.2562866, 578.2858276, -802.9729614, 727.6741943

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9640359, upper bound: 380.9652761
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9600478, upper bound: 380.9652217
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -96.6902390, 247.4056091, -130.3698730, 332.1929626, -428.8831787, 377.7754517
1: -244.2110748, 374.7010803, -327.2106934, 503.4120789, -747.6231689, 701.9117432
2: -157.7292480, 364.6114807, -211.8899994, 489.3280334, -647.0571899, 576.5014648
3: -263.9443054, 432.1766357, -355.8509827, 582.5105591, -846.4017944, 788.0273438
4: -230.3144684, 417.1680908, -310.3999939, 559.7120972, -790.0264893, 727.5681152

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9619292, upper bound: 380.9641369
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9617411, upper bound: 380.9629603
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -111.1197586, 283.4397278, -106.0314560, 271.4934387, -382.6131897, 389.4711914
1: -279.9325256, 428.7319946, -267.2953186, 411.1235046, -691.0560303, 696.0273438
2: -181.0849762, 418.2617493, -172.7518311, 400.9807129, -582.0656738, 591.0135498
3: -303.4072571, 495.4455261, -289.7905884, 474.8612671, -778.2685547, 785.2360229
4: -264.3697815, 477.8691406, -252.1583557, 458.3327026, -722.7025146, 730.0273438

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9612708, upper bound: 380.9597878
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9612708, upper bound: 380.9634975
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -109.1176300, 280.5442505, -110.3214645, 281.3713989, -390.4890137, 390.8657227
1: -275.5698853, 425.2963867, -277.8576355, 425.6194153, -701.1893311, 703.1540527
2: -177.6226501, 414.5187683, -179.8267670, 414.9303894, -592.5530396, 594.3455200
3: -298.3363342, 491.0006104, -301.1954346, 491.7821045, -790.1184082, 792.1960449
4: -259.1658020, 474.1903076, -262.6401672, 474.2566223, -733.4224243, 736.8304443

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9584756, upper bound: 380.9595161
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9616767, upper bound: 380.9603502
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -109.1176300, 280.5442505, -108.0874176, 277.8773193, -386.9949341, 388.6316528
1: -275.5698853, 425.2963867, -272.8939819, 421.2067871, -696.7766724, 698.1903687
2: -177.6226501, 414.5187683, -175.9844513, 410.3907471, -588.0133667, 590.5032349
3: -298.3363342, 491.0006104, -295.4751282, 486.2664490, -784.6027832, 786.4757080
4: -259.1658020, 474.1903076, -256.8855286, 469.6231995, -728.7890015, 731.0758057

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9596567, upper bound: 380.9595161
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9616767, upper bound: 380.9603502
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -107.4352646, 276.4808044, -110.1400146, 284.0924377, -391.5276794, 386.6208191
1: -271.4952698, 419.2215881, -279.3201599, 430.5152893, -702.0105591, 698.5416260
2: -174.8326263, 408.5609436, -179.8211823, 420.0940247, -594.9265747, 588.3821411
3: -293.9528809, 483.9515381, -301.7478943, 497.2974243, -791.2501831, 785.6994629
4: -255.0777435, 467.4588318, -261.5378113, 480.7334900, -735.8112183, 728.9966431

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9629380, upper bound: 380.9634920
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9667446, upper bound: 380.9664005
time: 1.39 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -108.2634201, 278.5458069, -110.2275925, 284.1570129, -392.4204407, 388.7733765
1: -273.4970703, 422.3041992, -279.1542664, 430.5336914, -704.0306396, 701.4584961
2: -176.1722870, 411.6236267, -179.7040100, 420.1418762, -596.3141479, 591.3275146
3: -296.1057739, 487.5267029, -301.7358093, 497.3587952, -793.4645996, 789.2624512
4: -257.0532532, 470.8860474, -261.7579956, 480.6918640, -737.7451172, 732.6440430

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9640755, upper bound: 380.9639235
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9675923, upper bound: 380.9677974
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -125.6412659, 320.7302856, -110.1400146, 284.0924377, -409.7336731, 430.8703003
1: -315.3043213, 486.2732239, -279.3201599, 430.5152893, -745.8195801, 765.5932617
2: -203.8670807, 472.4469299, -179.8211823, 420.0940247, -623.9608765, 652.2680054
3: -343.1812439, 562.3967896, -301.7478943, 497.2974243, -840.3242798, 863.7609863
4: -298.9297180, 540.5243530, -261.5378113, 480.7334900, -779.6630859, 802.0621338

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9600498, upper bound: 380.9583142
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9613180, upper bound: 380.9613398
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -126.1099548, 321.8115234, -110.2275925, 284.1570129, -410.2669678, 432.0391235
1: -316.4774780, 487.9237976, -279.1542664, 430.5336914, -747.0109863, 767.0780640
2: -204.6369934, 474.0578308, -179.7040100, 420.1418762, -624.7788086, 653.7618408
3: -344.3917542, 564.2827148, -301.7358093, 497.3587952, -841.6351929, 865.6095581
4: -300.0532227, 542.3577271, -261.7579956, 480.6918640, -780.7449951, 804.1157227

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9611821, upper bound: 380.9613673
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9624192, upper bound: 380.9590984
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9632020, upper bound: 380.9617439
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -107.7066269, 277.7078247, -99.4578857, 255.3170776, -363.0237122, 377.1657104
1: -272.6918640, 420.8123474, -250.9456177, 385.9647827, -658.6566162, 671.7579346
2: -175.6369324, 410.3365173, -161.9832306, 377.0029602, -552.6398315, 572.3197632
3: -294.7832642, 486.0839539, -271.8626099, 445.7557678, -740.5390625, 757.9465332
4: -255.9264984, 469.6476440, -236.4476318, 430.7850647, -686.7115479, 706.0952759

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9649602, upper bound: 380.9635386
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9649602, upper bound: 380.9635386
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -107.7066269, 277.7078247, -101.8801117, 261.7519836, -369.4585571, 379.5879211
1: -272.6918640, 420.8123474, -256.7658386, 395.8756104, -668.5675049, 677.5781860
2: -175.6369324, 410.3365173, -165.6361694, 386.3063049, -561.9431152, 575.9726562
3: -294.7832642, 486.0839539, -278.5220947, 456.9500427, -751.7331543, 764.6060181
4: -255.9264984, 469.6476440, -242.3437500, 441.4796753, -697.4061890, 711.9913940

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9649602, upper bound: 380.9638768
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9649602, upper bound: 380.9638768
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -107.1580582, 276.0041504, -107.0324860, 275.4232788, -382.5813293, 383.0366211
1: -271.0488892, 418.1104126, -270.4471130, 417.2114258, -688.2601929, 688.5574341
2: -174.5921173, 408.3463440, -174.2743073, 407.3002625, -581.8922729, 582.6205444
3: -292.9909668, 483.0111084, -292.5140381, 481.9382935, -774.9292603, 775.5251465
4: -254.3287201, 466.8004761, -254.2630768, 465.6774902, -720.0061646, 721.0635376

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9624216, upper bound: 380.9616209
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9624216, upper bound: 380.9616209
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -107.1580582, 276.0041504, -111.5034637, 286.9665833, -394.1246338, 387.5075989
1: -271.0488892, 418.1104126, -282.3046265, 434.7135925, -705.7624512, 700.4150391
2: -174.5921173, 408.3463440, -181.9177094, 424.1986694, -598.7907715, 590.2639771
3: -292.9909668, 483.0111084, -305.0166626, 502.2409363, -795.2318115, 788.0276489
4: -254.3287201, 466.8004761, -264.9156189, 485.3171997, -739.6458740, 731.7160645

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9624216, upper bound: 380.9616209
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9624216, upper bound: 380.9627077
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -109.1394806, 281.2298889, -108.9491272, 280.1452942, -389.2847900, 390.1790161
1: -276.7825623, 426.0996399, -277.3763123, 424.4076843, -701.1901855, 703.4759521
2: -178.2467346, 416.1659851, -178.6744690, 414.2894897, -592.5362549, 594.8403931
3: -298.6938782, 492.3502808, -298.2582703, 490.9793701, -789.6732178, 790.6085205
4: -258.9990845, 476.0036926, -258.5288696, 474.7673035, -733.7663574, 734.5324707

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9666425, upper bound: 380.9659014
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9666425, upper bound: 380.9659014
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -110.5343094, 284.7401733, -110.4797821, 284.2623596, -394.7966614, 395.2199402
1: -280.0118103, 431.4539490, -279.5233459, 430.7288208, -710.7406006, 710.9772339
2: -180.3761292, 421.0684509, -180.1308441, 420.0856018, -600.4617310, 601.1990967
3: -302.5249023, 498.4201050, -302.2250977, 497.5073242, -800.0321045, 800.6452026
4: -262.4782104, 481.6699219, -262.5693054, 480.6873779, -743.1655884, 744.2392578

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9670028, upper bound: 380.9659970
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9670028, upper bound: 380.9659970
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -109.2507782, 280.9021912, -127.4062958, 324.9772949, -434.2279968, 408.3084717
1: -275.9569092, 425.8200684, -319.7049255, 492.6313171, -768.5881958, 745.5249634
2: -177.8341370, 415.0439453, -206.8097992, 478.6806030, -656.5147705, 621.8535767
3: -298.7299194, 491.6083374, -347.8441772, 569.7676392, -868.0888062, 839.2614136
4: -259.4561157, 474.8166504, -303.1920776, 547.6320190, -807.0880127, 778.0086670

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9568602, upper bound: 380.9581371
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9569890, upper bound: 380.9577667
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -127.2834320, 324.6663208, -127.4062958, 324.9772949, -452.2607117, 452.0726318
1: -319.3868408, 492.1625366, -319.7049255, 492.6313171, -812.0180664, 811.8674316
2: -206.6075439, 478.2161560, -206.8097992, 478.6806030, -685.2881470, 685.0258789
3: -347.5069885, 569.2322998, -347.8441772, 569.7676392, -915.8171387, 915.6232910
4: -302.9024048, 547.1058960, -303.1920776, 547.6320190, -850.5344238, 850.2979736

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9587186, upper bound: 380.9597150
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9593982, upper bound: 380.9614935
time: 1.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -107.0709915, 274.6079712, -126.7825165, 323.4387817, -430.5097656, 401.3904724
1: -271.6510315, 416.3375549, -318.3457642, 490.1509094, -761.8018799, 734.6832275
2: -175.5291748, 406.3594666, -206.1176910, 476.8311462, -652.3603516, 612.4771118
3: -293.1677246, 481.7071838, -345.8911133, 567.1911621, -860.1213989, 827.3291016
4: -254.1489716, 464.9720154, -301.7779236, 545.1773071, -799.3262329, 766.7499390

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9638687, upper bound: 380.9614052
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9652012, upper bound: 380.9641060
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9652012, upper bound: 380.9641060
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -106.3285370, 272.7628479, -124.7954178, 318.2304077, -424.5589600, 397.5582581
1: -269.7420044, 413.5846252, -313.1323853, 482.2271118, -751.9689941, 726.7169800
2: -174.2807770, 403.5567322, -202.7620850, 469.0730591, -643.3538208, 606.3187256
3: -291.1631775, 478.5207825, -340.4318848, 558.0941772, -849.1763916, 818.7250366
4: -252.4328613, 461.8562927, -297.0601807, 536.2368774, -788.6697388, 758.9163818

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9638580, upper bound: 380.9613722
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9652012, upper bound: 380.9641060
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9652012, upper bound: 380.9641060
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -109.1070862, 280.0902100, -128.5230865, 327.9358521, -437.0429382, 408.6132812
1: -275.2328796, 424.7585144, -322.6440125, 496.9675293, -772.2003174, 747.4025269
2: -177.8209076, 414.2729797, -208.8832703, 483.2210999, -661.0419922, 623.1562500
3: -298.4584351, 490.7599182, -350.7309265, 574.9940796, -873.0538330, 841.2061768
4: -259.2329712, 473.3455505, -305.9734497, 552.6093750, -811.8423462, 779.3189697

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9650924, upper bound: 380.9652623
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9650924, upper bound: 380.9652019
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -108.3618469, 278.2790222, -126.7537842, 323.2989807, -431.6608276, 405.0328064
1: -273.2683105, 422.0718079, -317.9895325, 489.8952637, -763.1635742, 740.0613403
2: -176.5603027, 411.5358582, -205.8866119, 476.3372192, -652.8975220, 617.4224854
3: -296.4225159, 487.6260376, -345.8972168, 566.8494263, -862.9993896, 833.2415161
4: -257.5170288, 470.2795410, -301.7801819, 544.6549683, -802.1719360, 772.0596313

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9650924, upper bound: 380.9652049
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9650924, upper bound: 380.9650924
time: 0.90 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.80 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9587535, upper bound: 380.9593530
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9599721, upper bound: 380.9606485
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9680355, upper bound: 380.9676640
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9661361, upper bound: 380.9665691
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9614576, upper bound: 380.9621240
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9614576, upper bound: 380.9621240
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9680284, upper bound: 380.9665363
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9660232, upper bound: 380.9660232
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9613191, upper bound: 380.9629001
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9624830, upper bound: 380.9642185
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9695402, upper bound: 380.9693951
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9684547, upper bound: 380.9686943
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9611067, upper bound: 380.9614716
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9611067, upper bound: 380.9655841
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9591452, upper bound: 380.9576734
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9644142, upper bound: 380.9643693
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9618954, upper bound: 380.9610703
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9618954, upper bound: 380.9626498
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9618954, upper bound: 380.9610734
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9618954, upper bound: 380.9626888
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9618648, upper bound: 380.9629914
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9613607, upper bound: 380.9629259
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9602188, upper bound: 380.9618453
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9593732, upper bound: 380.9595326
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9640359, upper bound: 380.9652761
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9600478, upper bound: 380.9652217
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9619292, upper bound: 380.9641369
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9617411, upper bound: 380.9629603
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9612708, upper bound: 380.9597878
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9612708, upper bound: 380.9634975
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9584756, upper bound: 380.9595161
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9616767, upper bound: 380.9603502
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9596567, upper bound: 380.9595161
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9616767, upper bound: 380.9603502
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9629380, upper bound: 380.9634920
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9667446, upper bound: 380.9664005
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9640755, upper bound: 380.9639235
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9675923, upper bound: 380.9677974
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9600498, upper bound: 380.9583142
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9613180, upper bound: 380.9613398
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9624192, upper bound: 380.9590984
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9632020, upper bound: 380.9617439
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9649602, upper bound: 380.9635386
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9649602, upper bound: 380.9635386
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9649602, upper bound: 380.9638768
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9649602, upper bound: 380.9638768
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9624216, upper bound: 380.9616209
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9624216, upper bound: 380.9616209
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9624216, upper bound: 380.9616209
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9624216, upper bound: 380.9627077
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9666425, upper bound: 380.9659014
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9666425, upper bound: 380.9659014
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9670028, upper bound: 380.9659970
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9670028, upper bound: 380.9659970
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9568602, upper bound: 380.9581371
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9569890, upper bound: 380.9577667
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9587186, upper bound: 380.9597150
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9593982, upper bound: 380.9614935
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9652012, upper bound: 380.9641060
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9652012, upper bound: 380.9641060
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9652012, upper bound: 380.9641060
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9652012, upper bound: 380.9641060
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9650924, upper bound: 380.9652623
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9650924, upper bound: 380.9652019
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9650924, upper bound: 380.9652049
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 0, lower bound: -380.9650924, upper bound: 380.9650924

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -91.1372833, 232.2429352, -92.5372543, 236.3426971, -327.4798584, 324.7801819
1: -230.3450165, 351.1256104, -233.9887085, 357.5627747, -587.9077148, 585.1142578
2: -148.8572388, 342.0571594, -151.0711670, 348.2173157, -497.0745544, 493.1283264
3: -248.8926697, 405.1709290, -252.8257904, 412.5324097, -661.4248047, 657.9967041
4: -217.1373596, 390.9812622, -220.2815094, 398.2236633, -615.3610229, 611.2627563

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9673696, upper bound: 380.9664685
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9675051, upper bound: 380.9666057
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -90.0596313, 229.2843781, -91.7859268, 234.5295258, -324.5891724, 321.0702515
1: -227.5738831, 346.4954834, -232.0766602, 354.8210449, -582.3948975, 578.5721436
2: -147.0287628, 337.6989136, -149.8321075, 345.5057373, -492.5344849, 487.5310059
3: -245.8481293, 399.9549561, -250.7555084, 409.3832092, -655.2312012, 650.7102661
4: -214.4863434, 385.8372803, -218.5216522, 395.1695557, -609.6558838, 604.3589478

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9651747, upper bound: 380.9652000
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9652616, upper bound: 380.9653242
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -94.7555771, 242.7195129, -87.5037308, 223.6913147, -318.4468994, 330.2232056
1: -238.8876801, 367.4439697, -221.2258453, 338.9138184, -577.8014526, 588.6697998
2: -154.7351685, 358.1118164, -142.8191833, 328.7334900, -483.4686584, 500.9309998
3: -258.8259888, 423.7182007, -239.3416748, 390.5494995, -649.3754883, 663.0598145
4: -225.7136536, 408.8912659, -208.8246765, 376.5853271, -602.2988892, 617.7159424

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9378149, upper bound: 380.9423795
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9378149, upper bound: 380.9621240
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -93.3556442, 238.9171295, -86.9006958, 222.2000580, -315.5556335, 325.8178101
1: -235.2902832, 361.6282959, -219.6974792, 336.6990662, -571.9893799, 581.3258057
2: -152.4275665, 352.5953064, -141.8161316, 326.5264587, -478.9540100, 494.4114380
3: -254.9373169, 417.0751343, -237.6852112, 387.9970398, -642.9343262, 654.7603149
4: -222.2174377, 402.3880615, -207.4064484, 374.0957336, -596.3130493, 609.7944946

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9609979, upper bound: 380.9615080
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9599896, upper bound: 380.9610257
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -99.2967606, 253.6486664, -93.5963364, 239.8532867, -339.1500549, 347.2449341
1: -250.2639008, 383.7803955, -236.5359955, 363.4080200, -613.6719360, 620.3163452
2: -162.3755951, 374.5049438, -152.6754150, 353.3927917, -515.7683716, 527.1802979
3: -270.8992615, 442.8027344, -255.7105103, 419.0784912, -689.9777832, 698.5130615
4: -236.6607666, 427.1920166, -222.8290710, 404.5580750, -641.2188721, 650.0211182

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9680284, upper bound: 380.9665363
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9680284, upper bound: 380.9665363
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -97.9183350, 249.8751678, -92.7604294, 237.8158264, -335.7341614, 342.6355896
1: -246.7789307, 377.9357300, -234.4060211, 360.3379517, -607.1168823, 612.3416138
2: -160.1590729, 369.0235901, -151.2852173, 350.3123169, -510.4713135, 520.3087769
3: -267.0725098, 436.1832886, -253.4185791, 415.5461121, -682.6186523, 689.6018677
4: -233.2116547, 420.7821960, -220.8757019, 401.1087646, -634.3204346, 641.6578979

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9660232, upper bound: 380.9660232
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9660232, upper bound: 380.9660232
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -88.8879242, 226.4419708, -105.6178207, 271.0511780, -359.9390869, 332.0597839
1: -224.2434998, 342.4519348, -267.1042480, 410.0854492, -634.3289795, 609.5561523
2: -145.0949097, 333.1315613, -172.2396545, 400.1468811, -545.2418213, 505.3712158
3: -242.6197510, 395.0950012, -289.0032959, 473.7881165, -716.4078369, 684.0980835
4: -212.0589752, 380.8399048, -251.1965179, 457.5187073, -669.5776978, 632.0364380

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9611856, upper bound: 380.9629001
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9611856, upper bound: 380.9629001
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -89.3140335, 227.7156677, -105.7691498, 271.4531555, -360.7671814, 333.4848022
1: -225.6268768, 344.3666687, -267.5230713, 410.6981812, -636.3250732, 611.8897705
2: -145.9080963, 335.0718079, -172.4990540, 400.7363281, -546.6442261, 507.5708008
3: -243.8658447, 397.3309631, -289.4256897, 474.4949951, -718.3607788, 686.7566528
4: -212.9675293, 383.1667786, -251.5456390, 458.2208252, -671.1883545, 634.7124023

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9622489, upper bound: 380.9642185
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9622489, upper bound: 380.9642185
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -91.5725250, 233.2069244, -109.8393784, 281.7546082, -373.3271484, 343.0462952
1: -231.4204407, 352.5271912, -278.0728760, 426.2734070, -657.6937866, 630.6000366
2: -149.6280518, 343.4824524, -179.2859650, 416.2879944, -565.9160156, 522.7684326
3: -250.0230408, 406.8168640, -300.5494690, 492.5699463, -742.5928955, 707.3663330
4: -218.2477722, 392.5541077, -261.0576782, 475.9543152, -694.2020874, 653.6118164

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9688005, upper bound: 380.9683838
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9661727, upper bound: 380.9667700
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9681868, upper bound: 380.9681226
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -90.4413300, 230.1285858, -108.9043427, 279.4845276, -369.9258423, 339.0329285
1: -228.4244385, 347.7257996, -275.6619873, 422.9084167, -651.3328857, 623.3878174
2: -147.6817627, 338.9410095, -177.7061920, 412.8858032, -560.5675659, 516.6472168
3: -246.7895355, 401.4006653, -297.9833374, 488.6816711, -735.4711914, 699.3840332
4: -215.4800720, 387.1912231, -258.8763733, 472.1237793, -687.6037598, 646.0676270

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9676443, upper bound: 380.9675108
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9655938, upper bound: 380.9665043
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9655938, upper bound: 380.9673018
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -100.4049683, 256.4219971, -107.0585022, 275.6405334, -376.0455017, 363.4804688
1: -253.0372925, 387.9341431, -270.6826172, 417.5560608, -670.5933838, 658.6167603
2: -164.1944427, 378.6451721, -174.4622345, 407.0224915, -571.2168579, 553.1072998
3: -273.8925781, 447.6306763, -292.9433594, 482.2959290, -756.1883545, 740.5739746
4: -239.2581787, 431.8586121, -254.5028687, 465.7957458, -705.0538330, 686.3614502

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9606509, upper bound: 380.9655841
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9606509, upper bound: 380.9655841
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -99.1943436, 253.4449463, -110.3790207, 284.2957458, -383.4900513, 363.8239441
1: -250.0106201, 383.5177917, -279.2568359, 430.7322693, -680.7429199, 662.7746582
2: -162.2101746, 374.2223206, -179.8891754, 420.1025391, -582.3127441, 554.1114502
3: -270.6195068, 442.5226135, -302.0205994, 497.5307922, -768.1501465, 744.5432129
4: -236.4086609, 426.8773804, -262.1540833, 480.7769775, -717.1854248, 689.0314941

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9644142, upper bound: 380.9643693
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9644142, upper bound: 380.9643693
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -119.7734833, 303.9483032, -94.5898361, 241.9667969, -361.7402039, 398.5381470
1: -300.1729736, 459.5752563, -238.9221954, 366.5270996, -666.7000122, 698.4974365
2: -194.8818665, 447.3146667, -154.3707123, 356.6185608, -551.5004272, 601.6853027
3: -325.9441528, 531.4641724, -258.1803894, 422.7114868, -748.6556396, 789.6445312
4: -285.4195251, 510.7304077, -225.3290253, 408.0916748, -693.5109863, 736.0594482

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9602861, upper bound: 380.9609526
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9593328, upper bound: 380.9605712
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -119.7734833, 303.9483032, -97.1225739, 248.5167694, -368.2901917, 401.0708618
1: -300.1729736, 459.5752563, -245.3452148, 376.3760376, -676.5490112, 704.9204712
2: -194.8818665, 447.3146667, -158.4518280, 366.3197632, -561.2016602, 605.7664795
3: -325.9441528, 531.4641724, -265.1322632, 434.1132202, -760.0573120, 796.5964355
4: -285.4195251, 510.7304077, -231.3044128, 419.0727844, -704.4921875, 742.0347900

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9601786, upper bound: 380.9608117
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9593328, upper bound: 380.9607952
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -91.4433365, 232.8937378, -134.9739380, 342.2069092, -433.6502380, 367.8676453
1: -230.4667664, 352.5077515, -338.3598022, 517.8907471, -748.3574829, 690.8675537
2: -149.2877808, 343.1240845, -219.6260223, 503.5865479, -652.8743286, 562.7500000
3: -249.5083771, 406.7182312, -367.9349670, 599.6276245, -848.8291626, 774.6531372
4: -218.0499268, 392.2773132, -321.6629944, 575.6052246, -793.6551514, 713.9401855

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9592896, upper bound: 380.9588830
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9591547, upper bound: 380.9589839
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -91.6836700, 233.5021057, -134.9739380, 342.2069092, -433.8905334, 368.4760437
1: -231.1588287, 353.4169922, -338.3598022, 517.8907471, -749.0495605, 691.7767334
2: -149.7211151, 344.0307007, -219.6260223, 503.5865479, -653.3076782, 563.6567383
3: -250.2082520, 407.7929382, -367.9349670, 599.6276245, -849.5246582, 775.7279053
4: -218.5964508, 393.3457947, -321.6629944, 575.6052246, -794.2016602, 715.0087280

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9587258, upper bound: 380.9581685
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9586349, upper bound: 380.9583051
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -94.1818466, 239.9323120, -135.6339874, 343.7994080, -437.9812622, 375.5662842
1: -237.3945312, 363.0061340, -339.9566345, 520.2672119, -757.6617432, 702.9627686
2: -153.7109680, 353.5742798, -220.6919098, 506.0193787, -659.7303467, 574.2661743
3: -256.9894104, 418.9457397, -369.7017822, 602.3565063, -859.0433960, 788.6473389
4: -224.5119629, 404.1046448, -323.2562866, 578.2858276, -802.7977295, 727.3608398

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9620678, upper bound: 380.9623274
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9620392, upper bound: 380.9625694
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -94.4298859, 240.5539856, -135.6339874, 343.7994080, -438.2293091, 376.1879883
1: -238.1003723, 363.9363098, -339.9566345, 520.2672119, -758.3675537, 703.8929443
2: -154.1539001, 354.5000610, -220.6919098, 506.0193787, -660.1732178, 575.1918945
3: -257.7014465, 420.0483093, -369.7017822, 602.3565063, -859.7514648, 789.7500610
4: -225.0722961, 405.1933899, -323.2562866, 578.2858276, -803.3581543, 728.4495850

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9611021, upper bound: 380.9613751
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9610776, upper bound: 380.9616239
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -96.3227234, 246.4725952, -130.3698730, 332.1929626, -428.5156860, 376.8424683
1: -243.2267761, 373.3128052, -327.2106934, 503.4120789, -746.6388550, 700.5234985
2: -157.0952148, 363.2282715, -211.8899994, 489.3280334, -646.4231567, 575.1182861
3: -262.9323730, 430.5558472, -355.8509827, 582.5105591, -845.3728638, 786.4066162
4: -229.4614716, 415.5812073, -310.3999939, 559.7120972, -789.1734009, 725.9812012

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9606910, upper bound: 380.9623502
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9606910, upper bound: 380.9629603
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -96.0323410, 245.6160889, -130.0502014, 331.3729553, -427.4052734, 375.6662903
1: -242.1855011, 372.0948486, -326.3266296, 502.1818542, -744.3673706, 698.4214478
2: -156.4258881, 361.9343567, -211.3227997, 488.1193848, -644.5452881, 573.2571411
3: -262.1003723, 429.0476685, -354.9754028, 581.0729370, -843.1390991, 784.0230713
4: -228.8475342, 413.9517517, -309.6556702, 558.2941284, -787.1416016, 723.6074219

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9606911, upper bound: 380.9623502
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9606910, upper bound: 380.9629603
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -110.6813202, 282.2884521, -106.0314560, 271.4934387, -382.1747437, 388.3198853
1: -278.7929382, 427.0088196, -267.2953186, 411.1235046, -689.9163818, 694.3040771
2: -180.3539886, 416.5377197, -172.7518311, 400.9807129, -581.3345947, 589.2895508
3: -302.2182922, 493.4462280, -289.7905884, 474.8612671, -777.0795898, 783.2368164
4: -263.3414307, 475.9180298, -252.1583557, 458.3327026, -721.6740723, 728.0761719

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9583160, upper bound: 380.9616469
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9583160, upper bound: 380.9626769
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -107.8633881, 275.3341370, -107.0197830, 275.0494080, -382.9127808, 382.3539124
1: -271.7055359, 416.5843811, -270.9668579, 416.3994446, -688.1049194, 687.5512695
2: -175.6613770, 406.1555786, -174.7664948, 406.6406860, -582.3020020, 580.9220581
3: -294.6877136, 481.3615112, -293.1515808, 481.1953125, -775.8830566, 774.5129395
4: -256.6172180, 464.2065430, -254.3286438, 464.8841248, -721.5013428, 718.5351562

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9611143, upper bound: 380.9612655
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9609737, upper bound: 380.9614870
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -105.7860489, 272.2588196, -109.3645401, 282.1088562, -387.8948669, 381.6232605
1: -267.1632385, 412.8865356, -277.2778320, 427.5610657, -694.7241821, 690.1643677
2: -172.0659790, 402.1796570, -178.5101471, 417.1060791, -589.1719971, 580.6898193
3: -289.4107056, 476.5856323, -299.6076660, 493.8732300, -783.2839355, 776.1931763
4: -251.2345886, 460.2764587, -259.7262573, 477.3754883, -728.6099854, 720.0026855

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9648534, upper bound: 380.9639791
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9647221, upper bound: 380.9643989
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -108.7049332, 277.4133606, -107.2803116, 275.5717163, -384.2766418, 384.6935730
1: -273.7506714, 419.6701660, -271.2087097, 417.1111755, -690.8618164, 690.8787842
2: -177.0416412, 409.2250061, -174.9613495, 407.3739929, -584.4156494, 584.1863403
3: -296.8702087, 484.9075012, -293.5977783, 482.0536194, -778.9238281, 778.5052490
4: -258.6519470, 467.6621094, -254.9761810, 465.6159058, -724.2678223, 722.6382446

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9623049, upper bound: 380.9619061
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9612301, upper bound: 380.9618190
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -106.6278839, 274.3546448, -109.4200363, 282.0893555, -388.7172241, 383.7746887
1: -269.2043762, 416.0025635, -277.0406494, 427.4492798, -696.6536865, 693.0432129
2: -173.4428711, 405.2853699, -178.3335876, 417.0323181, -590.4750977, 583.6189575
3: -291.6018372, 480.2009888, -299.5111389, 493.7831726, -785.3847656, 779.7120361
4: -253.2572327, 463.7536011, -259.8719177, 477.1968689, -730.4539795, 723.6254883

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9656588, upper bound: 380.9654176
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9657164, upper bound: 380.9658537
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -122.7994232, 313.5521240, -107.8358688, 278.1732483, -400.9726562, 421.3880005
1: -308.6789246, 475.4497070, -273.3842773, 421.4595947, -730.1385498, 748.8339844
2: -199.5384064, 462.3560791, -175.9767303, 411.4767456, -611.0151367, 638.3328247
3: -335.3286743, 550.0422974, -295.3074646, 486.8722839, -822.1799927, 844.7553101
4: -291.8786316, 528.9519653, -256.0256348, 470.7769775, -762.6556396, 784.9776001

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9557577, upper bound: 380.9471076
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9607341, upper bound: 380.9575913
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9598927, upper bound: 380.9575866
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -125.8302536, 321.1357422, -110.0561066, 283.7294312, -409.5596924, 431.1918335
1: -315.8032532, 486.9098816, -278.7264709, 429.8832703, -745.6865234, 765.6363525
2: -204.1827087, 473.0654602, -179.4244690, 419.5112915, -623.6939697, 652.4899292
3: -343.6520386, 563.1169434, -301.2751770, 496.6083679, -840.0853271, 863.8967285
4: -299.3747559, 541.2402344, -261.3454590, 479.9786987, -779.3532715, 802.5856323

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9605657, upper bound: 380.9586687
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9618653, upper bound: 380.9591287
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -106.2320404, 273.9281311, -99.4578857, 255.3170776, -361.5491333, 373.3860168
1: -268.8046875, 415.1528931, -250.9456177, 385.9647827, -654.7694702, 666.0985107
2: -173.1204376, 404.5828247, -161.9832306, 377.0029602, -550.1234131, 566.5660400
3: -290.7230835, 479.5168762, -271.8626099, 445.7557678, -736.4788818, 751.3794556
4: -252.4868011, 463.2273254, -236.4476318, 430.7850647, -683.2718506, 699.6748047

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9648614, upper bound: 380.9633785
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9647060, upper bound: 380.9632475
time: 1.17 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -125.3252640, 320.1826477, -99.4578857, 255.3170776, -380.6423035, 419.6405334
1: -314.6661377, 485.3239441, -250.9456177, 385.9647827, -700.6308594, 736.2695312
2: -203.5398102, 471.6437988, -161.9832306, 377.0029602, -580.5427246, 633.6270142
3: -342.1969604, 561.5889282, -271.8626099, 445.7557678, -787.7904663, 832.6295166
4: -298.2131348, 539.6555176, -236.4476318, 430.7850647, -728.9981689, 776.1031494

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9648614, upper bound: 380.9633785
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9647060, upper bound: 380.9632475
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -106.2320404, 273.9281311, -101.8801117, 261.7519836, -367.9840088, 375.8082275
1: -268.8046875, 415.1528931, -256.7658386, 395.8756104, -664.6802979, 671.9187012
2: -173.1204376, 404.5828247, -165.6361694, 386.3063049, -559.4267578, 570.2189941
3: -290.7230835, 479.5168762, -278.5220947, 456.9500427, -747.6730957, 758.0388794
4: -252.4868011, 463.2273254, -242.3437500, 441.4796753, -693.9664307, 705.5709839

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9650613, upper bound: 380.9637595
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9644092, upper bound: 380.9633456
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -125.3252640, 320.1826477, -101.8801117, 261.7519836, -387.0771790, 422.0627441
1: -314.6661377, 485.3239441, -256.7658386, 395.8756104, -710.5417480, 742.0897827
2: -203.5398102, 471.6437988, -165.6361694, 386.3063049, -589.8460693, 637.2799072
3: -342.1969604, 561.5889282, -278.5220947, 456.9500427, -798.7084961, 839.4671021
4: -298.2131348, 539.6555176, -242.3437500, 441.4796753, -739.6926880, 781.9992676

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9650613, upper bound: 380.9637595
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9644092, upper bound: 380.9633456
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -105.4898682, 271.8018799, -107.0324860, 275.4232788, -380.9130859, 378.8343506
1: -266.6603088, 411.8117981, -270.4471130, 417.2114258, -683.8716431, 682.2589111
2: -171.7721252, 402.0112915, -174.2743073, 407.3002625, -579.0723877, 576.2855835
3: -288.4289551, 475.6906738, -292.5140381, 481.9382935, -770.3671875, 768.2045898
4: -250.4819183, 459.6999207, -254.2630768, 465.6774902, -716.1594238, 713.9630127

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9560077, upper bound: 380.9565650
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9594671, upper bound: 380.9590576
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -123.1535492, 314.1776733, -107.0324860, 275.4232788, -398.5768433, 421.2101440
1: -308.6651611, 476.3504028, -270.4471130, 417.2114258, -725.8764648, 746.7974243
2: -199.7452393, 463.1007080, -174.2743073, 407.3002625, -607.0453491, 637.3748779
3: -336.0541992, 551.1054688, -292.5140381, 481.9382935, -817.4268799, 842.8625488
4: -293.0440369, 529.4884033, -254.2630768, 465.6774902, -758.7213745, 783.7514648

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9560077, upper bound: 380.9565650
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9594671, upper bound: 380.9590576
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -105.4898682, 271.8018799, -111.5034637, 286.9665833, -392.4563904, 383.3053589
1: -266.6603088, 411.8117981, -282.3046265, 434.7135925, -701.3738403, 694.1164551
2: -171.7721252, 402.0112915, -181.9177094, 424.1986694, -595.9708252, 583.9290161
3: -288.4289551, 475.6906738, -305.0166626, 502.2409363, -790.6697388, 780.7070312
4: -250.4819183, 459.6999207, -264.9156189, 485.3171997, -735.7991333, 724.6155396

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9612764, upper bound: 380.9621301
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9628040, upper bound: 380.9624982
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -123.1535492, 314.1776733, -111.5034637, 286.9665833, -410.1201172, 425.6811523
1: -308.6651611, 476.3504028, -282.3046265, 434.7135925, -743.3787231, 758.6550293
2: -199.7452393, 463.1007080, -181.9177094, 424.1986694, -623.9438477, 645.0183105
3: -336.0541992, 551.1054688, -305.0166626, 502.2409363, -837.4172363, 855.4412231
4: -293.0440369, 529.4884033, -264.9156189, 485.3171997, -778.3610840, 794.4040527

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9612764, upper bound: 380.9621301
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9628040, upper bound: 380.9624982
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -107.6499252, 277.4054871, -108.9491272, 280.1452942, -387.7951660, 386.3545837
1: -272.8387451, 420.3706970, -277.3763123, 424.4076843, -697.2464600, 697.7470093
2: -175.7212982, 410.3366394, -178.6744690, 414.2894897, -590.0108032, 589.0111084
3: -294.6065369, 485.6777954, -298.2582703, 490.9793701, -785.5859375, 783.9360352
4: -255.5363770, 469.5085144, -258.5288696, 474.7673035, -730.3035278, 728.0373535

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663672, upper bound: 380.9658418
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9660596, upper bound: 380.9657476
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9629596, upper bound: 380.9641289
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9664550, upper bound: 380.9651052
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -124.6903076, 318.3165894, -108.9491272, 280.1452942, -404.8356018, 427.2657166
1: -312.8399658, 482.5302124, -277.3763123, 424.4076843, -737.2476196, 759.7819214
2: -202.4766388, 469.2543945, -178.6744690, 414.2894897, -616.7661133, 647.9288330
3: -340.1546936, 558.3353882, -298.2582703, 490.9793701, -830.3115845, 855.8569946
4: -296.7083435, 536.5305786, -258.5288696, 474.7673035, -771.4754639, 795.0593872

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663672, upper bound: 380.9658418
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9660596, upper bound: 380.9657476
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9629596, upper bound: 380.9641289
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9664550, upper bound: 380.9651052
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -109.0328903, 280.8949585, -110.4797821, 284.2623596, -393.2952271, 391.3747253
1: -276.0534363, 425.6777039, -279.5233459, 430.7288208, -706.7822266, 705.2010498
2: -177.8319244, 415.2210999, -180.1308441, 420.0856018, -597.9175415, 595.3517456
3: -298.3986511, 491.7099304, -302.2250977, 497.5073242, -795.9059448, 793.9350586
4: -258.9868164, 475.1401062, -262.5693054, 480.6873779, -739.6741333, 737.7092896

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9649450, upper bound: 380.9635083
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9649450, upper bound: 380.9659968
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -126.4421921, 322.8362732, -110.4797821, 284.2623596, -410.7045593, 433.3160400
1: -317.1414185, 489.3787231, -279.5233459, 430.7288208, -747.8702393, 768.9020996
2: -205.2606354, 475.6784668, -180.1308441, 420.0856018, -625.3462524, 655.8091431
3: -345.0083618, 566.1856689, -302.2250977, 497.5073242, -841.8560181, 867.5738525
4: -300.9461975, 543.9959106, -262.5693054, 480.6873779, -781.6335449, 806.5651855

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9649450, upper bound: 380.9635083
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9649450, upper bound: 380.9659968
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -106.6207504, 273.4873657, -126.7825165, 323.4387817, -430.0595398, 400.2698975
1: -270.5278931, 414.6541138, -318.3457642, 490.1509094, -760.6786499, 732.9998779
2: -174.7986755, 404.7173462, -206.1176910, 476.8311462, -651.6298218, 610.8350220
3: -291.9535217, 479.7592468, -345.8911133, 567.1911621, -858.9061890, 825.3847656
4: -253.0793610, 463.0970154, -301.7779236, 545.1773071, -798.2565918, 764.8749390

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9651955, upper bound: 380.9641451
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9651955, upper bound: 380.9641451
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -104.2301178, 267.4569702, -126.7825165, 323.4387817, -427.6688843, 394.2395020
1: -264.1344299, 405.4380188, -318.3457642, 490.1509094, -754.2852173, 723.7836304
2: -170.5861053, 395.8929749, -206.1176910, 476.8311462, -647.4171753, 602.0105591
3: -285.3962708, 469.1010742, -345.8911133, 567.1911621, -852.3626099, 814.8976440
4: -247.4682770, 452.8132019, -301.7779236, 545.1773071, -792.6454468, 754.5911255

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9651955, upper bound: 380.9641451
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9651955, upper bound: 380.9641451
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -106.6207504, 273.4873657, -124.7954178, 318.2304077, -424.8511658, 398.2827759
1: -270.5278931, 414.6541138, -313.1323853, 482.2271118, -752.7550049, 727.7864990
2: -174.7986755, 404.7173462, -202.7620850, 469.0730591, -643.8717041, 607.4793091
3: -291.9535217, 479.7592468, -340.4318848, 558.0941772, -849.9515381, 819.9053345
4: -253.0793610, 463.0970154, -297.0601807, 536.2368774, -789.3162231, 760.1571655

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9641060, upper bound: 380.9641060
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9641060, upper bound: 380.9641060
time: 0.73 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.82 + 418.11 = 420.93 seconds
