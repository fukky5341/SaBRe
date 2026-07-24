## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 339.77104719722996


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423)
1: (-124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621)
2: (-105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148)
3: (-110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960)
4: (-94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.77 + 2.25 = 3.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -339.8050277, upper bound: 339.8050277

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8041198, upper bound: 339.8047811
time: 1.07 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8040862, upper bound: 339.8040862
time: 0.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.00 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.00
Output dim: 0, lower bound: -339.8041198, upper bound: 339.8047811
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.00
Output dim: 0, lower bound: -339.8040862, upper bound: 339.8040862

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -85.0949707, 284.1704712, -87.5877380, 293.3702698, -378.4652100, 371.7582092
1: -119.3920059, 282.0884705, -122.9078217, 291.1408081, -410.5327759, 404.9962158
2: -101.2751236, 310.7092590, -104.2449875, 320.6318665, -421.9069214, 414.9542542
3: -106.2089005, 403.8296509, -109.3434143, 416.6802368, -522.8891602, 513.1729126
4: -90.6926956, 367.2429504, -93.3429947, 378.7810364, -469.4737244, 460.5859375

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7994896, upper bound: 339.7910547
time: 0.81 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7994896, upper bound: 339.8019178
time: 0.89 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -85.5067825, 287.2420959, -87.4125977, 293.1314087, -378.6381836, 374.6546936
1: -120.1114273, 284.9082947, -122.7049026, 290.8285217, -410.9399414, 407.6131897
2: -101.8361511, 313.7402954, -104.0567322, 320.2686768, -422.1048279, 417.7970276
3: -106.8467102, 407.9730225, -109.1592331, 416.3205872, -523.1672974, 517.1322632
4: -91.2148666, 370.7578735, -93.1842804, 378.4085083, -469.6233826, 463.9421387

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7981138, upper bound: 339.7909900
time: 0.91 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8009989, upper bound: 339.8009989
time: 0.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.43 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -339.7994896, upper bound: 339.7910547
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -339.7994896, upper bound: 339.8019178
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -339.7981138, upper bound: 339.7909900
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -339.8009989, upper bound: 339.8009989

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -81.0273743, 271.4459229, -79.4728928, 268.5468445, -349.5741577, 350.9188232
1: -113.6279831, 269.3906860, -111.4451370, 266.2232056, -379.8511047, 380.8358154
2: -96.3576508, 296.7964478, -94.4518890, 293.2672729, -389.6249390, 391.2483215
3: -101.1272354, 385.9545898, -99.2555389, 381.6227722, -482.7500000, 485.2101440
4: -86.3563309, 350.9807739, -84.7434769, 346.7669067, -433.1232300, 435.7242432

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7976358, upper bound: 339.7890603
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7953676, upper bound: 339.7887794
time: 1.03 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -84.7099152, 282.7786865, -86.7345047, 290.4049072, -375.1148071, 369.5131836
1: -118.8574142, 280.7217407, -121.7352142, 288.2131042, -407.0704346, 402.4569397
2: -100.8271713, 309.2071228, -103.2595596, 317.4109497, -418.2381287, 412.4666443
3: -105.7317886, 401.8627930, -108.2971039, 412.4870605, -518.2187500, 510.1599121
4: -90.2925262, 365.4902954, -92.4653625, 375.0029297, -465.2954102, 457.9556580

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7999681, upper bound: 339.7966620
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7972437, upper bound: 339.7963300
time: 0.95 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -81.6167603, 274.8861084, -79.5408783, 268.9675903, -350.5843506, 354.4270020
1: -114.5669022, 272.5852661, -111.5551987, 266.5870667, -381.1539612, 384.1404724
2: -97.1122131, 300.2497559, -94.5337830, 293.6547546, -390.7669373, 394.7835388
3: -101.9644089, 390.6507568, -99.3474731, 382.1955566, -484.1599731, 489.9982300
4: -87.0539474, 354.9667053, -84.8192596, 347.2721863, -434.3261414, 439.7859497

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7954838, upper bound: 339.7881552
time: 1.30 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7933735, upper bound: 339.7878474
time: 0.78 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -84.9605026, 285.3676147, -86.4718323, 289.8807678, -374.8412476, 371.8394165
1: -119.3579559, 283.0553589, -121.4108582, 287.6170654, -406.9750366, 404.4662170
2: -101.2033691, 311.7028809, -102.9692993, 316.7373352, -417.9407043, 414.6721802
3: -106.1753922, 405.3184509, -108.0052643, 411.7192688, -517.8945923, 513.3236694
4: -90.6518250, 368.3587952, -92.2150192, 374.2563171, -464.9080811, 460.5738220

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7978629, upper bound: 339.7956360
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7933735, upper bound: 339.7952777
time: 0.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.58 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -339.7976358, upper bound: 339.7890603
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -339.7953676, upper bound: 339.7887794
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -339.7999681, upper bound: 339.7966620
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -339.7972437, upper bound: 339.7963300
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -339.7954838, upper bound: 339.7881552
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -339.7933735, upper bound: 339.7878474
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -339.7978629, upper bound: 339.7956360
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -339.7933735, upper bound: 339.7952777

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -77.7003784, 259.7509460, -77.3642044, 261.2030640, -338.9034424, 337.1151123
1: -108.9674988, 257.7982788, -108.5003510, 258.9410706, -367.9085693, 366.2985535
2: -92.4377670, 283.9990540, -91.9659042, 285.2431641, -377.6808777, 375.9649658
3: -96.9742126, 369.1553345, -96.6317444, 371.0701599, -468.0443726, 465.7870178
4: -82.8533096, 335.8280029, -82.5157242, 337.2545776, -420.1078796, 418.3437195

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7755757
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -84.9602509, 282.9266968, -76.9909897, 259.9263916, -344.8866272, 359.9176636
1: -119.2234039, 280.8399963, -107.8894043, 257.6838684, -376.9072876, 388.7293701
2: -101.1231308, 309.4814148, -91.4469147, 283.9138184, -385.0369568, 400.9282532
3: -106.0317459, 401.8394165, -96.0956802, 369.3587646, -475.3905029, 497.9350586
4: -90.4856415, 365.9111633, -82.0592270, 335.7166443, -426.2022705, 447.9703979

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7910671, upper bound: 339.7842595
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7912053, upper bound: 339.7830153
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -81.4270096, 271.2620239, -84.7629852, 283.5292053, -364.9562073, 356.0249939
1: -114.2463760, 269.3209534, -118.9678726, 281.3923645, -395.6387329, 388.2888184
2: -96.9507446, 296.6494141, -100.9265823, 309.8970947, -406.8478088, 397.5759583
3: -101.6242142, 385.2831116, -105.8329086, 402.5982361, -504.2224121, 491.1160278
4: -86.8287430, 350.5486755, -90.3824387, 366.0799255, -452.9086304, 440.9310913

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7875591
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878528, upper bound: 339.7853487
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -88.5550766, 293.9342957, -84.0321198, 280.9253845, -369.4804688, 377.9663696
1: -124.2826233, 291.8616333, -117.8661575, 278.8472900, -403.1299133, 409.7277832
2: -105.4521561, 321.6031799, -99.9983368, 307.1615295, -412.6136780, 421.6015015
3: -110.4850540, 417.1892090, -104.8643723, 398.9664917, -509.4515381, 522.0535889
4: -94.2954025, 379.9549561, -89.5558853, 362.8458252, -457.1412354, 469.5108337

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7871465, upper bound: 339.7882640
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7868212, upper bound: 339.7851916
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -78.2733307, 263.2520447, -77.4189377, 261.5864563, -339.8598022, 340.6709595
1: -109.9019012, 261.0475464, -108.5918503, 259.2667847, -369.1686707, 369.6394043
2: -93.1696320, 287.5014343, -92.0304794, 285.5874939, -378.7571411, 379.5319214
3: -97.8078537, 373.9039612, -96.7070999, 371.5886841, -469.3965454, 470.6110535
4: -83.5288467, 339.8666382, -82.5761795, 337.7084656, -421.2373047, 422.4428101

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7817232, upper bound: 339.7764995
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7816871, upper bound: 339.7728223
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -85.3219757, 285.7596436, -77.0309753, 260.2617493, -345.5837097, 362.7905884
1: -119.8748550, 283.4420471, -107.9637146, 257.9598694, -377.8346252, 391.4057617
2: -101.6112900, 312.2992249, -91.4974823, 284.2034912, -385.8147888, 403.7966919
3: -106.6030655, 405.6167603, -96.1560516, 369.8211365, -476.4241943, 501.7728271
4: -90.9441605, 369.1127319, -82.1149826, 336.1181335, -427.0622559, 451.2277222

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7807334, upper bound: 339.7772110
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7804529, upper bound: 339.7726338
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -81.6745148, 273.9215393, -84.4935074, 282.9934692, -364.6679688, 358.4149475
1: -114.7613297, 271.7012329, -118.6365662, 280.7814941, -395.5428162, 390.3377686
2: -97.3231659, 299.1925964, -100.6289749, 309.2057190, -406.5288696, 399.8215332
3: -102.0799103, 388.8654480, -105.5345612, 401.8149414, -503.8948364, 494.4000244
4: -87.1870270, 353.5149536, -90.1263351, 365.3185425, -452.5055542, 443.6412354

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847890, upper bound: 339.7875964
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7844800, upper bound: 339.7828899
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -88.8190842, 296.5412292, -83.7410355, 280.3356628, -369.1547546, 380.2822571
1: -124.8070374, 294.2322998, -117.5065994, 278.1751709, -402.9822083, 411.7388916
2: -105.8310242, 324.1399231, -99.6743469, 306.3990173, -412.2300415, 423.8142090
3: -110.9448700, 420.7775269, -104.5397644, 398.1135254, -509.0584106, 525.3172607
4: -94.6583328, 382.9426270, -89.2750473, 362.0144043, -456.6727295, 472.2176819

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7807334, upper bound: 339.7877076
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7826346, upper bound: 339.7826346
time: 0.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.55 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7755757
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7910671, upper bound: 339.7842595
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7912053, upper bound: 339.7830153
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7875591
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7878528, upper bound: 339.7853487
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7871465, upper bound: 339.7882640
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7868212, upper bound: 339.7851916
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7817232, upper bound: 339.7764995
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7816871, upper bound: 339.7728223
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7807334, upper bound: 339.7772110
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7804529, upper bound: 339.7726338
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7847890, upper bound: 339.7875964
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7844800, upper bound: 339.7828899
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7807334, upper bound: 339.7877076
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -339.7826346, upper bound: 339.7826346

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -76.6747284, 256.5736389, -69.4078522, 235.1170502, -311.7916870, 325.9815063
1: -107.5928802, 254.6033630, -97.4621124, 233.0818634, -340.6747131, 352.0654602
2: -91.2643661, 280.4896545, -82.5630493, 256.7786560, -348.0430298, 363.0526733
3: -95.7421341, 364.5959778, -86.7975616, 334.1309814, -429.8731079, 451.3935547
4: -81.8008347, 331.6852417, -74.1446838, 303.6525269, -385.4533691, 405.8299255

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -75.5153275, 252.7481537, -73.0328522, 247.3614044, -322.8767395, 325.7809448
1: -105.7958374, 250.8136749, -102.2071838, 245.1500397, -350.9458618, 353.0208740
2: -89.7353287, 276.3110046, -86.6180420, 270.0850525, -359.8203735, 362.9289856
3: -94.1687775, 359.2999878, -91.0725708, 351.5776062, -445.7463989, 450.3725586
4: -80.4592285, 326.7872314, -77.8730087, 319.4046936, -399.8639221, 404.6602173

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7840463, upper bound: 339.7737438
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7840463, upper bound: 339.7737438
time: 1.31 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -83.6920242, 278.4622803, -72.2531967, 242.6130981, -326.3051147, 350.7154541
1: -117.4295731, 276.4407349, -100.5072174, 240.6424103, -358.0719604, 376.9479370
2: -99.6123276, 304.6630859, -85.2554398, 265.2538452, -364.8661194, 389.9185181
3: -104.4366150, 395.4643250, -89.6179657, 345.0139160, -449.4505310, 485.0822754
4: -89.1376114, 360.1421814, -76.7632217, 313.5218201, -402.6594238, 436.9053955

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7910177, upper bound: 339.7840661
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7910671, upper bound: 339.7842595
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -81.7373047, 272.3984985, -71.5301666, 241.5511169, -323.2884216, 343.9286194
1: -114.6030579, 270.6612244, -99.9368896, 239.9403687, -354.5434265, 370.5981140
2: -97.1505280, 298.3820801, -84.6651688, 264.5567932, -361.7073364, 383.0471802
3: -101.9924622, 387.3707886, -89.1217651, 344.0275269, -446.0199890, 476.4925537
4: -87.0233307, 352.7610779, -76.1830292, 312.8172607, -399.8405762, 428.9440918

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7846108, upper bound: 339.7683063
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7785737, upper bound: 339.7683894
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -80.4556808, 268.1766357, -76.7714005, 257.1851807, -337.6408386, 344.9480286
1: -112.9134216, 266.2287598, -107.7800674, 255.2827301, -368.1961670, 374.0087585
2: -95.8130035, 293.2504272, -91.3978348, 281.1499023, -376.9628906, 384.6482544
3: -100.4298248, 380.8751831, -95.8695374, 365.2928162, -465.7226562, 476.7446899
4: -85.8077087, 346.5280762, -81.8904114, 332.0989685, -417.9066772, 428.4183960

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
time: 1.24 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -79.2153702, 264.3002014, -80.4041138, 269.8348389, -349.0501404, 344.7043152
1: -111.1017075, 262.3894348, -112.7396698, 267.7330933, -378.8347778, 375.1290894
2: -94.2716599, 288.9962463, -95.6198196, 294.8318787, -389.1035461, 384.6159363
3: -98.8416214, 375.4968872, -100.3256454, 383.3066406, -482.1482544, 475.8225403
4: -84.4447632, 341.5718384, -85.6893311, 348.3633118, -432.8080750, 427.2611694

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878528, upper bound: 339.7853487
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878528, upper bound: 339.7853487
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -87.5744858, 290.7147217, -76.2843323, 255.3230743, -342.8975525, 366.9990540
1: -122.9220505, 288.6686401, -107.0297165, 253.5550995, -376.4771423, 395.6983643
2: -104.2951431, 318.0904236, -90.7616196, 279.3072205, -383.6023560, 408.8520508
3: -109.2712936, 412.6036682, -95.2181625, 362.6983643, -471.9696350, 507.8218384
4: -93.2599945, 375.7788086, -81.3283539, 329.8513489, -423.1113281, 457.1071472

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7542534, upper bound: 339.7756255
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7542534, upper bound: 339.7882640
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -86.1584396, 286.5064392, -79.8160553, 267.7255859, -353.8840332, 366.3225098
1: -120.9377975, 284.4754333, -111.8140259, 265.6779175, -386.6156921, 396.2894287
2: -102.5994263, 313.4352417, -94.8400192, 292.6589966, -395.2584229, 408.2752686
3: -107.5202637, 406.7664490, -99.5110092, 380.3600769, -487.8803406, 506.2774353
4: -91.7552719, 370.3927917, -85.0165176, 345.7670898, -437.5222778, 455.4093018

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7534731, upper bound: 339.7736242
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7534731, upper bound: 339.7851916
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -77.2548828, 260.1128235, -69.4035492, 235.2939453, -312.5488281, 329.5163574
1: -108.5510941, 257.8930969, -97.4747314, 233.2010803, -341.7521667, 355.3677979
2: -92.0158615, 284.0333557, -82.5573730, 256.9002991, -348.9161682, 366.5906982
3: -96.5964203, 369.4042053, -86.8042603, 334.3646545, -430.9610596, 456.2084656
4: -82.4929428, 335.7781677, -74.1432266, 303.8547974, -386.3477173, 409.9213562

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764730
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764995
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -76.1860657, 256.5781555, -73.0592804, 247.6810913, -323.8671570, 329.6374207
1: -106.8584824, 254.3876190, -102.2596817, 245.4048767, -352.2633667, 356.6473083
2: -90.5805664, 280.1688232, -86.6547241, 270.3575134, -360.9380798, 366.8235168
3: -95.1216278, 364.5124207, -91.1154327, 352.0030212, -447.1246338, 455.6278381
4: -81.2552109, 331.2555237, -77.9133224, 319.7744141, -401.0296326, 409.1688538

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
time: 1.23 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7728223
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -84.3186646, 282.5124207, -69.0703354, 234.1050568, -318.4237061, 351.5827637
1: -118.4889908, 280.2141113, -96.9434433, 232.1126099, -350.6015930, 377.1575623
2: -100.4323807, 308.7423401, -82.1017456, 255.7478333, -356.1802063, 390.8440857
3: -105.3661194, 400.9823303, -86.3438950, 332.7772217, -438.1433411, 487.3262329
4: -89.8884048, 364.8887329, -73.7513809, 302.4556274, -392.3440247, 438.6400452

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7645069, upper bound: 339.7489792
time: 1.22 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7714351, upper bound: 339.7764883
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7714351, upper bound: 339.7772110
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -83.0648804, 278.8061523, -72.8479538, 247.0541229, -330.1189880, 351.6541138
1: -116.7155914, 276.5238953, -101.8806534, 244.7858429, -361.5014343, 378.4045410
2: -98.9170990, 304.6532593, -86.3351059, 269.7480469, -368.6651611, 390.9883728
3: -103.8033066, 395.8731995, -90.7877808, 351.2036743, -455.0069885, 486.6609802
4: -88.5471954, 360.1473083, -77.6636581, 319.0796204, -407.6268005, 437.8109131

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7661746, upper bound: 339.7479243
time: 1.33 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7705713, upper bound: 339.7705714
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7705713, upper bound: 339.7726338
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -80.6942368, 270.8151245, -76.5349884, 256.7867737, -337.4810181, 347.3500977
1: -113.4213867, 268.5936584, -107.5017776, 254.7888947, -368.2102661, 376.0954285
2: -96.1806412, 295.7700500, -91.1387100, 280.5902100, -376.7707825, 386.9087524
3: -100.8810043, 384.4285278, -95.6157379, 364.6956787, -465.5766907, 480.0441895
4: -86.1621628, 349.4748230, -81.6669617, 331.5134277, -417.6755676, 431.1417847

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7820815
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7875860
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -79.5315857, 267.1955261, -80.1331406, 269.2861938, -348.8177795, 347.3286438
1: -111.6940842, 264.9929504, -112.3996048, 267.1072998, -378.8013916, 377.3925476
2: -94.7097931, 291.7889709, -95.3169937, 294.1314697, -388.8412476, 387.1059570
3: -99.3687973, 379.3954163, -100.0205536, 382.5018921, -481.8706970, 479.4159546
4: -84.8625107, 344.8090515, -85.4314728, 347.5932617, -432.4557495, 430.2405396

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7801626
time: 1.25 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7828899
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -87.8504639, 293.3922729, -76.0437393, 254.9332275, -342.7836914, 369.4360046
1: -123.4660645, 291.1073303, -106.7489624, 253.0709229, -376.5369568, 397.8562927
2: -104.6900940, 320.6931458, -90.5001068, 278.7565002, -383.4465637, 411.1931763
3: -109.7491074, 416.2899780, -94.9625549, 362.1279602, -471.8770752, 511.2525330
4: -93.6369247, 378.8467712, -81.1034927, 329.2878113, -422.9247131, 459.9502563

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7714351, upper bound: 339.7848819
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7714351, upper bound: 339.7877076
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -86.5557785, 289.6101685, -79.5781326, 267.2933350, -353.8491211, 369.1882935
1: -121.6444931, 287.3266907, -111.5234375, 265.1660461, -386.8105469, 398.8500977
2: -103.1306381, 316.5052185, -94.5764923, 292.0809021, -395.2114868, 411.0817261
3: -108.1412582, 411.0407410, -99.2505035, 379.7619324, -487.9031982, 510.2912292
4: -92.2553558, 373.9890442, -84.7941284, 345.1743164, -437.4296875, 458.7831726

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7705713, upper bound: 339.7804528
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7705713, upper bound: 339.7826346
time: 1.39 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.86 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7840463, upper bound: 339.7737438
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7840463, upper bound: 339.7737438
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7910177, upper bound: 339.7840661
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7910671, upper bound: 339.7842595
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7846108, upper bound: 339.7683063
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7785737, upper bound: 339.7683894
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7878528, upper bound: 339.7853487
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7878528, upper bound: 339.7853487
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7542534, upper bound: 339.7756255
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7542534, upper bound: 339.7882640
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7534731, upper bound: 339.7736242
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7534731, upper bound: 339.7851916
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764730
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764995
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7728223
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7714351, upper bound: 339.7764883
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7714351, upper bound: 339.7772110
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7705713, upper bound: 339.7705714
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7705713, upper bound: 339.7726338
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7820815
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7875860
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7801626
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7828899
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7714351, upper bound: 339.7848819
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7714351, upper bound: 339.7877076
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7705713, upper bound: 339.7804528
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -339.7705713, upper bound: 339.7826346

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -69.5513840, 232.9595032, -69.4078522, 235.1170502, -304.6684265, 302.3673706
1: -97.5931473, 231.2630310, -97.4621124, 233.0818634, -330.6749878, 328.7251587
2: -82.7497482, 254.8089752, -82.5630493, 256.7786560, -339.5284119, 337.3720093
3: -86.8460999, 331.2365417, -86.7975616, 334.1309814, -420.9770508, 418.0340881
4: -74.2284851, 301.3083496, -74.1446838, 303.6525269, -377.8809509, 375.4530029

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7755757
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7755757
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -73.2851791, 245.6114960, -69.4078522, 235.1170502, -308.4022217, 315.0193481
1: -102.5696869, 243.6932068, -97.4621124, 233.0818634, -335.6514587, 341.1553345
2: -86.9982605, 268.5047607, -82.5630493, 256.7786560, -343.7769165, 351.0678101
3: -91.3169250, 349.2504578, -86.7975616, 334.1309814, -425.4478760, 436.0480347
4: -78.0879364, 317.6050415, -74.1446838, 303.6525269, -381.7404175, 391.7497253

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7755757
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7755757
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -69.5513840, 232.9595032, -73.0328522, 247.3614044, -316.9127808, 305.9923706
1: -97.5931473, 231.2630310, -102.2071838, 245.1500397, -342.7431946, 333.4701843
2: -82.7497482, 254.8089752, -86.6180420, 270.0850525, -352.8347778, 341.4270020
3: -86.8460999, 331.2365417, -91.0725708, 351.5776062, -438.4236755, 422.3091125
4: -74.2284851, 301.3083496, -77.8730087, 319.4046936, -393.6331177, 379.1813354

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -73.2851791, 245.6114960, -73.0328522, 247.3614044, -320.6465759, 318.6443481
1: -102.5696869, 243.6932068, -102.2071838, 245.1500397, -347.7196960, 345.9003906
2: -86.9982605, 268.5047607, -86.6180420, 270.0850525, -357.0833130, 355.1227722
3: -91.3169250, 349.2504578, -91.0725708, 351.5776062, -442.8945007, 440.3230286
4: -78.0879364, 317.6050415, -77.8730087, 319.4046936, -397.4925842, 395.4780273

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -73.7456741, 246.8744202, -70.8726807, 237.8838959, -311.6294861, 317.7471008
1: -103.3255692, 244.9274597, -98.5952377, 235.9766235, -339.3020935, 343.5226440
2: -87.6685181, 270.0310974, -83.6537781, 260.1207886, -347.7893066, 353.6848755
3: -91.9267197, 350.9618835, -87.9100800, 338.2779541, -430.2046509, 438.8719482
4: -78.5416565, 319.5290222, -75.3193741, 307.4041748, -385.9458313, 394.8483276

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7909926, upper bound: 339.7840661
time: 1.24 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7909926, upper bound: 339.7840661
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -81.7432098, 271.9115295, -71.3849182, 239.7501831, -321.4934082, 343.2964478
1: -114.6574097, 269.9497986, -99.2537994, 237.7892761, -352.4466858, 369.2036133
2: -97.2616806, 297.5369873, -84.1946716, 262.1266479, -359.3883057, 381.7316284
3: -101.9812241, 386.0803833, -88.5113297, 340.9242859, -442.9055176, 474.5917053
4: -87.0161133, 351.6328735, -75.8290710, 309.7973328, -396.8134460, 427.4619141

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7910401, upper bound: 339.7842595
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7910401, upper bound: 339.7842595
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -74.5713882, 249.1286316, -70.4475174, 238.2738190, -312.8452148, 319.5761414
1: -104.5754700, 247.5596771, -98.5527115, 236.6694641, -341.2449341, 346.1123962
2: -88.6184082, 272.9008179, -83.4730530, 260.9183350, -349.5367432, 356.3738708
3: -93.0721588, 354.3959351, -87.8776550, 339.3608398, -432.4329224, 442.2735901
4: -79.4293060, 322.6373291, -75.0504303, 308.5489502, -387.9782410, 397.6877441

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7781413, upper bound: 339.7527548
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7781413, upper bound: 339.7683063
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -77.1751251, 257.7632141, -69.3447876, 234.6033936, -311.7785034, 327.1079712
1: -107.9988480, 256.1731873, -96.7582016, 233.0261993, -341.0250549, 352.9313354
2: -91.5330811, 282.3934937, -81.9662399, 256.9747925, -348.5078430, 364.3597412
3: -96.1434250, 366.8759155, -86.3112640, 334.2744141, -430.4178467, 453.1871338
4: -82.0976028, 333.9836121, -73.8565369, 303.8765564, -385.9741516, 407.8401184

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7394499, upper bound: 339.7515716
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7390666, upper bound: 339.7421101
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -73.3381424, 244.6101532, -76.7714005, 257.1851807, -330.5233154, 321.3815613
1: -102.9172745, 242.8930969, -107.7800674, 255.2827301, -358.2000122, 350.6731567
2: -87.3073273, 267.5639648, -91.3978348, 281.1499023, -368.4572144, 358.9617920
3: -91.5368042, 347.4905701, -95.8695374, 365.2928162, -456.8296204, 443.3600769
4: -78.2409210, 316.1669312, -81.8904114, 332.0989685, -410.3398743, 398.0573425

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7875591
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7875591
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -76.9758148, 257.1091003, -76.7714005, 257.1851807, -334.1609802, 333.8804932
1: -107.8613968, 255.2158203, -107.7800674, 255.2827301, -363.1441345, 362.9958801
2: -91.5137939, 281.1018677, -91.3978348, 281.1499023, -372.6636353, 372.4996948
3: -95.9746094, 365.3651733, -95.8695374, 365.2928162, -461.2674255, 461.2346802
4: -82.0231857, 332.3006287, -81.8904114, 332.0989685, -414.1221619, 414.1909790

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7875591
time: 1.08 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7875591
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -73.3381424, 244.6101532, -80.4041138, 269.8348389, -343.1729736, 325.0142822
1: -102.9172745, 242.8930969, -112.7396698, 267.7330933, -370.6503296, 355.6327515
2: -87.3073273, 267.5639648, -95.6198196, 294.8318787, -382.1391907, 363.1836853
3: -91.5368042, 347.4905701, -100.3256454, 383.3066406, -474.8434448, 447.8161926
4: -78.2409210, 316.1669312, -85.6893311, 348.3633118, -426.6042175, 401.8562622

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -76.9758148, 257.1091003, -80.4041138, 269.8348389, -346.8106079, 337.5132141
1: -107.8613968, 255.2158203, -112.7396698, 267.7330933, -375.5944824, 367.9554749
2: -91.5137939, 281.1018677, -95.6198196, 294.8318787, -386.3456726, 376.7216187
3: -95.9746094, 365.3651733, -100.3256454, 383.3066406, -479.2812500, 465.6908264
4: -82.0231857, 332.3006287, -85.6893311, 348.3633118, -430.3865051, 417.9899597

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -80.0753098, 268.5890198, -76.2843323, 255.3230743, -335.3983765, 344.8733521
1: -112.4296036, 266.3736267, -107.0297165, 253.5550995, -365.9847107, 373.4033508
2: -95.3291550, 293.5993958, -90.7616196, 279.3072205, -374.6363525, 384.3610229
3: -100.0343323, 381.2966309, -95.2181625, 362.6983643, -462.7326965, 476.5148010
4: -85.3743134, 347.0550537, -81.3283539, 329.8513489, -415.2256470, 428.3833923

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7540060, upper bound: 339.7756255
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7540060, upper bound: 339.7756255
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -87.1023941, 288.9736328, -76.2843323, 255.3230743, -342.4254761, 365.2579651
1: -122.2498779, 286.9699707, -107.0297165, 253.5550995, -375.8049927, 393.9996948
2: -103.7315826, 316.2297668, -90.7616196, 279.3072205, -383.0387878, 406.9913940
3: -108.6728821, 410.1506653, -95.2181625, 362.6983643, -471.3712463, 505.3688354
4: -92.7565842, 373.5724182, -81.3283539, 329.8513489, -422.6079407, 454.9007568

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7540060, upper bound: 339.7882640
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7540060, upper bound: 339.7882640
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -78.6936035, 264.4166870, -79.8160553, 267.7255859, -346.4191895, 344.2327271
1: -110.4570236, 262.2425232, -111.8140259, 265.6779175, -376.1349487, 374.0564880
2: -93.6443024, 289.0195618, -94.8400192, 292.6589966, -386.3032837, 383.8595886
3: -98.2981262, 375.5130920, -99.5110092, 380.3600769, -478.6582031, 475.0241089
4: -83.8868637, 341.7197571, -85.0165176, 345.7670898, -429.6539001, 426.7362671

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7534731, upper bound: 339.7736242
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7534731, upper bound: 339.7736242
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -85.6336975, 284.6076965, -79.8160553, 267.7255859, -353.3592529, 364.4237671
1: -120.1997910, 282.6163025, -111.8140259, 265.6779175, -385.8777161, 394.4303284
2: -101.9800720, 311.3958435, -94.8400192, 292.6589966, -394.6390076, 406.2358704
3: -106.8617401, 404.0912170, -99.5110092, 380.3600769, -487.2218018, 503.6022034
4: -91.2010727, 367.9867859, -85.0165176, 345.7670898, -436.9681396, 453.0032959

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7534731, upper bound: 339.7851916
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7534731, upper bound: 339.7851916
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -73.4968796, 249.0412445, -69.4035492, 235.2939453, -308.7908325, 318.4447937
1: -103.2814026, 246.7286530, -97.4747314, 233.2010803, -336.4824524, 344.2033691
2: -87.5033264, 271.7731018, -82.5573730, 256.9002991, -344.4036255, 354.3304749
3: -91.9600143, 353.6892395, -86.8042603, 334.3646545, -426.3246765, 440.4934998
4: -78.5316086, 321.4323425, -74.1432266, 303.8547974, -382.3863831, 395.5755615

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764730
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764730
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -80.1220779, 268.9065857, -69.4035492, 235.2939453, -315.4160156, 338.3101196
1: -112.6305847, 266.6913757, -97.4747314, 233.2010803, -345.8315735, 364.1660767
2: -95.5148392, 293.6755676, -82.5573730, 256.9002991, -352.4151306, 376.2329407
3: -100.1756516, 381.7141724, -86.8042603, 334.3646545, -434.5402832, 468.5184326
4: -85.5672302, 347.0112000, -74.1432266, 303.8547974, -389.4220276, 421.1544189

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764995
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764995
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -72.2763596, 244.9064636, -73.0592804, 247.6810913, -319.9574585, 317.9657288
1: -101.3383942, 242.6434326, -102.2596817, 245.4048767, -346.7432861, 344.9031067
2: -85.8648529, 267.2776184, -86.6547241, 270.3575134, -356.2223511, 353.9323425
3: -90.2656708, 347.9327393, -91.1154327, 352.0030212, -442.2686768, 439.0481567
4: -77.1345291, 316.1322021, -77.9133224, 319.7744141, -396.9089355, 394.0455322

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -78.8918686, 265.1601868, -73.0592804, 247.6810913, -326.5729675, 338.2194824
1: -110.8504257, 262.9640808, -102.2596817, 245.4048767, -356.2553101, 365.2236938
2: -93.9988098, 289.5491333, -86.6547241, 270.3575134, -364.3563232, 376.2038574
3: -98.6145859, 376.5031738, -91.1154327, 352.0030212, -450.6176147, 467.6185913
4: -84.2276993, 342.1849670, -77.9133224, 319.7744141, -404.0021057, 420.0982971

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
time: 1.43 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7728223
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -81.1012268, 273.4296875, -69.0703354, 234.1050568, -315.2062988, 342.5000000
1: -114.0062256, 270.9878845, -96.9434433, 232.1126099, -346.1188354, 367.9312439
2: -96.5905914, 298.6038208, -82.1017456, 255.7478333, -352.3384399, 380.7055664
3: -101.4230957, 388.1472778, -86.3438950, 332.7772217, -434.2003174, 474.4911499
4: -86.5237656, 353.1409302, -73.7513809, 302.4556274, -388.9794006, 426.8922729

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7574381
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7764883
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -87.0268021, 290.5809937, -69.0703354, 234.1050568, -321.1318665, 359.6513367
1: -122.3216095, 288.3269958, -96.9434433, 232.1126099, -354.4342041, 385.2704163
2: -103.7214050, 317.6370239, -82.1017456, 255.7478333, -359.4692383, 399.7387695
3: -108.7294540, 412.3012695, -86.3438950, 332.7772217, -441.5066528, 498.6451721
4: -92.7709885, 375.2309570, -73.7513809, 302.4556274, -395.2266235, 448.9822998

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7582153
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7766740
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -85.6666107, 286.6015625, -72.8479538, 247.0541229, -332.7207336, 359.4494629
1: -120.4145279, 284.3439331, -101.8806534, 244.7858429, -365.2003784, 386.2245789
2: -102.0890884, 313.2262878, -86.3351059, 269.7480469, -371.8370972, 399.5614014
3: -107.0440369, 406.7699280, -90.7877808, 351.2036743, -458.2477112, 497.5577087
4: -91.3236542, 370.1166992, -77.6636581, 319.0796204, -410.4032593, 447.7803040

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7648228, upper bound: 339.7541837
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7648228, upper bound: 339.7702184
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -73.4968796, 249.0412445, -76.5349884, 256.7867737, -330.2836304, 325.5762329
1: -103.2814026, 246.7286530, -107.5017776, 254.7888947, -358.0703125, 354.2304382
2: -87.5033264, 271.7731018, -91.1387100, 280.5902100, -368.0934753, 362.9118042
3: -91.9600143, 353.6892395, -95.6157379, 364.6956787, -456.6556702, 449.3049011
4: -78.5316086, 321.4323425, -81.6669617, 331.5134277, -410.0450439, 403.0993042

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7820815
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7820815
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -80.2049484, 269.1602173, -76.5349884, 256.7867737, -336.9917297, 345.6951904
1: -112.7416992, 266.9496155, -107.5017776, 254.7888947, -367.5305786, 374.4513855
2: -95.6105042, 293.9599915, -91.1387100, 280.5902100, -376.2007141, 385.0986938
3: -100.2752228, 382.0771179, -95.6157379, 364.6956787, -464.9708862, 477.6927795
4: -85.6529694, 347.3413086, -81.6669617, 331.5134277, -417.1663818, 429.0082703

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7875860
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7875860
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -72.2763596, 244.9064636, -80.1331406, 269.2861938, -341.5625610, 325.0395813
1: -101.3383942, 242.6434326, -112.3996048, 267.1072998, -368.4456787, 355.0430298
2: -85.8648529, 267.2776184, -95.3169937, 294.1314697, -379.9963379, 362.5946045
3: -90.2656708, 347.9327393, -100.0205536, 382.5018921, -472.7675781, 447.9532776
4: -77.1345291, 316.1322021, -85.4314728, 347.5932617, -424.7277832, 401.5636597

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7801626
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7801626
time: 1.20 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -79.0012131, 265.5031433, -80.1331406, 269.2861938, -348.2874146, 345.6362915
1: -110.9996643, 263.3114624, -112.3996048, 267.1072998, -378.1069336, 375.7110596
2: -94.1263504, 289.9316101, -95.3169937, 294.1314697, -388.2577820, 385.2485962
3: -98.7479401, 376.9936829, -100.0205536, 382.5018921, -481.2498169, 477.0141907
4: -84.3418961, 342.6320190, -85.4314728, 347.5932617, -431.9351196, 428.0634766

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7828899
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7828899
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -81.1014862, 273.4304810, -76.0437393, 254.9332275, -336.0346985, 349.4741821
1: -114.0065536, 270.9886475, -106.7489624, 253.0709229, -367.0774231, 377.7376099
2: -96.5908890, 298.6047363, -90.5001068, 278.7565002, -375.3473816, 389.1047974
3: -101.4233856, 388.1483154, -94.9625549, 362.1279602, -463.5513306, 483.1108398
4: -86.5240173, 353.1419067, -81.1034927, 329.2878113, -415.8118286, 434.2453918

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7848818
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7848819
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -87.2639465, 291.3030396, -76.0437393, 254.9332275, -342.1971741, 367.3467712
1: -122.6401215, 289.0608215, -106.7489624, 253.0709229, -375.7109985, 395.8097839
2: -103.9952545, 318.4460754, -90.5001068, 278.7565002, -382.7517395, 408.9461365
3: -109.0149155, 413.3319702, -94.9625549, 362.1279602, -471.1428528, 508.2945251
4: -93.0167007, 376.1705322, -81.1034927, 329.2878113, -422.3045044, 457.2740173

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7877076
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7877076
time: 1.36 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -79.8985214, 269.7968445, -79.5781326, 267.2933350, -347.1918640, 349.3749695
1: -112.2731857, 267.3858948, -111.5234375, 265.1660461, -377.4392090, 378.9093018
2: -95.1125793, 294.6107788, -94.5764923, 292.0809021, -387.1934509, 389.1872559
3: -99.8974915, 383.1125183, -99.2505035, 379.7619324, -479.6594238, 482.3630066
4: -85.2212143, 348.4922485, -84.7941284, 345.1743164, -430.3955078, 433.2863770

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7648228, upper bound: 339.7804527
time: 1.15 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7648228, upper bound: 339.7804528
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -85.9021225, 287.3211365, -79.5781326, 267.2933350, -353.1954651, 366.8992615
1: -120.7316055, 285.0752258, -111.5234375, 265.1660461, -385.8976135, 396.5986633
2: -102.3616028, 314.0325928, -94.5764923, 292.0809021, -394.4425049, 408.6090698
3: -107.3278961, 407.7969360, -99.2505035, 379.7619324, -487.0898132, 507.0473938
4: -91.5679703, 371.0538330, -84.7941284, 345.1743164, -436.7422791, 455.8479614

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7648228, upper bound: 339.7826346
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7648228, upper bound: 339.7826346
time: 1.03 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.94 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7755757
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7755757
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7755757
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7755757
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7909926, upper bound: 339.7840661
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7909926, upper bound: 339.7840661
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7910401, upper bound: 339.7842595
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7910401, upper bound: 339.7842595
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7781413, upper bound: 339.7527548
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7781413, upper bound: 339.7683063
NS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7394499, upper bound: 339.7515716
NS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7390666, upper bound: 339.7421101
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7875591
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7875591
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7875591
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7875591
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7540060, upper bound: 339.7756255
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7540060, upper bound: 339.7756255
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7540060, upper bound: 339.7882640
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7540060, upper bound: 339.7882640
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7534731, upper bound: 339.7736242
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7534731, upper bound: 339.7736242
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7534731, upper bound: 339.7851916
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7534731, upper bound: 339.7851916
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764730
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764730
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764995
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764995
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7728223
NS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7574381
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7764883
NS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7582153
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7766740
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7648228, upper bound: 339.7541837
NS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7648228, upper bound: 339.7702184
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7820815
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7820815
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7875860
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7875860
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7801626
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7801626
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7828899
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7828899
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7848818
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7848819
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7877076
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7877076
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7648228, upper bound: 339.7804527
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7648228, upper bound: 339.7804528
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7648228, upper bound: 339.7826346
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.94
Output dim: 0, lower bound: -339.7648228, upper bound: 339.7826346

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -69.5513840, 232.9595032, -68.0701523, 230.4742584, -300.0256348, 301.0296631
1: -97.5931473, 231.2630310, -95.5809097, 228.4632263, -326.0563660, 326.8439331
2: -82.7497482, 254.8089752, -80.9740677, 251.6937408, -334.4434814, 335.7830505
3: -86.8460999, 331.2365417, -85.1222763, 327.4763184, -414.3223877, 416.3587952
4: -74.2284851, 301.3083496, -72.7232666, 297.6366272, -371.8650818, 374.0316162

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7972748, upper bound: 339.7860748
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7973630, upper bound: 339.7863631
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -69.5513840, 232.9595032, -76.8233643, 258.7799683, -328.3313599, 309.7828674
1: -97.5931473, 231.2630310, -107.9044495, 256.5423279, -354.1354675, 339.1674805
2: -82.7497482, 254.8089752, -91.4312668, 282.6691895, -365.4189453, 346.2402344
3: -86.8460999, 331.2365417, -96.0066833, 367.3713074, -454.2174072, 427.2432251
4: -74.2284851, 301.3083496, -81.9241943, 334.1950989, -408.4235229, 383.2325134

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7972748, upper bound: 339.7860748
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7973630, upper bound: 339.7863631
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -73.2851791, 245.6114960, -68.0701523, 230.4742584, -303.7594299, 313.6816406
1: -102.5696869, 243.6932068, -95.5809097, 228.4632263, -331.0328979, 339.2741089
2: -86.9982605, 268.5047607, -80.9740677, 251.6937408, -338.6920166, 349.4788208
3: -91.3169250, 349.2504578, -85.1222763, 327.4763184, -418.7932129, 434.3727112
4: -78.0879364, 317.6050415, -72.7232666, 297.6366272, -375.7245178, 390.3283081

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7751993
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7826487, upper bound: 339.7755757
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -73.2851791, 245.6114960, -76.8233643, 258.7799683, -332.0651550, 322.4348450
1: -102.5696869, 243.6932068, -107.9044495, 256.5423279, -359.1119385, 351.5976257
2: -86.9982605, 268.5047607, -91.4312668, 282.6691895, -369.6674500, 359.9360046
3: -91.3169250, 349.2504578, -96.0066833, 367.3713074, -458.6882324, 445.2571411
4: -78.0879364, 317.6050415, -81.9241943, 334.1950989, -412.2829895, 399.5292358

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7751993
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7826487, upper bound: 339.7755757
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -69.5513840, 232.9595032, -71.8672867, 243.2808533, -312.8322449, 304.8267822
1: -97.5931473, 231.2630310, -100.5789566, 241.1149902, -338.7081299, 331.8419800
2: -82.7497482, 254.8089752, -85.2399216, 265.6368408, -348.3865356, 340.0488892
3: -86.8460999, 331.2365417, -89.6213989, 345.7196655, -432.5657043, 420.8579407
4: -74.2284851, 301.3083496, -76.6374817, 314.1193542, -388.3477783, 377.9458008

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7877576, upper bound: 339.7743731
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7842786, upper bound: 339.7735350
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -69.5513840, 232.9595032, -79.6831741, 268.6914062, -338.2427979, 312.6426697
1: -97.5931473, 231.2630310, -111.7580643, 266.3983765, -363.9915161, 343.0210876
2: -82.7497482, 254.8089752, -94.7053299, 293.5176697, -376.2674255, 349.5143127
3: -86.8460999, 331.2365417, -99.4631119, 381.6340332, -468.4801025, 430.6996460
4: -74.2284851, 301.3083496, -84.8803864, 347.1163635, -421.3448181, 386.1887207

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7877576, upper bound: 339.7743731
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7842786, upper bound: 339.7735350
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -73.2851791, 245.6114960, -71.8672867, 243.2808533, -316.5660400, 317.4787903
1: -102.5696869, 243.6932068, -100.5789566, 241.1149902, -343.6846619, 344.2721558
2: -86.9982605, 268.5047607, -85.2399216, 265.6368408, -352.6350708, 353.7446899
3: -91.3169250, 349.2504578, -89.6213989, 345.7196655, -437.0365295, 438.8718567
4: -78.0879364, 317.6050415, -76.6374817, 314.1193542, -392.2072449, 394.2424622

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7822449, upper bound: 339.7734592
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -73.2851791, 245.6114960, -79.6831741, 268.6914062, -341.9765930, 325.2946777
1: -102.5696869, 243.6932068, -111.7580643, 266.3983765, -368.9680481, 355.4512634
2: -86.9982605, 268.5047607, -94.7053299, 293.5176697, -380.5159302, 363.2100830
3: -91.3169250, 349.2504578, -99.4631119, 381.6340332, -472.9509277, 448.7135620
4: -78.0879364, 317.6050415, -84.8803864, 347.1163635, -425.2042542, 402.4854126

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7822449, upper bound: 339.7734592
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -73.7456741, 246.8744202, -68.9651871, 231.8384094, -305.5840759, 315.8395996
1: -103.3255692, 244.9274597, -96.0801392, 229.9191284, -333.2445679, 341.0075073
2: -87.6685181, 270.0310974, -81.5320206, 253.3316803, -341.0001526, 351.5631104
3: -91.9267197, 350.9618835, -85.6521530, 329.6236877, -421.5504150, 436.6140442
4: -78.5416565, 319.5290222, -73.3708191, 299.4720764, -378.0137329, 392.8998413

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7909926, upper bound: 339.7840236
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7908774, upper bound: 339.7840661
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -73.7456741, 246.8744202, -77.7421036, 260.0199585, -333.7656250, 324.6164856
1: -103.3255692, 244.9274597, -108.6034851, 257.9522705, -361.2778320, 353.5308533
2: -87.6685181, 270.0310974, -92.0875092, 284.2211304, -371.8895874, 362.1185913
3: -91.9267197, 350.9618835, -96.7045212, 369.3697815, -461.2964783, 447.6663818
4: -78.5416565, 319.5290222, -82.5567245, 335.9577637, -414.4994202, 402.0857239

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7909926, upper bound: 339.7840236
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7908774, upper bound: 339.7840661
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -81.7432098, 271.9115295, -69.3599930, 233.2927399, -315.0359497, 341.2715149
1: -114.6574097, 269.9497986, -96.5816193, 231.3170929, -345.9744873, 366.5314331
2: -97.2616806, 297.5369873, -81.9435349, 254.8822021, -352.1438904, 379.4804993
3: -101.9812241, 386.0803833, -86.1114883, 331.6745911, -433.6558228, 472.1918640
4: -87.0161133, 351.6328735, -73.7646103, 301.3293762, -388.3454895, 425.3974915

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7910401, upper bound: 339.7841593
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7910401, upper bound: 339.7842595
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -81.7432098, 271.9115295, -78.5047379, 262.7273865, -344.4705811, 350.4162598
1: -114.6574097, 269.9497986, -109.6294556, 260.6001892, -375.2575989, 379.5792542
2: -97.2616806, 297.5369873, -92.9448166, 287.1320190, -384.3936768, 390.4818115
3: -101.9812241, 386.0803833, -97.6291351, 373.2217102, -475.2029419, 483.7095337
4: -87.0161133, 351.6328735, -83.3447418, 339.4274292, -426.4435425, 434.9776001

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7910401, upper bound: 339.7841593
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7910401, upper bound: 339.7842595
time: 1.35 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -74.5713882, 249.1286316, -67.5324707, 227.8703918, -302.4417725, 316.6611023
1: -104.5754700, 247.5596771, -94.4100647, 226.4231110, -330.9985962, 341.9697266
2: -88.6184082, 272.9008179, -79.9923401, 249.6980438, -338.3164673, 352.8931580
3: -93.0721588, 354.3959351, -84.2051773, 324.7163086, -417.7883911, 438.6011047
4: -79.4293060, 322.6373291, -71.9569397, 295.3011169, -374.7304077, 394.5942688

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7530264, upper bound: 339.7499337
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7530264, upper bound: 339.7527548
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -74.5713882, 249.1286316, -68.8099670, 233.3446808, -307.9160461, 317.9385681
1: -104.5754700, 247.5596771, -96.3250580, 231.6776886, -336.2531738, 343.8847046
2: -88.6184082, 272.9008179, -81.5648117, 255.4143219, -344.0326843, 354.4656372
3: -93.0721588, 354.3959351, -85.8868408, 332.3657532, -425.4378662, 440.2827454
4: -79.4293060, 322.6373291, -73.3731766, 302.1296387, -381.5589294, 396.0104980

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7530264, upper bound: 339.7595768
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7530264, upper bound: 339.7683063
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -73.3381424, 244.6101532, -74.4024048, 248.2754364, -321.6135864, 319.0125732
1: -102.9172745, 242.8930969, -104.4156036, 246.5509491, -349.4682007, 347.3086853
2: -87.3073273, 267.5639648, -88.5706482, 271.5914612, -358.8988037, 356.1346130
3: -91.5368042, 347.4905701, -92.8714447, 352.7966309, -444.3334351, 440.3619995
4: -78.2409210, 316.1669312, -79.3689194, 320.9674683, -399.2083740, 395.5358582

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7964133, upper bound: 339.7897566
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7931687, upper bound: 339.7893216
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -73.3381424, 244.6101532, -74.4361725, 250.3334045, -323.6715088, 319.0463257
1: -102.9172745, 242.8930969, -104.6591797, 248.2941284, -351.2113953, 347.5522461
2: -87.3073273, 267.5639648, -88.7077789, 273.4312744, -360.7385864, 356.2717285
3: -91.5368042, 347.4905701, -93.0792694, 355.5361633, -447.0729675, 440.5698242
4: -78.2409210, 316.1669312, -79.5089569, 323.1212158, -401.3621216, 395.6759033

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7964133, upper bound: 339.7897566
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7931687, upper bound: 339.7893216
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -76.9758148, 257.1091003, -74.4024048, 248.2754364, -325.2512207, 331.5115051
1: -107.8613968, 255.2158203, -104.4156036, 246.5509491, -354.4123535, 359.6314087
2: -91.5137939, 281.1018677, -88.5706482, 271.5914612, -363.1052551, 369.6725159
3: -95.9746094, 365.3651733, -92.8714447, 352.7966309, -448.7712402, 458.2366333
4: -82.0231857, 332.3006287, -79.3689194, 320.9674683, -402.9906616, 411.6695557

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7876171, upper bound: 339.7875254
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7856238, upper bound: 339.7875591
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -76.9758148, 257.1091003, -74.4361725, 250.3334045, -327.3091431, 331.5452881
1: -107.8613968, 255.2158203, -104.6591797, 248.2941284, -356.1555176, 359.8749695
2: -91.5137939, 281.1018677, -88.7077789, 273.4312744, -364.9450684, 369.8096313
3: -95.9746094, 365.3651733, -93.0792694, 355.5361633, -451.5107727, 458.4444580
4: -82.0231857, 332.3006287, -79.5089569, 323.1212158, -405.1444092, 411.8095703

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7876171, upper bound: 339.7875254
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7856238, upper bound: 339.7875591
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -73.3381424, 244.6101532, -77.7785721, 260.0312500, -333.3693848, 322.3887329
1: -102.9172745, 242.8930969, -109.0360641, 258.1087036, -361.0259705, 351.9291687
2: -87.3073273, 267.5639648, -92.5042572, 284.2797852, -371.5870972, 360.0682373
3: -91.5368042, 347.4905701, -97.0183640, 369.6040649, -461.1408691, 444.5089111
4: -78.2409210, 316.1669312, -82.8901901, 336.1292725, -414.3701782, 399.0571289

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921803, upper bound: 339.7773076
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7867916, upper bound: 339.7759803
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -73.3381424, 244.6101532, -78.0771179, 262.9364319, -336.2745667, 322.6872559
1: -102.9172745, 242.8930969, -109.5995102, 260.7164917, -363.6337585, 352.4926147
2: -87.3073273, 267.5639648, -92.9267578, 287.0872498, -374.3945923, 360.4906616
3: -91.5368042, 347.4905701, -97.5246735, 373.4973450, -465.0341492, 445.0152283
4: -78.2409210, 316.1669312, -83.3126755, 339.3460388, -417.5869446, 399.4796143

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921803, upper bound: 339.7798186
time: 1.26 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7867916, upper bound: 339.7787307
time: 1.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -76.9758148, 257.1091003, -77.7785721, 260.0312500, -337.0070190, 334.8876648
1: -107.8613968, 255.2158203, -109.0360641, 258.1087036, -365.9700928, 364.2518921
2: -91.5137939, 281.1018677, -92.5042572, 284.2797852, -375.7935791, 373.6061401
3: -95.9746094, 365.3651733, -97.0183640, 369.6040649, -465.5786743, 462.3835449
4: -82.0231857, 332.3006287, -82.8901901, 336.1292725, -418.1524658, 415.1908264

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7789116, upper bound: 339.7801407
time: 1.20 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -76.9758148, 257.1091003, -78.0771179, 262.9364319, -339.9122009, 335.1862183
1: -107.8613968, 255.2158203, -109.5995102, 260.7164917, -368.5778809, 364.8153381
2: -91.5137939, 281.1018677, -92.9267578, 287.0872498, -378.6010437, 374.0285950
3: -95.9746094, 365.3651733, -97.5246735, 373.4973450, -469.4719543, 462.8898315
4: -82.0231857, 332.3006287, -83.3126755, 339.3460388, -421.3692322, 415.6133118

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7789116, upper bound: 339.7801407
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -80.0753098, 268.5890198, -73.9064636, 246.4675293, -326.5428467, 342.4954834
1: -112.4296036, 266.3736267, -103.6660690, 244.8640900, -357.2937012, 370.0397034
2: -95.3291550, 293.5993958, -87.9217072, 269.7864075, -365.1155701, 381.5210876
3: -100.0343323, 381.2966309, -92.2202454, 350.4021912, -450.4365234, 473.5168762
4: -85.3743134, 347.0550537, -78.8019485, 318.8732910, -404.2476196, 425.8569641

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7546880, upper bound: 339.7756255
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7546880, upper bound: 339.7756255
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -80.0753098, 268.5890198, -73.9311676, 248.4424591, -328.5177002, 342.5202026
1: -112.4296036, 266.3736267, -103.8889694, 246.5458984, -358.9754944, 370.2626038
2: -95.3291550, 293.5993958, -88.0497818, 271.5649719, -366.8941040, 381.6491699
3: -100.0343323, 381.2966309, -92.4103928, 352.9408264, -452.9751587, 473.7070312
4: -85.3743134, 347.0550537, -78.9307327, 320.8739624, -406.2482910, 425.9857483

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7546880, upper bound: 339.7756255
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7546880, upper bound: 339.7756255
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -87.1023941, 288.9736328, -73.9064636, 246.4675293, -333.5699158, 362.8800659
1: -122.2498779, 286.9699707, -103.6660690, 244.8640900, -367.1139526, 390.6360168
2: -103.7315826, 316.2297668, -87.9217072, 269.7864075, -373.5180054, 404.1514893
3: -108.6728821, 410.1506653, -92.2202454, 350.4021912, -459.0750732, 502.3709106
4: -92.7565842, 373.5724182, -78.8019485, 318.8732910, -411.6298828, 452.3743591

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7871465, upper bound: 339.7882640
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7871465, upper bound: 339.7882640
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -87.1023941, 288.9736328, -73.9311676, 248.4424591, -335.5448303, 362.9047852
1: -122.2498779, 286.9699707, -103.8889694, 246.5458984, -368.7957764, 390.8589478
2: -103.7315826, 316.2297668, -88.0497818, 271.5649719, -375.2965393, 404.2795410
3: -108.6728821, 410.1506653, -92.4103928, 352.9408264, -461.6137085, 502.5610657
4: -92.7565842, 373.5724182, -78.9307327, 320.8739624, -413.6305542, 452.5031433

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7871465, upper bound: 339.7882640
time: 1.19 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7871465, upper bound: 339.7882640
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -78.6936035, 264.4166870, -77.2436752, 258.1709290, -336.8645325, 341.6603088
1: -110.4570236, 262.2425232, -108.1991653, 256.3037415, -366.7607422, 370.4416809
2: -93.6443024, 289.0195618, -91.7878342, 282.3763123, -376.0205994, 380.8073425
3: -98.2981262, 375.5130920, -96.2838516, 367.0382690, -465.3363953, 471.7969360
4: -83.8868637, 341.7197571, -82.2781448, 333.8847656, -417.7715759, 423.9978943

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7034132, upper bound: 339.7310888
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7541837, upper bound: 339.7736242
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -78.6936035, 264.4166870, -77.5667801, 261.0824280, -339.7760315, 341.9834595
1: -110.4570236, 262.2425232, -108.7868271, 258.9263916, -369.3834229, 371.0293579
2: -93.6443024, 289.0195618, -92.2371902, 285.2008667, -378.8451538, 381.2567444
3: -98.2981262, 375.5130920, -96.8119812, 370.9766846, -469.2748108, 472.3250732
4: -83.8868637, 341.7197571, -82.7224503, 337.1283569, -421.0151978, 424.4421997

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7034132, upper bound: 339.7310888
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7541837, upper bound: 339.7736242
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -85.6336975, 284.6076965, -77.2436752, 258.1709290, -343.8045959, 361.8513184
1: -120.1997910, 282.6163025, -108.1991653, 256.3037415, -376.5035095, 390.8154602
2: -101.9800720, 311.3958435, -91.7878342, 282.3763123, -384.3563538, 403.1836243
3: -106.8617401, 404.0912170, -96.2838516, 367.0382690, -473.9000244, 500.3750610
4: -91.2010727, 367.9867859, -82.2781448, 333.8847656, -425.0858154, 450.2649231

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7866060, upper bound: 339.7851916
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847114, upper bound: 339.7839326
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -85.6336975, 284.6076965, -77.5667801, 261.0824280, -346.7160645, 362.1744690
1: -120.1997910, 282.6163025, -108.7868271, 258.9263916, -379.1261902, 391.4031372
2: -101.9800720, 311.3958435, -92.2371902, 285.2008667, -387.1808777, 403.6330261
3: -106.8617401, 404.0912170, -96.8119812, 370.9766846, -477.8384399, 500.9031982
4: -91.2010727, 367.9867859, -82.7224503, 337.1283569, -428.3294373, 450.7092285

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7866060, upper bound: 339.7851916
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847114, upper bound: 339.7839326
time: 1.22 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -73.4968796, 249.0412445, -68.0489349, 230.6139221, -304.1108093, 317.0901794
1: -103.2814026, 246.7286530, -95.5720901, 228.5440063, -331.8254089, 342.3007507
2: -87.5033264, 271.7731018, -80.9495544, 251.7684174, -339.2716980, 352.7226562
3: -91.9600143, 353.6892395, -85.1096954, 327.6518555, -419.6118469, 438.7989502
4: -78.5316086, 321.4323425, -72.7042313, 297.7798767, -376.3114929, 394.1365662

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764805
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764805
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -73.4968796, 249.0412445, -76.6811142, 258.6195679, -332.1164246, 325.7223511
1: -103.2814026, 246.7286530, -107.7439346, 256.3242493, -359.6055908, 354.4725952
2: -87.5033264, 271.7731018, -91.2761917, 282.4193726, -369.9226685, 363.0492859
3: -91.9600143, 353.6892395, -95.8580780, 367.1405640, -459.1005554, 449.5473022
4: -78.5316086, 321.4323425, -81.7952728, 333.9732361, -412.5048523, 403.2276001

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764805
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764805
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -80.1220779, 268.9065857, -68.0489349, 230.6139221, -310.7359619, 336.9555054
1: -112.6305847, 266.6913757, -95.5720901, 228.5440063, -341.1745911, 362.2634277
2: -95.5148392, 293.6755676, -80.9495544, 251.7684174, -347.2832642, 374.6250610
3: -100.1756516, 381.7141724, -85.1096954, 327.6518555, -427.8274536, 466.8238525
4: -85.5672302, 347.0112000, -72.7042313, 297.7798767, -383.3471069, 419.7154236

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7817231, upper bound: 339.7764995
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7817231, upper bound: 339.7764995
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -80.1220779, 268.9065857, -76.6811142, 258.6195679, -338.7415771, 345.5877075
1: -112.6305847, 266.6913757, -107.7439346, 256.3242493, -368.9547424, 374.4352722
2: -95.5148392, 293.6755676, -91.2761917, 282.4193726, -377.9342041, 384.9517517
3: -100.1756516, 381.7141724, -95.8580780, 367.1405640, -467.3161621, 477.5722351
4: -85.5672302, 347.0112000, -81.7952728, 333.9732361, -419.5404663, 428.8064575

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7817232, upper bound: 339.7764995
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7817232, upper bound: 339.7764995
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -72.2763596, 244.9064636, -71.8976974, 243.6161041, -315.8924561, 316.8041382
1: -101.3383942, 242.6434326, -100.6366119, 241.3848419, -342.7231750, 343.2800293
2: -85.8648529, 267.2776184, -85.2807693, 265.9253845, -351.7902222, 352.5583496
3: -90.2656708, 347.9327393, -89.6689453, 346.1664734, -436.4321289, 437.6016846
4: -77.1345291, 316.1322021, -76.6813736, 314.5084534, -391.6429749, 392.8135376

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
time: 1.46 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
time: 1.44 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -72.2763596, 244.9064636, -79.5246964, 268.4673157, -340.7436829, 324.4311523
1: -101.3383942, 242.6434326, -111.5745087, 266.1139221, -367.4523010, 354.2179565
2: -85.8648529, 267.2776184, -94.5337677, 293.1930237, -379.0578613, 361.8114014
3: -90.2656708, 347.9327393, -99.2946167, 381.3083496, -471.5740356, 447.2273560
4: -77.1345291, 316.1322021, -84.7382584, 346.8033447, -423.9378662, 400.8704529

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -78.8918686, 265.1601868, -71.8976974, 243.6161041, -322.5079651, 337.0578918
1: -110.8504257, 262.9640808, -100.6366119, 241.3848419, -352.2352600, 363.6006775
2: -93.9988098, 289.5491333, -85.2807693, 265.9253845, -359.9241943, 374.8298340
3: -98.6145859, 376.5031738, -89.6689453, 346.1664734, -444.7810364, 466.1721191
4: -84.2276993, 342.1849670, -76.6813736, 314.5084534, -398.7361450, 418.8663330

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7816871, upper bound: 339.7728223
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7816871, upper bound: 339.7728223
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -78.8918686, 265.1601868, -79.5246964, 268.4673157, -347.3591919, 344.6848755
1: -110.8504257, 262.9640808, -111.5745087, 266.1139221, -376.9643555, 374.5385742
2: -93.9988098, 289.5491333, -94.5337677, 293.1930237, -387.1918335, 384.0828857
3: -98.6145859, 376.5031738, -99.2946167, 381.3083496, -479.9229431, 475.7977905
4: -84.2276993, 342.1849670, -84.7382584, 346.8033447, -431.0310364, 426.9232178

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7816871, upper bound: 339.7728223
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7816871, upper bound: 339.7728223
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -81.1012268, 273.4296875, -67.3074341, 228.6159515, -309.7171631, 340.7370911
1: -114.0062256, 270.9878845, -94.5416565, 226.6139679, -340.6201782, 365.5294800
2: -96.5905914, 298.6038208, -80.0471954, 249.6912842, -346.2817078, 378.6509705
3: -101.4230957, 388.1472778, -84.2010803, 325.0054016, -426.4284973, 472.3482971
4: -86.5237656, 353.1409302, -71.9301529, 295.3530273, -381.8767700, 425.0710754

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7764883
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7764883
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -87.0268021, 290.5809937, -67.3074341, 228.6159515, -315.6427612, 357.8884277
1: -122.3216095, 288.3269958, -94.5416565, 226.6139679, -348.9355774, 382.8686218
2: -103.7214050, 317.6370239, -80.0471954, 249.6912842, -353.4125366, 397.6842041
3: -108.7294540, 412.3012695, -84.2010803, 325.0054016, -433.7348328, 496.5023193
4: -92.7709885, 375.2309570, -71.9301529, 295.3530273, -388.1240234, 447.1611023

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7723359, upper bound: 339.7766740
time: 1.13 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7723359, upper bound: 339.7766740
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -73.4968796, 249.0412445, -75.1074982, 251.8864746, -325.3833008, 324.1487427
1: -103.2814026, 246.7286530, -105.4976044, 249.8960571, -353.1774597, 352.2262573
2: -87.5033264, 271.7731018, -89.4473572, 275.1980591, -362.7012939, 361.2204590
3: -91.9600143, 353.6892395, -93.8290634, 357.6604614, -449.6204529, 447.5183105
4: -78.5316086, 321.4323425, -80.1569901, 325.1415405, -403.6731262, 401.5893250

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7820815
time: 1.37 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7820815
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -73.4968796, 249.0412445, -83.1277390, 277.2397766, -350.7366028, 332.1689758
1: -103.2814026, 246.7286530, -116.7270279, 275.2055054, -378.4869080, 363.4556885
2: -87.5033264, 271.7731018, -98.9958725, 303.1774902, -390.6807861, 370.7689819
3: -91.9600143, 353.6892395, -103.7656403, 393.4320984, -485.3920898, 457.4548645
4: -78.5316086, 321.4323425, -88.5655899, 358.0359802, -436.5675964, 409.9979248

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7820815
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7820815
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -80.2049484, 269.1602173, -75.1074982, 251.8864746, -332.0914001, 344.2677002
1: -112.7416992, 266.9496155, -105.4976044, 249.8960571, -362.6376953, 372.4472046
2: -95.6105042, 293.9599915, -89.4473572, 275.1980591, -370.8085327, 383.4072876
3: -100.2752228, 382.0771179, -93.8290634, 357.6604614, -457.9356689, 475.9061890
4: -85.6529694, 347.3413086, -80.1569901, 325.1415405, -410.7944946, 427.4982910

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847890, upper bound: 339.7875860
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847890, upper bound: 339.7875860
time: 1.15 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -80.2049484, 269.1602173, -83.1277390, 277.2397766, -357.4447021, 352.2879639
1: -112.7416992, 266.9496155, -116.7270279, 275.2055054, -387.9471741, 383.6766052
2: -95.6105042, 293.9599915, -98.9958725, 303.1774902, -398.7879944, 392.9558105
3: -100.2752228, 382.0771179, -103.7656403, 393.4320984, -493.7073059, 485.8427734
4: -85.6529694, 347.3413086, -88.5655899, 358.0359802, -443.6889343, 435.9068909

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847890, upper bound: 339.7875860
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847890, upper bound: 339.7875860
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -72.2763596, 244.9064636, -78.8922043, 264.9575806, -337.2339478, 323.7986755
1: -101.3383942, 242.6434326, -110.6651306, 262.8228455, -364.1611938, 353.3085632
2: -85.8648529, 267.2776184, -93.8518524, 289.4074707, -375.2723389, 361.1294250
3: -90.2656708, 347.9327393, -98.4748459, 376.2855225, -466.5512085, 446.4075928
4: -77.1345291, 316.1322021, -84.1213837, 341.9718933, -419.1064148, 400.2536011

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7801626
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7801626
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -72.2763596, 244.9064636, -85.7481537, 286.8530884, -359.1294556, 330.6545715
1: -101.3383942, 242.6434326, -120.4197540, 284.6775818, -386.0159912, 363.0631714
2: -85.8648529, 267.2776184, -102.1124573, 313.5783691, -399.4432373, 369.3900757
3: -90.2656708, 347.9327393, -107.0716705, 407.2160339, -497.4816895, 455.0043945
4: -77.1345291, 316.1322021, -91.3417511, 370.4882812, -447.6228027, 407.4739075

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7801626
time: 1.21 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7801626
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -79.0012131, 265.5031433, -78.8922043, 264.9575806, -343.9588013, 344.3953552
1: -110.9996643, 263.3114624, -110.6651306, 262.8228455, -373.8224487, 373.9765930
2: -94.1263504, 289.9316101, -93.8518524, 289.4074707, -383.5338135, 383.7833862
3: -98.7479401, 376.9936829, -98.4748459, 376.2855225, -475.0334473, 475.4685364
4: -84.3418961, 342.6320190, -84.1213837, 341.9718933, -426.3137817, 426.7533875

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7844799, upper bound: 339.7828899
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7844799, upper bound: 339.7828899
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -79.0012131, 265.5031433, -85.7481537, 286.8530884, -365.8543091, 351.2512817
1: -110.9996643, 263.3114624, -120.4197540, 284.6775818, -395.6772156, 383.7312012
2: -94.1263504, 289.9316101, -102.1124573, 313.5783691, -407.7046509, 392.0440369
3: -98.7479401, 376.9936829, -107.0716705, 407.2160339, -505.9639282, 484.0653381
4: -84.3418961, 342.6320190, -91.3417511, 370.4882812, -454.8301697, 433.9736938

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7844799, upper bound: 339.7828899
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7844799, upper bound: 339.7828899
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -81.1014862, 273.4304810, -73.9064636, 246.4675293, -327.5690308, 347.3368530
1: -114.0065536, 270.9886475, -103.6660690, 244.8640900, -358.8706360, 374.6547241
2: -96.5908890, 298.6047363, -87.9217072, 269.7864075, -366.3772888, 386.5264282
3: -101.4233856, 388.1483154, -92.2202454, 350.4021912, -451.8255615, 480.3685608
4: -86.5240173, 353.1419067, -78.8019485, 318.8732910, -405.3973083, 431.9438171

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7730560, upper bound: 339.7848820
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7730560, upper bound: 339.7848819
time: 1.27 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -81.1014862, 273.4304810, -73.9311676, 248.4424591, -329.5438843, 347.3616333
1: -114.0065536, 270.9886475, -103.8889694, 246.5458984, -360.5524292, 374.8776245
2: -96.5908890, 298.6047363, -88.0497818, 271.5649719, -368.1558228, 386.6545105
3: -101.4233856, 388.1483154, -92.4103928, 352.9408264, -454.3641968, 480.5587158
4: -86.5240173, 353.1419067, -78.9307327, 320.8739624, -407.3979797, 432.0726013

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7730560, upper bound: 339.7848819
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7730560, upper bound: 339.7848820
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -87.2639465, 291.3030396, -73.9064636, 246.4675293, -333.7314758, 365.2094727
1: -122.6401215, 289.0608215, -103.6660690, 244.8640900, -367.5042114, 392.7268982
2: -103.9952545, 318.4460754, -87.9217072, 269.7864075, -373.7816467, 406.3677673
3: -109.0149155, 413.3319702, -92.2202454, 350.4021912, -459.4171143, 505.5522156
4: -93.0167007, 376.1705322, -78.8019485, 318.8732910, -411.8899841, 454.9724426

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7836551, upper bound: 339.7877076
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7836551, upper bound: 339.7877076
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -87.2639465, 291.3030396, -73.9311676, 248.4424591, -335.7064209, 365.2341919
1: -122.6401215, 289.0608215, -103.8889694, 246.5458984, -369.1860046, 392.9497986
2: -103.9952545, 318.4460754, -88.0497818, 271.5649719, -375.5601807, 406.4958191
3: -109.0149155, 413.3319702, -92.4103928, 352.9408264, -461.9557190, 505.7423706
4: -93.0167007, 376.1705322, -78.9307327, 320.8739624, -413.8906555, 455.1012268

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7836551, upper bound: 339.7877076
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7836551, upper bound: 339.7877076
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -79.8985214, 269.7968445, -77.2317429, 258.1312256, -338.0297546, 347.0285950
1: -112.2731857, 267.3858948, -108.1820450, 256.2639160, -368.5370789, 375.5679321
2: -95.1125793, 294.6107788, -91.7735062, 282.3324280, -377.4450073, 386.3842773
3: -99.8974915, 383.1125183, -96.2686386, 366.9818726, -466.8793640, 479.3811646
4: -85.2212143, 348.4922485, -82.2653809, 333.8329773, -419.0541992, 430.7576294

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7726338, upper bound: 339.7804528
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7726338, upper bound: 339.7804528
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -79.8985214, 269.7968445, -77.5667801, 261.0824280, -340.9809570, 347.3636169
1: -112.2731857, 267.3858948, -108.7868271, 258.9263916, -371.1995544, 376.1727295
2: -95.1125793, 294.6107788, -92.2371902, 285.2008667, -380.3134460, 386.8479614
3: -99.8974915, 383.1125183, -96.8119812, 370.9766846, -470.8741760, 479.9244995
4: -85.2212143, 348.4922485, -82.7224503, 337.1283569, -422.3495483, 431.2146912

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7726338, upper bound: 339.7804528
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7726338, upper bound: 339.7804528
time: 1.44 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -85.9021225, 287.3211365, -77.2317429, 258.1312256, -344.0333557, 364.5528870
1: -120.7316055, 285.0752258, -108.1820450, 256.2639160, -376.9954834, 393.2572632
2: -102.3616028, 314.0325928, -91.7735062, 282.3324280, -384.6940308, 405.8060913
3: -107.3278961, 407.7969360, -96.2686386, 366.9818726, -474.3097534, 504.0655518
4: -91.5679703, 371.0538330, -82.2653809, 333.8329773, -425.4009399, 453.3192139

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7826324, upper bound: 339.7826346
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7826324, upper bound: 339.7826346
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -85.9021225, 287.3211365, -77.5667801, 261.0824280, -346.9845581, 364.8879089
1: -120.7316055, 285.0752258, -108.7868271, 258.9263916, -379.6579285, 393.8620605
2: -102.3616028, 314.0325928, -92.2371902, 285.2008667, -387.5624695, 406.2697754
3: -107.3278961, 407.7969360, -96.8119812, 370.9766846, -478.3045654, 504.6089172
4: -91.5679703, 371.0538330, -82.7224503, 337.1283569, -428.6963196, 453.7762756

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7826324, upper bound: 339.7826346
time: 1.50 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7826324, upper bound: 339.7826346
time: 0.84 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.60 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7972748, upper bound: 339.7860748
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7973630, upper bound: 339.7863631
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7972748, upper bound: 339.7860748
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7973630, upper bound: 339.7863631
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7751993
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7826487, upper bound: 339.7755757
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7751993
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7826487, upper bound: 339.7755757
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7877576, upper bound: 339.7743731
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7842786, upper bound: 339.7735350
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7877576, upper bound: 339.7743731
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7842786, upper bound: 339.7735350
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7822449, upper bound: 339.7734592
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7837514, upper bound: 339.7737438
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7822449, upper bound: 339.7734592
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7909926, upper bound: 339.7840236
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7908774, upper bound: 339.7840661
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7909926, upper bound: 339.7840236
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7908774, upper bound: 339.7840661
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7910401, upper bound: 339.7841593
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7910401, upper bound: 339.7842595
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7910401, upper bound: 339.7841593
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7910401, upper bound: 339.7842595
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7530264, upper bound: 339.7499337
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7530264, upper bound: 339.7527548
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7530264, upper bound: 339.7595768
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7530264, upper bound: 339.7683063
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7964133, upper bound: 339.7897566
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7931687, upper bound: 339.7893216
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7964133, upper bound: 339.7897566
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7931687, upper bound: 339.7893216
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7876171, upper bound: 339.7875254
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7856238, upper bound: 339.7875591
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7876171, upper bound: 339.7875254
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7856238, upper bound: 339.7875591
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7921803, upper bound: 339.7773076
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7867916, upper bound: 339.7759803
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7921803, upper bound: 339.7798186
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7867916, upper bound: 339.7787307
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7789116, upper bound: 339.7801407
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7789116, upper bound: 339.7801407
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7878250, upper bound: 339.7853487
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7546880, upper bound: 339.7756255
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7546880, upper bound: 339.7756255
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7546880, upper bound: 339.7756255
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7546880, upper bound: 339.7756255
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7871465, upper bound: 339.7882640
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7871465, upper bound: 339.7882640
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7871465, upper bound: 339.7882640
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7871465, upper bound: 339.7882640
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7034132, upper bound: 339.7310888
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7541837, upper bound: 339.7736242
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7034132, upper bound: 339.7310888
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7541837, upper bound: 339.7736242
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7866060, upper bound: 339.7851916
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7847114, upper bound: 339.7839326
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7866060, upper bound: 339.7851916
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7847114, upper bound: 339.7839326
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764805
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764805
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764805
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7727157, upper bound: 339.7764805
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7817231, upper bound: 339.7764995
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7817231, upper bound: 339.7764995
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7817232, upper bound: 339.7764995
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7817232, upper bound: 339.7764995
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7725175, upper bound: 339.7701429
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7816871, upper bound: 339.7728223
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7816871, upper bound: 339.7728223
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7816871, upper bound: 339.7728223
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7816871, upper bound: 339.7728223
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7764883
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7642921, upper bound: 339.7764883
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7723359, upper bound: 339.7766740
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7723359, upper bound: 339.7766740
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7820815
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7820815
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7820815
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7820815
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7847890, upper bound: 339.7875860
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7847890, upper bound: 339.7875860
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7847890, upper bound: 339.7875860
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7847890, upper bound: 339.7875860
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7801626
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7801626
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7801626
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7735576, upper bound: 339.7801626
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7844799, upper bound: 339.7828899
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7844799, upper bound: 339.7828899
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7844799, upper bound: 339.7828899
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7844799, upper bound: 339.7828899
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7730560, upper bound: 339.7848820
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7730560, upper bound: 339.7848819
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7730560, upper bound: 339.7848819
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7730560, upper bound: 339.7848820
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7836551, upper bound: 339.7877076
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7836551, upper bound: 339.7877076
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7836551, upper bound: 339.7877076
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7836551, upper bound: 339.7877076
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7726338, upper bound: 339.7804528
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7726338, upper bound: 339.7804528
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7726338, upper bound: 339.7804528
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7726338, upper bound: 339.7804528
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7826324, upper bound: 339.7826346
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7826324, upper bound: 339.7826346
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7826324, upper bound: 339.7826346
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -339.7826324, upper bound: 339.7826346

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -68.6890717, 229.9647522, -68.0416336, 230.3775940, -299.0666504, 298.0063782
1: -96.3805618, 228.3121796, -95.5412369, 228.3677979, -324.7483521, 323.8533936
2: -81.7298965, 251.5602570, -80.9405899, 251.5885925, -333.3184814, 332.5007629
3: -85.7649078, 326.9635315, -85.0868530, 327.3388977, -413.1038208, 412.0503845
4: -73.3111420, 297.4329834, -72.6932297, 297.5116577, -370.8227844, 370.1261902

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7902247, upper bound: 339.7771926
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7975678, upper bound: 339.7889345
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -71.0041809, 237.2990570, -67.9691467, 230.1302643, -301.1344604, 305.2681885
1: -99.5878906, 235.6744232, -95.4378815, 228.1231995, -327.7110901, 331.1122742
2: -84.4718628, 259.6318359, -80.8532562, 251.3188324, -335.7907104, 340.4850159
3: -88.6048279, 337.2171326, -84.9950104, 326.9856567, -415.5904846, 422.2121582
4: -75.7413788, 306.7381897, -72.6151733, 297.1901245, -372.9315186, 379.3533630

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878078, upper bound: 339.7765816
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7977167, upper bound: 339.7898709
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -68.6890717, 229.9647522, -76.7953110, 258.6853943, -327.3744507, 306.7600403
1: -96.3805618, 228.3121796, -107.8653412, 256.4485779, -352.8291321, 336.1775208
2: -81.7298965, 251.5602570, -91.3982468, 282.5658875, -364.2957764, 342.9584656
3: -85.7649078, 326.9635315, -95.9718552, 367.2364197, -453.0013428, 422.9353943
4: -73.3111420, 297.4329834, -81.8945770, 334.0723267, -407.3834839, 379.3275146

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7840508, upper bound: 339.7668510
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7840519, upper bound: 339.7860748
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -71.0041809, 237.2990570, -76.7269440, 258.4538879, -329.4580688, 314.0259705
1: -99.5878906, 235.6744232, -107.7687531, 256.2194824, -355.8073730, 343.4431152
2: -84.4718628, 259.6318359, -91.3165054, 282.3132629, -366.7851257, 350.9482422
3: -88.6048279, 337.2171326, -95.8859100, 366.9067993, -455.5116272, 433.1030273
4: -75.7413788, 306.7381897, -81.8214874, 333.7720337, -409.5134277, 388.5596924

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7840508, upper bound: 339.7668510
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7840519, upper bound: 339.7863631
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -64.5932999, 218.1203003, -66.9902420, 226.7276764, -291.3209839, 285.1105347
1: -90.1021271, 216.2742615, -94.0700684, 224.7812195, -314.8833313, 310.3443298
2: -76.4842606, 238.5247650, -79.7055435, 247.6503296, -324.1345825, 318.2303162
3: -80.3515015, 310.4765930, -83.7727203, 322.1438599, -402.4953613, 394.2492676
4: -68.8980179, 282.2430725, -71.5859604, 292.8000183, -361.6979980, 353.8290100

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7838043, upper bound: 339.7747434
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7837798, upper bound: 339.7784864
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -71.3068695, 239.0076141, -66.9663315, 226.8557434, -298.1625977, 305.9739380
1: -99.7075195, 237.1099548, -94.0193481, 224.8637848, -324.5711975, 331.1293030
2: -84.5704346, 261.3007202, -79.6532211, 247.7386780, -332.3091125, 340.9539490
3: -88.7838135, 339.7810364, -83.7375031, 322.3056335, -411.0894470, 423.5184937
4: -75.9495926, 308.9975891, -71.5481949, 292.9313354, -368.8808899, 380.5457458

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7824679, upper bound: 339.7747434
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7824679, upper bound: 339.7784371
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -64.5932999, 218.1203003, -75.4809189, 254.1148987, -318.7081909, 293.6012268
1: -90.1021271, 216.2742615, -106.0266495, 251.9378510, -342.0398865, 322.3009033
2: -76.4842606, 238.5247650, -89.8568192, 277.6153870, -354.0996399, 328.3815308
3: -80.3515015, 310.4765930, -94.3326492, 360.7068481, -441.0583496, 404.8092346
4: -68.8980179, 282.2430725, -80.5128098, 328.1613464, -397.0593567, 362.7558594

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7836158, upper bound: 339.7741732
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7798059, upper bound: 339.7549216
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -71.3068695, 239.0076141, -75.9382172, 255.9121857, -327.2190552, 314.9458313
1: -99.7075195, 237.1099548, -106.6436996, 253.6793976, -353.3868713, 343.7536621
2: -84.5704346, 261.3007202, -90.3612213, 279.5265808, -364.0969849, 351.6619263
3: -88.7838135, 339.7810364, -94.8945618, 363.2659912, -452.0498047, 434.6755981
4: -75.9495926, 308.9975891, -80.9715195, 330.4645691, -406.4141541, 389.9691162

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7824322, upper bound: 339.7744300
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7808207, upper bound: 339.7677703
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -68.6890717, 229.9647522, -71.8390808, 243.1854858, -311.8745117, 301.8038025
1: -96.3805618, 228.3121796, -100.5396194, 241.0207520, -337.4012756, 328.8518066
2: -81.7298965, 251.5602570, -85.2065430, 265.5329590, -347.2628479, 336.7667847
3: -85.7649078, 326.9635315, -89.5862579, 345.5840149, -431.3489380, 416.5498047
4: -73.3111420, 297.4329834, -76.6075668, 313.9960938, -387.3072205, 374.0404968

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7827086, upper bound: 339.7719070
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7827086, upper bound: 339.7741720
time: 1.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -71.0041809, 237.2990570, -71.7675323, 242.9426270, -313.9467773, 309.0665283
1: -99.5878906, 235.6744232, -100.4377747, 240.7796173, -340.3674927, 336.1121521
2: -84.4718628, 259.6318359, -85.1208572, 265.2667847, -349.7386475, 344.7526245
3: -88.6048279, 337.2171326, -89.4957199, 345.2359009, -433.8407288, 426.7128601
4: -75.7413788, 306.7381897, -76.5312424, 313.6790161, -389.4204102, 383.2694092

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7842786, upper bound: 339.7739193
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7834881, upper bound: 339.7741513
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -68.6890717, 229.9647522, -79.6534348, 268.5899353, -337.2789612, 309.6181946
1: -96.3805618, 228.3121796, -111.7165527, 266.2981262, -362.6786804, 340.0287476
2: -81.7298965, 251.5602570, -94.6702423, 293.4072876, -375.1371765, 346.2304993
3: -85.7649078, 326.9635315, -99.4261398, 381.4895325, -467.2544556, 426.3896484
4: -73.3111420, 297.4329834, -84.8489075, 346.9851990, -420.2963257, 382.2818909

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7829813, upper bound: 339.7719600
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7827086, upper bound: 339.7735350
time: 1.14 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.02 + 417.29 = 420.31 seconds
