## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 176.11861014800002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625)
1: (-113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706)
2: (-160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044)
3: (-81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450)
4: (-173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692)

## BASE Result
execution time: IAR + LP analysis = 2.11 + 1.59 = 3.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -191.4338822, upper bound: 191.4338822


# Binary Search by BASE starts (time budget: 1196.29 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=279.9653625488281
rel_dist={0: [-191.43388219474815, 191.4338821947481]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=279.9653625488281
rel_dist={0: [-191.43347747308115, 191.4334774730812]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=279.9653625488281
rel_dist={0: [-191.43286165766193, 191.43286165766187]}

## Binary search (step 3) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=279.9653625488281
rel_dist={0: [-191.43192722771326, 191.43192722771323]}

## Binary search (step 4) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=279.9653625488281
rel_dist={0: [-191.4314584819337, 191.43145848193365]}

## Binary search (step 5) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=279.9653625488281
rel_dist={0: [-191.4312218785597, 191.4312218785597]}

## Binary search (step 6) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=279.9653625488281
rel_dist={0: [-191.43110226199482, 191.43110226199485]}

## Binary search (step 7) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=279.9653625488281
rel_dist={0: [-191.43104245379186, 191.43104245379186]}

## Binary search (step 8) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=279.9653625488281
rel_dist={0: [-191.4310076995096, 191.43100769950962]}

## Binary search (step 9) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=279.9653625488281
rel_dist={0: [-191.43098889644278, 191.43098889644278]}

## Binary search (step 10) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=279.9653625488281
rel_dist={0: [-191.4309794949129, 191.4309794949129]}

## Binary search (step 11) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=279.9653625488281
rel_dist={0: [-191.43097479415502, 191.43097479415508]}

## Binary search (step 12) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=279.9653625488281
rel_dist={0: [-191.43097244379015, 191.4309724437902]}

## Binary search (step 13) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=279.9653625488281
rel_dist={0: [-191.43097126863566, 191.43097126863574]}

## Binary search (step 14) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=279.9653625488281
rel_dist={0: [-191.43097068111345, 191.43097068111354]}

## Binary search (step 15) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=279.9653625488281
rel_dist={0: [-191.4309703874591, 191.43097038745907]}

## Binary search (step 16) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=279.9653625488281
rel_dist={0: [-191.43097024083306, 191.43097024083306]}

## Binary search (step 17) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=279.9653625488281
rel_dist={0: [-191.4309702318314, 191.43097076613515]}

## Binary search (step 18) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=279.9653625488281
rel_dist={0: [-191.4309717958775, 191.43097291164423]}

## Binary Search Result
Binary search time: 69.40 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1126.89 seconds

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.6756343, upper bound: 188.5359257
time: 0.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1353896, upper bound: 191.1353919
time: 0.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.35 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 0, lower bound: -179.6756343, upper bound: 188.5359257
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 0, lower bound: -191.1353896, upper bound: 191.1353919

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -165.7317810, 169.9891968, -131.9812012, 138.2930145, -304.0247803, 301.9703979
1: -136.9342194, 156.5604401, -109.0557861, 126.4673157, -263.4014893, 265.6161804
2: -192.2410583, 168.2551727, -153.1938171, 138.4591217, -330.7001343, 321.4489746
3: -91.3198547, 188.0007629, -80.2488022, 150.3314209, -241.6512756, 268.2495117
4: -208.4816437, 177.5048828, -166.3941803, 146.1310120, -354.6125488, 343.8990479

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.7747242, upper bound: 177.7747242
time: 0.73 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.7747242, upper bound: 188.5359257
time: 0.63 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -138.0393982, 141.9259644, -268.2368774, 272.8351440
1: -104.3817749, 123.2117767, -113.9848785, 129.8824921, -234.2642670, 237.1966400
2: -146.5827942, 135.2793121, -160.1121521, 141.6186523, -288.2014465, 295.3914795
3: -78.7137909, 144.2006378, -81.8182144, 156.6791382, -235.3929138, 226.0188599
4: -159.2684784, 142.8177032, -173.9111633, 149.4060059, -308.6744995, 316.7288818

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.5359257, upper bound: 179.6756343
time: 0.61 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.5359257, upper bound: 191.1353919
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.40 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -177.7747242, upper bound: 177.7747242
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -177.7747242, upper bound: 188.5359257
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -188.5359257, upper bound: 179.6756343
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -188.5359257, upper bound: 191.1353919

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -165.7317810, 169.9891968, -165.7308350, 169.9760437, -335.7078247, 335.7199402
1: -136.9342194, 156.5604401, -136.9331818, 156.5503845, -293.4845886, 293.4936218
2: -192.2410583, 168.2551727, -192.2384796, 168.2447968, -360.4857178, 360.4936523
3: -91.3198547, 188.0007629, -91.3181000, 187.9923859, -279.3122253, 279.3188171
4: -208.4816437, 177.5048828, -208.4785309, 177.4915466, -385.9732056, 385.9833984

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.1128964, upper bound: 171.9770942
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.3929760, upper bound: 170.3929760
time: 0.65 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -165.7317810, 169.9891968, -126.0566635, 134.6618042, -300.3935852, 296.0458374
1: -136.9342194, 156.5604401, -104.1840591, 123.0892410, -260.0234070, 260.7445068
2: -192.2410583, 168.2551727, -146.3088684, 135.1582336, -327.3991699, 314.5640259
3: -91.3198547, 188.0007629, -78.6480179, 143.9691772, -235.2890320, 266.6487122
4: -208.4816437, 177.5048828, -158.9604797, 142.6918640, -351.1735229, 336.4653320

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.1128964, upper bound: 184.2725163
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -170.3929760, upper bound: 180.8551704
time: 0.57 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -165.7317810, 169.9891968, -296.3001099, 300.5275269
1: -104.3817749, 123.2117767, -136.9342194, 156.5604401, -260.9421997, 260.1459961
2: -146.5827942, 135.2793121, -192.2410583, 168.2551727, -314.8379517, 327.5202942
3: -78.7137909, 144.2006378, -91.3198547, 188.0007629, -266.7145081, 235.5204926
4: -159.2684784, 142.8177032, -208.4816437, 177.5048828, -336.7733765, 351.2993469

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.5022987, upper bound: 173.8431734
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.8551670, upper bound: 173.6338378
time: 0.70 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -126.3109131, 134.7957306, -261.1066284, 261.1066284
1: -104.3817749, 123.2117767, -104.3817749, 123.2117767, -227.5935364, 227.5935211
2: -146.5827942, 135.2793121, -146.5827942, 135.2793121, -281.8620911, 281.8620911
3: -78.7137909, 144.2006378, -78.7137909, 144.2006378, -222.9144287, 222.9144287
4: -159.2684784, 142.8177032, -159.2684784, 142.8177032, -302.0861816, 302.0861816

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.5023009, upper bound: 186.7113725
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.8551704, upper bound: 186.6846139
time: 0.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.72 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.72
Output dim: 0, lower bound: -176.1128964, upper bound: 171.9770942
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.72
Output dim: 0, lower bound: -170.3929760, upper bound: 170.3929760
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 0, lower bound: -176.1128964, upper bound: 184.2725163
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 0, lower bound: -170.3929760, upper bound: 180.8551704
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 0, lower bound: -184.5022987, upper bound: 173.8431734
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 0, lower bound: -180.8551670, upper bound: 173.6338378
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 0, lower bound: -184.5023009, upper bound: 186.7113725
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 0, lower bound: -180.8551704, upper bound: 186.6846139

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -150.1845093, 158.3305664, -126.0566635, 134.6618042, -284.8463135, 284.3871765
1: -124.1818390, 145.5034790, -104.1840591, 123.0892410, -247.2710266, 249.6875305
2: -174.3076477, 156.9754639, -146.3088684, 135.1582336, -309.4658203, 303.2843323
3: -86.1170044, 170.7208557, -78.6480179, 143.9691772, -230.0861816, 249.3688507
4: -188.9908142, 165.6596985, -158.9604797, 142.6918640, -331.6826782, 324.6201172

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.6338378, upper bound: 180.8551670
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.6338378, upper bound: 180.8551670
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -149.3858643, 158.0441895, -125.8630981, 134.5245056, -283.9103699, 283.9072571
1: -123.6372223, 145.3979187, -104.0259247, 122.9601822, -246.5973969, 249.4238434
2: -173.3423462, 156.5578461, -146.0834656, 135.0372009, -308.3795471, 302.6412659
3: -85.9223938, 170.1979218, -78.5887680, 143.7519073, -229.6743011, 248.7866516
4: -187.6243896, 165.1948853, -158.7129211, 142.5664978, -330.1908875, 323.9078064

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.5939908, upper bound: 180.5254645
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.6338378, upper bound: 180.8551670
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -112.0257797, 124.0801163, -165.7317810, 169.9891968, -282.0149231, 289.8118896
1: -92.6331711, 113.0716934, -136.9342194, 156.5604401, -249.1936035, 250.0059204
2: -130.0409088, 125.1827774, -192.2410583, 168.2551727, -298.2960205, 317.4237366
3: -74.0341339, 128.2513275, -91.3198547, 188.0007629, -262.0348511, 219.5711823
4: -141.3389282, 132.2405701, -208.4816437, 177.5048828, -318.8438110, 340.7221680

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.4491149, upper bound: 173.7918620
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.8551670, upper bound: 173.6338378
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.8551670, upper bound: 173.6338378
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -109.6574478, 121.9550476, -165.5601349, 169.8690491, -279.5264893, 287.5151672
1: -90.7688217, 111.0725479, -136.7947083, 156.4484558, -247.2172852, 247.8672485
2: -127.1920471, 123.9425659, -192.0430756, 168.1382599, -295.3303223, 315.9856262
3: -73.1300201, 125.2666092, -91.2646484, 187.8170013, -260.9470215, 216.5312500
4: -137.8858032, 130.8748932, -208.2643738, 177.3821106, -315.2678528, 339.1392822

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.7225891, upper bound: 173.5655171
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.8551670, upper bound: 173.6338378
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.8551670, upper bound: 173.6338378
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -112.0257797, 124.0801163, -126.3109131, 134.7957306, -246.8215027, 250.3910065
1: -92.6331711, 113.0716934, -104.3817749, 123.2117767, -215.8449402, 217.4534607
2: -130.0409088, 125.1827774, -146.5827942, 135.2793121, -265.3201599, 271.7655640
3: -74.0341339, 128.2513275, -78.7137909, 144.2006378, -218.2347717, 206.9651031
4: -141.3389282, 132.2405701, -159.2684784, 142.8177032, -284.1566162, 291.5090332

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.6839557, upper bound: 186.6846104
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.6839557, upper bound: 186.6846104
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -109.6574478, 121.9550476, -126.1161804, 134.6578674, -244.3152924, 248.0712280
1: -90.7688217, 111.0725479, -104.2228088, 123.0821915, -213.8510132, 215.2953491
2: -127.1920471, 123.9425659, -146.3562622, 135.1577148, -262.3497314, 270.2988281
3: -73.1300201, 125.2666092, -78.6542053, 143.9823303, -217.1123505, 203.9208069
4: -137.8858032, 130.8748932, -159.0196686, 142.6917114, -280.5773926, 289.8945618

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7843643, upper bound: 186.6389382
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347796, upper bound: 185.7386919
time: 0.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.45 seconds
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -173.6338378, upper bound: 180.8551670
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -173.6338378, upper bound: 180.8551670
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -173.5939908, upper bound: 180.5254645
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -173.6338378, upper bound: 180.8551670
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -180.8551670, upper bound: 173.6338378
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -180.8551670, upper bound: 173.6338378
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -180.8551670, upper bound: 173.6338378
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -180.8551670, upper bound: 173.6338378
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -186.6839557, upper bound: 186.6846104
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -186.6839557, upper bound: 186.6846104
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -185.7843643, upper bound: 186.6389382
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -185.7347796, upper bound: 185.7386919

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -150.1845093, 158.3305664, -111.9550552, 124.0426331, -274.2271423, 270.2856140
1: -124.1818390, 145.5034790, -92.5776672, 113.0374374, -237.2192383, 238.0811462
2: -174.3076477, 156.9754639, -129.9642029, 125.1493378, -299.4569702, 286.9396667
3: -86.1170044, 170.7208557, -74.0159073, 128.1865387, -214.3035126, 244.7367554
4: -188.9908142, 165.6596985, -141.2522125, 132.2058716, -321.1966553, 306.9119263

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.4879075, upper bound: 184.1712877
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.4165011, upper bound: 183.1923507
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8738464, upper bound: 182.8624044
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -150.1845093, 158.3305664, -109.4051819, 121.8282166, -272.0127258, 267.7357178
1: -124.1818390, 145.5034790, -90.5728226, 110.9506683, -235.1325073, 236.0762939
2: -174.3076477, 156.9754639, -126.9218140, 123.8235092, -298.1310730, 283.8972778
3: -86.1170044, 170.7208557, -73.0652924, 125.0379715, -211.1549377, 243.7861481
4: -188.9908142, 165.6596985, -137.5807800, 130.7518005, -319.7426147, 303.2404480

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.4879075, upper bound: 184.1712836
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.4165011, upper bound: 183.1923548
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8738464, upper bound: 182.8624044
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -148.9946136, 157.7889252, -100.0438232, 117.3660202, -266.3606262, 257.8327332
1: -123.3195496, 145.1629944, -82.9620209, 107.0355606, -230.3551025, 228.1250153
2: -172.8952789, 156.3115387, -116.5249939, 119.4649353, -292.3602295, 272.8365173
3: -85.7932739, 169.7784729, -70.6660461, 116.1025085, -201.8957825, 240.4445190
4: -187.1355743, 164.9366608, -126.4828644, 126.3498535, -313.4854126, 291.4195251

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.5204922, upper bound: 180.3896600
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.6487517, upper bound: 179.6080733
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.3443597, upper bound: 180.2808492
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -149.3858643, 158.0441895, -118.4635773, 129.3534241, -278.7392578, 276.5076599
1: -123.6372223, 145.3979187, -97.9063873, 118.1397858, -241.7769928, 243.3043060
2: -173.3423462, 156.5578461, -137.4764557, 130.5394287, -303.8817749, 294.0342407
3: -85.9223938, 170.1979218, -76.3804016, 135.5371704, -221.4595642, 246.5783234
4: -187.6243896, 165.1948853, -149.4171600, 137.8908386, -325.5152283, 314.6120605

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.5655171, upper bound: 180.7225891
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.7266227, upper bound: 180.0535393
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.3918844, upper bound: 180.6152132
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -112.0257797, 124.0801163, -150.1845093, 158.3305664, -270.3562927, 274.2646179
1: -92.6331711, 113.0716934, -124.1818390, 145.5034790, -238.1366577, 237.2535400
2: -130.0409088, 125.1827774, -174.3076477, 156.9754639, -287.0163574, 299.4903870
3: -74.0341339, 128.2513275, -86.1170044, 170.7208557, -244.7549896, 214.3683319
4: -141.3389282, 132.2405701, -188.9908142, 165.6596985, -306.9986267, 321.2313843

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.5022987, upper bound: 173.8428402
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.5022987, upper bound: 173.8431734
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -112.0257797, 124.0801163, -149.3858643, 158.0441895, -270.0698853, 273.4659424
1: -92.6331711, 113.0716934, -123.6372223, 145.3979187, -238.0310974, 236.7089233
2: -130.0409088, 125.1827774, -173.3423462, 156.5578461, -286.5986633, 298.5251160
3: -74.0341339, 128.2513275, -85.9223938, 170.1979218, -244.2320557, 214.1737213
4: -141.3389282, 132.2405701, -187.6243896, 165.1948853, -306.5338135, 319.8649597

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.5022987, upper bound: 173.8428402
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.5022987, upper bound: 173.8431734
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -109.6574478, 121.9550476, -150.1845093, 158.3305664, -267.9880066, 272.1395569
1: -90.7688217, 111.0725479, -124.1818390, 145.5034790, -236.2723083, 235.2543945
2: -127.1920471, 123.9425659, -174.3076477, 156.9754639, -284.1675110, 298.2501831
3: -73.1300201, 125.2666092, -86.1170044, 170.7208557, -243.8508759, 211.3836060
4: -137.8858032, 130.8748932, -188.9908142, 165.6596985, -303.5454102, 319.8657227

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.5254645, upper bound: 173.5939908
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.8551670, upper bound: 173.6338378
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -109.6574478, 121.9550476, -149.3858643, 158.0441895, -267.7015686, 271.3409119
1: -90.7688217, 111.0725479, -123.6372223, 145.3979187, -236.1667480, 234.7097626
2: -127.1920471, 123.9425659, -173.3423462, 156.5578461, -283.7498779, 297.2849121
3: -73.1300201, 125.2666092, -85.9223938, 170.1979218, -243.3279419, 211.1889954
4: -137.8858032, 130.8748932, -187.6243896, 165.1948853, -303.0806580, 318.4992676

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.5254645, upper bound: 173.5939908
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.8551670, upper bound: 173.6338378
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -112.0257797, 124.0801163, -112.0257797, 124.0801163, -236.1058807, 236.1058960
1: -92.6331711, 113.0716934, -92.6331711, 113.0716934, -205.7048645, 205.7048645
2: -130.0409088, 125.1827774, -130.0409088, 125.1827774, -255.2236938, 255.2236328
3: -74.0341339, 128.2513275, -74.0341339, 128.2513275, -202.2854614, 202.2854614
4: -141.3389282, 132.2405701, -141.3389282, 132.2405701, -273.5794983, 273.5794678

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6610607, upper bound: 185.7965131
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.5343429, upper bound: 185.7966922
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -112.0257797, 124.0801163, -109.6574478, 121.9550476, -233.9808044, 233.7375641
1: -92.6331711, 113.0716934, -90.7688217, 111.0725479, -203.7057190, 203.8405151
2: -130.0409088, 125.1827774, -127.1920471, 123.9425659, -253.9834747, 252.3748169
3: -74.0341339, 128.2513275, -73.1300201, 125.2666092, -199.3007507, 201.3813477
4: -141.3389282, 132.2405701, -137.8858032, 130.8748932, -272.2138062, 270.1262817

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6610607, upper bound: 185.7965131
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.5343429, upper bound: 185.7966922
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -109.2380371, 121.6821213, -100.0438232, 117.3660202, -226.6040649, 221.7259064
1: -90.4246216, 110.8098526, -82.9620209, 107.0355606, -197.4601746, 193.7718658
2: -126.7065048, 123.6747360, -116.5249939, 119.4649353, -246.1714478, 240.1997070
3: -72.9934311, 124.8013992, -70.6660461, 116.1025085, -189.0959473, 195.4674377
4: -137.3587341, 130.5968323, -126.4828644, 126.3498535, -263.7085571, 257.0797119

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -109.6574478, 121.9550476, -118.6066208, 129.4276428, -239.0850830, 240.5616760
1: -90.7688217, 111.0725479, -98.0182877, 118.2079926, -208.9768066, 209.0908356
2: -127.1920471, 123.9425659, -137.6315918, 130.6060791, -257.7980957, 261.5741577
3: -73.1300201, 125.2666092, -76.4170532, 135.6676178, -208.7976379, 201.6836548
4: -137.8858032, 130.8748932, -149.5916443, 137.9600220, -275.8457336, 280.4665527

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347796, upper bound: 185.7386919
time: 0.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.49 seconds
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -178.4165011, upper bound: 183.1923507
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -177.8738464, upper bound: 182.8624044
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -178.4165011, upper bound: 183.1923548
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -177.8738464, upper bound: 182.8624044
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -172.6487517, upper bound: 179.6080733
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -173.3443597, upper bound: 180.2808492
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -172.7266227, upper bound: 180.0535393
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -173.3918844, upper bound: 180.6152132
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -184.5022987, upper bound: 173.8428402
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -184.5022987, upper bound: 173.8431734
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -184.5022987, upper bound: 173.8428402
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -184.5022987, upper bound: 173.8431734
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -180.5254645, upper bound: 173.5939908
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -180.8551670, upper bound: 173.6338378
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -180.5254645, upper bound: 173.5939908
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -180.8551670, upper bound: 173.6338378
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -190.6610607, upper bound: 185.7965131
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -190.5343429, upper bound: 185.7966922
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -190.6610607, upper bound: 185.7965131
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -190.5343429, upper bound: 185.7966922
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -185.7347796, upper bound: 185.7386919

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -133.2034607, 147.1067047, -111.9550552, 124.0426331, -257.2460938, 259.0617371
1: -110.4020233, 134.9165955, -92.5776672, 113.0374374, -223.4394073, 227.4942627
2: -154.8717651, 145.9841461, -129.9642029, 125.1493378, -280.0211182, 275.9483643
3: -81.0192032, 152.4372559, -74.0159073, 128.1865387, -209.2057495, 226.4531555
4: -167.6655273, 154.3106689, -141.2522125, 132.2058716, -299.8713379, 295.5628662

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.5065452, upper bound: 188.1651016
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.5065452, upper bound: 188.0401401
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -138.3767853, 149.9827576, -111.9550552, 124.0426331, -262.4194336, 261.9378052
1: -114.4383926, 137.5594788, -92.5776672, 113.0374374, -227.4758148, 230.1371460
2: -160.5395813, 148.8502045, -129.9642029, 125.1493378, -285.6889038, 278.8143921
3: -82.6676102, 156.7407684, -74.0159073, 128.1865387, -210.8541260, 230.7566833
4: -174.0542450, 157.1458282, -141.2522125, 132.2058716, -306.2601013, 298.3980408

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.9159582, upper bound: 187.9315055
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.9159582, upper bound: 187.8383189
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -133.2034607, 147.1067047, -109.4051819, 121.8282166, -255.0316772, 256.5118408
1: -110.4020233, 134.9165955, -90.5728226, 110.9506683, -221.3526611, 225.4894104
2: -154.8717651, 145.9841461, -126.9218140, 123.8235092, -278.6952209, 272.9059448
3: -81.0192032, 152.4372559, -73.0652924, 125.0379715, -206.0571747, 225.5025482
4: -167.6655273, 154.3106689, -137.5807800, 130.7518005, -298.4173279, 291.8913879

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.4165018, upper bound: 183.1923548
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.4165011, upper bound: 182.2516076
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -138.3767853, 149.9827576, -109.4051819, 121.8282166, -260.2049866, 259.3879395
1: -114.4383926, 137.5594788, -90.5728226, 110.9506683, -225.3890686, 228.1322937
2: -160.5395813, 148.8502045, -126.9218140, 123.8235092, -284.3630371, 275.7720337
3: -82.6676102, 156.7407684, -73.0652924, 125.0379715, -207.7055511, 229.8060608
4: -174.0542450, 157.1458282, -137.5807800, 130.7518005, -304.8060303, 294.7265320

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8738464, upper bound: 182.8624044
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8711018, upper bound: 181.8741509
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -129.7377014, 146.7494049, -98.4322052, 116.4923096, -246.2299957, 245.1816101
1: -107.0335541, 135.3816223, -81.6003571, 106.0244370, -213.0579681, 216.9819794
2: -149.9635010, 146.3835754, -114.6106796, 118.6053085, -268.5688171, 260.9942627
3: -80.5001678, 147.9834290, -70.2596436, 114.2749023, -194.7750549, 218.2430725
4: -163.0087891, 154.3884277, -124.4631577, 125.4651566, -288.4739380, 278.8515930

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.6487517, upper bound: 179.6080693
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.6487517, upper bound: 179.6080693
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -144.7516327, 155.0637665, -100.0438232, 117.3660202, -262.1175842, 255.1075592
1: -119.8011017, 142.6269684, -82.9620209, 107.0355606, -226.8366699, 225.5889740
2: -167.9449463, 153.7201385, -116.5249939, 119.4649353, -287.4098816, 270.2451172
3: -84.3235550, 165.0267029, -70.6660461, 116.1025085, -200.4260559, 235.6927490
4: -181.7902832, 162.2114105, -126.4828644, 126.3498535, -308.1401062, 288.6942444

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.3443591, upper bound: 180.2808452
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.3443591, upper bound: 180.2808492
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -130.0292511, 146.9473877, -116.3291397, 127.9847565, -258.0140076, 263.2765198
1: -107.2738342, 135.5659943, -96.1358871, 116.8663025, -224.1400757, 231.7018738
2: -150.3018036, 146.5797424, -134.9826050, 129.4120178, -279.7138062, 281.5623474
3: -80.5982056, 148.3070068, -75.8034286, 133.1335754, -213.7317810, 224.1104279
4: -163.3715210, 154.5930634, -146.7257538, 136.7198792, -300.0914001, 301.3187866

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.7266227, upper bound: 180.0535361
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.7266227, upper bound: 180.0535393
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -145.1236725, 155.3142090, -118.4635773, 129.3534241, -274.4770813, 273.7777710
1: -120.1042175, 142.8575745, -97.9063873, 118.1397858, -238.2439728, 240.7639618
2: -168.3727112, 153.9620667, -137.4764557, 130.5394287, -298.9121399, 291.4384766
3: -84.4498138, 165.4356689, -76.3804016, 135.5371704, -219.9869843, 241.8160706
4: -182.2562256, 162.4646301, -149.4171600, 137.8908386, -320.1470642, 311.8817444

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.3918844, upper bound: 180.6152132
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.3918848, upper bound: 180.6152166
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -149.7921143, 158.0719452, -247.3513641, 259.1456299
1: -74.0222931, 98.9946747, -123.8619766, 145.2650299, -219.2873230, 222.8566284
2: -103.9738922, 111.5052643, -173.8566284, 156.7256470, -260.6995239, 285.3618774
3: -67.1468887, 104.1213226, -85.9854584, 170.2961273, -237.4430084, 190.1067810
4: -113.0477753, 118.0451050, -188.4992981, 165.3972015, -278.4449768, 306.5444031

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3892508, upper bound: 179.6645814
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1650989, upper bound: 179.5065452
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.9315055, upper bound: 178.9159582
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -150.1845093, 158.3305664, -263.3902283, 269.3696594
1: -86.8530884, 108.5048981, -124.1818390, 145.5034790, -232.3565674, 232.6867371
2: -121.9298782, 120.9879761, -174.3076477, 156.9754639, -278.9053345, 295.2955933
3: -71.9708786, 120.4962387, -86.1170044, 170.7208557, -242.6917114, 206.6132507
4: -132.5959625, 127.8747635, -188.9908142, 165.6596985, -298.2556152, 316.8655396

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2462810, upper bound: 179.6643055
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.0401401, upper bound: 179.5065452
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8383189, upper bound: 178.9159582
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -148.9946136, 157.7889252, -247.0683594, 258.3481140
1: -74.0222931, 98.9946747, -123.3195496, 145.1629944, -219.1852875, 222.3141785
2: -103.9738922, 111.5052643, -172.8952789, 156.3115387, -260.2854309, 284.4005432
3: -67.1468887, 104.1213226, -85.7932739, 169.7784729, -236.9253540, 189.9145966
4: -113.0477753, 118.0451050, -187.1355743, 164.9366608, -277.9844360, 305.1806641

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.4491149, upper bound: 173.7916243
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.6226537, upper bound: 172.9928904
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.2673015, upper bound: 173.6097772
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -149.3858643, 158.0441895, -263.1038513, 268.5710144
1: -86.8530884, 108.5048981, -123.6372223, 145.3979187, -232.2510071, 232.1421204
2: -121.9298782, 120.9879761, -173.3423462, 156.5578461, -278.4877319, 294.3303223
3: -71.9708786, 120.4962387, -85.9223938, 170.1979218, -242.1687775, 206.4186401
4: -132.5959625, 127.8747635, -187.6243896, 165.1948853, -297.7908325, 315.4991455

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.4491149, upper bound: 173.7918620
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.6226537, upper bound: 172.9951741
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.2673015, upper bound: 173.6109362
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -149.7921143, 158.0719452, -245.7755737, 258.5236511
1: -72.8436890, 98.2632751, -123.8619766, 145.2650299, -218.1087036, 222.1252441
2: -102.0504456, 110.7943802, -173.8566284, 156.7256470, -258.7760925, 284.6509705
3: -66.5469589, 102.0576096, -85.9854584, 170.2961273, -236.8430786, 188.0430603
4: -110.6790924, 117.2479630, -188.4992981, 165.3972015, -276.0762939, 305.7472534

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.1712836, upper bound: 179.4879075
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.1923507, upper bound: 178.4165011
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.8624044, upper bound: 177.8738464
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -150.1845093, 158.3305664, -261.4943542, 268.1583252
1: -85.3876343, 106.9490433, -124.1818390, 145.5034790, -230.8910980, 231.1308899
2: -119.6347275, 120.0215302, -174.3076477, 156.9754639, -276.6101990, 294.3291626
3: -71.2133255, 118.0576172, -86.1170044, 170.7208557, -241.9341736, 204.1746063
4: -129.7596588, 126.8023834, -188.9908142, 165.6596985, -295.4193420, 315.7931519

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.4990397, upper bound: 179.4454842
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.2516076, upper bound: 178.4165011
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8741509, upper bound: 177.8711018
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -148.9946136, 157.7889252, -245.4925537, 257.7261353
1: -72.8436890, 98.2632751, -123.3195496, 145.1629944, -218.0066681, 221.5827789
2: -102.0504456, 110.7943802, -172.8952789, 156.3115387, -258.3619690, 283.6896362
3: -66.5469589, 102.0576096, -85.7932739, 169.7784729, -236.3254242, 187.8508911
4: -110.6790924, 117.2479630, -187.1355743, 164.9366608, -275.6157532, 304.3835144

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.3896600, upper bound: 173.5204922
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.6080693, upper bound: 172.6487517
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.2808452, upper bound: 173.3443591
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -149.3858643, 158.0441895, -261.2079773, 267.3596802
1: -85.3876343, 106.9490433, -123.6372223, 145.3979187, -230.7855225, 230.5862427
2: -119.6347275, 120.0215302, -173.3423462, 156.5578461, -276.1925354, 293.3638916
3: -71.2133255, 118.0576172, -85.9223938, 170.1979218, -241.4112396, 203.9800110
4: -129.7596588, 126.8023834, -187.6243896, 165.1948853, -294.9545288, 314.4267578

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.7225891, upper bound: 173.5655171
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.0535361, upper bound: 172.7266227
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.6152132, upper bound: 173.3918844
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -111.5843811, 123.7866211, -213.0660553, 220.9378815
1: -74.0222931, 98.9946747, -92.2709656, 112.7995148, -186.8218079, 191.2656097
2: -103.9738922, 111.5052643, -129.5300293, 124.9066772, -228.8805695, 241.0352936
3: -67.1468887, 104.1213226, -73.8924255, 127.7664642, -194.9133606, 178.0137482
4: -113.0477753, 118.0451050, -140.7826996, 131.9527435, -245.0004883, 258.8277893

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9715596, upper bound: 190.9691246
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9715596, upper bound: 190.9691246
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -112.0257797, 124.0801163, -229.1398010, 231.2109375
1: -86.8530884, 108.5048981, -92.6331711, 113.0716934, -199.9247742, 201.1380615
2: -121.9298782, 120.9879761, -130.0409088, 125.1827774, -247.1126556, 251.0288849
3: -71.9708786, 120.4962387, -74.0341339, 128.2513275, -200.2221985, 194.5303650
4: -132.5959625, 127.8747635, -141.3389282, 132.2405701, -264.8365173, 269.2136841

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9715596, upper bound: 190.9693645
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9715596, upper bound: 190.9693645
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -109.2380371, 121.6821213, -210.9615479, 218.5915375
1: -74.0222931, 98.9946747, -90.4246216, 110.8098526, -184.8321533, 189.4192810
2: -103.9738922, 111.5052643, -126.7065048, 123.6747360, -227.6486206, 238.2117615
3: -67.1468887, 104.1213226, -72.9934311, 124.8013992, -191.9482880, 177.1147461
4: -113.0477753, 118.0451050, -137.3587341, 130.5968323, -243.6446075, 255.4038239

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.8286070, upper bound: 185.7965131
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.8286134, upper bound: 185.7965131
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -109.6574478, 121.9550476, -227.0147400, 228.8425903
1: -86.8530884, 108.5048981, -90.7688217, 111.0725479, -197.9256287, 199.2737122
2: -121.9298782, 120.9879761, -127.1920471, 123.9425659, -245.8724365, 248.1800232
3: -71.9708786, 120.4962387, -73.1300201, 125.2666092, -197.2374878, 193.6262512
4: -132.5959625, 127.8747635, -137.8858032, 130.8748932, -263.4708557, 265.7605286

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.8286070, upper bound: 185.7966922
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.8286134, upper bound: 185.7966922
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -100.0438232, 117.3660202, -205.0696564, 208.7753296
1: -72.8436890, 98.2632751, -82.9620209, 107.0355606, -179.8792419, 181.2252655
2: -102.0504456, 110.7943802, -116.5249939, 119.4649353, -221.5153809, 227.3193512
3: -66.5469589, 102.0576096, -70.6660461, 116.1025085, -182.6494751, 172.7236633
4: -110.6790924, 117.2479630, -126.4828644, 126.3498535, -237.0289459, 243.7308044

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7843643, upper bound: 186.6389382
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7843642, upper bound: 186.6389382
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -100.0438232, 117.3660202, -220.5298615, 218.0176392
1: -85.3876343, 106.9490433, -82.9620209, 107.0355606, -192.4231873, 189.9110413
2: -119.6347275, 120.0215302, -116.5249939, 119.4649353, -239.0996704, 236.5464783
3: -71.2133255, 118.0576172, -70.6660461, 116.1025085, -187.3158264, 188.7236633
4: -129.7596588, 126.8023834, -126.4828644, 126.3498535, -256.1094666, 253.2852478

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7843642, upper bound: 186.6389382
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7843643, upper bound: 186.6389382
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -118.6066208, 129.4276428, -217.1312866, 227.3381653
1: -72.8436890, 98.2632751, -98.0182877, 118.2079926, -191.0516815, 196.2815247
2: -102.0504456, 110.7943802, -137.6315918, 130.6060791, -232.6565247, 248.4259491
3: -66.5469589, 102.0576096, -76.4170532, 135.6676178, -202.2145691, 178.4746704
4: -110.6790924, 117.2479630, -149.5916443, 137.9600220, -248.6391144, 266.8395691

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347796, upper bound: 185.7386919
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -118.6066208, 129.4276428, -232.5914917, 236.5804291
1: -85.3876343, 106.9490433, -98.0182877, 118.2079926, -203.5956268, 204.9673157
2: -119.6347275, 120.0215302, -137.6315918, 130.6060791, -250.2407990, 257.6531372
3: -71.2133255, 118.0576172, -76.4170532, 135.6676178, -206.8809509, 194.4746704
4: -129.7596588, 126.8023834, -149.5916443, 137.9600220, -267.7196655, 276.3940125

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347796, upper bound: 185.7386919
time: 0.67 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.37 seconds
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -179.5065452, upper bound: 188.1651016
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -179.5065452, upper bound: 188.0401401
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -178.9159582, upper bound: 187.9315055
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -178.9159582, upper bound: 187.8383189
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -178.4165018, upper bound: 183.1923548
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -178.4165011, upper bound: 182.2516076
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -177.8738464, upper bound: 182.8624044
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -177.8711018, upper bound: 181.8741509
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -172.6487517, upper bound: 179.6080693
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -172.6487517, upper bound: 179.6080693
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -173.3443591, upper bound: 180.2808452
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -173.3443591, upper bound: 180.2808492
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -172.7266227, upper bound: 180.0535361
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -172.7266227, upper bound: 180.0535393
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -173.3918844, upper bound: 180.6152132
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -173.3918848, upper bound: 180.6152166
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -188.1650989, upper bound: 179.5065452
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -187.9315055, upper bound: 178.9159582
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -188.0401401, upper bound: 179.5065452
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -187.8383189, upper bound: 178.9159582
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -183.6226537, upper bound: 172.9928904
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -184.2673015, upper bound: 173.6097772
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -183.6226537, upper bound: 172.9951741
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -184.2673015, upper bound: 173.6109362
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -183.1923507, upper bound: 178.4165011
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -182.8624044, upper bound: 177.8738464
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -182.2516076, upper bound: 178.4165011
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -181.8741509, upper bound: 177.8711018
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -179.6080693, upper bound: 172.6487517
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -180.2808452, upper bound: 173.3443591
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -180.0535361, upper bound: 172.7266227
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -180.6152132, upper bound: 173.3918844
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -190.9715596, upper bound: 190.9691246
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -190.9715596, upper bound: 190.9691246
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -190.9715596, upper bound: 190.9693645
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -190.9715596, upper bound: 190.9693645
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -189.8286070, upper bound: 185.7965131
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -189.8286134, upper bound: 185.7965131
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -189.8286070, upper bound: 185.7966922
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -189.8286134, upper bound: 185.7966922
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -185.7843643, upper bound: 186.6389382
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -185.7843642, upper bound: 186.6389382
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -185.7843642, upper bound: 186.6389382
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -185.7843643, upper bound: 186.6389382
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -185.7347796, upper bound: 185.7386919
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.37
Output dim: 0, lower bound: -185.7347796, upper bound: 185.7386919

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -132.8635101, 146.8815002, -89.2794342, 109.3535004, -242.2170105, 236.1609192
1: -110.1251068, 134.7084198, -74.0222931, 98.9946747, -209.1197510, 208.7307129
2: -154.4828949, 145.7666321, -103.9738922, 111.5052643, -265.9881592, 249.7405090
3: -80.9037628, 152.0749207, -67.1468887, 104.1213226, -185.0250854, 219.2218018
4: -167.2418518, 154.0827026, -113.0477753, 118.0451050, -285.2869568, 267.1304932

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.4557785, upper bound: 188.0951582
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.9443451, upper bound: 183.7606617
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.3228483, upper bound: 187.9890995
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -133.2034607, 147.1067047, -105.0596924, 119.1851501, -252.3886108, 252.1663971
1: -110.4020233, 134.9165955, -86.8530884, 108.5048981, -218.9069214, 221.7696838
2: -154.8717651, 145.9841461, -121.9298782, 120.9879761, -275.8597412, 267.9140015
3: -81.0192032, 152.4372559, -71.9708786, 120.4962387, -201.5154419, 224.4081421
4: -167.6655273, 154.3106689, -132.5959625, 127.8747635, -295.5402527, 286.9066162

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.4555057, upper bound: 187.9680341
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.1002860, upper bound: 177.3466620
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.0655435, upper bound: 177.0643786
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -137.9932709, 149.7316742, -89.2794342, 109.3535004, -247.3467712, 239.0110931
1: -114.1260529, 137.3279266, -74.0222931, 98.9946747, -213.1207275, 211.3502197
2: -160.1003113, 148.6080170, -103.9738922, 111.5052643, -271.6055908, 252.5819092
3: -82.5416260, 156.3286591, -67.1468887, 104.1213226, -186.6629333, 223.4755554
4: -173.5762329, 156.8913116, -113.0477753, 118.0451050, -291.6213379, 269.9390869

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.8814175, upper bound: 187.8552001
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1314048, upper bound: 184.1309148
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.7003468, upper bound: 187.7758645
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -138.3767853, 149.9827576, -105.0596924, 119.1851501, -257.5619507, 255.0424500
1: -114.4383926, 137.5594788, -86.8530884, 108.5048981, -222.9432983, 224.4125519
2: -160.5395813, 148.8502045, -121.9298782, 120.9879761, -281.5275269, 270.7800903
3: -82.6676102, 156.7407684, -71.9708786, 120.4962387, -203.1638489, 228.7116394
4: -174.0542450, 157.1458282, -132.5959625, 127.8747635, -301.9289856, 289.7417297

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.8811418, upper bound: 187.7578689
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.7906176, upper bound: 177.2279754
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9681374, upper bound: 177.0089863
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -132.8635101, 146.8815002, -87.7036362, 108.7315369, -241.5950470, 234.5851135
1: -110.1251068, 134.7084198, -72.8436890, 98.2632751, -208.3883514, 207.5520935
2: -154.4828949, 145.7666321, -102.0504456, 110.7943802, -265.2772827, 247.8170776
3: -80.9037628, 152.0749207, -66.5469589, 102.0576096, -182.9613647, 218.6218872
4: -167.2418518, 154.0827026, -110.6790924, 117.2479630, -284.4897766, 264.7617493

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.3414380, upper bound: 183.0918288
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.2645398, upper bound: 179.7339467
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.1987269, upper bound: 182.9989340
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -133.2034607, 147.1067047, -103.0622864, 117.9308777, -251.1343384, 250.1689758
1: -110.4020233, 134.9165955, -85.3088760, 106.9066620, -217.3086548, 220.2254639
2: -154.8717651, 145.9841461, -119.5274124, 119.9788513, -274.8505554, 265.5115356
3: -81.0192032, 152.4372559, -71.1931610, 117.9752502, -198.9944458, 223.6303864
4: -167.6655273, 154.3106689, -129.6396484, 126.7591858, -294.4247131, 283.9503174

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.3380755, upper bound: 182.1517859
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.7550359, upper bound: 170.3114668
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -173.4577211, upper bound: 170.0929664
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -137.9932709, 149.7316742, -87.7036362, 108.7315369, -246.7248077, 237.4353027
1: -114.1260529, 137.3279266, -72.8436890, 98.2632751, -212.3893280, 210.1715851
2: -160.1003113, 148.6080170, -102.0504456, 110.7943802, -270.8946838, 250.6584167
3: -82.5416260, 156.3286591, -66.5469589, 102.0576096, -184.5992126, 222.8756104
4: -173.5762329, 156.8913116, -110.6790924, 117.2479630, -290.8241882, 267.5703735

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8092794, upper bound: 182.7494483
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.3365456, upper bound: 179.8042574
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.6243083, upper bound: 182.6443136
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -138.3767853, 149.9827576, -103.0622864, 117.9308777, -256.3076782, 253.0450439
1: -114.4383926, 137.5594788, -85.3088760, 106.9066620, -221.3450623, 222.8683472
2: -160.5395813, 148.8502045, -119.5274124, 119.9788513, -280.5183716, 268.3775940
3: -82.6676102, 156.7407684, -71.1931610, 117.9752502, -200.6428528, 227.9338837
4: -174.0542450, 157.1458282, -129.6396484, 126.7591858, -300.8134155, 286.7854614

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8008059, upper bound: 181.7689472
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.1649656, upper bound: 169.9993655
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -173.2631699, upper bound: 169.9293796
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -129.7377014, 146.7494049, -87.7436447, 108.5790100, -238.3167114, 234.4930420
1: -107.0335541, 135.3816223, -72.7271118, 98.1704102, -205.2039490, 208.1087189
2: -149.9635010, 146.3835754, -102.1450348, 110.7397079, -260.7031860, 248.5286102
3: -80.5001678, 147.9834290, -66.7829132, 102.3765106, -182.8766785, 214.7663422
4: -163.0087891, 154.3884277, -111.1220703, 117.2613220, -280.2700806, 265.5104980

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.5747506, upper bound: 179.4597892
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.6392012, upper bound: 179.3958632
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -170.9971788, upper bound: 178.1760282
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -129.7377014, 146.7494049, -86.1618347, 107.9491882, -237.6868896, 232.9112244
1: -107.0335541, 135.3816223, -71.5395966, 97.4825668, -204.5160828, 206.9212189
2: -149.9635010, 146.3835754, -100.2137604, 110.0167923, -259.9802856, 246.5973358
3: -80.5001678, 147.9834290, -66.1684647, 100.3223419, -180.8225098, 214.1518860
4: -163.0087891, 154.3884277, -108.7500687, 116.4526978, -279.4614868, 263.1384888

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.5747506, upper bound: 179.4597933
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.6392012, upper bound: 179.3958632
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -170.9971788, upper bound: 178.1760282
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -144.7516327, 155.0637665, -89.2794342, 109.3535004, -254.1050873, 244.3432007
1: -119.8011017, 142.6269684, -74.0222931, 98.9946747, -218.7957458, 216.6492615
2: -167.9449463, 153.7201385, -103.9738922, 111.5052643, -279.4501953, 257.6940308
3: -84.3235550, 165.0267029, -67.1468887, 104.1213226, -188.4448853, 232.1735840
4: -181.7902832, 162.2114105, -113.0477753, 118.0451050, -299.8353577, 275.2591858

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.2696756, upper bound: 180.1450075
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.3084916, upper bound: 179.9816038
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -168.2812351, upper bound: 175.6214869
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -144.7516327, 155.0637665, -87.7036362, 108.7315369, -253.4831543, 242.7673950
1: -119.8011017, 142.6269684, -72.8436890, 98.2632751, -218.0643616, 215.4706573
2: -167.9449463, 153.7201385, -102.0504456, 110.7943802, -278.7392883, 255.7705688
3: -84.3235550, 165.0267029, -66.5469589, 102.0576096, -186.3811646, 231.5736694
4: -181.7902832, 162.2114105, -110.6790924, 117.2479630, -299.0382080, 272.8904419

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.2696756, upper bound: 180.1450075
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.3084916, upper bound: 179.9816038
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -168.2812351, upper bound: 175.6214869
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -130.0292511, 146.9473877, -103.0650635, 117.8751068, -247.9043579, 250.0124512
1: -107.2738342, 135.5659943, -85.1862640, 107.2890472, -214.5628662, 220.7522583
2: -150.3018036, 146.5797424, -119.5905457, 119.9286118, -270.2303467, 266.1702881
3: -80.5982056, 148.3070068, -71.4306717, 118.2265091, -198.8247070, 219.7376709
4: -163.3715210, 154.5930634, -130.0896301, 126.7739029, -290.1454163, 284.6826782

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.6494725, upper bound: 179.9093310
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.6416784, upper bound: 179.3115764
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.0165182, upper bound: 178.3205941
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -130.0292511, 146.9473877, -101.4494019, 117.0731659, -247.1024170, 248.3967896
1: -107.2738342, 135.5659943, -83.9501190, 106.0555115, -213.3293304, 219.5161133
2: -150.3018036, 146.5797424, -117.6158981, 119.1305237, -269.4323120, 264.1956482
3: -80.5982056, 148.3070068, -70.7912598, 116.1366425, -196.7348328, 219.0982666
4: -163.3715210, 154.5930634, -127.6161880, 125.8867798, -289.2582703, 282.2092590

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.6494725, upper bound: 179.9093310
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.6416784, upper bound: 179.3115731
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -171.0165182, upper bound: 178.3205941
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -145.1236725, 155.3142090, -105.0596924, 119.1851501, -264.3088379, 260.3739014
1: -120.1042175, 142.8575745, -86.8530884, 108.5048981, -228.6091156, 229.7106628
2: -168.3727112, 153.9620667, -121.9298782, 120.9879761, -289.3606873, 275.8919373
3: -84.4498138, 165.4356689, -71.9708786, 120.4962387, -204.9460449, 237.4065552
4: -182.2562256, 162.4646301, -132.5959625, 127.8747635, -310.1309814, 295.0605164

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.3159738, upper bound: 180.4815677
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.2879496, upper bound: 179.7651124
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -168.3688669, upper bound: 176.3271476
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -145.1236725, 155.3142090, -103.0622864, 117.9308777, -263.0545654, 258.3764954
1: -120.1042175, 142.8575745, -85.3088760, 106.9066620, -227.0108490, 228.1664429
2: -168.3727112, 153.9620667, -119.5274124, 119.9788513, -288.3515625, 273.4894409
3: -84.4498138, 165.4356689, -71.1931610, 117.9752502, -202.4250641, 236.6287994
4: -182.2562256, 162.4646301, -129.6396484, 126.7591858, -309.0154114, 292.1042480

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.3159738, upper bound: 180.4815677
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.2879496, upper bound: 179.7651161
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -168.3688664, upper bound: 176.3271442
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -132.8635101, 146.8815002, -236.1609192, 242.2170105
1: -74.0222931, 98.9946747, -110.1251068, 134.7084198, -208.7307129, 209.1197510
2: -103.9738922, 111.5052643, -154.4828949, 145.7666321, -249.7405090, 265.9881592
3: -67.1468887, 104.1213226, -80.9037628, 152.0749207, -219.2218018, 185.0250854
4: -113.0477753, 118.0451050, -167.2418518, 154.0827026, -267.1304932, 285.2869568

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7864280, upper bound: 178.9154693
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7864280, upper bound: 178.9159582
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -137.9932709, 149.7316742, -239.0110931, 247.3467712
1: -74.0222931, 98.9946747, -114.1260529, 137.3279266, -211.3502197, 213.1207275
2: -103.9738922, 111.5052643, -160.1003113, 148.6080170, -252.5818787, 271.6055908
3: -67.1468887, 104.1213226, -82.5416260, 156.3286591, -223.4755554, 186.6629486
4: -113.0477753, 118.0451050, -173.5762329, 156.8913116, -269.9390869, 291.6213379

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7864280, upper bound: 178.9154693
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7864280, upper bound: 178.9159582
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -133.2034607, 147.1067047, -252.1663971, 252.3886108
1: -86.8530884, 108.5048981, -110.4020233, 134.9165955, -221.7696838, 218.9069214
2: -121.9298782, 120.9879761, -154.8717651, 145.9841461, -267.9140015, 275.8597107
3: -71.9708786, 120.4962387, -81.0192032, 152.4372559, -224.4081421, 201.5154419
4: -132.5959625, 127.8747635, -167.6655273, 154.3106689, -286.9066162, 295.5402527

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6638617, upper bound: 178.9154693
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6638617, upper bound: 178.9159582
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -138.3767853, 149.9827576, -255.0424500, 257.5619507
1: -86.8530884, 108.5048981, -114.4383926, 137.5594788, -224.4125519, 222.9432983
2: -121.9298782, 120.9879761, -160.5395813, 148.8502045, -270.7800903, 281.5275269
3: -71.9708786, 120.4962387, -82.6676102, 156.7407684, -228.7116394, 203.1638489
4: -132.5959625, 127.8747635, -174.0542450, 157.1458282, -289.7417297, 301.9289856

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6638617, upper bound: 178.9154693
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6638617, upper bound: 178.9159582
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -87.7436447, 108.5790100, -129.7377014, 146.7494049, -234.4930420, 238.3167114
1: -72.7271118, 98.1704102, -107.0335541, 135.3816223, -208.1087341, 205.2039490
2: -102.1450348, 110.7397079, -149.9635010, 146.3835754, -248.5285950, 260.7031555
3: -66.7829132, 102.3765106, -80.5001678, 147.9834290, -214.7663422, 182.8766785
4: -111.1220703, 117.2613220, -163.0087891, 154.3884277, -265.5104980, 280.2701111

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.4301817, upper bound: 169.3588714
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.4881298, upper bound: 169.3587994
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -144.7516327, 155.0637665, -244.3432007, 254.1050873
1: -74.0222931, 98.9946747, -119.8011017, 142.6269684, -216.6492615, 218.7957458
2: -103.9738922, 111.5052643, -167.9449463, 153.7201385, -257.6940308, 279.4501953
3: -67.1468887, 104.1213226, -84.3235550, 165.0267029, -232.1735840, 188.4448853
4: -113.0477753, 118.0451050, -181.7902832, 162.2114105, -275.2591858, 299.8353882

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.1716838, upper bound: 171.0110435
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 7

Time for candidate selection: 5.95 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.2673015, upper bound: 173.6097772
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.2280531, upper bound: 173.5946257
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -103.0650635, 117.8751068, -130.0292511, 146.9473877, -250.0124054, 247.9043579
1: -85.1862640, 107.2890472, -107.2738342, 135.5659943, -220.7522583, 214.5628662
2: -119.5905457, 119.9286118, -150.3018036, 146.5797424, -266.1702881, 270.2303467
3: -71.4306717, 118.2265091, -80.5982056, 148.3070068, -219.7376709, 198.8247070
4: -130.0896301, 126.7739029, -163.3715210, 154.5930634, -284.6826782, 290.1454163

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.5433084, upper bound: 169.3911889
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.5097742, upper bound: 169.3593102
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -145.1236725, 155.3142090, -260.3739014, 264.3088379
1: -86.8530884, 108.5048981, -120.1042175, 142.8575745, -229.7106476, 228.6091156
2: -121.9298782, 120.9879761, -168.3727112, 153.9620667, -275.8919373, 289.3606873
3: -71.9708786, 120.4962387, -84.4498138, 165.4356689, -237.4065399, 204.9460449
4: -132.5959625, 127.8747635, -182.2562256, 162.4646301, -295.0605164, 310.1309814

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.1716838, upper bound: 171.0158121
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.4981183, upper bound: 168.9264776
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -132.8635101, 146.8815002, -234.5851440, 241.5950470
1: -72.8436890, 98.2632751, -110.1251068, 134.7084198, -207.5521088, 208.3883514
2: -102.0504456, 110.7943802, -154.4828949, 145.7666321, -247.8170776, 265.2772827
3: -66.5469589, 102.0576096, -80.9037628, 152.0749207, -218.6218872, 182.9613647
4: -110.6790924, 117.2479630, -167.2418518, 154.0827026, -264.7617493, 284.4898071

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.8514284, upper bound: 177.8738464
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.8514284, upper bound: 177.8738464
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -137.9932709, 149.7316742, -237.4353027, 246.7248077
1: -72.8436890, 98.2632751, -114.1260529, 137.3279266, -210.1716003, 212.3893280
2: -102.0504456, 110.7943802, -160.1003113, 148.6080170, -250.6584473, 270.8946838
3: -66.5469589, 102.0576096, -82.5416260, 156.3286591, -222.8756104, 184.5992126
4: -110.6790924, 117.2479630, -173.5762329, 156.8913116, -267.5704041, 290.8241882

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.8514284, upper bound: 177.8738464
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.8514284, upper bound: 177.8738464
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -133.2034607, 147.1067047, -250.2705383, 251.1772766
1: -85.3876343, 106.9490433, -110.4020233, 134.9165955, -220.3042297, 217.3510284
2: -119.6347275, 120.0215302, -154.8717651, 145.9841461, -265.6188660, 274.8932800
3: -71.2133255, 118.0576172, -81.0192032, 152.4372559, -223.6505737, 199.0768127
4: -129.7596588, 126.8023834, -167.6655273, 154.3106689, -284.0703125, 294.4678650

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8741509, upper bound: 177.8711018
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8741509, upper bound: 177.8711018
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -138.3767853, 149.9827576, -253.1465912, 256.3505859
1: -85.3876343, 106.9490433, -114.4383926, 137.5594788, -222.9470825, 221.3874359
2: -119.6347275, 120.0215302, -160.5395813, 148.8502045, -268.4849243, 280.5610962
3: -71.2133255, 118.0576172, -82.6676102, 156.7407684, -227.9540863, 200.7252045
4: -129.7596588, 126.8023834, -174.0542450, 157.1458282, -286.9054565, 300.8565979

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8741509, upper bound: 177.8711018
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8741509, upper bound: 177.8711018
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -86.1618347, 107.9491882, -129.7377014, 146.7494049, -232.9112396, 237.6868896
1: -71.5395966, 97.4825668, -107.0335541, 135.3816223, -206.9212189, 204.5160828
2: -100.2137604, 110.0167923, -149.9635010, 146.3835754, -246.5973206, 259.9802856
3: -66.1684647, 100.3223419, -80.5001678, 147.9834290, -214.1518860, 180.8225098
4: -108.7500687, 116.4526978, -163.0087891, 154.3884277, -263.1384888, 279.4614868

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.4433831, upper bound: 170.1834046
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.1263185, upper bound: 168.8722589
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.8605814, upper bound: 168.7041185
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -144.7516327, 155.0637665, -242.7673950, 253.4831696
1: -72.8436890, 98.2632751, -119.8011017, 142.6269684, -215.4706421, 218.0643616
2: -102.0504456, 110.7943802, -167.9449463, 153.7201385, -255.7705841, 278.7392883
3: -66.5469589, 102.0576096, -84.3235550, 165.0267029, -231.5736694, 186.3811646
4: -110.6790924, 117.2479630, -181.7902832, 162.2114105, -272.8904724, 299.0382080

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.0139382, upper bound: 170.6038760
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 7

Time for candidate selection: 6.36 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.2717347, upper bound: 173.3381576
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.2808452, upper bound: 173.3443591
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -101.5166855, 117.1017532, -130.0292511, 146.9473877, -248.4640656, 247.1310120
1: -84.0027618, 106.0837326, -107.2738342, 135.5659943, -219.5687561, 213.3575134
2: -117.6876831, 119.1589813, -150.3018036, 146.5797424, -264.2674255, 269.4607544
3: -70.8047180, 116.1916962, -80.5982056, 148.3070068, -219.1117249, 196.7898865
4: -127.6964569, 125.9155655, -163.3715210, 154.5930634, -282.2894897, 289.2870789

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9258154, upper bound: 168.9714893
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5211717, upper bound: 168.9150856
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -145.1236725, 155.3142090, -258.4780273, 263.0974731
1: -85.3876343, 106.9490433, -120.1042175, 142.8575745, -228.2451782, 227.0532227
2: -119.6347275, 120.0215302, -168.3727112, 153.9620667, -273.5967102, 288.3942261
3: -71.2133255, 118.0576172, -84.4498138, 165.4356689, -236.6489868, 202.5074310
4: -129.7596588, 126.8023834, -182.2562256, 162.4646301, -292.2242126, 309.0585938

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.7757158, upper bound: 170.7075773
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -169.9322959, upper bound: 168.7024555
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -89.2794342, 109.3535004, -198.6329346, 198.6329346
1: -74.0222931, 98.9946747, -74.0222931, 98.9946747, -173.0169678, 173.0169678
2: -103.9738922, 111.5052643, -103.9738922, 111.5052643, -215.4791565, 215.4791565
3: -67.1468887, 104.1213226, -67.1468887, 104.1213226, -171.2682190, 171.2682190
4: -113.0477753, 118.0451050, -113.0477753, 118.0451050, -231.0928802, 231.0928802

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7881632, upper bound: 190.8119671
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9471445, upper bound: 190.8153525
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -105.0596924, 119.1851501, -208.4645844, 214.4131927
1: -74.0222931, 98.9946747, -86.8530884, 108.5048981, -182.5271912, 185.8477325
2: -103.9738922, 111.5052643, -121.9298782, 120.9879761, -224.9618683, 233.4351501
3: -67.1468887, 104.1213226, -71.9708786, 120.4962387, -187.6431274, 176.0921936
4: -113.0477753, 118.0451050, -132.5959625, 127.8747635, -240.9225464, 250.6410675

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7881629, upper bound: 190.8119671
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9471445, upper bound: 190.8153525
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -89.2794342, 109.3535004, -214.4131927, 208.4645844
1: -86.8530884, 108.5048981, -74.0222931, 98.9946747, -185.8477325, 182.5271912
2: -121.9298782, 120.9879761, -103.9738922, 111.5052643, -233.4351501, 224.9618683
3: -71.9708786, 120.4962387, -67.1468887, 104.1213226, -176.0921936, 187.6431274
4: -132.5959625, 127.8747635, -113.0477753, 118.0451050, -250.6410675, 240.9225464

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6749022, upper bound: 190.8088039
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8173254, upper bound: 190.8153819
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -105.0596924, 119.1851501, -224.2448425, 224.2448425
1: -86.8530884, 108.5048981, -86.8530884, 108.5048981, -195.3579865, 195.3579865
2: -121.9298782, 120.9879761, -121.9298782, 120.9879761, -242.9178467, 242.9178467
3: -71.9708786, 120.4962387, -71.9708786, 120.4962387, -192.4671173, 192.4671173
4: -132.5959625, 127.8747635, -132.5959625, 127.8747635, -260.4707336, 260.4707336

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6749020, upper bound: 190.8088039
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8173254, upper bound: 190.8153819
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -87.7036362, 108.7315369, -198.0109711, 197.0571289
1: -74.0222931, 98.9946747, -72.8436890, 98.2632751, -172.2855682, 171.8383179
2: -103.9738922, 111.5052643, -102.0504456, 110.7943802, -214.7682495, 213.5557098
3: -67.1468887, 104.1213226, -66.5469589, 102.0576096, -169.2044983, 170.6682739
4: -113.0477753, 118.0451050, -110.6790924, 117.2479630, -230.2957306, 228.7241974

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.3964494, upper bound: 185.1759076
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.7239554, upper bound: 185.2178909
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -103.1638489, 117.9738159, -207.2532501, 212.5173492
1: -74.0222931, 98.9946747, -85.3876343, 106.9490433, -180.9713440, 184.3822479
2: -103.9738922, 111.5052643, -119.6347275, 120.0215302, -223.9953766, 231.1399841
3: -67.1468887, 104.1213226, -71.2133255, 118.0576172, -185.2044983, 175.3346558
4: -113.0477753, 118.0451050, -129.7596588, 126.8023834, -239.8501587, 247.8047333

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.3964494, upper bound: 185.1759076
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.7239554, upper bound: 185.2178909
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -87.7036362, 108.7315369, -213.7912292, 206.8887939
1: -86.8530884, 108.5048981, -72.8436890, 98.2632751, -185.1163483, 181.3485870
2: -121.9298782, 120.9879761, -102.0504456, 110.7943802, -232.7242126, 223.0384216
3: -71.9708786, 120.4962387, -66.5469589, 102.0576096, -174.0284882, 187.0431976
4: -132.5959625, 127.8747635, -110.6790924, 117.2479630, -249.8439026, 238.5538635

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.3695785, upper bound: 185.1759076
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.5992036, upper bound: 185.2183074
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -103.1638489, 117.9738159, -223.0335083, 222.3489990
1: -86.8530884, 108.5048981, -85.3876343, 106.9490433, -193.8021240, 193.8925323
2: -121.9298782, 120.9879761, -119.6347275, 120.0215302, -241.9513702, 240.6227112
3: -71.9708786, 120.4962387, -71.2133255, 118.0576172, -190.0284882, 191.7095642
4: -132.5959625, 127.8747635, -129.7596588, 126.8023834, -259.3983459, 257.6343994

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.3695785, upper bound: 185.1759076
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.5992036, upper bound: 185.2183074
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -89.2794342, 109.3535004, -197.0571289, 198.0109711
1: -72.8436890, 98.2632751, -74.0222931, 98.9946747, -171.8383179, 172.2855682
2: -102.0504456, 110.7943802, -103.9738922, 111.5052643, -213.5557098, 214.7682648
3: -66.5469589, 102.0576096, -67.1468887, 104.1213226, -170.6682739, 169.2044983
4: -110.6790924, 117.2479630, -113.0477753, 118.0451050, -228.7241974, 230.2957153

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.3371662, upper bound: 186.2192850
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -87.7036362, 108.7315369, -196.4351807, 196.4351807
1: -72.8436890, 98.2632751, -72.8436890, 98.2632751, -171.1069336, 171.1069183
2: -102.0504456, 110.7943802, -102.0504456, 110.7943802, -212.8447876, 212.8448029
3: -66.5469589, 102.0576096, -66.5469589, 102.0576096, -168.6045685, 168.6045685
4: -110.6790924, 117.2479630, -110.6790924, 117.2479630, -227.9270630, 227.9270630

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.3371653, upper bound: 186.2192850
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -89.2794342, 109.3535004, -212.5173492, 207.2532501
1: -85.3876343, 106.9490433, -74.0222931, 98.9946747, -184.3822479, 180.9713440
2: -119.6347275, 120.0215302, -103.9738922, 111.5052643, -231.1399841, 223.9953918
3: -71.2133255, 118.0576172, -67.1468887, 104.1213226, -175.3346558, 185.2044983
4: -129.7596588, 126.8023834, -113.0477753, 118.0451050, -247.8047638, 239.8501587

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1753641
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2284433, upper bound: 186.1667493
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -87.7036362, 108.7315369, -211.8953705, 205.6774597
1: -85.3876343, 106.9490433, -72.8436890, 98.2632751, -183.6508484, 179.7926941
2: -119.6347275, 120.0215302, -102.0504456, 110.7943802, -230.4291077, 222.0719147
3: -71.2133255, 118.0576172, -66.5469589, 102.0576096, -173.2709351, 184.6045685
4: -129.7596588, 126.8023834, -110.6790924, 117.2479630, -247.0075989, 237.4814758

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1753641
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2284435, upper bound: 186.1667493
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -105.0596924, 119.1851501, -206.8887939, 213.7912292
1: -72.8436890, 98.2632751, -86.8530884, 108.5048981, -181.3485870, 185.1163330
2: -102.0504456, 110.7943802, -121.9298782, 120.9879761, -223.0384216, 232.7242432
3: -66.5469589, 102.0576096, -71.9708786, 120.4962387, -187.0431976, 174.0284882
4: -110.6790924, 117.2479630, -132.5959625, 127.8747635, -238.5538635, 249.8439178

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.2651397, upper bound: 185.1840151
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1418162, upper bound: 185.2035257
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -103.1638489, 117.9738159, -205.6774597, 211.8953857
1: -72.8436890, 98.2632751, -85.3876343, 106.9490433, -179.7927094, 183.6508484
2: -102.0504456, 110.7943802, -119.6347275, 120.0215302, -222.0719299, 230.4291077
3: -66.5469589, 102.0576096, -71.2133255, 118.0576172, -184.6045685, 173.2709351
4: -110.6790924, 117.2479630, -129.7596588, 126.8023834, -237.4814758, 247.0075836

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.2651397, upper bound: 185.1840151
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1418162, upper bound: 185.2035257
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -105.0596924, 119.1851501, -222.3489990, 223.0335083
1: -85.3876343, 106.9490433, -86.8530884, 108.5048981, -193.8925323, 193.8021240
2: -119.6347275, 120.0215302, -121.9298782, 120.9879761, -240.6227112, 241.9513550
3: -71.2133255, 118.0576172, -71.9708786, 120.4962387, -191.7095642, 190.0285034
4: -129.7596588, 126.8023834, -132.5959625, 127.8747635, -257.6344299, 259.3983459

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.3653759, upper bound: 185.1759076
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.1759076, upper bound: 185.1759076
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -103.1638489, 117.9738159, -221.1376648, 221.1376648
1: -85.3876343, 106.9490433, -85.3876343, 106.9490433, -192.3366394, 192.3366241
2: -119.6347275, 120.0215302, -119.6347275, 120.0215302, -239.6562195, 239.6562042
3: -71.2133255, 118.0576172, -71.2133255, 118.0576172, -189.2709351, 189.2709351
4: -129.7596588, 126.8023834, -129.7596588, 126.8023834, -256.5620117, 256.5620117

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.3653759, upper bound: 185.1759076
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.1759076, upper bound: 185.1759076
time: 0.66 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.76 seconds
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -174.9443451, upper bound: 183.7606617
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -179.3228483, upper bound: 187.9890995
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -177.1002860, upper bound: 177.3466620
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -176.0655435, upper bound: 177.0643786
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -175.1314048, upper bound: 184.1309148
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -178.7003468, upper bound: 187.7758645
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -176.7906176, upper bound: 177.2279754
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -175.9681374, upper bound: 177.0089863
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -174.2645398, upper bound: 179.7339467
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -178.1987269, upper bound: 182.9989340
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -174.7550359, upper bound: 170.3114668
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -173.4577211, upper bound: 170.0929664
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -174.3365456, upper bound: 179.8042574
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -177.6243083, upper bound: 182.6443136
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -174.1649656, upper bound: 169.9993655
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -173.2631699, upper bound: 169.9293796
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -172.6392012, upper bound: 179.3958632
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -170.9971788, upper bound: 178.1760282
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -172.6392012, upper bound: 179.3958632
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -170.9971788, upper bound: 178.1760282
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -173.3084916, upper bound: 179.9816038
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -168.2812351, upper bound: 175.6214869
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -173.3084916, upper bound: 179.9816038
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -168.2812351, upper bound: 175.6214869
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -172.6416784, upper bound: 179.3115764
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -171.0165182, upper bound: 178.3205941
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -172.6416784, upper bound: 179.3115731
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -171.0165182, upper bound: 178.3205941
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -173.2879496, upper bound: 179.7651124
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -168.3688669, upper bound: 176.3271476
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -173.2879496, upper bound: 179.7651161
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -168.3688664, upper bound: 176.3271442
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -187.7864280, upper bound: 178.9154693
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -187.7864280, upper bound: 178.9159582
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -187.7864280, upper bound: 178.9154693
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -187.7864280, upper bound: 178.9159582
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -187.6638617, upper bound: 178.9154693
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -187.6638617, upper bound: 178.9159582
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -187.6638617, upper bound: 178.9154693
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -187.6638617, upper bound: 178.9159582
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -180.4301817, upper bound: 169.3588714
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -180.4881298, upper bound: 169.3587994
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -184.2673015, upper bound: 173.6097772
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -184.2280531, upper bound: 173.5946257
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -180.5433084, upper bound: 169.3911889
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -180.5097742, upper bound: 169.3593102
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -182.1716838, upper bound: 171.0158121
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -174.4981183, upper bound: 168.9264776
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -182.8514284, upper bound: 177.8738464
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -182.8514284, upper bound: 177.8738464
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -182.8514284, upper bound: 177.8738464
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -182.8514284, upper bound: 177.8738464
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -181.8741509, upper bound: 177.8711018
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -181.8741509, upper bound: 177.8711018
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -181.8741509, upper bound: 177.8711018
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -181.8741509, upper bound: 177.8711018
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -176.1263185, upper bound: 168.8722589
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -174.8605814, upper bound: 168.7041185
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -180.2717347, upper bound: 173.3381576
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -180.2808452, upper bound: 173.3443591
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -176.9258154, upper bound: 168.9714893
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -176.5211717, upper bound: 168.9150856
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -178.7757158, upper bound: 170.7075773
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -169.9322959, upper bound: 168.7024555
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -190.7881632, upper bound: 190.8119671
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -190.9471445, upper bound: 190.8153525
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -190.7881629, upper bound: 190.8119671
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -190.9471445, upper bound: 190.8153525
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -190.6749022, upper bound: 190.8088039
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -190.8173254, upper bound: 190.8153819
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -190.6749020, upper bound: 190.8088039
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -190.8173254, upper bound: 190.8153819
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -189.3964494, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -189.7239554, upper bound: 185.2178909
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -189.3964494, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -189.7239554, upper bound: 185.2178909
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -189.3695785, upper bound: 185.1759076
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -189.5992036, upper bound: 185.2183074
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -189.3695785, upper bound: 185.1759076
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -189.5992036, upper bound: 185.2183074
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -186.3371662, upper bound: 186.2192850
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -186.3371653, upper bound: 186.2192850
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1753641
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -185.2284433, upper bound: 186.1667493
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1753641
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -185.2284435, upper bound: 186.1667493
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -186.2651397, upper bound: 185.1840151
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -186.1418162, upper bound: 185.2035257
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -186.2651397, upper bound: 185.1840151
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -186.1418162, upper bound: 185.2035257
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -185.3653759, upper bound: 185.1759076
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -185.1759076, upper bound: 185.1759076
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -185.3653759, upper bound: 185.1759076
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -185.1759076, upper bound: 185.1759076

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -116.5353088, 138.2101135, -87.7436447, 108.5790100, -225.1143188, 225.9537354
1: -96.1892548, 126.9677124, -72.7271118, 98.1704102, -194.3596497, 199.6947937
2: -134.9284210, 138.2520905, -102.1450348, 110.7397079, -245.6681213, 240.3970947
3: -76.8945007, 133.9988556, -66.7829132, 102.3765106, -179.2709961, 200.7817688
4: -146.8789215, 145.9765930, -111.1220703, 117.2613220, -264.1402588, 257.0986633

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.9434033, upper bound: 183.6456288
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.9434033, upper bound: 183.7606633
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -128.8031006, 144.1958160, -89.2794342, 109.3535004, -238.1566010, 233.4752502
1: -106.7664566, 132.2233582, -74.0222931, 98.9946747, -205.7611237, 206.2456512
2: -149.7733765, 143.2122955, -103.9738922, 111.5052643, -261.2786255, 247.1861572
3: -79.4585190, 147.4990845, -67.1468887, 104.1213226, -183.5798340, 214.6459656
4: -162.1636963, 151.3956604, -113.0477753, 118.0451050, -280.2087708, 264.4434204

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.3221374, upper bound: 187.9113827
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.3221374, upper bound: 187.9890995
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -128.4142303, 143.5745087, -105.0596924, 119.1851501, -247.5993805, 248.6342010
1: -106.4365082, 131.5967712, -86.8530884, 108.5048981, -214.9414062, 218.4498596
2: -149.3158417, 142.5776672, -121.9298782, 120.9879761, -270.3037415, 264.5075378
3: -79.4790649, 147.1745758, -71.9708786, 120.4962387, -199.9753113, 219.1454468
4: -161.6869507, 150.7080078, -132.5959625, 127.8747635, -289.5617065, 283.3039551

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.0157750, upper bound: 176.4225407
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.0157751, upper bound: 177.3466627
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -124.8078156, 140.3845520, -105.0081100, 119.1510162, -243.9588318, 245.3926239
1: -103.1713562, 128.6421967, -86.8102570, 108.4728546, -211.6441803, 215.4524536
2: -144.8646851, 139.7669525, -121.8700790, 120.9590988, -265.8237915, 261.6370239
3: -78.2998886, 142.7011108, -71.9566269, 120.4403534, -198.7402344, 214.6577301
4: -157.2565308, 147.6992645, -132.5320129, 127.8445892, -285.1011353, 280.2312622

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7884296, upper bound: 176.0425981
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.7884296, upper bound: 177.0643786
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -122.4806519, 141.9398346, -87.7436447, 108.5790100, -231.0596619, 229.6834717
1: -100.9000854, 130.4472656, -72.7271118, 98.1704102, -199.0704651, 203.1743469
2: -141.5088959, 142.0330505, -102.1450348, 110.7397079, -252.2485962, 244.1780701
3: -78.6786652, 139.5681915, -66.7829132, 102.3765106, -181.0551758, 206.3511047
4: -154.2557373, 149.7192078, -111.1220703, 117.2613220, -271.5170593, 260.8412476

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.0614635, upper bound: 183.9014693
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.0614645, upper bound: 184.0970439
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -133.8843231, 146.9945526, -89.2794342, 109.3535004, -243.2378235, 236.2739868
1: -110.7203522, 134.7954102, -74.0222931, 98.9946747, -209.7150269, 208.8177032
2: -155.3194275, 146.0055084, -103.9738922, 111.5052643, -266.8247070, 249.9794006
3: -81.0639038, 151.6560516, -67.1468887, 104.1213226, -185.1852264, 218.8029480
4: -168.4227295, 154.1468506, -113.0477753, 118.0451050, -286.4678040, 267.1946411

Time for backsubstitution: 2.16 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=279.9653625488281
rel_dist={0: [-191.43388219474815, 191.4338821947481]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.3499641, upper bound: 183.6521416
time: 0.60 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1345495, upper bound: 191.1345518
time: 0.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.33 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 0, lower bound: -179.3499641, upper bound: 183.6521416
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 0, lower bound: -191.1345495, upper bound: 191.1345518

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -165.7317810, 169.9891968, -115.4142151, 128.1499329, -293.8816833, 285.4034119
1: -136.9342194, 156.5604401, -95.3663254, 116.9463272, -253.8805237, 251.9267273
2: -192.2410583, 168.2551727, -133.9529266, 129.7927856, -322.0337524, 302.2080688
3: -91.3198547, 188.0007629, -75.9782944, 132.4578400, -223.7776947, 263.9790649
4: -208.4816437, 177.5048828, -145.6999359, 137.1465759, -345.6282043, 323.2047729

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.7747242, upper bound: 177.7747242
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.7747242, upper bound: 183.6521416
time: 0.65 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -138.0393982, 141.9259644, -268.2368774, 272.8351440
1: -104.3817749, 123.2117767, -113.9848785, 129.8824921, -234.2642670, 237.1966400
2: -146.5827942, 135.2793121, -160.1121521, 141.6186523, -288.2014465, 295.3914795
3: -78.7137909, 144.2006378, -81.8182144, 156.6791382, -235.3929138, 226.0188599
4: -159.2684784, 142.8177032, -173.9111633, 149.4060059, -308.6744995, 316.7288818

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.6521416, upper bound: 179.3499641
time: 0.66 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.6521416, upper bound: 191.1345518
time: 0.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.24 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 0, lower bound: -177.7747242, upper bound: 177.7747242
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 0, lower bound: -177.7747242, upper bound: 183.6521416
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 0, lower bound: -183.6521416, upper bound: 179.3499641
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 0, lower bound: -183.6521416, upper bound: 191.1345518

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -165.7317810, 169.9891968, -164.5511169, 169.5321350, -335.2639160, 334.5403137
1: -136.9342194, 156.5604401, -136.0207825, 156.1482239, -293.0823975, 292.5811462
2: -192.2410583, 168.2551727, -190.9581909, 167.8390503, -360.0800476, 359.2133789
3: -91.3198547, 188.0007629, -91.0548401, 187.1302185, -278.4500732, 279.0556030
4: -208.4816437, 177.5048828, -207.0550385, 177.0473022, -385.5289001, 384.5599365

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.8846409, upper bound: 171.9110543
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.3929760, upper bound: 170.3929760
time: 0.55 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -165.7317810, 169.9891968, -125.4949493, 134.3617401, -300.0935059, 295.4841309
1: -136.9342194, 156.5604401, -103.7368469, 122.8197327, -259.7539673, 260.2973022
2: -192.2410583, 168.2551727, -145.6806335, 134.8824158, -327.1233826, 313.9357910
3: -91.3198547, 188.0007629, -78.4876633, 143.4431458, -234.7630005, 266.4883728
4: -208.4816437, 177.5048828, -158.2714539, 142.3994293, -350.8809814, 335.7763367

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.8846409, upper bound: 178.2398534
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.3929760, upper bound: 176.0355022
time: 0.59 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -165.7317810, 169.9891968, -296.3001099, 300.5275269
1: -104.3817749, 123.2117767, -136.9342194, 156.5604401, -260.9421997, 260.1459961
2: -146.5827942, 135.2793121, -192.2410583, 168.2551727, -314.8379517, 327.5202942
3: -78.7137909, 144.2006378, -91.3198547, 188.0007629, -266.7145081, 235.5204926
4: -159.2684784, 142.8177032, -208.4816437, 177.5048828, -336.7733765, 351.2993469

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0021666, upper bound: 172.9652348
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0354979, upper bound: 172.5805158
time: 0.50 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -126.3109131, 134.7957306, -261.1066284, 261.1066284
1: -104.3817749, 123.2117767, -104.3817749, 123.2117767, -227.5935364, 227.5935211
2: -146.5827942, 135.2793121, -146.5827942, 135.2793121, -281.8620911, 281.8620911
3: -78.7137909, 144.2006378, -78.7137909, 144.2006378, -222.9144287, 222.9144287
4: -159.2684784, 142.8177032, -159.2684784, 142.8177032, -302.0861816, 302.0861816

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.0021681, upper bound: 186.7054696
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.0355022, upper bound: 186.6846139
time: 0.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.32 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 0, lower bound: -175.8846409, upper bound: 171.9110543
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 0, lower bound: -170.3929760, upper bound: 170.3929760
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -175.8846409, upper bound: 178.2398534
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 0, lower bound: -170.3929760, upper bound: 176.0355022
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -179.0021666, upper bound: 172.9652348
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 0, lower bound: -176.0354979, upper bound: 172.5805158
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -179.0021681, upper bound: 186.7054696
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -176.0355022, upper bound: 186.6846139

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -150.1845093, 158.3305664, -125.2735138, 134.1932678, -284.3777466, 283.6040344
1: -124.1818390, 145.5034790, -103.5544662, 122.6605148, -246.8423462, 249.0579529
2: -174.3076477, 156.9754639, -145.4239197, 134.7227478, -309.0303955, 302.3993835
3: -86.1170044, 170.7208557, -78.4142761, 143.1940613, -229.3110657, 249.1351166
4: -188.9908142, 165.6596985, -157.9934540, 142.2321472, -331.2229614, 323.6531372

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.5805158, upper bound: 176.0354979
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.5805158, upper bound: 176.0354979
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -112.0257797, 124.0801163, -165.4802399, 169.8005676, -281.8263245, 289.5603333
1: -92.6331711, 113.0716934, -136.7282410, 156.3819275, -249.0151062, 249.7999268
2: -130.0409088, 125.1827774, -191.9507599, 168.0728912, -298.1137695, 317.1334534
3: -74.0341339, 128.2513275, -91.2358551, 187.7215576, -261.7556458, 219.4871826
4: -141.3389282, 132.2405701, -208.1654358, 177.3133545, -318.6522827, 340.4060059

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0354979, upper bound: 172.5805158
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0354979, upper bound: 172.5805158
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -112.0257797, 124.0801163, -126.0740814, 134.6199341, -246.6457214, 250.1541748
1: -92.6331711, 113.0716934, -104.1872864, 123.0459442, -215.6791077, 217.2589722
2: -130.0409088, 125.1827774, -146.3089752, 135.1129150, -265.1537781, 271.4917297
3: -74.0341339, 128.2513275, -78.6363983, 143.9376984, -217.9718323, 206.8877258
4: -141.3389282, 132.2405701, -158.9718628, 142.6432495, -283.9821777, 291.2124329

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.9475802, upper bound: 186.7046327
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.1768747, upper bound: 185.7909980
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -109.6574478, 121.9550476, -119.2828140, 129.7453613, -239.4027710, 241.2378540
1: -90.7688217, 111.0725479, -98.6307831, 118.4716263, -209.2404480, 209.7033386
2: -127.1920471, 123.9425659, -138.4092102, 130.8295746, -258.0216064, 262.3517761
3: -73.1300201, 125.2666092, -76.5505295, 136.2505646, -209.3805847, 201.8171387
4: -137.8858032, 130.8748932, -150.2953796, 138.2123871, -276.0981445, 281.1702881

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7843642, upper bound: 186.6389382
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
time: 0.62 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.67 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.67
Output dim: 0, lower bound: -172.5805158, upper bound: 176.0354979
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.67
Output dim: 0, lower bound: -172.5805158, upper bound: 176.0354979
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.67
Output dim: 0, lower bound: -176.0354979, upper bound: 172.5805158
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.67
Output dim: 0, lower bound: -176.0354979, upper bound: 172.5805158
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -188.9475802, upper bound: 186.7046327
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -189.1768747, upper bound: 185.7909980
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -185.7843642, upper bound: 186.6389382
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -104.8343201, 119.2564621, -99.9963074, 117.3188477, -222.1531677, 219.2527618
1: -86.7180481, 108.6027222, -82.9214325, 106.9956284, -193.7136688, 191.5241089
2: -121.7082596, 120.6615601, -116.4698334, 119.4181442, -241.1264038, 237.1313934
3: -71.7078094, 120.3225098, -70.6472015, 116.0528564, -187.7606354, 190.9697113
4: -132.2850800, 127.5225296, -126.4263992, 126.3003159, -258.5853882, 253.9489136

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.9475857, upper bound: 185.7909980
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.9475857, upper bound: 185.7909980
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -112.0257797, 124.0801163, -118.5624695, 129.3886108, -241.4143829, 242.6425323
1: -92.6331711, 113.0716934, -97.9810104, 118.1707611, -210.8039246, 211.0527039
2: -130.0409088, 125.1827774, -137.5818481, 130.5605011, -260.6014099, 262.7646179
3: -74.0341339, 128.2513275, -76.3987427, 135.6210175, -209.6551514, 204.6500549
4: -141.3389282, 132.2405701, -149.5412292, 137.9108276, -279.2497559, 281.7817993

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.1768747, upper bound: 185.7909980
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.1768747, upper bound: 185.7909980
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -103.3335114, 117.8350906, -95.0599060, 114.1239319, -217.4574432, 212.8949890
1: -85.5760803, 107.0865479, -78.8560333, 103.5713425, -189.1474304, 185.9425812
2: -119.8707428, 119.8880768, -110.6966553, 116.2222672, -236.0929718, 230.5847168
3: -71.0732193, 118.3061981, -69.1467743, 110.4701080, -181.5433350, 187.4529724
4: -129.9538269, 126.6679916, -120.1393127, 123.0008087, -252.9546356, 246.8072815

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -109.6574478, 121.9550476, -112.3044662, 124.8482590, -234.5056915, 234.2594910
1: -90.7688217, 111.0725479, -92.8569336, 113.9030457, -204.6718750, 203.9294739
2: -127.1920471, 123.9425659, -130.3015137, 126.5989151, -253.7909241, 254.2440796
3: -73.1300201, 125.2666092, -74.4742126, 128.4913177, -201.6213379, 199.7408142
4: -137.8858032, 130.8748932, -141.5330048, 133.8156281, -271.7013855, 272.4078979

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347796, upper bound: 185.7386919
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.30 seconds
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -188.9475857, upper bound: 185.7909980
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -188.9475857, upper bound: 185.7909980
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -189.1768747, upper bound: 185.7909980
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -189.1768747, upper bound: 185.7909980
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -185.7347796, upper bound: 185.7386919
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -99.9963074, 117.3188477, -206.5982819, 209.3498077
1: -74.0222931, 98.9946747, -82.9214325, 106.9956284, -181.0179138, 181.9160461
2: -103.9738922, 111.5052643, -116.4698334, 119.4181442, -223.3920288, 227.9750977
3: -67.1468887, 104.1213226, -70.6472015, 116.0528564, -183.1997375, 174.7685242
4: -113.0477753, 118.0451050, -126.4263992, 126.3003159, -239.3480835, 244.4714966

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.9475857, upper bound: 186.7046327
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.9475857, upper bound: 186.7046327
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -99.9963074, 117.3188477, -222.3785400, 219.1814575
1: -86.8530884, 108.5048981, -82.9214325, 106.9956284, -193.8487091, 191.4263306
2: -121.9298782, 120.9879761, -116.4698334, 119.4181442, -241.3480225, 237.4577942
3: -71.9708786, 120.4962387, -70.6472015, 116.0528564, -188.0237274, 191.1434326
4: -132.5959625, 127.8747635, -126.4263992, 126.3003159, -258.8962708, 254.3011627

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.9475857, upper bound: 186.7046327
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.9475857, upper bound: 186.7046327
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -118.5624695, 129.3886108, -218.6680450, 227.9159698
1: -74.0222931, 98.9946747, -97.9810104, 118.1707611, -192.1930542, 196.9756622
2: -103.9738922, 111.5052643, -137.5818481, 130.5605011, -234.5343933, 249.0871124
3: -67.1468887, 104.1213226, -76.3987427, 135.6210175, -202.7679138, 180.5200653
4: -113.0477753, 118.0451050, -149.5412292, 137.9108276, -250.9586029, 267.5863342

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.9475857, upper bound: 185.7909980
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.9475857, upper bound: 185.7909980
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -118.5624695, 129.3886108, -234.4483032, 237.7475739
1: -86.8530884, 108.5048981, -97.9810104, 118.1707611, -205.0238342, 206.4859009
2: -121.9298782, 120.9879761, -137.5818481, 130.5605011, -252.4903870, 258.5698242
3: -71.9708786, 120.4962387, -76.3987427, 135.6210175, -207.5918884, 196.8949890
4: -132.5959625, 127.8747635, -149.5412292, 137.9108276, -270.5067749, 277.4159851

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.9475857, upper bound: 185.7909980
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.9475857, upper bound: 185.7909980
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -95.0599060, 114.1239319, -201.8275757, 203.7914429
1: -72.8436890, 98.2632751, -78.8560333, 103.5713425, -176.4150238, 177.1193085
2: -102.0504456, 110.7943802, -110.6966553, 116.2222672, -218.2726593, 221.4910278
3: -66.5469589, 102.0576096, -69.1467743, 110.4701080, -177.0170593, 171.2043762
4: -110.6790924, 117.2479630, -120.1393127, 123.0008087, -233.6798859, 237.3872528

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7843642, upper bound: 186.6389382
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7843642, upper bound: 186.6389382
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -95.0599060, 114.1239319, -217.2877808, 213.0337219
1: -85.3876343, 106.9490433, -78.8560333, 103.5713425, -188.9589539, 185.8050842
2: -119.6347275, 120.0215302, -110.6966553, 116.2222672, -235.8569794, 230.7181702
3: -71.2133255, 118.0576172, -69.1467743, 110.4701080, -181.6834106, 187.2043762
4: -129.7596588, 126.8023834, -120.1393127, 123.0008087, -252.7604218, 246.9416809

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7843642, upper bound: 186.6389382
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7843642, upper bound: 186.6389382
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -112.3044662, 124.8482590, -212.5518951, 221.0359802
1: -72.8436890, 98.2632751, -92.8569336, 113.9030457, -186.7467346, 191.1201477
2: -102.0504456, 110.7943802, -130.3015137, 126.5989151, -228.6493225, 241.0958557
3: -66.5469589, 102.0576096, -74.4742126, 128.4913177, -195.0382690, 176.5318298
4: -110.6790924, 117.2479630, -141.5330048, 133.8156281, -244.4947205, 258.7809448

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -112.3044662, 124.8482590, -228.0121002, 230.2782593
1: -85.3876343, 106.9490433, -92.8569336, 113.9030457, -199.2906647, 199.8059235
2: -119.6347275, 120.0215302, -130.3015137, 126.5989151, -246.2336273, 250.3230133
3: -71.2133255, 118.0576172, -74.4742126, 128.4913177, -199.7046356, 192.5318298
4: -129.7596588, 126.8023834, -141.5330048, 133.8156281, -263.5752563, 268.3353882

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
time: 0.62 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.63 seconds
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -188.9475857, upper bound: 186.7046327
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -188.9475857, upper bound: 186.7046327
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -188.9475857, upper bound: 186.7046327
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -188.9475857, upper bound: 186.7046327
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -188.9475857, upper bound: 185.7909980
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -188.9475857, upper bound: 185.7909980
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -188.9475857, upper bound: 185.7909980
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -188.9475857, upper bound: 185.7909980
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -185.7843642, upper bound: 186.6389382
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -185.7843642, upper bound: 186.6389382
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -185.7843642, upper bound: 186.6389382
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -185.7843642, upper bound: 186.6389382
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -89.2794342, 109.3535004, -198.6329346, 198.6329346
1: -74.0222931, 98.9946747, -74.0222931, 98.9946747, -173.0169678, 173.0169678
2: -103.9738922, 111.5052643, -103.9738922, 111.5052643, -215.4791565, 215.4791565
3: -67.1468887, 104.1213226, -67.1468887, 104.1213226, -171.2682190, 171.2682190
4: -113.0477753, 118.0451050, -113.0477753, 118.0451050, -231.0928802, 231.0928802

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.4025253, upper bound: 186.2202914
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.5459993, upper bound: 186.2426172
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -87.7036362, 108.7315369, -198.0109711, 197.0571289
1: -74.0222931, 98.9946747, -72.8436890, 98.2632751, -172.2855682, 171.8383179
2: -103.9738922, 111.5052643, -102.0504456, 110.7943802, -214.7682495, 213.5557098
3: -67.1468887, 104.1213226, -66.5469589, 102.0576096, -169.2044983, 170.6682739
4: -113.0477753, 118.0451050, -110.6790924, 117.2479630, -230.2957306, 228.7241974

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.4025253, upper bound: 186.2202914
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.5459993, upper bound: 186.2426172
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -89.2794342, 109.3535004, -214.4131927, 208.4645844
1: -86.8530884, 108.5048981, -74.0222931, 98.9946747, -185.8477325, 182.5271912
2: -121.9298782, 120.9879761, -103.9738922, 111.5052643, -233.4351501, 224.9618683
3: -71.9708786, 120.4962387, -67.1468887, 104.1213226, -176.0921936, 187.6431274
4: -132.5959625, 127.8747635, -113.0477753, 118.0451050, -250.6410675, 240.9225464

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2274761, upper bound: 186.2190757
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2559100, upper bound: 186.2334862
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -87.7036362, 108.7315369, -213.7912292, 206.8887939
1: -86.8530884, 108.5048981, -72.8436890, 98.2632751, -185.1163483, 181.3485870
2: -121.9298782, 120.9879761, -102.0504456, 110.7943802, -232.7242126, 223.0384216
3: -71.9708786, 120.4962387, -66.5469589, 102.0576096, -174.0284882, 187.0431976
4: -132.5959625, 127.8747635, -110.6790924, 117.2479630, -249.8439026, 238.5538635

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2274761, upper bound: 186.2190757
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2559100, upper bound: 186.2334862
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -105.0596924, 119.1851501, -208.4645844, 214.4131927
1: -74.0222931, 98.9946747, -86.8530884, 108.5048981, -182.5271912, 185.8477325
2: -103.9738922, 111.5052643, -121.9298782, 120.9879761, -224.9618683, 233.4351501
3: -67.1468887, 104.1213226, -71.9708786, 120.4962387, -187.6431274, 176.0921936
4: -113.0477753, 118.0451050, -132.5959625, 127.8747635, -240.9225464, 250.6410675

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3528107, upper bound: 185.1759076
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.5015563, upper bound: 185.2059581
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -89.2794342, 109.3535004, -103.1638489, 117.9738159, -207.2532501, 212.5173492
1: -74.0222931, 98.9946747, -85.3876343, 106.9490433, -180.9713440, 184.3822479
2: -103.9738922, 111.5052643, -119.6347275, 120.0215302, -223.9953766, 231.1399841
3: -67.1468887, 104.1213226, -71.2133255, 118.0576172, -185.2044983, 175.3346558
4: -113.0477753, 118.0451050, -129.7596588, 126.8023834, -239.8501587, 247.8047333

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3528107, upper bound: 185.1759076
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.5015563, upper bound: 185.2059581
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -105.0596924, 119.1851501, -224.2448425, 224.2448425
1: -86.8530884, 108.5048981, -86.8530884, 108.5048981, -195.3579865, 195.3579865
2: -121.9298782, 120.9879761, -121.9298782, 120.9879761, -242.9178467, 242.9178467
3: -71.9708786, 120.4962387, -71.9708786, 120.4962387, -192.4671173, 192.4671173
4: -132.5959625, 127.8747635, -132.5959625, 127.8747635, -260.4707336, 260.4707336

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1780227, upper bound: 185.1759076
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2129283, upper bound: 185.2059581
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -105.0596924, 119.1851501, -103.1638489, 117.9738159, -223.0335083, 222.3489990
1: -86.8530884, 108.5048981, -85.3876343, 106.9490433, -193.8021240, 193.8925323
2: -121.9298782, 120.9879761, -119.6347275, 120.0215302, -241.9513702, 240.6227112
3: -71.9708786, 120.4962387, -71.2133255, 118.0576172, -190.0284882, 191.7095642
4: -132.5959625, 127.8747635, -129.7596588, 126.8023834, -259.3983459, 257.6343994

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1780227, upper bound: 185.1759076
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2129283, upper bound: 185.2059581
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -89.2794342, 109.3535004, -197.0571289, 198.0109711
1: -72.8436890, 98.2632751, -74.0222931, 98.9946747, -171.8383179, 172.2855682
2: -102.0504456, 110.7943802, -103.9738922, 111.5052643, -213.5557098, 214.7682648
3: -66.5469589, 102.0576096, -67.1468887, 104.1213226, -170.6682739, 169.2044983
4: -110.6790924, 117.2479630, -113.0477753, 118.0451050, -228.7241974, 230.2957153

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.3371653, upper bound: 186.2192850
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -87.7036362, 108.7315369, -196.4351807, 196.4351807
1: -72.8436890, 98.2632751, -72.8436890, 98.2632751, -171.1069336, 171.1069183
2: -102.0504456, 110.7943802, -102.0504456, 110.7943802, -212.8447876, 212.8448029
3: -66.5469589, 102.0576096, -66.5469589, 102.0576096, -168.6045685, 168.6045685
4: -110.6790924, 117.2479630, -110.6790924, 117.2479630, -227.9270630, 227.9270630

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.3371653, upper bound: 186.2192850
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -89.2794342, 109.3535004, -212.5173492, 207.2532501
1: -85.3876343, 106.9490433, -74.0222931, 98.9946747, -184.3822479, 180.9713440
2: -119.6347275, 120.0215302, -103.9738922, 111.5052643, -231.1399841, 223.9953918
3: -71.2133255, 118.0576172, -67.1468887, 104.1213226, -175.3346558, 185.2044983
4: -129.7596588, 126.8023834, -113.0477753, 118.0451050, -247.8047638, 239.8501587

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1753641
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2284433, upper bound: 186.1667493
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -87.7036362, 108.7315369, -211.8953705, 205.6774597
1: -85.3876343, 106.9490433, -72.8436890, 98.2632751, -183.6508484, 179.7926941
2: -119.6347275, 120.0215302, -102.0504456, 110.7943802, -230.4291077, 222.0719147
3: -71.2133255, 118.0576172, -66.5469589, 102.0576096, -173.2709351, 184.6045685
4: -129.7596588, 126.8023834, -110.6790924, 117.2479630, -247.0075989, 237.4814758

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1753641
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2284433, upper bound: 186.1667493
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -105.0596924, 119.1851501, -206.8887939, 213.7912292
1: -72.8436890, 98.2632751, -86.8530884, 108.5048981, -181.3485870, 185.1163330
2: -102.0504456, 110.7943802, -121.9298782, 120.9879761, -223.0384216, 232.7242432
3: -66.5469589, 102.0576096, -71.9708786, 120.4962387, -187.0431976, 174.0284882
4: -110.6790924, 117.2479630, -132.5959625, 127.8747635, -238.5538635, 249.8439178

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.2651397, upper bound: 185.1840151
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1418167, upper bound: 185.2002712
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -103.1638489, 117.9738159, -205.6774597, 211.8953857
1: -72.8436890, 98.2632751, -85.3876343, 106.9490433, -179.7927094, 183.6508484
2: -102.0504456, 110.7943802, -119.6347275, 120.0215302, -222.0719299, 230.4291077
3: -66.5469589, 102.0576096, -71.2133255, 118.0576172, -184.6045685, 173.2709351
4: -110.6790924, 117.2479630, -129.7596588, 126.8023834, -237.4814758, 247.0075836

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.2651397, upper bound: 185.1840151
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1418162, upper bound: 185.2002713
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -105.0596924, 119.1851501, -222.3489990, 223.0335083
1: -85.3876343, 106.9490433, -86.8530884, 108.5048981, -193.8925323, 193.8021240
2: -119.6347275, 120.0215302, -121.9298782, 120.9879761, -240.6227112, 241.9513550
3: -71.2133255, 118.0576172, -71.9708786, 120.4962387, -191.7095642, 190.0285034
4: -129.7596588, 126.8023834, -132.5959625, 127.8747635, -257.6344299, 259.3983459

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.3653759, upper bound: 185.1759076
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.1759076, upper bound: 185.1759076
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -103.1638489, 117.9738159, -221.1376648, 221.1376648
1: -85.3876343, 106.9490433, -85.3876343, 106.9490433, -192.3366394, 192.3366241
2: -119.6347275, 120.0215302, -119.6347275, 120.0215302, -239.6562195, 239.6562042
3: -71.2133255, 118.0576172, -71.2133255, 118.0576172, -189.2709351, 189.2709351
4: -129.7596588, 126.8023834, -129.7596588, 126.8023834, -256.5620117, 256.5620117

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.3653759, upper bound: 185.1759076
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.1759076, upper bound: 185.1759076
time: 0.64 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.75 seconds
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.4025253, upper bound: 186.2202914
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.5459993, upper bound: 186.2426172
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.4025253, upper bound: 186.2202914
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.5459993, upper bound: 186.2426172
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.2274761, upper bound: 186.2190757
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.2559100, upper bound: 186.2334862
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.2274761, upper bound: 186.2190757
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.2559100, upper bound: 186.2334862
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.3528107, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.5015563, upper bound: 185.2059581
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.3528107, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.5015563, upper bound: 185.2059581
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.1780227, upper bound: 185.1759076
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.2129283, upper bound: 185.2059581
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.1780227, upper bound: 185.1759076
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -188.2129283, upper bound: 185.2059581
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -186.3371653, upper bound: 186.2192850
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -186.3371653, upper bound: 186.2192850
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1753641
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -185.2284433, upper bound: 186.1667493
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1753641
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -185.2284433, upper bound: 186.1667493
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -186.2651397, upper bound: 185.1840151
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -186.1418167, upper bound: 185.2002712
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -186.2651397, upper bound: 185.1840151
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -186.1418162, upper bound: 185.2002713
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -185.3653759, upper bound: 185.1759076
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -185.1759076, upper bound: 185.1759076
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -185.3653759, upper bound: 185.1759076
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -185.1759076, upper bound: 185.1759076

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -85.4144974, 107.1782455, -185.8087158, 188.6262512
1: -65.3912277, 92.9626999, -70.8506393, 96.7967911, -162.1880188, 163.8133087
2: -91.6809158, 105.3891754, -99.4731140, 109.3250351, -201.0059052, 204.8622894
3: -64.0711365, 92.9304047, -66.1114426, 99.9281235, -163.9992523, 159.0418396
4: -99.6445236, 111.7989883, -108.1985321, 115.8302917, -215.4747620, 219.9974823

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7842274, upper bound: 190.7842949
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7842274, upper bound: 190.8927040
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -78.9156494, 103.2667618, -86.8181000, 107.9173050, -186.8329468, 190.0848389
1: -65.4742050, 92.8440399, -72.0066376, 97.5073929, -162.9815979, 164.8506775
2: -91.7911377, 105.4159927, -101.1018982, 110.0635376, -201.8546753, 206.5178833
3: -64.3398361, 92.0656891, -66.4979248, 101.3387527, -165.6785889, 158.5636139
4: -99.9847870, 111.7970657, -109.9459457, 116.5819244, -216.5667114, 221.7430115

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8926026, upper bound: 190.7849502
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8926025, upper bound: 190.9486574
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -83.9908295, 106.6378632, -185.2683258, 187.2025909
1: -65.3912277, 92.9626999, -69.7955551, 96.1882095, -161.5794373, 162.7582397
2: -91.6809158, 105.3891754, -97.7412949, 108.6794891, -200.3603821, 203.1304626
3: -64.0711365, 92.9304047, -65.5384064, 98.0504990, -162.1216125, 158.4688110
4: -99.6445236, 111.7989883, -106.0335846, 115.1053162, -214.7498016, 217.8325043

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.4025253, upper bound: 186.2202914
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.4025253, upper bound: 186.2202914
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -78.9156494, 103.2667618, -85.3004150, 107.3088379, -186.2244873, 188.5671692
1: -65.4742050, 92.8440399, -70.8585739, 96.8421860, -162.3163757, 163.7026062
2: -91.7911377, 105.4159927, -99.2187729, 109.3601913, -201.1513367, 204.6347351
3: -64.3398361, 92.0656891, -65.8997116, 99.3301086, -163.6699524, 157.9653931
4: -99.9847870, 111.7970657, -107.6202774, 115.7935944, -215.7783508, 219.4173431

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.5460000, upper bound: 186.2426173
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.5460000, upper bound: 186.2426173
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -91.9630280, 110.9765549, -85.4144974, 107.1782455, -199.1412659, 196.3910370
1: -76.2160187, 100.2644730, -70.8506393, 96.7967911, -173.0128174, 171.1150970
2: -106.8935013, 113.1315613, -99.4731140, 109.3250351, -216.2184906, 212.6046753
3: -68.0785675, 106.5302124, -66.1114426, 99.9281235, -168.0066833, 172.6416626
4: -116.0856934, 119.8276749, -108.1985321, 115.8302917, -231.9159546, 228.0261841

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6395273, upper bound: 190.7612561
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6395273, upper bound: 190.8803285
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -94.0209885, 111.9910202, -86.8181000, 107.9173050, -201.9382935, 198.8091125
1: -77.7521286, 101.0731201, -72.0066376, 97.5073929, -175.2595215, 173.0797577
2: -109.0427399, 114.1736069, -101.1018982, 110.0635376, -219.1062317, 215.2755127
3: -68.8223877, 107.5189819, -66.4979248, 101.3387527, -170.1611328, 174.0169067
4: -118.6816559, 120.7862167, -109.9459457, 116.5819244, -235.2635498, 230.7321625

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7385821, upper bound: 190.7612561
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7385821, upper bound: 190.9472436
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -91.9630280, 110.9765549, -83.9908295, 106.6378632, -198.6008911, 194.9673767
1: -76.2160187, 100.2644730, -69.7955551, 96.1882095, -172.4042358, 170.0600281
2: -106.8935013, 113.1315613, -97.7412949, 108.6794891, -215.5729980, 210.8728638
3: -68.0785675, 106.5302124, -65.5384064, 98.0504990, -166.1290436, 172.0686188
4: -116.0856934, 119.8276749, -106.0335846, 115.1053162, -231.1909943, 225.8612061

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2274761, upper bound: 186.2190757
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2274761, upper bound: 186.2190757
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -94.0209885, 111.9910202, -85.3004150, 107.3088379, -201.3298340, 197.2914429
1: -77.7521286, 101.0731201, -70.8585739, 96.8421860, -174.5943146, 171.9317017
2: -109.0427399, 114.1736069, -99.2187729, 109.3601913, -218.4028778, 213.3923645
3: -68.8223877, 107.5189819, -65.8997116, 99.3301086, -168.1524963, 173.4186707
4: -118.6816559, 120.7862167, -107.6202774, 115.7935944, -234.4752045, 228.4064636

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2559101, upper bound: 186.2334862
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2559101, upper bound: 186.2334862
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -100.2111282, 116.1518860, -194.7823486, 203.4228821
1: -65.3912277, 92.9626999, -82.9235992, 105.4323578, -170.8235779, 175.8862610
2: -91.6809158, 105.3891754, -116.4361649, 118.2331543, -209.9140625, 221.8253174
3: -64.0711365, 92.9304047, -70.6579437, 115.3432693, -179.4143982, 163.5883331
4: -99.6445236, 111.7989883, -126.5574417, 125.0758667, -224.7203522, 238.3564301

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7656684, upper bound: 190.6395234
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7656684, upper bound: 190.7334616
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -78.9156494, 103.2667618, -102.2045746, 117.1111603, -196.0268097, 205.4713135
1: -65.4742050, 92.8440399, -84.4977341, 106.5265961, -172.0007935, 177.3417664
2: -91.7911377, 105.4159927, -118.5997162, 119.2017136, -210.9928589, 224.0157166
3: -64.3398361, 92.0656891, -71.1448975, 117.0975189, -181.4373474, 163.2105560
4: -99.9847870, 111.7970657, -128.9840851, 126.0116959, -225.9964905, 240.7811584

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8787489, upper bound: 190.6400635
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8787489, upper bound: 190.8151866
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -99.0893936, 115.7138519, -194.3443298, 202.3011475
1: -65.3912277, 92.9626999, -82.0695496, 104.7281342, -170.1193542, 175.0322418
2: -91.6809158, 105.3891754, -115.0334167, 117.7662201, -209.4470978, 220.4225922
3: -64.0711365, 92.9304047, -70.1660004, 113.8812714, -177.9524078, 163.0964050
4: -99.6445236, 111.7989883, -124.7635498, 124.4892731, -224.1337738, 236.5625000

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3528107, upper bound: 185.1759076
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3528107, upper bound: 185.1759076
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -78.9156494, 103.2667618, -100.7103958, 116.3573151, -195.2729645, 203.9771576
1: -65.4742050, 92.8440399, -83.3643799, 105.3513718, -170.8255768, 176.2084198
2: -91.7911377, 105.4159927, -116.7628937, 118.4084244, -210.1995544, 222.1788788
3: -64.3398361, 92.0656891, -70.4987793, 115.1464844, -179.4863281, 162.5644684
4: -99.9847870, 111.7970657, -126.6582642, 125.1467361, -225.1315308, 238.4553223

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.5015571, upper bound: 185.2059582
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.5015571, upper bound: 185.2059582
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -91.9630280, 110.9765549, -100.2111282, 116.1518860, -208.1148987, 211.1876831
1: -76.2160187, 100.2644730, -82.9235992, 105.4323578, -181.6483765, 183.1880493
2: -106.8935013, 113.1315613, -116.4361649, 118.2331543, -225.1266479, 229.5677185
3: -68.0785675, 106.5302124, -70.6579437, 115.3432693, -183.4218445, 177.1881409
4: -116.0856934, 119.8276749, -126.5574417, 125.0758667, -241.1615448, 246.3851166

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5652229, upper bound: 180.5086886
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.1578288, upper bound: 176.7120679
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -94.0209885, 111.9910202, -102.2045746, 117.1111603, -211.1321411, 214.1955872
1: -77.7521286, 101.0731201, -84.4977341, 106.5265961, -184.2787170, 185.5708618
2: -109.0427399, 114.1736069, -118.5997162, 119.2017136, -228.2444458, 232.7733154
3: -68.8223877, 107.5189819, -71.1448975, 117.0975189, -185.9199066, 178.6638336
4: -118.6816559, 120.7862167, -128.9840851, 126.0116959, -244.6933441, 249.7702484

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.3765857, upper bound: 182.0589109
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.4503646, upper bound: 177.4808780
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -91.9630280, 110.9765549, -99.0893936, 115.7138519, -207.6768799, 210.0659485
1: -76.2160187, 100.2644730, -82.0695496, 104.7281342, -180.9441528, 182.3340149
2: -106.8935013, 113.1315613, -115.0334167, 117.7662201, -224.6597137, 228.1649780
3: -68.0785675, 106.5302124, -70.1660004, 113.8812714, -181.9598389, 176.6962128
4: -116.0856934, 119.8276749, -124.7635498, 124.4892731, -240.5749664, 244.5912170

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.7201194, upper bound: 174.0959647
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.7571669, upper bound: 171.1328535
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -94.0209885, 111.9910202, -100.7103958, 116.3573151, -210.3782959, 212.7014160
1: -77.7521286, 101.0731201, -83.3643799, 105.3513718, -183.1035004, 184.4375000
2: -109.0427399, 114.1736069, -116.7628937, 118.4084244, -227.4511414, 230.9364929
3: -68.8223877, 107.5189819, -70.4987793, 115.1464844, -183.9688721, 178.0177612
4: -118.6816559, 120.7862167, -126.6582642, 125.1467361, -243.8283844, 247.4444733

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6203134, upper bound: 175.7790428
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.3273570, upper bound: 171.6090485
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -78.4691620, 103.4701385, -85.4144974, 107.1782455, -185.6473999, 188.8846283
1: -65.3832474, 93.1269760, -70.8506393, 96.7967911, -162.1800385, 163.9775848
2: -91.4410248, 105.5574799, -99.4731140, 109.3250351, -200.7660370, 205.0305786
3: -63.9262199, 92.4040985, -66.1114426, 99.9281235, -163.8543396, 158.5155334
4: -99.1128998, 111.9080811, -108.1985321, 115.8302917, -214.9431610, 220.1065826

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.3148545, upper bound: 188.4174430
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.3148545, upper bound: 188.5523977
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -77.8985825, 102.8410950, -86.8181000, 107.9173050, -185.8158569, 189.6591644
1: -64.7397842, 92.3455353, -72.0066376, 97.5073929, -162.2471771, 164.3521423
2: -90.4897461, 104.8518677, -101.1018982, 110.0635376, -200.5532837, 205.9537354
3: -63.8641090, 90.5127716, -66.4979248, 101.3387527, -165.2028656, 157.0106964
4: -98.2341003, 111.2121277, -109.9459457, 116.5819244, -214.8160248, 221.1580811

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.3148545, upper bound: 188.4174430
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.3148545, upper bound: 188.5523977
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -78.4691620, 103.4701385, -83.9908295, 106.6378632, -185.1070251, 187.4609680
1: -65.3832474, 93.1269760, -69.7955551, 96.1882095, -161.5714569, 162.9225311
2: -91.4410248, 105.5574799, -97.7412949, 108.6794891, -200.1205139, 203.2987671
3: -63.9262199, 92.4040985, -65.5384064, 98.0504990, -161.9766998, 157.9424896
4: -99.1128998, 111.9080811, -106.0335846, 115.1053162, -214.2182007, 217.9416046

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -77.8985825, 102.8410950, -85.3004150, 107.3088379, -185.2074127, 188.1415100
1: -64.7397842, 92.3455353, -70.8585739, 96.8421860, -161.5819702, 163.2041016
2: -90.4897461, 104.8518677, -99.2187729, 109.3601913, -199.8499451, 204.0705872
3: -63.8641090, 90.5127716, -65.8997116, 99.3301086, -163.1942139, 156.4124756
4: -98.2341003, 111.2121277, -107.6202774, 115.7935944, -214.0276947, 218.8323975

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -90.9992142, 110.7701797, -85.4144974, 107.1782455, -198.1774292, 196.1846771
1: -75.5363464, 99.9865494, -70.8506393, 96.7967911, -172.3331299, 170.8371887
2: -105.7453613, 112.8695450, -99.4731140, 109.3250351, -215.0703735, 212.3426514
3: -67.7385864, 105.2213135, -66.1114426, 99.9281235, -167.6667175, 171.3327637
4: -114.5958176, 119.4744339, -108.1985321, 115.8302917, -230.4261017, 227.6729736

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2998358, upper bound: 188.2364407
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2998358, upper bound: 188.3927434
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -93.1209030, 111.6165161, -86.8181000, 107.9173050, -201.0381470, 198.4346008
1: -77.1096115, 100.6684418, -72.0066376, 97.5073929, -174.6170044, 172.6750488
2: -107.9064789, 113.7237091, -101.1018982, 110.0635376, -217.9700165, 214.8256073
3: -68.3730927, 106.1663742, -66.4979248, 101.3387527, -169.7118530, 172.6643066
4: -117.1297531, 120.2549515, -109.9459457, 116.5819244, -233.7116699, 230.2008972

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2998358, upper bound: 188.2364408
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2998363, upper bound: 188.3927434
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -90.9992142, 110.7701797, -83.9908295, 106.6378632, -197.6370697, 194.7610168
1: -75.5363464, 99.9865494, -69.7955551, 96.1882095, -171.7245483, 169.7821045
2: -105.7453613, 112.8695450, -97.7412949, 108.6794891, -214.4248505, 210.6108398
3: -67.7385864, 105.2213135, -65.5384064, 98.0504990, -165.7890778, 170.7597198
4: -114.5958176, 119.4744339, -106.0335846, 115.1053162, -229.7011261, 225.5079956

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2284433, upper bound: 186.1667493
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2284433, upper bound: 186.1667493
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -93.1209030, 111.6165161, -85.3004150, 107.3088379, -200.4297028, 196.9169312
1: -77.1096115, 100.6684418, -70.8585739, 96.8421860, -173.9517975, 171.5270081
2: -107.9064789, 113.7237091, -99.2187729, 109.3601913, -217.2666626, 212.9424438
3: -68.3730927, 106.1663742, -65.8997116, 99.3301086, -167.7031708, 172.0660858
4: -117.1297531, 120.2549515, -107.6202774, 115.7935944, -232.9233093, 227.8752289

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2284433, upper bound: 186.1667493
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2284433, upper bound: 186.1667493
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -78.4691620, 103.4701385, -100.2111282, 116.1518860, -194.6210175, 203.6812744
1: -65.3832474, 93.1269760, -82.9235992, 105.4323578, -170.8156128, 176.0505524
2: -91.4410248, 105.5574799, -116.4361649, 118.2331543, -209.6741791, 221.9936218
3: -63.9262199, 92.4040985, -70.6579437, 115.3432693, -179.2694855, 163.0620117
4: -99.1128998, 111.9080811, -126.5574417, 125.0758667, -224.1887512, 238.4655151

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.3140482, upper bound: 188.2355797
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.3140482, upper bound: 188.2727689
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -77.8985825, 102.8410950, -102.2045746, 117.1111603, -195.0097198, 205.0456696
1: -64.7397842, 92.3455353, -84.4977341, 106.5265961, -171.2663879, 176.8432465
2: -90.4897461, 104.8518677, -118.5997162, 119.2017136, -209.6914673, 223.4515533
3: -63.8641090, 90.5127716, -71.1448975, 117.0975189, -180.9616241, 161.6576538
4: -98.2341003, 111.2121277, -128.9840851, 126.0116959, -224.2457886, 240.1962128

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.3140482, upper bound: 188.2355797
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.3140482, upper bound: 188.2727689
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -78.4691620, 103.4701385, -99.0893936, 115.7138519, -194.1830139, 202.5595398
1: -65.3832474, 93.1269760, -82.0695496, 104.7281342, -170.1113892, 175.1965332
2: -91.4410248, 105.5574799, -115.0334167, 117.7662201, -209.2072449, 220.5908966
3: -63.9262199, 92.4040985, -70.1660004, 113.8812714, -177.8074951, 162.5700989
4: -99.1128998, 111.9080811, -124.7635498, 124.4892731, -223.6021729, 236.6716003

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1418167, upper bound: 185.1840151
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1418167, upper bound: 185.1840151
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -77.8985825, 102.8410950, -100.7103958, 116.3573151, -194.2558594, 203.5514832
1: -64.7397842, 92.3455353, -83.3643799, 105.3513718, -170.0911407, 175.7098999
2: -90.4897461, 104.8518677, -116.7628937, 118.4084244, -208.8981628, 221.6147156
3: -63.8641090, 90.5127716, -70.4987793, 115.1464844, -179.0105896, 161.0115509
4: -98.2341003, 111.2121277, -126.6582642, 125.1467361, -223.3808289, 237.8703918

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1418162, upper bound: 185.2002713
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.7294622, upper bound: 181.2276249
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.3827466, upper bound: 179.2465985
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -90.9992142, 110.7701797, -100.2111282, 116.1518860, -207.1510773, 210.9813080
1: -75.5363464, 99.9865494, -82.9235992, 105.4323578, -180.9687042, 182.9101562
2: -105.7453613, 112.8695450, -116.4361649, 118.2331543, -223.9785156, 229.3056946
3: -67.7385864, 105.2213135, -70.6579437, 115.3432693, -183.0818481, 175.8792419
4: -114.5958176, 119.4744339, -126.5574417, 125.0758667, -239.6716919, 246.0318756

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -172.8127820, upper bound: 177.3116897
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.5389429, upper bound: 173.7323660
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -93.1209030, 111.6165161, -102.2045746, 117.1111603, -210.2320251, 213.8210907
1: -77.1096115, 100.6684418, -84.4977341, 106.5265961, -183.6362000, 185.1661682
2: -107.9064789, 113.7237091, -118.5997162, 119.2017136, -227.1081848, 232.3234253
3: -68.3730927, 106.1663742, -71.1448975, 117.0975189, -185.4705963, 177.3112640
4: -117.1297531, 120.2549515, -128.9840851, 126.0116959, -243.1414490, 249.2390442

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.1884184, upper bound: 177.7072497
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.9754837, upper bound: 174.5816869
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -90.9992142, 110.7701797, -99.0893936, 115.7138519, -206.7130585, 209.8595734
1: -75.5363464, 99.9865494, -82.0695496, 104.7281342, -180.2644806, 182.0560913
2: -105.7453613, 112.8695450, -115.0334167, 117.7662201, -223.5115814, 227.9029541
3: -67.7385864, 105.2213135, -70.1660004, 113.8812714, -181.6198578, 175.3873138
4: -114.5958176, 119.4744339, -124.7635498, 124.4892731, -239.0850830, 244.2379761

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.0606027, upper bound: 174.8512004
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.1328535, upper bound: 171.2871997
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -93.1209030, 111.6165161, -100.7103958, 116.3573151, -209.4781494, 212.3269043
1: -77.1096115, 100.6684418, -83.3643799, 105.3513718, -182.4609833, 184.0328217
2: -107.9064789, 113.7237091, -116.7628937, 118.4084244, -226.3149109, 230.4865875
3: -68.3730927, 106.1663742, -70.4987793, 115.1464844, -183.5195770, 176.6651611
4: -117.1297531, 120.2549515, -126.6582642, 125.1467361, -242.2764893, 246.9132080

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.7743060, upper bound: 175.5706920
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.6090525, upper bound: 171.6090495
time: 0.78 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.52 seconds
IS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -190.7842274, upper bound: 190.7842949
IS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -190.7842274, upper bound: 190.8927040
IS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -190.8926026, upper bound: 190.7849502
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -190.8926025, upper bound: 190.9486574
IS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -188.4025253, upper bound: 186.2202914
IS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -188.4025253, upper bound: 186.2202914
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -188.5460000, upper bound: 186.2426173
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -188.5460000, upper bound: 186.2426173
IS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -190.6395273, upper bound: 190.7612561
IS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -190.6395273, upper bound: 190.8803285
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -190.7385821, upper bound: 190.7612561
IS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -190.7385821, upper bound: 190.9472436
IS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -188.2274761, upper bound: 186.2190757
IS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -188.2274761, upper bound: 186.2190757
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -188.2559101, upper bound: 186.2334862
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -188.2559101, upper bound: 186.2334862
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -190.7656684, upper bound: 190.6395234
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -190.7656684, upper bound: 190.7334616
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -190.8787489, upper bound: 190.6400635
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -190.8787489, upper bound: 190.8151866
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -188.3528107, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -188.3528107, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -188.5015571, upper bound: 185.2059582
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -188.5015571, upper bound: 185.2059582
IS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -176.5652229, upper bound: 180.5086886
IS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -176.1578288, upper bound: 176.7120679
IS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -178.3765857, upper bound: 182.0589109
IS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -177.4503646, upper bound: 177.4808780
IS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.52
Output dim: 0, lower bound: -172.7201194, upper bound: 174.0959647
IS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.52
Output dim: 0, lower bound: -172.7571669, upper bound: 171.1328535
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.52
Output dim: 0, lower bound: -174.6203134, upper bound: 175.7790428
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.52
Output dim: 0, lower bound: -174.3273570, upper bound: 171.6090485
IS_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -186.3148545, upper bound: 188.4174430
IS_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -186.3148545, upper bound: 188.5523977
IS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -186.3148545, upper bound: 188.4174430
IS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -186.3148545, upper bound: 188.5523977
IS_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
IS_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -186.2192850, upper bound: 186.2192850
IS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -185.2998358, upper bound: 188.2364407
IS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -185.2998358, upper bound: 188.3927434
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -185.2998358, upper bound: 188.2364408
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -185.2998363, upper bound: 188.3927434
IS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -185.2284433, upper bound: 186.1667493
IS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -185.2284433, upper bound: 186.1667493
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -185.2284433, upper bound: 186.1667493
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -185.2284433, upper bound: 186.1667493
IS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -186.3140482, upper bound: 188.2355797
IS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -186.3140482, upper bound: 188.2727689
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -186.3140482, upper bound: 188.2355797
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -186.3140482, upper bound: 188.2727689
IS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -186.1418167, upper bound: 185.1840151
IS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -186.1418167, upper bound: 185.1840151
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -182.7294622, upper bound: 181.2276249
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -182.3827466, upper bound: 179.2465985
IS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -172.8127820, upper bound: 177.3116897
IS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.52
Output dim: 0, lower bound: -171.5389429, upper bound: 173.7323660
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.52
Output dim: 0, lower bound: -173.1884184, upper bound: 177.7072497
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.52
Output dim: 0, lower bound: -171.9754837, upper bound: 174.5816869
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.52
Output dim: 0, lower bound: -172.0606027, upper bound: 174.8512004
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.52
Output dim: 0, lower bound: -171.1328535, upper bound: 171.2871997
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.52
Output dim: 0, lower bound: -172.7743060, upper bound: 175.5706920
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.52
Output dim: 0, lower bound: -171.6090525, upper bound: 171.6090495

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -78.6304703, 103.2117615, -181.8422241, 181.8422241
1: -65.3912277, 92.9626999, -65.3912277, 92.9626999, -158.3539124, 158.3539276
2: -91.6809158, 105.3891754, -91.6809158, 105.3891754, -197.0700989, 197.0700989
3: -64.0711365, 92.9304047, -64.0711365, 92.9304047, -157.0015259, 157.0015259
4: -99.6445236, 111.7989883, -99.6445236, 111.7989883, -211.4434662, 211.4434662

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.5784878, upper bound: 188.0646152
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.9471810, upper bound: 187.9471810
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -78.9156494, 103.2667618, -181.8972321, 182.1274109
1: -65.3912277, 92.9626999, -65.4742050, 92.8440399, -158.2352600, 158.4368896
2: -91.6809158, 105.3891754, -91.7911377, 105.4159927, -197.0969086, 197.1803131
3: -64.0711365, 92.9304047, -64.3398361, 92.0656891, -156.1367950, 157.2702332
4: -99.6445236, 111.7989883, -99.9847870, 111.7970657, -211.4415741, 211.7837524

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.5784878, upper bound: 190.8173867
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.9471810, upper bound: 188.1371534
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -78.9156494, 103.2667618, -78.6304703, 103.2117615, -182.1274109, 181.8972168
1: -65.4742050, 92.8440399, -65.3912277, 92.9626999, -158.4368744, 158.2352600
2: -91.7911377, 105.4159927, -91.6809158, 105.3891754, -197.1803131, 197.0969086
3: -64.3398361, 92.0656891, -64.0711365, 92.9304047, -157.2702332, 156.1368103
4: -99.9847870, 111.7970657, -99.6445236, 111.7989883, -211.7837524, 211.4415741

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6322418, upper bound: 188.0624693
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7293823, upper bound: 188.0625456
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -78.9156494, 103.2667618, -78.9156494, 103.2667618, -182.1824036, 182.1824036
1: -65.4742050, 92.8440399, -65.4742050, 92.8440399, -158.3182373, 158.3182373
2: -91.7911377, 105.4159927, -91.7911377, 105.4159927, -197.2071228, 197.2071228
3: -64.3398361, 92.0656891, -64.3398361, 92.0656891, -156.4055176, 156.4055176
4: -99.9847870, 111.7970657, -99.9847870, 111.7970657, -211.7818604, 211.7818604

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6322424, upper bound: 190.7722345
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7293843, upper bound: 190.9298024
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -78.4691620, 103.4701385, -182.1006165, 181.6809235
1: -65.3912277, 92.9626999, -65.3832474, 93.1269760, -158.5182037, 158.3459473
2: -91.6809158, 105.3891754, -91.4410248, 105.5574799, -197.2383728, 196.8302002
3: -64.0711365, 92.9304047, -63.9262199, 92.4040985, -156.4751892, 156.8566132
4: -99.6445236, 111.7989883, -99.1128998, 111.9080811, -211.5525513, 210.9118652

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 7

Time for candidate selection: 4.66 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.4025253, upper bound: 185.4677375
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.4025253, upper bound: 186.2202914
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -77.8985825, 102.8410950, -181.4715576, 181.1103363
1: -65.3912277, 92.9626999, -64.7397842, 92.3455353, -157.7367554, 157.7024689
2: -91.6809158, 105.3891754, -90.4897461, 104.8518677, -196.5327454, 195.8789215
3: -64.0711365, 92.9304047, -63.8641090, 90.5127716, -154.5839081, 156.7945099
4: -99.6445236, 111.7989883, -98.2341003, 111.2121277, -210.8566284, 210.0330811

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 7

Time for candidate selection: 4.97 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.4025253, upper bound: 185.4677280
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.4025253, upper bound: 186.2202914
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -78.9156494, 103.2667618, -78.4691620, 103.4701385, -182.3857880, 181.7359161
1: -65.4742050, 92.8440399, -65.3832474, 93.1269760, -158.6011658, 158.2272949
2: -91.7911377, 105.4159927, -91.4410248, 105.5574799, -197.3486176, 196.8570251
3: -64.3398361, 92.0656891, -63.9262199, 92.4040985, -156.7439117, 155.9918976
4: -99.9847870, 111.7970657, -99.1128998, 111.9080811, -211.8928528, 210.9099731

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 7

Time for candidate selection: 4.84 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.4025253, upper bound: 186.2426139
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.4647134, upper bound: 186.2426172
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -78.9156494, 103.2667618, -77.8985825, 102.8410950, -181.7567444, 181.1653290
1: -65.4742050, 92.8440399, -64.7397842, 92.3455353, -157.8197174, 157.5838318
2: -91.7911377, 105.4159927, -90.4897461, 104.8518677, -196.6429901, 195.9057312
3: -64.3398361, 92.0656891, -63.8641090, 90.5127716, -154.8526001, 155.9297943
4: -99.9847870, 111.7970657, -98.2341003, 111.2121277, -211.1969147, 210.0311584

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 7

Time for candidate selection: 5.32 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.5460000, upper bound: 186.2426173
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.4647139, upper bound: 186.2426173
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -91.9630280, 110.9765549, -78.6304703, 103.2117615, -195.1747894, 189.6070251
1: -76.2160187, 100.2644730, -65.3912277, 92.9626999, -169.1787109, 165.6557007
2: -106.8935013, 113.1315613, -91.6809158, 105.3891754, -212.2826843, 204.8124695
3: -68.0785675, 106.5302124, -64.0711365, 92.9304047, -161.0089722, 170.6013489
4: -116.0856934, 119.8276749, -99.6445236, 111.7989883, -227.8846588, 219.4721527

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.4404875, upper bound: 188.0613311
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4592063, upper bound: 187.8187003
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -91.9630280, 110.9765549, -78.9156494, 103.2667618, -195.2297974, 189.8922119
1: -76.2160187, 100.2644730, -65.4742050, 92.8440399, -169.0600586, 165.7386780
2: -106.8935013, 113.1315613, -91.7911377, 105.4159927, -212.3094940, 204.9226990
3: -68.0785675, 106.5302124, -64.3398361, 92.0656891, -160.1442566, 170.8700562
4: -116.0856934, 119.8276749, -99.9847870, 111.7970657, -227.8827515, 219.8124542

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.2015326, upper bound: 190.8802899
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7700070, upper bound: 186.2901740
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -94.0209885, 111.9910202, -78.6304703, 103.2117615, -197.2327576, 190.6214905
1: -77.7521286, 101.0731201, -65.3912277, 92.9626999, -170.7148132, 166.4643555
2: -109.0427399, 114.1736069, -91.6809158, 105.3891754, -214.4319000, 205.8545227
3: -68.8223877, 107.5189819, -64.0711365, 92.9304047, -161.7527924, 171.5900879
4: -118.6816559, 120.7862167, -99.6445236, 111.7989883, -230.4806061, 220.4306793

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.4537318, upper bound: 188.0436639
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.4178588, upper bound: 188.0279253
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -94.0209885, 111.9910202, -78.9156494, 103.2667618, -197.2877502, 190.9066772
1: -77.7521286, 101.0731201, -65.4742050, 92.8440399, -170.5961609, 166.5473328
2: -109.0427399, 114.1736069, -91.7911377, 105.4159927, -214.4587097, 205.9647522
3: -68.8223877, 107.5189819, -64.3398361, 92.0656891, -160.8880768, 171.8587952
4: -118.6816559, 120.7862167, -99.9847870, 111.7970657, -230.4787292, 220.7709961

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.4537327, upper bound: 190.7413183
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.4178596, upper bound: 190.8907712
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -91.9630280, 110.9765549, -78.4691620, 103.4701385, -195.4331665, 189.4457092
1: -76.2160187, 100.2644730, -65.3832474, 93.1269760, -169.3429871, 165.6477203
2: -106.8935013, 113.1315613, -91.4410248, 105.5574799, -212.4509583, 204.5725861
3: -68.0785675, 106.5302124, -63.9262199, 92.4040985, -160.4826355, 170.4564362
4: -116.0856934, 119.8276749, -99.1128998, 111.9080811, -227.9937439, 218.9405518

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28

Time for candidate selection: 5.24 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1892122, upper bound: 185.0598853
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2274761, upper bound: 186.2190757
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -91.9630280, 110.9765549, -77.8985825, 102.8410950, -194.8041229, 188.8751373
1: -76.2160187, 100.2644730, -64.7397842, 92.3455353, -168.5615387, 165.0042572
2: -106.8935013, 113.1315613, -90.4897461, 104.8518677, -211.7453308, 203.6213074
3: -68.0785675, 106.5302124, -63.8641090, 90.5127716, -158.5913391, 170.3943176
4: -116.0856934, 119.8276749, -98.2341003, 111.2121277, -227.2978210, 218.0617676

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28

Time for candidate selection: 5.48 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1892122, upper bound: 185.0598948
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2274761, upper bound: 186.2190757
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -94.0209885, 111.9910202, -78.4691620, 103.4701385, -197.4911194, 190.4601746
1: -77.7521286, 101.0731201, -65.3832474, 93.1269760, -170.8791046, 166.4563599
2: -109.0427399, 114.1736069, -91.4410248, 105.5574799, -214.6001740, 205.6146240
3: -68.8223877, 107.5189819, -63.9262199, 92.4040985, -161.2264709, 171.4451904
4: -118.6816559, 120.7862167, -99.1128998, 111.9080811, -230.5897064, 219.8991089

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.6444116, upper bound: 182.7419514
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.1906930, upper bound: 186.2005149
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -94.0209885, 111.9910202, -77.8985825, 102.8410950, -196.8620911, 189.8896027
1: -77.7521286, 101.0731201, -64.7397842, 92.3455353, -170.0976562, 165.8128967
2: -109.0427399, 114.1736069, -90.4897461, 104.8518677, -213.8945465, 204.6633606
3: -68.8223877, 107.5189819, -63.8641090, 90.5127716, -159.3351593, 171.3830872
4: -118.6816559, 120.7862167, -98.2341003, 111.2121277, -229.8937836, 219.0203247

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 28

Time for candidate selection: 5.45 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2419197, upper bound: 186.1760501
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2559101, upper bound: 186.2334862
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -91.9630280, 110.9765549, -189.6070251, 195.1747894
1: -65.3912277, 92.9626999, -76.2160187, 100.2644730, -165.6557007, 169.1787109
2: -91.6809158, 105.3891754, -106.8935013, 113.1315613, -204.8124695, 212.2826843
3: -64.0711365, 92.9304047, -68.0785675, 106.5302124, -170.6013489, 161.0089722
4: -99.6445236, 111.7989883, -116.0856934, 119.8276749, -219.4721527, 227.8846436

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.5411685, upper bound: 187.9923981
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.5442279, upper bound: 187.6037955
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.9129211, upper bound: 187.4860365
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -94.0209885, 111.9910202, -190.6214905, 197.2327576
1: -65.3912277, 92.9626999, -77.7521286, 101.0731201, -166.4643555, 170.7148285
2: -91.6809158, 105.3891754, -109.0427399, 114.1736069, -205.8545227, 214.4319000
3: -64.0711365, 92.9304047, -68.8223877, 107.5189819, -171.5900879, 161.7527771
4: -99.6445236, 111.7989883, -118.6816559, 120.7862167, -220.4307098, 230.4806061

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.5411685, upper bound: 190.7334616
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.5442279, upper bound: 190.6606020
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.9129211, upper bound: 188.0272241
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -78.9156494, 103.2667618, -91.9630280, 110.9765549, -189.8922119, 195.2297974
1: -65.4742050, 92.8440399, -76.2160187, 100.2644730, -165.7386780, 169.0600586
2: -91.7911377, 105.4159927, -106.8935013, 113.1315613, -204.9226990, 212.3094940
3: -64.3398361, 92.0656891, -68.0785675, 106.5302124, -170.8700562, 160.1442566
4: -99.9847870, 111.7970657, -116.0856934, 119.8276749, -219.8124390, 227.8827515

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6580434, upper bound: 187.9927601
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7186708, upper bound: 187.9927601
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -78.9156494, 103.2667618, -94.0209885, 111.9910202, -190.9066772, 197.2877502
1: -65.4742050, 92.8440399, -77.7521286, 101.0731201, -166.5473328, 170.5961609
2: -91.7911377, 105.4159927, -109.0427399, 114.1736069, -205.9647522, 214.4587097
3: -64.3398361, 92.0656891, -68.8223877, 107.5189819, -171.8588257, 160.8880768
4: -99.9847870, 111.7970657, -118.6816559, 120.7862167, -220.7709961, 230.4787140

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6580446, upper bound: 190.7322546
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7186727, upper bound: 190.8147050
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -90.9992142, 110.7701797, -189.4006500, 194.2109528
1: -65.3912277, 92.9626999, -75.5363464, 99.9865494, -165.3777771, 168.4990387
2: -91.6809158, 105.3891754, -105.7453613, 112.8695450, -204.5504456, 211.1345367
3: -64.0711365, 92.9304047, -67.7385864, 105.2213135, -169.2924347, 160.6689911
4: -99.6445236, 111.7989883, -114.5958176, 119.4744339, -219.1189423, 226.3947754

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9969110, upper bound: 179.6202485
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.9064154, upper bound: 176.6788072
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 7

Time for candidate selection: 6.90 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3528107, upper bound: 184.6633666
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3528107, upper bound: 185.1759076
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -93.1209030, 111.6165161, -190.2469788, 196.3326416
1: -65.3912277, 92.9626999, -77.1096115, 100.6684418, -166.0596619, 170.0722961
2: -91.6809158, 105.3891754, -107.9064789, 113.7237091, -205.4046173, 213.2956543
3: -64.0711365, 92.9304047, -68.3730927, 106.1663742, -170.2375183, 161.3034821
4: -99.6445236, 111.7989883, -117.1297531, 120.2549515, -219.8994446, 228.9287415

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9969110, upper bound: 183.9200554
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.9064154, upper bound: 182.5404546
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 7

Time for candidate selection: 6.98 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3528107, upper bound: 184.6633696
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3528107, upper bound: 185.1759076
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -78.9156494, 103.2667618, -90.9992142, 110.7701797, -189.6858215, 194.2659454
1: -65.4742050, 92.8440399, -75.5363464, 99.9865494, -165.4607544, 168.3803864
2: -91.7911377, 105.4159927, -105.7453613, 112.8695450, -204.6606750, 211.1613464
3: -64.3398361, 92.0656891, -67.7385864, 105.2213135, -169.5611572, 159.8042755
4: -99.9847870, 111.7970657, -114.5958176, 119.4744339, -219.4592285, 226.3928833

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=279.9653625488281
rel_dist={0: [-191.43347747308115, 191.4334774730812]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.7668396, upper bound: 180.8103516
time: 0.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1338263, upper bound: 191.1338286
time: 0.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.55 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 0, lower bound: -178.7668396, upper bound: 180.8103516
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 0, lower bound: -191.1338263, upper bound: 191.1338286

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -165.7317810, 169.9891968, -99.5717773, 119.4847870, -285.2165527, 269.5609436
1: -136.9342194, 156.5604401, -82.2094345, 108.2213745, -245.1555939, 238.7698669
2: -192.2410583, 168.2551727, -115.4599609, 121.4488373, -313.6898804, 283.7151489
3: -91.3198547, 188.0007629, -71.8553314, 115.1565094, -206.4763641, 259.8560486
4: -208.4816437, 177.5048828, -125.8874283, 128.5055542, -336.9871826, 303.3922424

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.7654832, upper bound: 177.7654832
time: 0.54 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.7654832, upper bound: 180.8103516
time: 0.68 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -135.6846161, 140.4945374, -266.8054504, 270.4803467
1: -104.3817749, 123.2117767, -112.0749969, 128.5360260, -232.9178009, 235.2867432
2: -146.5827942, 135.2793121, -157.4221497, 140.3601532, -286.9429321, 292.7014771
3: -78.7137909, 144.2006378, -81.2042694, 154.1939697, -232.9077301, 225.4048920
4: -159.2684784, 142.8177032, -170.9760895, 148.0922089, -307.3606873, 313.7937927

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.8103516, upper bound: 178.7668396
time: 0.57 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -180.8103516, upper bound: 191.1338286
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.38 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -177.7654832, upper bound: 177.7654832
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -177.7654832, upper bound: 180.8103516
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -180.8103516, upper bound: 178.7668396
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -180.8103516, upper bound: 191.1338286

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -165.7317810, 169.9891968, -162.0709229, 167.6504669, -333.3822327, 332.0600891
1: -136.9342194, 156.5604401, -134.1833801, 154.4241333, -291.3583374, 290.7438354
2: -192.2410583, 168.2551727, -188.2958679, 166.0926361, -358.3336182, 356.5510254
3: -91.3198547, 188.0007629, -90.5365677, 184.9062195, -276.2260742, 278.5373230
4: -208.4816437, 177.5048828, -203.9703979, 174.9989166, -383.4804993, 381.4752808

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -173.2113163, upper bound: 171.1920654
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.3794217, upper bound: 170.3794217
time: 0.56 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -165.7317810, 169.9891968, -120.0776215, 130.6659546, -296.3977356, 290.0667725
1: -136.9342194, 156.5604401, -99.2898560, 119.4401779, -256.3743286, 255.8502808
2: -192.2410583, 168.2551727, -139.4512329, 132.0033112, -324.2442322, 307.7063599
3: -91.3198547, 188.0007629, -77.1049347, 137.7709656, -229.0908203, 265.1056824
4: -208.4816437, 177.5048828, -151.5433502, 139.3815155, -347.8630981, 329.0482178

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -173.2113163, upper bound: 174.3704125
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.3794217, upper bound: 173.2171620
time: 0.70 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -165.7308350, 169.9760437, -296.2869568, 300.5265503
1: -104.3817749, 123.2117767, -136.9331818, 156.5503845, -260.9321594, 260.1449585
2: -146.5827942, 135.2793121, -192.2384796, 168.2447968, -314.8275757, 327.5177917
3: -78.7137909, 144.2006378, -91.3181000, 187.9923859, -266.7061462, 235.5187378
4: -159.2684784, 142.8177032, -208.4785309, 177.4915466, -336.7600098, 351.2962341

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.8768175, upper bound: 171.7687831
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -173.2171577, upper bound: 171.5441458
time: 0.59 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -126.3109131, 134.7957306, -126.3109131, 134.7957306, -261.1066284, 261.1066284
1: -104.3817749, 123.2117767, -104.3817749, 123.2117767, -227.5935364, 227.5935211
2: -146.5827942, 135.2793121, -146.5827942, 135.2793121, -281.8620911, 281.8620911
3: -78.7137909, 144.2006378, -78.7137909, 144.2006378, -222.9144287, 222.9144287
4: -159.2684784, 142.8177032, -159.2684784, 142.8177032, -302.0861816, 302.0861816

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -174.8768194, upper bound: 186.6974360
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -173.2171620, upper bound: 186.6846139
time: 0.53 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.19 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.19
Output dim: 0, lower bound: -173.2113163, upper bound: 171.1920654
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.19
Output dim: 0, lower bound: -170.3794217, upper bound: 170.3794217
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.19
Output dim: 0, lower bound: -173.2113163, upper bound: 174.3704125
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.19
Output dim: 0, lower bound: -170.3794217, upper bound: 173.2171620
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.19
Output dim: 0, lower bound: -174.8768175, upper bound: 171.7687831
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.19
Output dim: 0, lower bound: -173.2171577, upper bound: 171.5441458
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -174.8768194, upper bound: 186.6974360
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -173.2171620, upper bound: 186.6846139

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -112.0257797, 124.0801163, -120.1664124, 130.2270508, -242.2528381, 244.2464905
1: -92.6331711, 113.0716934, -99.3317642, 118.8982620, -211.5314331, 212.4034576
2: -130.0409088, 125.1827774, -139.4687500, 130.9649506, -261.0057983, 264.6515198
3: -74.0341339, 128.2513275, -76.7070618, 137.3722534, -211.4063873, 204.9583740
4: -141.3389282, 132.2405701, -151.5593262, 138.2962036, -279.6351318, 283.7998352

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6545023, upper bound: 186.6569586
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7383466, upper bound: 185.7844461
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -109.6574478, 121.9550476, -113.2088928, 125.3538437, -235.0112915, 235.1639404
1: -90.7688217, 111.0725479, -93.6585236, 114.3264389, -205.0952606, 204.7310638
2: -127.1920471, 123.9425659, -131.3547974, 126.9600601, -254.1520996, 255.2973633
3: -73.1300201, 125.2666092, -74.6668701, 129.3716583, -202.5016785, 199.9334717
4: -137.8858032, 130.8748932, -142.5425262, 134.1998291, -272.0856018, 273.4174194

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7833257, upper bound: 186.6336781
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
time: 0.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.76 seconds
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -187.6545023, upper bound: 186.6569586
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -187.7383466, upper bound: 185.7844461
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -185.7833257, upper bound: 186.6336781
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -98.7407684, 115.1411285, -95.5981750, 114.0148392, -212.7556152, 210.7392731
1: -81.6861649, 104.7747421, -79.2648010, 103.6729965, -185.3591614, 184.0395508
2: -114.6209946, 116.8553162, -111.3381195, 116.1443024, -230.7652893, 228.1934357
3: -69.7889023, 113.6269913, -69.1974487, 111.1452408, -180.9340973, 182.8244324
4: -124.6354828, 123.5493011, -120.9247665, 122.8834839, -247.5189667, 244.4740448

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8518784, upper bound: 186.0944179
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -109.7476730, 122.4863892, -113.0142975, 125.2319336, -234.9795837, 235.5006866
1: -90.7455292, 111.5849380, -93.4145203, 114.2324066, -204.9779358, 204.9994049
2: -127.3901825, 123.8168182, -131.1571350, 126.6466599, -254.0368347, 254.9739532
3: -73.3614655, 125.7217407, -74.5843582, 129.4272156, -202.7886658, 200.3060913
4: -138.4773560, 130.8197937, -142.5761414, 133.8083649, -272.2857056, 273.3959351

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8508673, upper bound: 185.1954048
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -97.8751526, 114.3515625, -90.3036423, 111.0928421, -208.9679413, 204.6552124
1: -81.0732651, 103.6351013, -74.9432373, 100.5904846, -181.6637573, 178.5783081
2: -113.5271912, 116.4141922, -105.1401672, 113.1799774, -226.7071686, 221.5543213
3: -69.3547440, 112.3326721, -67.7220383, 105.1223450, -174.4770813, 180.0547028
4: -123.1190033, 123.0731430, -114.0968018, 119.8596802, -242.9786682, 237.1699524

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2171631, upper bound: 186.1290138
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -107.5599899, 120.6634979, -106.4291382, 120.8401184, -228.4000854, 227.0926056
1: -89.0310516, 109.6851196, -88.0397873, 109.8472443, -198.8782959, 197.7248840
2: -124.7508926, 122.6711884, -123.4730682, 122.8363190, -247.5871887, 246.1441956
3: -72.5076752, 122.9317780, -72.6447067, 121.8211136, -194.3287811, 195.5764771
4: -135.2579956, 129.5533447, -134.0461426, 129.9118805, -265.1697998, 263.5994873

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7347796, upper bound: 185.7386919
time: 0.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.70 seconds
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -186.8518784, upper bound: 186.0944179
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -186.8508673, upper bound: 185.1954048
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -185.2171631, upper bound: 186.1290138
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -185.7347798, upper bound: 185.7386919
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -185.7347796, upper bound: 185.7386919

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -87.9776001, 108.2874222, -88.3502502, 109.8533401, -197.8309021, 196.6376343
1: -72.9800644, 98.0002899, -73.3083496, 99.3782806, -172.3583374, 171.3086243
2: -102.3379059, 110.4846039, -102.9094849, 111.9728394, -214.3107452, 213.3940887
3: -66.6643219, 102.4660034, -67.2305756, 103.2851105, -169.9494324, 169.6965790
4: -111.1385574, 117.0449600, -111.8236847, 118.6308212, -229.7693787, 228.8686371

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.9019678, upper bound: 186.1745118
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.9019678, upper bound: 186.1745118
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -88.2088928, 108.2571411, -89.6991425, 110.3468018, -198.5556641, 197.9562683
1: -72.9888840, 97.7906876, -74.3725586, 99.8483734, -172.8371735, 172.1632385
2: -102.3158417, 110.4738388, -104.3472443, 112.4774094, -214.7932434, 214.8210602
3: -66.8833466, 101.3135910, -67.5621033, 104.3029633, -171.1863098, 168.8757019
4: -111.3934021, 116.9380264, -113.4251633, 119.1407928, -230.5341949, 230.3631897

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8518784, upper bound: 186.0944179
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8518784, upper bound: 186.0944179
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -96.1349487, 113.4527512, -103.8675003, 119.3779678, -215.5128632, 217.3202515
1: -79.6951218, 102.8921051, -85.9543915, 108.3832474, -188.0783539, 188.8464966
2: -111.7815094, 115.6093063, -120.6897736, 121.3948212, -233.1763306, 236.2990723
3: -69.2827454, 111.1328735, -72.0616913, 119.5430298, -188.8257751, 183.1945648
4: -121.3083954, 122.3943939, -131.1181488, 128.4261780, -249.7345734, 253.5125427

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -97.8884659, 114.3051682, -105.9971771, 120.2535248, -218.1419830, 220.3023376
1: -80.9806976, 103.5719223, -87.6332092, 109.4822769, -190.4629669, 191.2050934
2: -113.5681839, 116.4872894, -122.9797516, 122.2892838, -235.8574677, 239.4670410
3: -69.9507446, 111.7727356, -72.5682373, 121.1228104, -191.0735474, 184.3409729
4: -123.5076218, 123.1841125, -133.6939697, 129.2659607, -252.7735901, 256.8780823

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8508673, upper bound: 185.1954048
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8508673, upper bound: 185.1954048
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -87.7265015, 108.4833603, -83.1872559, 107.1249390, -194.8514404, 191.6705933
1: -72.9088745, 97.9554443, -69.1092682, 96.6483917, -169.5572662, 167.0647125
2: -102.0393219, 110.6087418, -96.8773117, 109.1823425, -211.2216339, 207.4860535
3: -66.5177765, 101.9232483, -65.8218460, 97.4388657, -163.9566345, 167.7450867
4: -110.5511169, 117.1125717, -105.1892700, 115.8016968, -226.3527985, 222.3018494

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -87.9035645, 108.1686783, -84.4253387, 107.5278625, -195.4314270, 192.5940247
1: -72.8521805, 97.5076752, -70.0861816, 97.0342560, -169.8864441, 167.5938568
2: -101.8921814, 110.2828140, -98.1606827, 109.5966187, -211.4888000, 208.4434967
3: -66.5803909, 100.5617676, -66.1128311, 98.3252258, -164.9056091, 166.6745758
4: -110.5906982, 116.7047806, -106.6078262, 116.2161407, -226.8068085, 223.3126068

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2171631, upper bound: 186.1290138
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2171631, upper bound: 186.1290138
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -87.7036362, 108.7315369, -106.4291382, 120.8401184, -208.5437622, 215.1606750
1: -72.8436890, 98.2632751, -88.0397873, 109.8472443, -182.6909027, 186.3030243
2: -102.0504456, 110.7943802, -123.4730682, 122.8363190, -224.8867493, 234.2674408
3: -66.5469589, 102.0576096, -72.6447067, 121.8211136, -188.3680725, 174.7023163
4: -110.6790924, 117.2479630, -134.0461426, 129.9118805, -240.5909729, 251.2940826

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.1845224, upper bound: 185.3823186
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.1759076, upper bound: 185.1759076
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -103.1638489, 117.9738159, -106.4291382, 120.8401184, -224.0039520, 224.4029388
1: -85.3876343, 106.9490433, -88.0397873, 109.8472443, -195.2348480, 194.9888153
2: -119.6347275, 120.0215302, -123.4730682, 122.8363190, -242.4710388, 243.4945526
3: -71.2133255, 118.0576172, -72.6447067, 121.8211136, -193.0344391, 190.7023315
4: -129.7596588, 126.8023834, -134.0461426, 129.9118805, -259.6715088, 260.8485107

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.1845224, upper bound: 185.3823186
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.1759076, upper bound: 185.1759076
time: 0.64 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.42 seconds
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -186.9019678, upper bound: 186.1745118
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -186.9019678, upper bound: 186.1745118
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -186.8518784, upper bound: 186.0944179
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -186.8518784, upper bound: 186.0944179
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -186.8508673, upper bound: 185.1954048
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -186.8508673, upper bound: 185.1954048
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -185.2171631, upper bound: 186.1290138
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -185.2171631, upper bound: 186.1290138
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -185.1845224, upper bound: 185.3823186
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -185.1759076, upper bound: 185.1759076
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -185.1845224, upper bound: 185.3823186
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.42
Output dim: 0, lower bound: -185.1759076, upper bound: 185.1759076

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -87.9776001, 108.2874222, -82.0610886, 105.3424301, -193.3199921, 190.3484802
1: -72.9800644, 98.0002899, -68.1042328, 94.9625549, -167.9426270, 166.1045227
2: -102.3379059, 110.4846039, -95.5750885, 107.4659119, -209.8038025, 206.0596924
3: -66.6643219, 102.4660034, -65.2240982, 96.2926483, -162.9569244, 167.6900940
4: -111.1385574, 117.0449600, -103.9902267, 113.9467926, -225.0853424, 221.0351562

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -87.9776001, 108.2874222, -81.0018234, 104.9991684, -192.9767456, 189.2892456
1: -72.9800644, 98.0002899, -67.3439331, 94.5523834, -167.5324402, 165.3442230
2: -102.3379059, 110.4846039, -94.2812119, 107.0258408, -209.3637390, 204.7658081
3: -66.6643219, 102.4660034, -64.7617645, 94.7947769, -161.4590759, 167.2277527
4: -111.1385574, 117.0449600, -102.2960052, 113.4245834, -224.5631409, 219.3409729

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -88.2088928, 108.2571411, -83.6409683, 106.0165482, -194.2254333, 191.8981018
1: -72.9888840, 97.7906876, -69.3598175, 95.5994415, -168.5882721, 167.1505127
2: -102.3158417, 110.4738388, -97.2918549, 108.1578751, -210.4737091, 207.7656860
3: -66.8833466, 101.3135910, -65.6236954, 97.5932922, -164.4766388, 166.9372864
4: -111.3934021, 116.9380264, -105.8965683, 114.6326447, -226.0260162, 222.8345947

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8518784, upper bound: 186.0944171
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8518784, upper bound: 186.0944179
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -88.2088928, 108.2571411, -82.1276627, 105.3648529, -193.5737152, 190.3847961
1: -72.9888840, 97.7906876, -68.2199402, 94.8961868, -167.8850403, 166.0106201
2: -102.3158417, 110.4738388, -95.4157639, 107.3954773, -209.7113190, 205.8896027
3: -66.8833466, 101.3135910, -65.0046082, 95.5868912, -162.4702454, 166.3181915
4: -111.3934021, 116.9380264, -103.5401535, 113.8027725, -225.1961670, 220.4781647

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8518784, upper bound: 186.0944179
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8518784, upper bound: 186.0944179
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -96.1349487, 113.4527512, -96.7277985, 114.0814133, -210.2163391, 210.1805420
1: -79.6951218, 102.8921051, -80.0583954, 103.1765671, -182.8716736, 182.9505005
2: -111.7815094, 115.6093063, -112.3797913, 116.1788254, -227.9603271, 227.9890747
3: -69.2827454, 111.1328735, -69.6931305, 111.5153809, -180.7981262, 180.8260040
4: -121.3083954, 122.3943939, -122.1692734, 122.9708328, -244.2792358, 244.5636444

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -96.1349487, 113.4527512, -95.9936829, 113.9485397, -210.0834961, 209.4464417
1: -79.6951218, 102.8921051, -79.5642395, 102.9904327, -182.6855316, 182.4563446
2: -111.7815094, 115.6093063, -111.4978180, 116.0074005, -227.7889099, 227.1071167
3: -69.2827454, 111.1328735, -69.3579865, 110.5664749, -179.8492126, 180.4908600
4: -121.3083954, 122.3943939, -120.9444962, 122.6787872, -243.9871826, 243.3388672

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -97.8884659, 114.3051682, -98.7661514, 114.9155502, -212.8040161, 213.0713043
1: -80.9806976, 103.5719223, -81.6709747, 104.1926575, -185.1733398, 185.2428894
2: -113.5681839, 116.4872894, -114.5949554, 117.0547256, -230.6229095, 231.0822296
3: -69.9507446, 111.7727356, -70.1461182, 113.0315857, -182.9823303, 181.9188385
4: -123.5076218, 123.1841125, -124.6522980, 123.7725372, -247.2801514, 247.8364105

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8508673, upper bound: 185.1954048
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8508673, upper bound: 185.1954048
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -97.8884659, 114.3051682, -97.4446030, 114.2804108, -212.1688538, 211.7497559
1: -80.9806976, 103.5719223, -80.6750259, 103.3011398, -184.2818146, 184.2469482
2: -113.5681839, 116.4872894, -112.9487915, 116.3551331, -229.9233093, 229.4360809
3: -69.9507446, 111.7727356, -69.5654144, 111.2999115, -181.2506561, 181.3381348
4: -123.5076218, 123.1841125, -122.5433197, 123.0025177, -246.5101318, 245.7274323

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8508673, upper bound: 185.1954048
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8508673, upper bound: 185.1954048
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -87.7265015, 108.4833603, -82.0610886, 105.3424301, -193.0689392, 190.5444183
1: -72.9088745, 97.9554443, -68.1042328, 94.9625549, -167.8714294, 166.0596771
2: -102.0393219, 110.6087418, -95.5750885, 107.4659119, -209.5052185, 206.1838379
3: -66.5177765, 101.9232483, -65.2240982, 96.2926483, -162.8104095, 167.1473389
4: -110.5511169, 117.1125717, -103.9902267, 113.9467926, -224.4978790, 221.1027527

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -87.7265015, 108.4833603, -81.0372849, 105.0178909, -192.7443848, 189.5206451
1: -72.9088745, 97.9554443, -67.3724594, 94.5705261, -167.4794006, 165.3279114
2: -102.0393219, 110.6087418, -94.3196259, 107.0446167, -209.0839386, 204.9283752
3: -66.5177765, 101.9232483, -64.7711792, 94.8270264, -161.3448029, 166.6944275
4: -110.5511169, 117.1125717, -102.3381653, 113.4438095, -223.9949036, 219.4507294

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -87.9035645, 108.1686783, -83.6409683, 106.0165482, -193.9201050, 191.8096466
1: -72.8521805, 97.5076752, -69.3598175, 95.5994415, -168.4515991, 166.8674927
2: -101.8921814, 110.2828140, -97.2918549, 108.1578751, -210.0500488, 207.5746765
3: -66.5803909, 100.5617676, -65.6236954, 97.5932922, -164.1736755, 166.1854553
4: -110.5906982, 116.7047806, -105.8965683, 114.6326447, -225.2233124, 222.6013489

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2171631, upper bound: 186.1290124
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2171631, upper bound: 186.1290138
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -87.9035645, 108.1686783, -82.1276627, 105.3648529, -193.2684021, 190.2963409
1: -72.8521805, 97.5076752, -68.2199402, 94.8961868, -167.7483673, 165.7276001
2: -101.8921814, 110.2828140, -95.4157639, 107.3954773, -209.2876587, 205.6985779
3: -66.5803909, 100.5617676, -65.0046082, 95.5868912, -162.1672516, 165.5663452
4: -110.5906982, 116.7047806, -103.5401535, 113.8027725, -224.3934631, 220.2449188

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2171631, upper bound: 186.1290138
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.2171631, upper bound: 186.1290138
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -81.0372849, 105.0178909, -93.3521652, 112.9497070, -193.9869995, 198.3700562
1: -67.3724594, 94.5705261, -77.4321442, 102.1221848, -169.4946442, 172.0026703
2: -94.3196259, 107.0446167, -108.5130081, 115.0706940, -209.3903198, 215.5576172
3: -64.7711792, 94.8270264, -68.7948303, 107.9435425, -172.7147064, 163.6218567
4: -102.3381653, 113.4438095, -117.6463470, 121.9186478, -224.2568054, 231.0901489

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1325237, upper bound: 185.1820568
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1325237, upper bound: 185.1929490
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -82.1276627, 105.3648529, -94.9664917, 113.5634918, -195.6911621, 200.3313446
1: -68.2199402, 94.8961868, -78.6105652, 102.5655670, -170.7854767, 173.5067291
2: -95.4157639, 107.3954773, -110.1304245, 115.6814346, -211.0971985, 217.5258942
3: -65.0046082, 95.5868912, -69.3339691, 108.3247299, -173.3293152, 164.9208527
4: -103.5401535, 113.8027725, -119.6191406, 122.4709244, -226.0110779, 233.4219055

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8530616, upper bound: 179.0995434
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.3436480, upper bound: 179.0416369
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -95.9973450, 113.9500885, -93.3521652, 112.9497070, -208.9470520, 207.3022461
1: -79.5670929, 102.9919281, -77.4321442, 102.1221848, -181.6892700, 180.4240723
2: -111.5017166, 116.0089340, -108.5130081, 115.0706940, -226.5724030, 224.5219116
3: -69.3587112, 110.5694885, -68.7948303, 107.9435425, -177.3022461, 179.3643036
4: -120.9488831, 122.6803436, -117.6463470, 121.9186478, -242.8675232, 240.3266754

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.9754283, upper bound: 171.1469466
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.1328535, upper bound: 171.1328535
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -97.4881897, 114.2985687, -94.9664917, 113.5634918, -211.0516205, 209.2650604
1: -80.7088242, 103.3190536, -78.6105652, 102.5655670, -183.2743835, 181.9296265
2: -112.9948425, 116.3731842, -110.1304245, 115.6814346, -228.6762695, 226.5035858
3: -69.5740128, 111.3350372, -69.3339691, 108.3247299, -177.8987274, 180.6690063
4: -122.5949707, 123.0207062, -119.6191406, 122.4709244, -245.0658875, 242.6398163

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.7450254, upper bound: 172.0186522
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.6090531, upper bound: 171.6090525
time: 0.85 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.84 seconds
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.8518784, upper bound: 186.0944171
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.8518784, upper bound: 186.0944179
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.8518784, upper bound: 186.0944179
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.8518784, upper bound: 186.0944179
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.8508673, upper bound: 185.1954048
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.8508673, upper bound: 185.1954048
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.8508673, upper bound: 185.1954048
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.8508673, upper bound: 185.1954048
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -185.2171631, upper bound: 186.1290124
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -185.2171631, upper bound: 186.1290138
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -185.2171631, upper bound: 186.1290138
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -185.2171631, upper bound: 186.1290138
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.1325237, upper bound: 185.1820568
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -186.1325237, upper bound: 185.1929490
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -181.8530616, upper bound: 179.0995434
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 0, lower bound: -181.3436480, upper bound: 179.0416369
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.84
Output dim: 0, lower bound: -171.9754283, upper bound: 171.1469466
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.84
Output dim: 0, lower bound: -171.1328535, upper bound: 171.1328535
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.84
Output dim: 0, lower bound: -172.7450254, upper bound: 172.0186522
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.84
Output dim: 0, lower bound: -171.6090531, upper bound: 171.6090525

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -82.0610886, 105.3424301, -183.9729004, 185.2728424
1: -65.3912277, 92.9626999, -68.1042328, 94.9625549, -160.3537903, 161.0669098
2: -91.6809158, 105.3891754, -95.5750885, 107.4659119, -199.1468048, 200.9642639
3: -64.0711365, 92.9304047, -65.2240982, 96.2926483, -160.3637543, 158.1544952
4: -99.6445236, 111.7989883, -103.9902267, 113.9467926, -213.5912781, 215.7891541

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6727376, upper bound: 190.7253771
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6727373, upper bound: 190.7667056
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -91.3995895, 110.6041260, -82.0610886, 105.3424301, -196.7420197, 192.6652069
1: -75.7478790, 99.9164047, -68.1042328, 94.9625549, -170.7104340, 168.0206146
2: -106.2185059, 112.7644730, -95.5750885, 107.4659119, -213.6844025, 208.3395538
3: -67.8935394, 105.8188629, -65.2240982, 96.2926483, -164.1861725, 171.0429535
4: -115.3436584, 119.4522095, -103.9902267, 113.9467926, -229.2904510, 223.4423981

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6727376, upper bound: 190.7253771
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6727373, upper bound: 190.7667056
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -81.0018234, 104.9991684, -183.6296387, 184.2135925
1: -65.3912277, 92.9626999, -67.3439331, 94.5523834, -159.9436035, 160.3066406
2: -91.6809158, 105.3891754, -94.2812119, 107.0258408, -198.7067566, 199.6703796
3: -64.0711365, 92.9304047, -64.7617645, 94.7947769, -158.8659058, 157.6921387
4: -99.6445236, 111.7989883, -102.2960052, 113.4245834, -213.0690918, 214.0949707

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.9019678, upper bound: 186.1745118
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -91.3995895, 110.6041260, -81.0018234, 104.9991684, -196.3987579, 191.6059570
1: -75.7478790, 99.9164047, -67.3439331, 94.5523834, -170.3002625, 167.2603302
2: -106.2185059, 112.7644730, -94.2812119, 107.0258408, -213.2443542, 207.0456848
3: -67.8935394, 105.8188629, -64.7617645, 94.7947769, -162.6883240, 170.5806274
4: -115.3436584, 119.4522095, -102.2960052, 113.4245834, -228.7682495, 221.7481842

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.9019678, upper bound: 186.1745118
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -78.9156494, 103.2667618, -83.6409683, 106.0165482, -184.9321899, 186.9077301
1: -65.4742050, 92.8440399, -69.3598175, 95.5994415, -161.0736237, 162.2038574
2: -91.7911377, 105.4159927, -97.2918549, 108.1578751, -199.9490051, 202.7078552
3: -64.3398361, 92.0656891, -65.6236954, 97.5932922, -161.9331360, 157.6893768
4: -99.9847870, 111.7970657, -105.8965683, 114.6326447, -214.6174164, 217.6936340

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7422150, upper bound: 190.7731926
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8156361, upper bound: 190.9288452
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -92.7008362, 111.1121902, -83.6409683, 106.0165482, -198.7173767, 194.7531586
1: -76.6542511, 100.2565689, -69.3598175, 95.5994415, -172.2536621, 169.6163635
2: -107.4650955, 113.3041992, -97.2918549, 108.1578751, -215.6229401, 210.5960541
3: -68.3835297, 105.8630829, -65.6236954, 97.5932922, -165.9768219, 171.4867859
4: -116.9510498, 119.9003525, -105.8965683, 114.6326447, -231.5836792, 225.7969208

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7422150, upper bound: 190.7731926
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8156361, upper bound: 190.9288452
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -78.9156494, 103.2667618, -82.1276627, 105.3648529, -184.2805023, 185.3944244
1: -65.4742050, 92.8440399, -68.2199402, 94.8961868, -160.3703918, 161.0639801
2: -91.7911377, 105.4159927, -95.4157639, 107.3954773, -199.1866150, 200.8317566
3: -64.3398361, 92.0656891, -65.0046082, 95.5868912, -159.9267273, 157.0702972
4: -99.9847870, 111.7970657, -103.5401535, 113.8027725, -213.7875671, 215.3372192

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.0466610, upper bound: 181.5898125
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.0470103, upper bound: 180.1373937
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -92.7008362, 111.1121902, -82.1276627, 105.3648529, -198.0656433, 193.2398529
1: -76.6542511, 100.2565689, -68.2199402, 94.8961868, -171.5504303, 168.4764862
2: -107.4650955, 113.3041992, -95.4157639, 107.3954773, -214.8605499, 208.7199707
3: -68.3835297, 105.8630829, -65.0046082, 95.5868912, -163.9704132, 170.8676910
4: -116.9510498, 119.9003525, -103.5401535, 113.8027725, -230.7538147, 223.4405060

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.0466610, upper bound: 181.5898125
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.0470103, upper bound: 180.1373937
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -96.7277985, 114.0814133, -192.7118683, 199.9395599
1: -65.3912277, 92.9626999, -80.0583954, 103.1765671, -168.5677948, 173.0210724
2: -91.6809158, 105.3891754, -112.3797913, 116.1788254, -207.8597412, 217.7689667
3: -64.0711365, 92.9304047, -69.6931305, 111.5153809, -175.5865173, 162.6235352
4: -99.6445236, 111.7989883, -122.1692734, 122.9708328, -222.6153564, 233.9682007

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6727376, upper bound: 190.6405225
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6727376, upper bound: 190.6906324
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -91.9630280, 110.9765549, -96.7277985, 114.0814133, -206.0444183, 207.7043457
1: -76.2160187, 100.2644730, -80.0583954, 103.1765671, -179.3925781, 180.3228455
2: -106.8935013, 113.1315613, -112.3797913, 116.1788254, -223.0723267, 225.5113525
3: -68.0785675, 106.5302124, -69.6931305, 111.5153809, -179.5939484, 176.2233429
4: -116.0856934, 119.8276749, -122.1692734, 122.9708328, -239.0565186, 241.9969330

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6727376, upper bound: 190.6405774
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6727376, upper bound: 190.6906324
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -95.9936829, 113.9485397, -192.5790100, 199.2054443
1: -65.3912277, 92.9626999, -79.5642395, 102.9904327, -168.3816528, 172.5269318
2: -91.6809158, 105.3891754, -111.4978180, 116.0074005, -207.6883240, 216.8869934
3: -64.0711365, 92.9304047, -69.3579865, 110.5664749, -174.6376038, 162.2883759
4: -99.6445236, 111.7989883, -120.9444962, 122.6787872, -222.3232880, 232.7434692

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -91.9630280, 110.9765549, -95.9936829, 113.9485397, -205.9115601, 206.9702454
1: -76.2160187, 100.2644730, -79.5642395, 102.9904327, -179.2064514, 179.8287048
2: -106.8935013, 113.1315613, -111.4978180, 116.0074005, -222.9009094, 224.6293793
3: -68.0785675, 106.5302124, -69.3579865, 110.5664749, -178.6450500, 175.8881989
4: -116.0856934, 119.8276749, -120.9444962, 122.6787872, -238.7644653, 240.7721558

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -78.9156494, 103.2667618, -98.7661514, 114.9155502, -193.8312073, 202.0329132
1: -65.4742050, 92.8440399, -81.6709747, 104.1926575, -169.6668549, 174.5150146
2: -91.7911377, 105.4159927, -114.5949554, 117.0547256, -208.8458557, 220.0109406
3: -64.3398361, 92.0656891, -70.1461182, 113.0315857, -177.3714294, 162.2117920
4: -99.9847870, 111.7970657, -124.6522980, 123.7725372, -223.7572937, 236.4493713

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6917133, upper bound: 190.6399897
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6917148, upper bound: 190.8145539
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -94.0209885, 111.9910202, -98.7661514, 114.9155502, -208.9365387, 210.7571716
1: -77.7521286, 101.0731201, -81.6709747, 104.1926575, -181.9447937, 182.7440948
2: -109.0427399, 114.1736069, -114.5949554, 117.0547256, -226.0974121, 228.7685547
3: -68.8223877, 107.5189819, -70.1461182, 113.0315857, -181.8539734, 177.6650848
4: -118.6816559, 120.7862167, -124.6522980, 123.7725372, -242.4541931, 245.4385071

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6917133, upper bound: 190.6399897
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6917148, upper bound: 190.8145539
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -78.9156494, 103.2667618, -97.4446030, 114.2804108, -193.1960602, 200.7113342
1: -65.4742050, 92.8440399, -80.6750259, 103.3011398, -168.7753448, 173.5190735
2: -91.7911377, 105.4159927, -112.9487915, 116.3551331, -208.1462708, 218.3647766
3: -64.3398361, 92.0656891, -69.5654144, 111.2999115, -175.6397400, 161.6310883
4: -99.9847870, 111.7970657, -122.5433197, 123.0025177, -222.9873047, 234.3403931

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8508666, upper bound: 185.1954047
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8508675, upper bound: 185.1954049
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -94.0209885, 111.9910202, -97.4446030, 114.2804108, -208.3013916, 209.4356232
1: -77.7521286, 101.0731201, -80.6750259, 103.3011398, -181.0532684, 181.7481384
2: -109.0427399, 114.1736069, -112.9487915, 116.3551331, -225.3978271, 227.1224060
3: -68.8223877, 107.5189819, -69.5654144, 111.2999115, -180.1222992, 177.0843658
4: -118.6816559, 120.7862167, -122.5433197, 123.0025177, -241.6841583, 243.3295135

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8508666, upper bound: 185.1954047
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8508675, upper bound: 185.1954049
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -78.4691620, 103.4701385, -82.0610886, 105.3424301, -183.8115845, 185.5312195
1: -65.3832474, 93.1269760, -68.1042328, 94.9625549, -160.3457947, 161.2312012
2: -91.4410248, 105.5574799, -95.5750885, 107.4659119, -198.9069366, 201.1325531
3: -63.9262199, 92.4040985, -65.2240982, 96.2926483, -160.2188416, 157.6281586
4: -99.1128998, 111.9080811, -103.9902267, 113.9467926, -213.0596924, 215.8982544

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4766841, upper bound: 187.3748861
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4766841, upper bound: 187.3748861
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -90.3582993, 110.3141403, -82.0610886, 105.3424301, -195.7007294, 192.3752289
1: -75.0085526, 99.5621262, -68.1042328, 94.9625549, -169.9710846, 167.6663513
2: -104.9821396, 112.4037552, -95.5750885, 107.4659119, -212.4480591, 207.9788513
3: -67.5116730, 104.3866425, -65.2240982, 96.2926483, -163.8042908, 169.6107178
4: -113.7537079, 119.0166550, -103.9902267, 113.9467926, -227.7004852, 223.0068512

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4766841, upper bound: 187.3748825
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4766843, upper bound: 187.3748861
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -78.4691620, 103.4701385, -81.0372849, 105.0178909, -183.4870605, 184.5074158
1: -65.3832474, 93.1269760, -67.3724594, 94.5705261, -159.9537659, 160.4994354
2: -91.4410248, 105.5574799, -94.3196259, 107.0446167, -198.4856415, 199.8771057
3: -63.9262199, 92.4040985, -64.7711792, 94.8270264, -158.7532501, 157.1752319
4: -99.1128998, 111.9080811, -102.3381653, 113.4438095, -212.5567017, 214.2462311

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -90.3582993, 110.3141403, -81.0372849, 105.0178909, -195.3761902, 191.3514252
1: -75.0085526, 99.5621262, -67.3724594, 94.5705261, -169.5790710, 166.9345856
2: -104.9821396, 112.4037552, -94.3196259, 107.0446167, -212.0267639, 206.7233887
3: -67.5116730, 104.3866425, -64.7711792, 94.8270264, -162.3386993, 169.1578064
4: -113.7537079, 119.0166550, -102.3381653, 113.4438095, -227.1975098, 221.3548126

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -77.8985825, 102.8410950, -83.6409683, 106.0165482, -183.9151306, 186.4820557
1: -64.7397842, 92.3455353, -69.3598175, 95.5994415, -160.3392181, 161.7053223
2: -90.4897461, 104.8518677, -97.2918549, 108.1578751, -198.6476135, 202.1436920
3: -63.8641090, 90.5127716, -65.6236954, 97.5932922, -161.4573975, 156.1364746
4: -98.2341003, 111.2121277, -105.8965683, 114.6326447, -212.8667450, 217.1087036

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.9302808, upper bound: 181.0668038
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.9273405, upper bound: 181.0556650
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -91.8061829, 110.6801300, -83.6409683, 106.0165482, -197.8227234, 194.3211060
1: -76.0213242, 99.7979431, -69.3598175, 95.5994415, -171.6207275, 169.1577606
2: -106.3325272, 112.7706985, -97.2918549, 108.1578751, -214.4903870, 210.0625610
3: -67.9105835, 104.4501953, -65.6236954, 97.5932922, -165.5038757, 170.0738831
4: -115.3986740, 119.3196564, -105.8965683, 114.6326447, -230.0312958, 225.2162170

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.9302808, upper bound: 181.0668038
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.9273433, upper bound: 181.0556800
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -77.8985825, 102.8410950, -82.1276627, 105.3648529, -183.2634125, 184.9687500
1: -64.7397842, 92.3455353, -68.2199402, 94.8961868, -159.6359711, 160.5654449
2: -90.4897461, 104.8518677, -95.4157639, 107.3954773, -197.8852234, 200.2676086
3: -63.8641090, 90.5127716, -65.0046082, 95.5868912, -159.4509888, 155.5173798
4: -98.2341003, 111.2121277, -103.5401535, 113.8027725, -212.0368652, 214.7522736

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.8552974, upper bound: 180.5367489
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.7295280, upper bound: 179.6840441
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -91.8061829, 110.6801300, -82.1276627, 105.3648529, -197.1710205, 192.8078003
1: -76.0213242, 99.7979431, -68.2199402, 94.8961868, -170.9174805, 168.0178680
2: -106.3325272, 112.7706985, -95.4157639, 107.3954773, -213.7279968, 208.1864624
3: -67.9105835, 104.4501953, -65.0046082, 95.5868912, -163.4974670, 169.4548035
4: -115.3986740, 119.3196564, -103.5401535, 113.8027725, -229.2014465, 222.8598022

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.8552974, upper bound: 180.5367489
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.7295280, upper bound: 179.6840441
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -78.4691620, 103.4701385, -93.3521652, 112.9497070, -191.4188690, 196.8222961
1: -65.3832474, 93.1269760, -77.4321442, 102.1221848, -167.5054321, 170.5591125
2: -91.4410248, 105.5574799, -108.5130081, 115.0706940, -206.5117188, 214.0704956
3: -63.9262199, 92.4040985, -68.7948303, 107.9435425, -171.8697662, 161.1988983
4: -99.1128998, 111.9080811, -117.6463470, 121.9186478, -221.0315552, 229.5543976

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1622562, upper bound: 185.4343195
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1622562, upper bound: 185.4343195
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -77.8985825, 102.8410950, -93.3521652, 112.9497070, -190.8482971, 196.1932678
1: -64.7397842, 92.3455353, -77.4321442, 102.1221848, -166.8619690, 169.7776794
2: -90.4897461, 104.8518677, -108.5130081, 115.0706940, -205.5604401, 213.3648529
3: -63.8641090, 90.5127716, -68.7948303, 107.9435425, -171.8076477, 159.3075867
4: -98.2341003, 111.2121277, -117.6463470, 121.9186478, -220.1527405, 228.8584747

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1622562, upper bound: 185.4343195
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1622564, upper bound: 185.4343195
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -77.5166702, 103.0111008, -91.7879944, 111.8798828, -189.3965454, 194.7991028
1: -64.4592285, 92.5544739, -75.9953384, 100.9152451, -165.3744812, 168.5498047
2: -90.1261292, 105.0026550, -106.4514160, 113.9817581, -204.1078796, 211.4540710
3: -63.8193588, 90.6829224, -68.4920578, 104.8522949, -168.6716614, 159.1749878
4: -97.8379517, 111.3594055, -115.6609268, 120.7251892, -218.5631409, 227.0203247

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8530616, upper bound: 179.0995434
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8530616, upper bound: 179.0995434
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -77.6817398, 103.7266083, -88.9362869, 110.2420197, -187.9237671, 192.6629028
1: -64.6789856, 93.3507996, -73.6263809, 99.3425446, -164.0214844, 166.9771576
2: -90.5085526, 105.6324921, -103.1387787, 112.3660126, -202.8745422, 208.7712708
3: -63.9413872, 91.5904388, -67.7007828, 102.0129547, -165.9543457, 159.2912140
4: -98.1209641, 112.0538025, -112.1567383, 119.0524521, -217.1734009, 224.2105408

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.3436480, upper bound: 179.0416369
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.3436480, upper bound: 179.0416369
time: 0.77 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.94 seconds
IS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.6727376, upper bound: 190.7253771
IS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.6727373, upper bound: 190.7667056
IS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.6727376, upper bound: 190.7253771
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.6727373, upper bound: 190.7667056
IS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
IS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.9019678, upper bound: 186.1745118
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.9019678, upper bound: 186.1745118
IS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.7422150, upper bound: 190.7731926
IS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.8156361, upper bound: 190.9288452
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.7422150, upper bound: 190.7731926
IS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.8156361, upper bound: 190.9288452
IS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -183.0466610, upper bound: 181.5898125
IS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -182.0470103, upper bound: 180.1373937
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -183.0466610, upper bound: 181.5898125
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -182.0470103, upper bound: 180.1373937
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.6727376, upper bound: 190.6405225
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.6727376, upper bound: 190.6906324
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.6727376, upper bound: 190.6405774
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.6727376, upper bound: 190.6906324
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
IS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.6917133, upper bound: 190.6399897
IS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.6917148, upper bound: 190.8145539
IS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.6917133, upper bound: 190.6399897
IS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -190.6917148, upper bound: 190.8145539
IS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.8508666, upper bound: 185.1954047
IS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.8508675, upper bound: 185.1954049
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.8508666, upper bound: 185.1954047
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.8508675, upper bound: 185.1954049
IS_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -185.4766841, upper bound: 187.3748861
IS_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -185.4766841, upper bound: 187.3748861
IS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -185.4766841, upper bound: 187.3748825
IS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -185.4766843, upper bound: 187.3748861
IS_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
IS_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
IS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -178.9302808, upper bound: 181.0668038
IS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -178.9273405, upper bound: 181.0556650
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -178.9302808, upper bound: 181.0668038
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -178.9273433, upper bound: 181.0556800
IS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -178.8552974, upper bound: 180.5367489
IS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -178.7295280, upper bound: 179.6840441
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -178.8552974, upper bound: 180.5367489
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -178.7295280, upper bound: 179.6840441
IS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.1622562, upper bound: 185.4343195
IS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.1622562, upper bound: 185.4343195
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.1622562, upper bound: 185.4343195
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -186.1622564, upper bound: 185.4343195
IS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -181.8530616, upper bound: 179.0995434
IS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -181.8530616, upper bound: 179.0995434
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -181.3436480, upper bound: 179.0416369
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 0, lower bound: -181.3436480, upper bound: 179.0416369

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -78.6304703, 103.2117615, -181.8422241, 181.8422241
1: -65.3912277, 92.9626999, -65.3912277, 92.9626999, -158.3539124, 158.3539276
2: -91.6809158, 105.3891754, -91.6809158, 105.3891754, -197.0700989, 197.0700989
3: -64.0711365, 92.9304047, -64.0711365, 92.9304047, -157.0015259, 157.0015259
4: -99.6445236, 111.7989883, -99.6445236, 111.7989883, -211.4434662, 211.4434662

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.2921014, upper bound: 188.0556386
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.9471810, upper bound: 187.9471810
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -78.6304703, 103.2117615, -78.9156494, 103.2667618, -181.8972321, 182.1274109
1: -65.3912277, 92.9626999, -65.4742050, 92.8440399, -158.2352600, 158.4368896
2: -91.6809158, 105.3891754, -91.7911377, 105.4159927, -197.0969086, 197.1803131
3: -64.0711365, 92.9304047, -64.3398361, 92.0656891, -156.1367950, 157.2702332
4: -99.6445236, 111.7989883, -99.9847870, 111.7970657, -211.4415741, 211.7837524

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.2921014, upper bound: 190.7723243
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.9471810, upper bound: 188.1021125
time: 0.60 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 6.39 seconds
IS_A2_B2_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 6.39
Output dim: 0, lower bound: -190.2921014, upper bound: 188.0556386
IS_A2_B2_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 6.39
Output dim: 0, lower bound: -187.9471810, upper bound: 187.9471810
IS_A2_B2_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 6.39
Output dim: 0, lower bound: -190.2921014, upper bound: 190.7723243
IS_A2_B2_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 6.39
Output dim: 0, lower bound: -187.9471810, upper bound: 188.1021125
IS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -190.6727376, upper bound: 190.7253771
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -190.6727373, upper bound: 190.7667056
IS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
IS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.9019678, upper bound: 186.1745118
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.9019680, upper bound: 186.1745118
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.9019678, upper bound: 186.1745118
IS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -190.7422150, upper bound: 190.7731926
IS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -190.8156361, upper bound: 190.9288452
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -190.7422150, upper bound: 190.7731926
IS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -190.8156361, upper bound: 190.9288452
IS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -183.0466610, upper bound: 181.5898125
IS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -182.0470103, upper bound: 180.1373937
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -183.0466610, upper bound: 181.5898125
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -182.0470103, upper bound: 180.1373937
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -190.6727376, upper bound: 190.6405225
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -190.6727376, upper bound: 190.6906324
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -190.6727376, upper bound: 190.6405774
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -190.6727376, upper bound: 190.6906324
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.8747459, upper bound: 185.1759076
IS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -190.6917133, upper bound: 190.6399897
IS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -190.6917148, upper bound: 190.8145539
IS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -190.6917133, upper bound: 190.6399897
IS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -190.6917148, upper bound: 190.8145539
IS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.8508666, upper bound: 185.1954047
IS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.8508675, upper bound: 185.1954049
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.8508666, upper bound: 185.1954047
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.8508675, upper bound: 185.1954049
IS_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -185.4766841, upper bound: 187.3748861
IS_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -185.4766841, upper bound: 187.3748861
IS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -185.4766841, upper bound: 187.3748825
IS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -185.4766843, upper bound: 187.3748861
IS_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
IS_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -185.4348543, upper bound: 186.1652154
IS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -178.9302808, upper bound: 181.0668038
IS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -178.9273405, upper bound: 181.0556650
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -178.9302808, upper bound: 181.0668038
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -178.9273433, upper bound: 181.0556800
IS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -178.8552974, upper bound: 180.5367489
IS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -178.7295280, upper bound: 179.6840441
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -178.8552974, upper bound: 180.5367489
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -178.7295280, upper bound: 179.6840441
IS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.1622562, upper bound: 185.4343195
IS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.1622562, upper bound: 185.4343195
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.1622562, upper bound: 185.4343195
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -186.1622564, upper bound: 185.4343195
IS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -181.8530616, upper bound: 179.0995434
IS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -181.8530616, upper bound: 179.0995434
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -181.3436480, upper bound: 179.0416369
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 0, lower bound: -181.3436480, upper bound: 179.0416369
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=279.9653625488281
rel_dist={0: [-191.43286165766193, 191.43286165766187]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1127.95 seconds
